"""The run loop: one thread driving the whole pipeline for one run.

COUNT MODE — per frame: ingest /frame -> persons /detect -> tracker /track ->
faces /detect (within the tracked boxes) -> exclusion zones (operator-drawn
no-count polygons; see :meth:`RunLoop._apply_zones`) -> local quality gate ->
embed /embed -> match /match per face -> unique count, with the TRACK HEAL
(:meth:`RunLoop._maybe_heal`) folding back a mint the same track disowns
moments later.  Every stage is timed; aggregates and
sampled raw rows flush to the planner every ``flush_interval_s`` seconds.  On
top of that (CONTRACTS.md v1): annotated debug frames + structured taps go up
every ``tap_interval_s``, and operator feedback is polled every
``feedback_poll_s`` and applied to the live gallery (merge / split / mark-staff
/ missed), adjusting the unique count.  The matcher checks the site STAFF store
first when the run carries a ``siteId``: a staff hit is counted as a staff
crossing and kept out of the guest unique count, but the track stays visible
upstream.

**THE COUNT IS THE PRODUCT; reporting is secondary.**  Every planner call made
from inside the frame loop is best-effort in both senses — errors are
swallowed AND latency is bounded (short-timeout transports, plus a whole-round
budget on taps).  A planner restart mid-event used to raise out of the stats
flush, fail the run and stop counting; restarting then gave the run a fresh
planner id and therefore a fresh EMPTY gallery, so every guest already counted
was counted again.  A five-second laptop hiccup corrupted the event total.  It
must cost nothing but a gap in the charts, which the next upsert repairs.

ENROL MODE — a professional-enrolment walk-through: capture faces, keep the
best ``enrol_best_n`` by quality, write them to the site staff store, and PUT
the sample count back to the planner.  No counting, but it DOES open a
``mode:'enrol'`` pipeline run: erasure has a permanent ledger
(``staff_tombstones``) and the capture that creates the biometric had none.

End-of-source: ingest serves the LATEST frame with a monotonically increasing
``seq`` and never signals EOF explicitly — a **stalled seq** is the signal
(see services/ingest).  The loop also accepts a stub-friendly explicit end
(``{"ended": true}``, missing ``imageB64``, or HTTP 204/404/410).  RTSP
sources stall only on network loss; stop those via POST /runs/:id/stop.

End-of-run: per-run state downstream has an owner and is HANDED BACK — the
ingest capture slot, the tracker's SortLite, and the gallery file (which holds
real guests' face embeddings, so leaving it behind is a retention liability,
not just disk).  See :meth:`RunLoop._release_run_state`.
"""

import contextlib
import threading
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
from heco_common.geometry import dedupe_boxes, point_in_polygon
from heco_common.imaging import decode_jpeg_b64
from heco_common.logs import RunLog, safe, setup_logging
from heco_common.planner import FileTransport, PlannerClient, PlannerError, Transport
from heco_common.schemas import Sample

from . import annotate, gate, taps
from .config import Settings
from .feedback import plan_action
from .stats import SampleBuffer, StatsBoard


def source_label(source: dict) -> str:
    """A human label for a source that NEVER carries credentials.

    The planner slugs this label into the run row's PERMANENT id, so a
    userinfo'd RTSP URL here would bake the camera password into every run
    URL and export forever (the planner redacts the stored config, but the
    label reaches it first). URLs reduce to scheme://host[:port]/path;
    file sources pass through as their path.
    """
    url = (source or {}).get("url")
    if url:
        p = urlsplit(str(url))
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://{host}{port}{p.path}"
    path = (source or {}).get("path")
    return str(path) if path else "unknown source"


class StageError(RuntimeError):
    """A pipeline stage refused a call; carries its status and own message.

    ``raise_for_status`` alone would report only the status code, which turns
    a deliberate, explanatory refusal (ingest's 409 "the capture slot is held
    by run X") into an opaque failure the operator cannot act on.  The status
    is kept because the two families mean opposite things to a retry: 4xx is
    "understood you, no" (settle it), 5xx is "try me again".
    """

    def __init__(self, message: str, status_code: int = 0) -> None:
        """Record the message and the HTTP status the stage answered with."""
        super().__init__(message)
        self.status_code = status_code


def httpx_transport(client: httpx.Client) -> Transport:
    """Adapt an httpx client to the PlannerClient transport callable.

    One adapter, several clients: the runner builds a long-timeout client for
    stage calls and short-timeout ones for planner traffic (see RunManager),
    and tests route everything through one httpx.MockTransport.
    """

    def transport(method: str, url: str, payload: dict | None) -> tuple[int, dict]:
        r = client.request(method, url, json=payload)
        return r.status_code, (r.json() if r.content else {})

    return transport


def httpx_file_transport(client: httpx.Client) -> FileTransport:
    """Adapt an httpx client to the PlannerClient multipart file transport.

    Debug frames are ``multipart/form-data`` (a ``stage`` field + the JPEG
    ``file``), so they need their own adapter separate from the JSON transport.
    """

    def transport(
        url: str, fields: dict, filename: str, content: bytes, content_type: str
    ) -> tuple[int, dict]:
        r = client.post(url, data=fields, files={"file": (filename, content, content_type)})
        return r.status_code, (r.json() if r.content else {})

    return transport


def _now_iso() -> str:
    """ISO-8601 UTC timestamp for planner reports (e.g. enrolledAt)."""
    return datetime.now(UTC).isoformat()


def enrol_score(face: dict) -> float:
    """Rank one candidate enrolment face; higher is a better template.

    Box width was the old ranking and it is the wrong axis: it rewards whoever
    stands closest to the camera, not whoever is most recognisable.  The faces
    service already emits the two signals recognition actually depends on —
    ``iedPx`` (inter-eye distance, the size measure FR standards use) and
    ``frontality`` (0..1 from how centred the nose sits between the eyes) — and
    their product prefers a large, square-on face over a large, side-on one.
    Width is the fallback for a detector reply without landmarks; the faces
    service emits both signals for every face or neither, so a shortlist is
    never ranked on mixed scales.
    """
    if "iedPx" in face:
        return float(face["iedPx"]) * float(face.get("frontality", 1.0))
    return float(face["box"]["w"])


class RunLoop:
    """Owns one run: the pipeline loop, its stats, and its lifecycle."""

    def __init__(
        self,
        run_id: str,
        request: dict,
        settings: Settings,
        client: httpx.Client,
        planner: PlannerClient,
        is_live_run=None,
    ) -> None:
        """Prepare a run; `request` is the validated POST /runs body as a dict.

        `client` is injected so tests can pass an httpx.MockTransport-backed
        client and drive the loop against a fully fake pipeline; `planner`
        should speak through the same client (see httpx_transport).
        ``is_live_run`` (run_id -> bool) is the registry's LIVENESS test, used
        only to tell a STALE capture-slot holder (a corpse from a killed runner
        process, or a sibling that settled hours ago) from a run still counting
        — see _open_source.  It asks "is that run still going?", never "have I
        ever heard of it": a settled sibling that never managed its /close used
        to hold the camera for the life of the process.
        """
        self.run_id = run_id
        self.request = request
        self.s = settings
        self.client = client
        self.planner = planner
        self._is_live_run = is_live_run
        # Resolved once: the gate's floors cannot change under a running run,
        # and re-reading them per face would let a config reload move the
        # discard threshold half way through a count.
        self.gate = gate.GateThresholds.from_settings(settings)
        self.board = StatsBoard()
        self.samples = SampleBuffer(cap=settings.sample_batch_max)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_seq: int | None = None
        self._last_ms: float = 0.0
        # personKey -> how many templates that identity last reported holding.
        # Guarded by _lock, like _status, because the settle path reads it.
        self._template_n: dict[str, int] = {}
        # Operator-drawn no-count polygons, validated at the API edge
        # (app.main.ExclusionZone): [{label, points: [[x,y] normalized 0..1]}].
        self.zones: list[dict] = list(request.get("exclusionZones") or [])
        # TRACK-HEAL bookkeeping: track_id -> {key, cosine, at} recorded when a
        # verdict on that track minted a NEW identity (isNew=true).  Guarded by
        # _lock like _template_n.  Entries older than heal_window_s are pruned
        # on every mint, so the dict is bounded by the tracks active within one
        # window, not by the night's traffic.
        self._minted: dict[int, dict] = {}
        # Snapshot of the frame just processed, for debug taps (loop-thread only).
        self._last: dict | None = None
        self._last_tap: float = 0.0
        self._last_feedback: float = 0.0
        self._last_purge: float = 0.0
        # Why the frame loop stopped; decides how the run settles and
        # whether this run's embeddings are safe to delete.
        self._end_reason: str = "source-ended"
        # Every line this run writes carries its id, so a night running
        # several gates can be read back one run at a time.
        self.log = RunLog(setup_logging("runner"), run_id)
        self._handled_feedback: set = set()
        # fid -> outcome we already carried out but could not tell the planner
        # about.  Without this a dropped PUT made the next poll re-execute an
        # already-applied correction, which then failed (the key is gone) and
        # was recorded as REJECTED next to a count that had visibly changed.
        self._settled_feedback: dict = {}
        # staffId -> monotonic time last seen, for crossing debounce.
        self._staff_last_seen: dict = {}
        self._status: dict = {
            "runId": run_id,
            "plannerRunId": None,
            "mode": request.get("mode", "count"),
            "siteId": request.get("siteId"),
            "state": "starting",
            "frames": 0,
            "unique": 0,
            "matches": 0,
            "staffCrossings": 0,
            "staffFaceFrames": 0,
            "subCanonMatches": 0,
            "subCanonShare": 0.0,
            # Is multi-template actually doing anything?  Without these the
            # only way to answer was to open the gallery by hand.  A run where
            # every guest ends on ONE template means the crossings never
            # supplied a second usable view, and no threshold change will fix
            # that; a healthy run shows most guests holding several.
            "templatesEnrolled": 0,
            "singleTemplateGuests": 0,
            "templateNMax": 0,
            # TRACK HEAL: mints this run folded back after the same track
            # matched a different existing identity comfortably (see
            # _maybe_heal).  Each one is a guest the delivered number would
            # otherwise have over-counted by exactly one.
            "healedSplits": 0,
            # EXCLUSION ZONES.  excludedByZone counts faces whose box centre
            # fell inside an operator-drawn no-count polygon — dropped before
            # the gate, so they are never embedded or matched, but kept
            # visible in the face-detect tap so the operator can see what the
            # zone is eating.  zoneUnmeasured mirrors the gatedUnmeasured
            # honesty pattern: faces that passed ARMED zones untested because
            # the frame carried no dimensions to scale the polygons by.
            "excludedByZone": 0,
            "zoneUnmeasured": 0,
            "feedbackApplied": 0,
            "feedbackRejected": 0,
            "manualAdditions": 0,
            "staffPurged": 0,
            "plannerAuthFailures": 0,
            "sampleCount": 0,
            "samplesDropped": 0,
            "multiFaceFramesSkipped": 0,
            "plannerReportErrors": 0,
            "tapRoundsAbandoned": 0,
            # Whole-run gate accounting, one counter per reason a face was
            # discarded.  A gate that only says "N faces dropped" cannot be
            # audited: an operator who armed an IED floor and lost a third of a
            # wedding needs to see WHICH floor did it, and
            # ``gatedUnmeasured`` says how many faces passed an armed floor
            # only because nobody could measure them.
            "gatedByWidth": 0,
            "gatedByIed": 0,
            "gatedByFrontality": 0,
            "gatedBySharpness": 0,
            "gatedUnmeasured": 0,
            # The floors this run is actually enforcing beyond width, so the
            # status says what gate produced the number, not just the number.
            # A tuple, because status() hands out a SHALLOW copy of this dict.
            "gateArmed": self.gate.armed,
            "error": None,
        }

    # ------------------------------------------------------------- lifecycle

    def stop(self) -> None:
        """Ask the loop to end after the current frame (idempotent)."""
        self._stop.set()

    def status(self) -> dict:
        """Return a copy of the live status (safe from any thread)."""
        with self._lock:
            return dict(self._status)

    def _set(self, **kv) -> None:
        with self._lock:
            self._status.update(kv)

    def _bump(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._status[key] = self._status.get(key, 0) + by

    # ------------------------------------------------------------------ HTTP

    def _post(self, url: str, body: dict) -> dict:
        """POST to a stage service, raising a StageError that quotes its reply."""
        r = self.client.post(url, json=body)
        if r.status_code >= 400:
            raise StageError(
                f"POST {url} -> HTTP {r.status_code}: {_detail(r)}", r.status_code
            )
        return r.json() if r.content else {}

    def _next_frame(self) -> dict | None:
        """Poll ingest for a frame with a NEW seq; None once the source stops.

        Sets ``self._end_reason`` to WHY it stopped, because the three cases
        are not the same event and must not be settled the same way:

        * ``"source-ended"`` — ingest said so (204/404/410, ``ended``, or no
          image). A file finished. The count is complete.
        * ``"operator-stopped"`` — the stop flag. The count is complete as far
          as anybody asked it to go.
        * ``"source-stalled"`` — the frame sequence stopped advancing for
          ``source_stall_s``. **This is not an end.** It is a camera that
          blinked: a Wi-Fi dropout, a PoE bounce, an RTSP reconnect. Ingest is
          still retrying in the background.

        Collapsing the third into the first is how a five-second network blip
        at a wedding used to close the run as "ended" and delete the gallery,
        after which restarting re-counted every guest already through the gate.
        """
        deadline = time.monotonic() + self.s.source_stall_s
        while not self._stop.is_set() and time.monotonic() < deadline:
            r = self.client.get(f"{self.s.ingest_url}/frame")
            if r.status_code in (204, 404, 410):
                self._end_reason = "source-ended"  # stub-style explicit end
                return None
            if r.status_code == 503:  # no frame decoded yet — retry
                time.sleep(self.s.source_poll_s)
                continue
            r.raise_for_status()
            body = r.json()
            if body.get("ended") or not body.get("imageB64"):
                self._end_reason = "source-ended"
                return None
            seq = body.get("seq")
            if seq is None or seq != self._last_seq:
                self._last_seq = seq
                return body
            time.sleep(self.s.source_poll_s)  # same frame again — source idle
        self._end_reason = "operator-stopped" if self._stop.is_set() else "source-stalled"
        return None

    def _open_source(self) -> None:
        """Claim ingest's exclusive capture slot for this run.

        The slot is single: without an owner a second run silently replaced a
        live run's camera and both loops read the wrong frames.  A 409 names
        the holder; whether to fail depends on whether that run is STILL
        COUNTING:

        - a run this runner has LIVE -> a sibling really is using the camera;
          fail loudly, exactly as before (seizing it would corrupt that run's
          count);
        - anything else -> a corpse.  Either a killed runner process (which
          never sends /close, so its slot outlives it), or a run of THIS
          process that settled without its /close landing — ingest was down or
          restarting at that moment.  Both used to 409 every later start until
          someone restarted ingest by hand, while the error message told the
          operator to stop a run that had already stopped.  Seize it with
          takeover=true and note the seizure in the run status.
        """
        body = {**self.request["source"], "owner": self.run_id}
        try:
            self._post(f"{self.s.ingest_url}/open", body)
            return
        except StageError as e:
            if e.status_code != 409 or self._is_live_run is None:
                raise
        holder = self._slot_holder()
        if not holder or self._is_live_run(holder):
            raise StageError(
                f"capture slot is held by live run '{holder}' on this runner — "
                "stop that run before starting another on the same camera",
                409,
            )
        self.log.warning(
            f"seized the capture slot from stale run '{holder}' — if that run is "
            "actually alive, two runs are now fighting for one camera"
        )
        self._post(f"{self.s.ingest_url}/open", {**body, "takeover": True})
        self._set(seizedSlotFrom=holder)

    def _slot_holder(self) -> str | None:
        """Who ingest says owns the capture slot right now (its /health)."""
        try:
            r = self.client.get(f"{self.s.ingest_url}/health")
            return (r.json() or {}).get("owner") if r.status_code < 400 else None
        except Exception as e:  # noqa: BLE001 — a health probe must not mask the 409
            self.log.warning(f"could not ask ingest who holds the slot: {e}")
            return None

    # ------------------------------------------------------------- the loop

    def run(self) -> dict:
        """Execute the whole run (count or enrol); returns the final status."""
        try:
            if self.request.get("mode") == "enrol":
                self._run_enrol()
            else:
                self._run()
        except Exception as e:  # noqa: BLE001 — a run must always settle
            self.log.error(f"run failed: {type(e).__name__}: {e}")
            self._set(state="failed", error=f"{type(e).__name__}: {e}")
            self._release_run_state()
            if self.planner.run_id is not None:
                # Best effort: the planner may be down too; local status
                # already says failed either way.
                with contextlib.suppress(Exception):
                    # safe(): a stage's reply can quote the source URL, and
                    # these notes are permanent, served to the browser, and
                    # exported. Belt and braces with ingest's own redaction.
                    self.planner.end_run(status="failed", notes=safe(str(e)))
        return self.status()

    def _run(self) -> None:
        req = self.request
        label = req.get("label") or f"runner {source_label(req['source'])}"
        planner_run_id = str(
            self.planner.create_run(
                req["eventId"],
                placement_id=req.get("placementId"),
                label=label,
                config={
                    "source": req["source"],
                    "mode": "count",
                    "siteId": req.get("siteId"),
                    **self._gate_config(),
                    "geometry": "poc-2.8mm-2.0m-close-zone",  # CONTRACTS.md POC geometry
                },
            )
        )
        self._set(plannerRunId=planner_run_id, state="running")

        # Fresh per-run state downstream, then open the source.
        self._post(f"{self.s.match_url}/reset", {"runId": planner_run_id})
        self.log.info(
            f"counting from {source_label(self.request['source'])} "
            f"(plannerRun={planner_run_id}, site={self.request.get('siteId') or 'none'})"
        )
        # Honour any pending consent withdrawals BEFORE the first frame is
        # matched: purge tombstoned staff from the site store now, so a member
        # deleted while the pipeline was off never matches again.
        self._maybe_purge_staff()  # before the first frame is matched
        self._post(f"{self.s.tracker_url}/reset", {"runId": planner_run_id})
        self._open_source()

        t0 = time.monotonic()
        last_flush = t0
        self._last_tap = t0
        self._last_feedback = t0
        self._last_purge = t0
        frames = 0

        while not self._stop.is_set():
            frame = self._timed("ingest", "ingestMs", self._next_frame)
            if frame is None:
                break
            self.board.frame("ingest")
            frames += 1
            t_ms = int(frame.get("tMs", (time.monotonic() - t0) * 1000.0))

            self._pipeline_step(planner_run_id, frame, t_ms)
            self._set(frames=frames)

            now = time.monotonic()
            if now - last_flush >= self.s.flush_interval_s:
                self._flush(now - t0)
                last_flush = now
            self._maybe_tap(now)  # best-effort debug frames + payloads
            self._maybe_feedback(now)  # best-effort operator corrections
            self._maybe_purge_staff(now)  # best-effort consent-withdrawal purge

        self._flush(max(time.monotonic() - t0, 1e-9))
        st = self.status()
        notes = (
            f"unique={st['unique']} frames={frames} matches={st['matches']} "
            f"staffCrossings={st['staffCrossings']} "
            f"staffFaceFrames={st['staffFaceFrames']} "
            f"manualAdditions={st['manualAdditions']} "
            f"subCanonShare={st['subCanonShare']:.2f} "
            # Read this beside `unique`. Guests stuck on one template mean the
            # crossings supplied no second view, so the gallery is effectively
            # pre-M1 and the split identities are NOT a threshold problem.
            f"singleTemplateGuests={st['singleTemplateGuests']}/{st['unique']} "
            f"templatesEnrolled={st['templatesEnrolled']} "
            # Two corrections the loop applied ITSELF, so a report reading
            # `unique` months later can see how much of the number the heal
            # and the operator's zones shaped (CONTRACTS.md heal + zones).
            f"healedSplits={st['healedSplits']} "
            f"excludedByZone={st['excludedByZone']} "
            f"(POC geometry: 2.8mm @2.0m close-zone, faces ~64-85px, floor 56px)"
        )
        # A STALL IS NOT AN END. If the camera merely blinked, this run's
        # embeddings are the only record of who has already come through the
        # gate — deleting them means a restart counts every one of those guests
        # a second time. So the gallery survives a stall, and the run settles
        # as `failed` with the reason attached: a run that stopped because the
        # camera vanished must not present itself as a complete count.
        stalled = self._end_reason == "source-stalled"
        notes = f"{notes} endReason={self._end_reason}"
        self._release_run_state(keep_gallery=stalled)
        status = "failed" if stalled else "ended"
        results = {
            "unique": st["unique"],
            "staffCrossings": st["staffCrossings"],
            "manualAdditions": st["manualAdditions"],
            "frames": frames,
            "matches": st["matches"],
            "healedSplits": st["healedSplits"],
            "excludedByZone": st["excludedByZone"],
        }
        try:
            self.planner.end_run(
                status=status, notes=notes, results=results,
                end_reason=self._end_reason,
            )
        except PlannerError as e:  # the count happened; only the report is lost
            self.log.error(f"counted {st['unique']} but the planner was not told: {e}")
            self._bump("plannerReportErrors")
            self._set(error=f"run ended but the planner was not told: {e}")
        self.log.info(
            f"settled {status}: unique={st['unique']} frames={frames} "
            f"reason={self._end_reason}"
            + (" (gallery kept for a resume)" if stalled else "")
        )
        self._set(state=status, endReason=self._end_reason)

    def _gate_config(self) -> dict:
        """The gate this run enforced, for the run row's permanent config.

        The count is an invoice figure, so "which floors were in force when
        this number was produced" has to be recoverable from the record months
        later — not inferred from whatever the environment happens to hold at
        the time somebody asks.  Unarmed floors are recorded as 0.0 rather than
        omitted, so a config with no IED key means an OLD run, not an unarmed
        one.
        """
        return {
            "qualityMinPx": self.s.quality_min_px,
            "qualityCanonPx": self.s.quality_canon_px,
            "qualityMinIedPx": self.s.quality_min_ied_px,
            "qualityMinFrontality": self.s.quality_min_frontality,
            "qualityMinSharpness": self.s.quality_min_sharpness,
            "gateArmed": list(self.gate.armed),
        }

    def _release_run_state(self, keep_gallery: bool = False) -> None:
        """Give back every piece of per-run state this run created downstream.

        ``keep_gallery`` preserves this run's face embeddings — used when the
        source only STALLED, so that resuming does not re-count every guest
        already through the gate. The match service's /gallery/sweep remains
        the backstop that eventually reclaims it.

        Nothing here can fail the run — it has already produced its number —
        so each call is independently best-effort.  Three owners:

        * ingest's capture slot, so the next run can claim the camera;
        * the tracker's per-run SortLite, which otherwise stays resident until
          the process restarts (one leak per run, forever);
        * the gallery file, which holds this event's guests' face embeddings.
          A per-run file that nobody deletes is a retention liability; the
          match service's /gallery/sweep is only the backstop for runs that
          die before reaching here.
        """
        planner_run_id = self.status().get("plannerRunId")
        with contextlib.suppress(Exception):
            self._post(f"{self.s.ingest_url}/close", {"owner": self.run_id})
        if not planner_run_id or self.request.get("mode") == "enrol":
            # An enrolment now has a planner row (so the capture that creates a
            # biometric leaves a record) but it never created tracker state or
            # a gallery under that id, so there is nothing else of ours to hand
            # back.  Releasing them anyway would be a pair of pointless calls
            # against ids that were never opened.
            return
        with contextlib.suppress(Exception):
            self._post(f"{self.s.tracker_url}/release", {"runId": planner_run_id})
        if not keep_gallery:
            with contextlib.suppress(Exception):
                self._post(f"{self.s.match_url}/reset", {"runId": planner_run_id})

    def _pipeline_step(self, planner_run_id: str, frame: dict, t_ms: int) -> None:
        """Run stages 2..8 for one frame, timing and measuring each."""
        s, board, samples = self.s, self.board, self.samples
        image_b64 = frame["imageB64"]
        site_id = self.request.get("siteId")

        # person-detect
        persons = self._timed(
            "person-detect",
            "personDetectMs",
            lambda: self._post(f"{s.persons_url}/detect", {"imageB64": image_b64}),
        )
        board.frame("person-detect")
        boxes = persons.get("boxes", [])
        for b in boxes:
            board.observe("person-detect", "personBoxHPx", float(b["h"]))
            samples.add("person-detect", t_ms, {"personBoxHPx": float(b["h"])})

        # track (stateful per run — the tracker keys its state on runId)
        tracked = self._timed(
            "track",
            "trackMs",
            lambda: self._post(
                f"{s.tracker_url}/track",
                {"runId": planner_run_id, "tMs": t_ms, "boxes": boxes},
            ),
        )
        board.frame("track")
        tracks = tracked.get("tracks", [])
        board.observe("track", "tracksActive", float(len(tracks)))

        # Face-detect region = every RAW person box this frame, deduped against
        # the track boxes.
        #
        # The bug this replaced (finding D1): `[t["box"] for t in tracks] or
        # boxes` searched ONLY confirmed-track boxes once any track existed.
        # The tracker reports a track only after min_hits frames (sort.py
        # SortLite.step), so a guest who has just walked in is a detection with
        # no reported track — their face was never searched, and because the
        # count is match.isNew, a face never searched is a guest never counted.
        #
        # Raw boxes are therefore the recall-bearing half and go FIRST. The
        # track half is redundant while the tracker reports only misses == 0
        # tracks, whose boxes are snapped to the matched detection
        # (TrackState.update) and so duplicate a raw box. It is kept because
        # that invariant is the tracker's to change: if coasting tracks are
        # ever reported, their predicted boxes cover people the detector missed
        # this frame, which is real recall. Boxes-first ordering means a fresh
        # detection always wins the dedupe over a stale prediction.
        within = dedupe_boxes(
            list(boxes) + [t["box"] for t in tracks], iou_thr=0.6
        )
        faces_out = self._timed(
            "face-detect",
            "faceDetectMs",
            lambda: self._post(
                f"{s.faces_url}/detect", {"imageB64": image_b64, "within": within}
            ),
        )
        board.frame("face-detect")
        faces = faces_out.get("faces", [])
        for f in faces:
            board.observe("face-detect", "faceBoxWPx", float(f["box"]["w"]))
            sample = {"faceBoxWPx": float(f["box"]["w"])}
            # The signals the gate is composed from (accuracy R&D F1):
            # recognition size is honestly read in inter-eye distance, not box
            # width, and size says nothing about pose or focus.  They are
            # measured on every face whether or not their floors are armed, so
            # a floor can be priced from a run that was not gated on it.
            for key, metric in (
                ("iedPx", "faceIedPx"),
                ("frontality", "frontality"),
                ("sharpness", "sharpness"),
            ):
                if key in f:
                    board.observe("face-detect", metric, float(f[key]))
                    sample[metric] = float(f[key])
            samples.add("face-detect", t_ms, sample)

        # EXCLUSION ZONES — before the gate, so an excluded face is never
        # gated, embedded or matched.  The face stays in `faces` (stamped
        # excludedByZone) so the taps and the annotated frame can show the
        # operator exactly what their polygon is eating.
        countable = self._apply_zones(faces, frame)

        # QUALITY GATE (local) — the pipeline's only irreversible discard.
        # Width floor 56 px as always; the IED / frontality / sharpness floors
        # ride alongside it and are unarmed by default (see app/gate.py), so
        # this is today's behaviour until an operator opts in.  Every face is
        # stamped with WHY it was rejected, for the taps and the overlay.
        tq = time.perf_counter()
        outcome = gate.gate_faces(countable, self.gate)
        kept = outcome.kept
        board.frame("quality")
        board.observe("quality", "qualityMs", (time.perf_counter() - tq) * 1000.0)
        for f in kept:
            board.observe("quality", "faceBoxWPx", float(f["box"]["w"]))
        self._count_gated(outcome)

        if not kept:
            self._remember(image_b64, frame.get("seq"), t_ms, boxes, tracks, faces, [])
            self._count_stage()
            return

        # embed (only gate survivors — the expensive stage sees the fewest crops)
        embedded = self._timed(
            "embed",
            "embedMs",
            lambda: self._post(
                f"{s.embed_url}/embed", {"imageB64": image_b64, "faces": kept}
            ),
        )
        board.frame("embed")
        samples.add("embed", t_ms, {"embedMs": self._last_ms})
        embeddings = embedded.get("embeddings", [])
        # HOW MANY FACE CROPS ACTUALLY REACHED THE EMBEDDER — a funnel rung
        # nothing else reports. The stage's own counter counts FRAMES, and the
        # gate-survivor count is what was OFFERED, not what came back: the
        # zip(strict=False) below silently drops any shortfall if the embedder
        # returns fewer vectors than crops. Without this, an eval harness can
        # see faces pass the gate and matches happen but cannot say whether the
        # embed stage ran at all, which is precisely the rung that went to zero
        # when ingest was downscaled and the pipeline stopped counting anyone.
        # Width (not a new metric name) so the same key means the same thing at
        # face-detect, quality and embed, and the size of what was embedded is
        # priced alongside the size of what was seen.
        for face in kept[: len(embeddings)]:
            board.observe("embed", "faceBoxWPx", float(face["box"]["w"]))

        # match (one call per embedding; gallery keyed by the planner run id;
        # staff checked first when the run carries a siteId)
        verdicts: list[dict] = []
        for face, emb in zip(kept, embeddings, strict=False):
            w = float(face["box"]["w"])
            body = {"runId": planner_run_id, "embedding": emb, "quality": w}
            if site_id:
                body["siteId"] = site_id
            m = self._timed(
                "match", "matchMs", lambda b=body: self._post(f"{s.match_url}/match", b)
            )
            board.frame("match")
            if m.get("cosine") is not None:
                board.observe("match", "matchCosine", float(m["cosine"]))
                samples.add("match", t_ms, {"matchCosine": float(m["cosine"])})
            verdicts.append(
                {
                    "personKey": m.get("personKey"),
                    "cosine": m.get("cosine"),
                    "isNew": m.get("isNew"),
                    "isStaff": bool(m.get("isStaff", False)),
                    "subCanon": bool(m.get("subCanon", False)),
                    # How many views the identity holds, and whether this
                    # sighting became one.  A run's counts cannot be read
                    # without them: most guests holding a single template
                    # means the crossings are not supplying diverse views and
                    # multi-template is doing nothing, which is worth knowing
                    # BEFORE anyone blames the threshold.  None on a staff hit.
                    "templateN": m.get("templateN"),
                    "templateAdded": bool(m.get("templateAdded", False)),
                    "box": face.get("box"),
                }
            )
            if m.get("isStaff"):
                # A staff hit never records a mint and never heals INTO staff:
                # the staff store is operator-supervised evidence, and folding
                # a guest key into it from track behaviour would silently
                # shrink the guest count against the strongest identities in
                # the system (CONTRACTS.md heal scope).
                self._count_staff(m)
                continue
            with self._lock:
                st = self._status
                st["matches"] += 1
                if m.get("subCanon"):
                    st["subCanonMatches"] += 1
                if m.get("isNew"):
                    st["unique"] += 1
                st["subCanonShare"] = (
                    st["subCanonMatches"] / st["matches"] if st["matches"] else 0.0
                )
                if m.get("templateAdded"):
                    st["templatesEnrolled"] += 1
                self._note_templates(m.get("personKey"), m.get("templateN"))
            # TRACK HEAL bookkeeping: only verdicts that can be pinned to a
            # track participate (no containing track = no heal for this one).
            track_id = self._track_for(face, tracks)
            if track_id is None:
                continue
            if m.get("isNew"):
                self._record_mint(track_id, m)
            else:
                self._maybe_heal(planner_run_id, track_id, m)

        self._remember(image_b64, frame.get("seq"), t_ms, boxes, tracks, faces, verdicts)
        self._count_stage()

    #: gate reason -> the status counter that records it.
    _GATE_COUNTER = {
        "width": "gatedByWidth",
        "ied": "gatedByIed",
        "frontality": "gatedByFrontality",
        "sharpness": "gatedBySharpness",
    }

    def _count_gated(self, outcome) -> None:
        """Record this frame's rejections against their reasons.

        The gate is where a guest stops being countable, so the run status has
        to be able to answer "what did the gate cost us, and which floor did
        it".  One counter per reason rather than a nested dict because
        ``status()`` hands out a shallow copy and a nested dict would be shared
        across threads.
        """
        if not outcome.gated_by and not outcome.unmeasured:
            return
        with self._lock:
            for reason, n in outcome.gated_by.items():
                key = self._GATE_COUNTER.get(reason)
                if key:
                    self._status[key] += n
            self._status["gatedUnmeasured"] += outcome.unmeasured

    # ------------------------------------------------ exclusion zones + heal

    def _apply_zones(self, faces: list, frame: dict) -> list:
        """Drop faces whose box CENTRE falls inside an operator-drawn zone.

        Returns the faces that remain countable; excluded ones are stamped
        ``excludedByZone`` in place and stay in the caller's list, because the
        operator must be able to SEE what their polygon is eating (in the tap
        and on the annotated frame) — an invisible filter on an invoice figure
        is not correctable.

        Why zones exist at all: the live bench minted p00004 from an 87 px
        face seen THROUGH A FROSTED GLASS PARTITION — someone inside an
        office, not at the gate, scoring 0.3203 against their true owner.  No
        quality floor can reject a face for being in the wrong PLACE; only the
        operator knows where the partitions, mirrors and TV screens are, so
        they draw them and the loop enforces them here, before the gate, so an
        excluded face is never embedded or matched.

        The centre rule is deliberate: a box straddling a zone edge counts by
        where its middle is, so a real guest walking PAST a partition (box
        clipping the zone) is still counted while a face BEHIND it (centre
        inside) is not.

        Honesty when it cannot run: zone points are normalized 0..1 and need
        the frame's pixel dimensions to scale by.  A frame that carries none
        gets NO exclusion — under-counting is the dominant failure mode, and a
        filter guessing at geometry is worse than a filter reporting it could
        not run — and every face that passed untested is counted in
        ``zoneUnmeasured``, mirroring the ``gatedUnmeasured`` pattern.
        """
        if not self.zones or not faces:
            return faces
        w = frame.get("w")
        h = frame.get("h")
        if not w or not h:
            self._bump("zoneUnmeasured", len(faces))
            return faces
        w, h = float(w), float(h)
        countable: list = []
        excluded = 0
        for f in faces:
            box = f.get("box", {})
            cx = float(box.get("x", 0.0)) + float(box.get("w", 0.0)) / 2.0
            cy = float(box.get("y", 0.0)) + float(box.get("h", 0.0)) / 2.0
            hit = next(
                (
                    z
                    for z in self.zones
                    if point_in_polygon(
                        cx, cy, [[p[0] * w, p[1] * h] for p in z["points"]]
                    )
                ),
                None,
            )
            if hit is None:
                countable.append(f)
            else:
                f["excludedByZone"] = True
                f["excludedZone"] = hit.get("label")
                excluded += 1
        if excluded:
            self._bump("excludedByZone", excluded)
        return countable

    @staticmethod
    def _track_for(face: dict, tracks: list) -> int | None:
        """The track a face belongs to: box centre inside the track box.

        Ties (overlapping tracks both containing the centre) go to the track
        whose box CENTRE is nearest the face centre.  None when no track
        contains it — a face with no track cannot carry heal bookkeeping,
        because "the SAME track matched someone else" is the entire evidence
        the heal acts on.
        """
        box = face.get("box") or {}
        cx = float(box.get("x", 0.0)) + float(box.get("w", 0.0)) / 2.0
        cy = float(box.get("y", 0.0)) + float(box.get("h", 0.0)) / 2.0
        best_id: int | None = None
        best_d = 0.0
        for t in tracks:
            tb = t.get("box") or {}
            tx, ty = float(tb.get("x", 0.0)), float(tb.get("y", 0.0))
            tw, th = float(tb.get("w", 0.0)), float(tb.get("h", 0.0))
            if not (tx <= cx <= tx + tw and ty <= cy <= ty + th):
                continue
            d = (tx + tw / 2.0 - cx) ** 2 + (ty + th / 2.0 - cy) ** 2
            if best_id is None or d < best_d:
                best_id, best_d = t.get("id"), d
        return best_id

    def _record_mint(self, track_id: int, verdict: dict) -> None:
        """Remember that this track just minted a new identity (heal evidence).

        Expired entries are pruned here — the only growth point — so the
        bookkeeping is bounded by the tracks active within one heal window.
        """
        if self.s.heal_window_s <= 0:
            return  # healing disabled: record nothing rather than leak
        now = time.monotonic()
        with self._lock:
            for tid in [
                t for t, e in self._minted.items()
                if now - e["at"] > self.s.heal_window_s
            ]:
                del self._minted[tid]
            self._minted[track_id] = {
                "key": verdict.get("personKey"),
                "cosine": verdict.get("cosine"),
                "at": now,
            }

    def _maybe_heal(self, planner_run_id: str, track_id: int, verdict: dict) -> None:
        """Fold a junk mint back when its own track disowns it (TRACK HEAL).

        The failure this kills, measured live: the bench (3 real people,
        counted 5) minted p00002 on a re-entry frame at cosine 0.3084 — below
        the 0.363 threshold, so a fresh key — and the SAME physical track
        matched the correct identity p00001 at 0.69 four seconds later.  The
        evidence that the mint was junk arrived almost immediately, and
        nothing acted on it.  This does: when a track that recently minted a
        new identity later matches a DIFFERENT existing identity comfortably
        (cosine >= heal_min_cosine, within heal_window_s of the mint), the
        mint is folded into the matched identity via ``POST /merge`` with
        ``onlyIfSingleton: true`` and the unique count steps back down.

        The guards, each load-bearing:

        * ``onlyIfSingleton`` — the match service refuses (inside its own
          transaction) if the mint has accumulated a second template, because
          that means the gallery independently re-sighted it and the "junk"
          theory is contradicted.  A ``merged=false`` (that, or an operator's
          cannot-link split) drops the bookkeeping and counts NOTHING — the
          refusal is the system working, not an error.
        * the 0.45 cosine floor — above the 0.377 measured impostor ceiling
          on this camera with margin (tonight's heal evidence was 0.69), so a
          heal can only ride evidence stronger than any impostor pair we have
          seen.  Same-person misses measured 0.294/0.308/0.361, i.e. the
          impostor and genuine distributions OVERLAP and no match-threshold
          value separates them — which is exactly why this is a heal on later
          evidence and not a threshold tweak.
        * the 20 s window and staff exemption (staff verdicts never reach
          here — see the verdict loop).

        KNOWN RESIDUAL RISK, documented rather than solved (CONTRACTS.md): a
        tracker identity swap — two people crossing paths — can hand a track
        from person A to person B.  If A's mint is still a singleton inside
        the window and B matches at >= 0.45, the heal folds A into B: an
        under-count of one.  The guards above bound how often that can happen
        (singleton-only, 20 s, 0.45 floor, cannot_link respected); the hard
        fix is track-quality gating and is out of scope tonight.  The reverse
        ordering (track matches Y first, mints X later) is deliberately NOT
        healed — the mint came after the evidence, so the evidence says
        nothing about it.
        """
        if self.s.heal_window_s <= 0:
            return
        with self._lock:
            entry = self._minted.get(track_id)
        if entry is None:
            return
        now = time.monotonic()
        if now - entry["at"] > self.s.heal_window_s:
            with self._lock:
                self._minted.pop(track_id, None)
            return
        matched_key = verdict.get("personKey")
        minted_key = entry["key"]
        if not matched_key or not minted_key or matched_key == minted_key:
            return  # the track still resolves to its own mint — nothing to heal
        cosine = verdict.get("cosine")
        if cosine is None or float(cosine) < self.s.heal_min_cosine:
            return  # not comfortable enough to outrank the impostor ceiling
        try:
            r = self._post(
                f"{self.s.match_url}/merge",
                {
                    "runId": planner_run_id,
                    "keep": matched_key,
                    "drop": minted_key,
                    "onlyIfSingleton": True,
                },
            )
        except Exception as e:  # noqa: BLE001 — a heal must never stop the count
            # Transient (match hiccup): keep the bookkeeping; the next verdict
            # on this track retries while the window lasts.
            self.log.warning(f"heal attempt failed, will retry in-window: {e}")
            return
        with self._lock:
            self._minted.pop(track_id, None)
        if r.get("merged"):
            self._dec_unique()
            self._bump("healedSplits")
            self.log.info(
                f"healed split: track {track_id} minted {minted_key} "
                f"(cosine={_fmt(entry['cosine'])}) then matched {matched_key} "
                f"(cosine={_fmt(cosine)}) — mint folded, unique -1"
            )
        else:
            self.log.info(
                f"heal refused for track {track_id}: {minted_key} is no longer "
                f"a singleton or is split from {matched_key} — count unchanged, "
                "which is the refusal doing its job"
            )

    def _note_templates(self, person_key: object, template_n: object) -> None:
        """Track how many views each guest holds, and summarise it in the status.

        The point is to make a silent failure loud.  Multi-template only helps
        when a crossing supplies a SECOND usable view of the same person; when
        it does not, every guest ends the run on one template, the gallery is
        no better than it was before M1, and the counts look exactly the same
        as if the threshold were at fault.  ``singleTemplateGuests`` is that
        distinction, reported on every run rather than discovered by opening a
        SQLite file at midnight.

        Called with the lock already held (from the verdict path).  A staff hit
        reports ``templateN: null`` by contract and is not a guest, so both are
        ignored rather than counted as a one-template person.
        """
        if not isinstance(person_key, str) or not isinstance(template_n, int):
            return
        self._template_n[person_key] = template_n
        st = self._status
        st["singleTemplateGuests"] = sum(1 for n in self._template_n.values() if n <= 1)
        st["templateNMax"] = max(self._template_n.values())

    def _count_staff(self, verdict: dict) -> None:
        """Record one staff sighting: a face-frame always, a crossing sometimes.

        ``staffCrossings`` used to be incremented per matched face per frame,
        so one waiter passing the gate 30 times reported hundreds — a number
        that moved with the frame rate rather than with staff behaviour, and
        that an operator could not read (3 busy waiters looked like 30 staff).
        A CROSSING is now one appearance: the same staff member seen again
        within ``staff_cooldown_s`` of their last sighting is still the same
        pass.  The raw per-frame figure survives as ``staffFaceFrames`` because
        it is genuinely useful when debugging recall — it is simply not a
        crossing count and no longer pretends to be.
        """
        staff_id = verdict.get("staffId") or verdict.get("personKey") or "staff:unknown"
        now = time.monotonic()
        last = self._staff_last_seen.get(staff_id)
        self._staff_last_seen[staff_id] = now
        with self._lock:
            self._status["staffFaceFrames"] += 1
            if last is None or (now - last) >= self.s.staff_cooldown_s:
                self._status["staffCrossings"] += 1

    def _remember(
        self,
        image_b64: str,
        seq: int | None,
        t_ms: int,
        boxes: list,
        tracks: list,
        faces: list,
        verdicts: list,
    ) -> None:
        """Store the last frame's outputs so a debug tap can render them.

        ``zones`` rides along so the annotated face-detect frame can draw the
        exclusion polygons the excluded faces were dropped by — a flag with no
        visible boundary would leave the operator unable to check their own
        drawing.
        """
        self._last = {
            "image_b64": image_b64,
            "seq": seq,
            "t_ms": t_ms,
            "boxes": boxes,
            "tracks": tracks,
            "faces": faces,
            "verdicts": verdicts,
            "zones": self.zones,
        }

    def _count_stage(self) -> None:
        """Record the count stage: the running unique total, once per frame."""
        self.board.frame("count")
        self.board.observe("count", "uniqueTotal", float(self.status()["unique"]))

    # ---------------------------------------------------------- debug taps

    def _maybe_tap(self, now: float) -> None:
        """Post per-stage payloads and annotated frames if the interval elapsed.

        Entirely best-effort: structured payloads always attempted; annotated
        frames only when the frame decodes (a stub/opaque frame simply skips the
        image upload).  Nothing here raises into the loop.

        A tap round is up to 5 payload POSTs plus 5 JPEG uploads, so bounding
        each call's timeout is not enough — a planner answering slowly but
        successfully would still cost ten timeouts per tick.  The whole round
        therefore shares ``tap_budget_s``: when it is spent the rest of the
        round is dropped (counted in ``tapRoundsAbandoned``) and the loop goes
        back to counting.  Debug frames are worth less than a guest.
        """
        if now - self._last_tap < self.s.tap_interval_s:
            return
        self._last_tap = now
        last = self._last
        if last is None:
            return
        deadline = time.monotonic() + self.s.tap_budget_s
        st = self.status()
        payloads = taps.build_payloads(
            last, self.s.quality_min_px, self.s.quality_canon_px,
            st["unique"], st["staffCrossings"],
            staff_face_frames=st["staffFaceFrames"],
            manual_additions=st["manualAdditions"],
        )
        for stage, payload in payloads.items():
            if time.monotonic() >= deadline:
                self._bump("tapRoundsAbandoned")
                return
            self.planner.post_tap(stage, payload)

        try:
            img = decode_jpeg_b64(last["image_b64"])
        except Exception:  # noqa: BLE001 — opaque/stub frame: skip the image upload
            return
        for stage in annotate.STAGES:
            if time.monotonic() >= deadline:
                self._bump("tapRoundsAbandoned")
                return
            try:
                jpeg = annotate.render(
                    stage, img, last, self.s.quality_min_px, self.s.quality_canon_px
                )
            except Exception:  # noqa: BLE001 — one bad overlay must not stop the rest
                continue
            self.planner.post_frame(stage, jpeg)

    # ------------------------------------------------------ operator feedback

    def _maybe_purge_staff(self, now: float | None = None) -> None:
        """Purge tombstoned staff from the site store (best-effort, idempotent).

        The erasure half of the consented-whitelist contract: the planner
        tombstones a deleted roster member, we relay the ids to the match
        service (which owns ``data/staff-<siteId>.db``) and confirm each purge
        so the planner's ledger records that the withdrawal was honoured.

        Called once before the first frame (``now=None`` forces the check, so a
        member deleted while the pipeline was off can never match again) and
        then on ``feedback_poll_s`` during the run, so an erasure requested
        mid-event lands within seconds.  Every step is best-effort: a planner
        or match hiccup leaves the tombstone open and the next cycle retries.
        """
        site_id = self.request.get("siteId")
        if not site_id:
            return
        if now is not None:
            if now - self._last_purge < self.s.feedback_poll_s:
                return
            self._last_purge = now
        try:
            tombs = self.planner.staff_tombstones(site_id)
        except Exception as e:  # noqa: BLE001 - erasure must never kill the loop
            self.log.warning(f"could not poll the erasure ledger: {e}")
            return
        # A rejected token makes the poll look like "nothing to erase" forever.
        # Erasure silently not happening is the exact failure this whole chain
        # exists to prevent, so it is reported, not shrugged off.
        if self.planner.auth_failures:
            self._set(plannerAuthFailures=self.planner.auth_failures)
        if not tombs:
            return
        ids = [t["staffId"] for t in tombs if t.get("staffId")]
        if not ids:
            return
        try:
            r = self._post(f"{self.s.match_url}/staff/purge", {"siteId": site_id, "staffIds": ids})
        except Exception as e:  # noqa: BLE001 - match hiccup: tombstones stay open, retried
            self.log.warning(f"staff purge failed, tombstones stay open: {e}")
            return
        removed = r.get("removed", {})
        with self._lock:
            self._status["staffPurged"] += sum(1 for v in removed.values())
        for t in tombs:
            if t.get("staffId") in removed:
                self.planner.confirm_tombstone(t["id"])

    def _maybe_feedback(self, now: float) -> None:
        """Poll the planner for open corrections and apply what we can."""
        if now - self._last_feedback < self.s.feedback_poll_s:
            return
        self._last_feedback = now
        items = self.planner.poll_feedback()
        if items:
            self._apply_feedback(items)

    def _apply_feedback(self, items: list[dict]) -> None:
        """Apply each open feedback item to the live gallery, then PUT status.

        Two guarantees, and the second one is why ``_settled_feedback`` exists.

        1. A transient match-service error leaves the item OPEN (unresolved and
           unsettled), so it is retried on the next poll — nothing is lost.
        2. An item this runner has already settled is NEVER executed twice, and
           is always re-reported with the SAME status.  ``resolve_feedback`` is
           single-shot, so a dropped PUT used to leave the item open, the next
           poll re-ran the correction, the re-run failed (the merged key no
           longer exists / the templates have already moved), and the item was
           recorded REJECTED beside a unique count that had visibly dropped.
           The operator could not tell what actually happened, which is the
           entire purpose of the feedback ledger.  Worse for ``missed``, which
           is not idempotent at all: a re-run would add a second person.
        """
        for item in items:
            if item.get("status", "open") != "open":
                continue
            fid = item.get("id")
            if fid in self._handled_feedback:
                continue
            settled = self._settled_feedback.get(fid)
            if settled is None:
                action = plan_action(item, has_site_id=bool(self.request.get("siteId")))
                applied = self._execute_action(action)
                if applied is None:
                    continue  # transient error — leave open, retry next poll
                settled = "applied" if applied else "rejected"
                self._settled_feedback[fid] = settled
                self._bump("feedbackRejected" if settled == "rejected" else "feedbackApplied")
            if self.planner.resolve_feedback(fid, settled):
                self._handled_feedback.add(fid)
                self._settled_feedback.pop(fid, None)

    def _execute_action(self, action) -> bool | None:
        """Run one correction against the match service.

        Returns True (applied, gallery changed), False (rejected / no-op) or
        None (transient error — caller should retry).  Merge and mark-staff
        decrement the live unique count only when the gallery confirms a
        change; ``count-missed`` increments it.  ``invalid`` never touches the
        gallery and is reported rejected: claiming "applied" for something the
        runner could not do is exactly the lie this loop must not tell.
        """
        run = self.planner.run_id
        site_id = self.request.get("siteId")
        try:
            if action.kind == "merge":
                r = self._post(
                    f"{self.s.match_url}/merge",
                    {"runId": run, "keep": action.key_a, "drop": action.key_b},
                )
                if r.get("merged"):
                    self._dec_unique()
                    return True
                return False
            if action.kind == "split":
                self._post(
                    f"{self.s.match_url}/split",
                    {"runId": run, "a": action.key_a, "b": action.key_b},
                )
                return True
            if action.kind == "mark-staff":
                body = {"runId": run, "personKey": action.person_key, "siteId": site_id}
                if action.staff_id:
                    body["staffId"] = action.staff_id
                r = self._post(f"{self.s.match_url}/mark-staff", body)
                if r.get("moved", 0) > 0:
                    self._dec_unique()
                    return True
                return False
            if action.kind == "count-missed":
                body = {"runId": run}
                if action.note:
                    body["note"] = action.note
                r = self._post(f"{self.s.match_url}/count/manual", body)
                if r.get("personKey"):
                    self._inc_unique()
                    return True
                return False
            # ack (note): filed, gallery untouched.
            # invalid: rejected without a gallery call.
            return action.kind == "ack"
        except StageError as e:
            # 4xx: the service understood us and refused (a mark-staff with no
            # staff store, a malformed key) — retrying cannot help, settle it
            # as rejected.  5xx: the service is unwell; leave the item open.
            return False if 400 <= e.status_code < 500 else None
        except Exception as e:  # noqa: BLE001 — match hiccup: retry on the next poll
            self.log.warning(f"could not apply a correction, will retry: {e}")
            return None

    def _dec_unique(self) -> None:
        """Drop the live unique count by one after a confirmed correction."""
        with self._lock:
            st = self._status
            st["unique"] = max(0, st["unique"] - 1)

    def _inc_unique(self) -> None:
        """Add one operator-attested person the pipeline missed (*missed*).

        Counted separately as ``manualAdditions`` so the report can always say
        how much of the headline number a human put there by hand — an
        attested count and a detected one are different evidence and must not
        be silently blended.
        """
        with self._lock:
            st = self._status
            st["unique"] += 1
            st["manualAdditions"] += 1

    # ------------------------------------------------------------ enrol mode

    def _run_enrol(self) -> None:
        """Capture a staff walk-through and write the best samples to the store.

        No counting: the deliverable is the site staff store plus a
        ``PUT /api/staff/:id`` reporting the sample count, so the operator can
        confirm the enrolment in the UI before the next person.

        **IT DOES LEAVE A RUN RECORD**, and used to leave none.  That asymmetry
        was the defect: ``staff_tombstones`` gives ERASURE a permanent ledger,
        while ENROLMENT — the act that creates the biometric in the first place
        — wrote nothing but an ``enrolled_at`` timestamp that the next
        re-enrolment silently overwrote.  A failed enrolment left no trace at
        all, and no record anywhere said from which camera, at which event, or
        over how many frames a staff template was captured.  Consent is only
        auditable if the capture is.  So an enrolment opens a ``mode:'enrol'``
        pipeline run and settles it with structured results.

        The row is BEST-EFFORT in both directions: a planner outage must not
        fail an enrolment (the samples are the deliverable and they are already
        stored), and a failed capture must still settle its row.

        **SINGLE-SUBJECT RULE.** A frame contributes a sample only when EXACTLY
        ONE face passes the quality gate; frames with two or more are skipped
        and counted in ``multiFaceFramesSkipped``.  Enrolment writes to a
        PERSISTENT, per-site whitelist and supersedes the member's previous
        templates, so a bystander caught in the walk-through does not just add
        noise — they become permanently recognised as that staff member at
        every future event at the venue, silently excluded from every guest
        count, while the real member may stop matching.  There is no signal
        that this happened and no way to notice it later.  A refusal the
        operator can see (and redo the walk-through for) is strictly better
        than a poisoned whitelist nobody knows about, so an enrolment that
        captured nothing FAILS loudly instead of reporting zero samples.

        Within the surviving single-subject frames the shortlist is ranked by
        :func:`enrol_score` — inter-eye distance x frontality, not box width.
        """
        req = self.request
        site_id, staff_id = req["siteId"], req["staffId"]
        self._set(state="enrolling")
        # BEFORE the camera is claimed, so a slot conflict or a dead source
        # settles a row that exists rather than vanishing without trace.
        self._open_enrol_run(site_id, staff_id)
        self._open_source()

        captured: list[tuple[float, list]] = []  # (score, embedding)
        frames = 0
        faces_seen = 0
        skipped = 0

        while not self._stop.is_set():
            frame = self._next_frame()
            if frame is None:
                break
            frames += 1
            image_b64 = frame["imageB64"]
            boxes = self._post(
                f"{self.s.persons_url}/detect", {"imageB64": image_b64}
            ).get("boxes", [])
            faces = self._post(
                f"{self.s.faces_url}/detect",
                {"imageB64": image_b64, "within": boxes or None},
            ).get("faces", [])
            # The SAME gate as counting: a template captured from a face the
            # count would have discarded is a template that will not match.
            kept = gate.gate_faces(faces, self.gate).kept
            if len(kept) != 1:
                if len(kept) > 1:
                    skipped += 1
                self._set(frames=frames, multiFaceFramesSkipped=skipped)
                continue
            faces_seen += 1
            embeddings = self._post(
                f"{self.s.embed_url}/embed", {"imageB64": image_b64, "faces": kept}
            ).get("embeddings", [])
            for face, emb in zip(kept, embeddings, strict=False):
                captured.append((enrol_score(face), emb))
            # Keep only the running best-N by quality to bound memory.
            captured.sort(key=lambda qe: qe[0], reverse=True)
            del captured[self.s.enrol_best_n :]
            self._set(
                frames=frames,
                facesSeen=faces_seen,
                sampleCount=len(captured),
                multiFaceFramesSkipped=skipped,
            )

        self._release_run_state()  # capture is done: hand the camera back
        samples = [{"embedding": emb, "quality": q} for q, emb in captured[: self.s.enrol_best_n]]
        tally = {
            "facesSeen": faces_seen,
            "frames": frames,
            "multiFaceFramesSkipped": skipped,
        }
        if not samples:
            error = (
                f"enrolment captured no single-subject frames "
                f"({skipped} frame(s) had more than one face, {frames} seen) — "
                f"walk {staff_id} through alone and re-run the enrolment"
            )
            self._set(state="failed", sampleCount=0, error=error, **tally)
            # A failed enrolment used to leave nothing anywhere. Now it settles
            # its own row with the reason, so "we tried and it did not work" is
            # a fact somebody can find later.
            self._end_enrol_run("failed", staff_id, 0, tally, error)
            return
        out = self._post(
            f"{self.s.match_url}/staff/enrol",
            {"siteId": site_id, "staffId": staff_id, "samples": samples},
        )
        n = int(out.get("sampleCount", len(samples)))
        # Report to the planner (retrying, but a planner outage must not fail
        # the enrolment itself — the samples are already stored).
        with contextlib.suppress(Exception):
            self.planner.report_enrolment(staff_id, _now_iso(), n)
        self._set(state="ended", sampleCount=n, **tally)
        self._end_enrol_run("ended", staff_id, n, tally, None)

    def _open_enrol_run(self, site_id: str, staff_id: str) -> None:
        """Open the planner run row that records this enrolment.

        Best-effort ON PURPOSE.  In count mode a failed ``create_run`` fails
        the run, because the planner id keys the gallery and there is no count
        without it.  An enrolment has no such dependency: the deliverable is
        the staff store write, the operator has already walked the member
        through, and refusing to enrol because the laptop was rebooting would
        be a worse outcome than an enrolment with a missing record.  So a
        failure is logged, ``plannerRunId`` stays None, and the walk-through
        proceeds exactly as it did before this record existed.
        """
        req = self.request
        label = req.get("label") or f"enrol {staff_id} {source_label(req['source'])}"
        try:
            planner_run_id = str(
                self.planner.create_run(
                    req["eventId"],
                    placement_id=req.get("placementId"),
                    label=label,
                    config={
                        "source": req["source"],
                        "mode": "enrol",
                        "siteId": site_id,
                        "staffId": staff_id,
                        "enrolBestN": self.s.enrol_best_n,
                        **self._gate_config(),
                        "geometry": "poc-2.8mm-2.0m-close-zone",
                    },
                )
            )
        except Exception as e:  # noqa: BLE001 — the enrolment matters more
            self.log.warning(
                f"enrolling {staff_id} without a planner record: {safe(str(e))}"
            )
            return
        self._set(plannerRunId=planner_run_id)
        self.log.info(
            f"enrolling {staff_id} from {source_label(req['source'])} "
            f"(plannerRun={planner_run_id}, site={site_id})"
        )

    def _end_enrol_run(
        self, status: str, staff_id: str, sample_count: int, tally: dict, error: str | None
    ) -> None:
        """Settle the enrolment's run row with a structured, readable record.

        ``results`` is the machine-readable half — how many templates this
        walk-through actually created — and ``notes`` the human sentence beside
        it.  Both are best-effort: the templates are already written, so a
        planner that is down costs the record, never the enrolment.
        """
        if self.planner.run_id is None:
            return
        notes = (
            f"enrol staffId={staff_id} samples={sample_count} "
            f"facesSeen={tally['facesSeen']} frames={tally['frames']} "
            f"multiFaceFramesSkipped={tally['multiFaceFramesSkipped']} "
            f"source={source_label(self.request['source'])} "
            f"endReason={self._end_reason}"
        )
        if error:
            notes = f"{notes} — {safe(error)}"
        results = {"sampleCount": float(sample_count)}
        results.update({k: float(v) for k, v in tally.items()})
        try:
            self.planner.end_run(
                status=status,
                notes=notes,
                results=results,
                end_reason=self._end_reason,
            )
        except Exception as e:  # noqa: BLE001 — the templates are already stored
            self._bump("plannerReportErrors")
            self.log.warning(f"enrolment {status} but the planner was not told: {e}")

    # -------------------------------------------------------------- plumbing

    def _timed(self, stage: str, metric: str, fn):
        """Run fn, record its wall time under stage/metric, return its result."""
        t = time.perf_counter()
        out = fn()
        self._last_ms = (time.perf_counter() - t) * 1000.0
        self.board.observe(stage, metric, self._last_ms)
        return out

    def _flush(self, elapsed_s: float) -> None:
        """Push per-stage aggregates and the sample batch to the planner.

        NEVER raises into the loop.  Stats are upserted aggregates and samples
        are supplementary, so a failed flush costs a gap the next successful
        flush repairs — whereas letting PlannerError out of here failed the
        whole run, and a restarted run gets a fresh empty gallery, re-counting
        every guest already counted.  A planner restart must cost chart data,
        never the count.
        """
        try:
            for body in self.board.snapshot(elapsed_s):
                self.planner.post_stats(
                    body["stage"], frames=body["frames"], fps=body["fps"], metrics=body["metrics"]
                )
            rows = self.samples.drain()
            if rows:
                self.planner.post_samples([Sample(**row) for row in rows])
        except PlannerError:
            self._bump("plannerReportErrors")
        except Exception:  # noqa: BLE001 — reporting must never stop counting
            self._bump("plannerReportErrors")
        self._set(samplesDropped=self.samples.dropped)


def _fmt(x) -> str:
    """A cosine for a log line; None-safe (a gallery's first mint has none)."""
    return "none" if x is None else f"{float(x):.4f}"


def _detail(response: httpx.Response) -> str:
    """Best readable explanation a stage gave for refusing a call."""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 — not JSON; fall back to raw text
        return response.text[:300]
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return str(body)[:300]
