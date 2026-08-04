"""The run loop: one thread driving the whole pipeline for one run.

COUNT MODE — per frame: ingest /frame -> persons /detect -> tracker /track ->
faces /detect (within the tracked boxes) -> local quality gate -> embed /embed
-> match /match per face -> unique count.  Every stage is timed; aggregates and
sampled raw rows flush to the planner every ``flush_interval_s`` seconds.  On
top of that (CONTRACTS.md v1, all best-effort so a planner hiccup never blocks
or crashes the loop): annotated debug frames + structured taps go up every
``tap_interval_s``, and operator feedback is polled every ``feedback_poll_s``
and applied to the live gallery (merge / split / mark-staff), adjusting the
unique count.  The matcher checks the site STAFF store first when the run
carries a ``siteId``: a staff hit is counted as a staffCrossing and kept out of
the guest unique count, but the track stays visible upstream.

ENROL MODE — a professional-enrolment walk-through: capture faces, keep the
best ``enrol_best_n`` by quality, write them to the site staff store, and PUT
the sample count back to the planner (no pipeline_run record, no counting).

End-of-source: ingest serves the LATEST frame with a monotonically increasing
``seq`` and never signals EOF explicitly — a **stalled seq** is the signal
(see services/ingest).  The loop also accepts a stub-friendly explicit end
(``{"ended": true}``, missing ``imageB64``, or HTTP 204/404/410).  RTSP
sources stall only on network loss; stop those via POST /runs/:id/stop.
"""

import contextlib
import threading
import time
from datetime import UTC, datetime

import httpx
from heco_common.imaging import decode_jpeg_b64
from heco_common.geometry import dedupe_boxes
from heco_common.planner import FileTransport, PlannerClient, Transport
from heco_common.schemas import Sample

from . import annotate, taps
from .config import Settings
from .feedback import plan_action
from .stats import SampleBuffer, StatsBoard


def httpx_transport(client: httpx.Client) -> Transport:
    """Adapt an httpx client to the PlannerClient transport callable.

    Lets production share one connection pool for stages and planner, and
    lets tests route planner traffic through the same httpx.MockTransport
    that fakes the stage services.
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


class RunLoop:
    """Owns one run: the pipeline loop, its stats, and its lifecycle."""

    def __init__(
        self,
        run_id: str,
        request: dict,
        settings: Settings,
        client: httpx.Client,
        planner: PlannerClient,
    ) -> None:
        """Prepare a run; `request` is the validated POST /runs body as a dict.

        `client` is injected so tests can pass an httpx.MockTransport-backed
        client and drive the loop against a fully fake pipeline; `planner`
        should speak through the same client (see httpx_transport).
        """
        self.run_id = run_id
        self.request = request
        self.s = settings
        self.client = client
        self.planner = planner
        self.board = StatsBoard()
        self.samples = SampleBuffer(cap=settings.sample_batch_max)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_seq: int | None = None
        self._last_ms: float = 0.0
        # Snapshot of the frame just processed, for debug taps (loop-thread only).
        self._last: dict | None = None
        self._last_tap: float = 0.0
        self._last_feedback: float = 0.0
        self._handled_feedback: set = set()
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
            "subCanonMatches": 0,
            "subCanonShare": 0.0,
            "feedbackApplied": 0,
            "sampleCount": 0,
            "samplesDropped": 0,
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

    # ------------------------------------------------------------------ HTTP

    def _post(self, url: str, body: dict) -> dict:
        r = self.client.post(url, json=body)
        r.raise_for_status()
        return r.json()

    def _next_frame(self) -> dict | None:
        """Poll ingest for a frame with a NEW seq; None once the source ends.

        503 (source warming up) and a stalled seq both retry every
        ``source_poll_s`` until ``source_stall_s`` passes without progress.
        """
        deadline = time.monotonic() + self.s.source_stall_s
        while not self._stop.is_set() and time.monotonic() < deadline:
            r = self.client.get(f"{self.s.ingest_url}/frame")
            if r.status_code in (204, 404, 410):
                return None  # stub-style explicit end
            if r.status_code == 503:  # no frame decoded yet — retry
                time.sleep(self.s.source_poll_s)
                continue
            r.raise_for_status()
            body = r.json()
            if body.get("ended") or not body.get("imageB64"):
                return None
            seq = body.get("seq")
            if seq is None or seq != self._last_seq:
                self._last_seq = seq
                return body
            time.sleep(self.s.source_poll_s)  # same frame again — source idle
        return None  # stalled past source_stall_s (or stopped): source ended

    # ------------------------------------------------------------- the loop

    def run(self) -> dict:
        """Execute the whole run (count or enrol); returns the final status."""
        try:
            if self.request.get("mode") == "enrol":
                self._run_enrol()
            else:
                self._run()
        except Exception as e:  # noqa: BLE001 — a run must always settle
            self._set(state="failed", error=f"{type(e).__name__}: {e}")
            if self.planner.run_id is not None:
                # Best effort: the planner may be down too; local status
                # already says failed either way.
                with contextlib.suppress(Exception):
                    self.planner.end_run(status="failed", notes=str(e))
        return self.status()

    def _run(self) -> None:
        req = self.request
        label = req.get("label") or f"runner {req['source']}"
        planner_run_id = str(
            self.planner.create_run(
                req["eventId"],
                placement_id=req.get("placementId"),
                label=label,
                config={
                    "source": req["source"],
                    "mode": "count",
                    "siteId": req.get("siteId"),
                    "qualityMinPx": self.s.quality_min_px,
                    "qualityCanonPx": self.s.quality_canon_px,
                    "geometry": "poc-2.8mm-2.0m-close-zone",  # CONTRACTS.md POC geometry
                },
            )
        )
        self._set(plannerRunId=planner_run_id, state="running")

        # Fresh per-run state downstream, then open the source.
        self._post(f"{self.s.match_url}/reset", {"runId": planner_run_id})
        self._post(f"{self.s.tracker_url}/reset", {"runId": planner_run_id})
        self._post(f"{self.s.ingest_url}/open", req["source"])

        t0 = time.monotonic()
        last_flush = t0
        self._last_tap = t0
        self._last_feedback = t0
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

        self._flush(max(time.monotonic() - t0, 1e-9))
        st = self.status()
        notes = (
            f"unique={st['unique']} frames={frames} matches={st['matches']} "
            f"staffCrossings={st['staffCrossings']} "
            f"subCanonShare={st['subCanonShare']:.2f} "
            f"(POC geometry: 2.8mm @2.0m close-zone, faces ~64-85px, floor 56px)"
        )
        self.planner.end_run(status="ended", notes=notes)
        self._set(state="ended")

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

        # face-detect region: the DEDUPED UNION of the raw person boxes and the
        # confirmed-track boxes. Searching only confirmed tracks (the old
        # `[t["box"] for t in tracks] or boxes`) silently dropped every
        # unconfirmed, coasting or short-dwell person from face search once ANY
        # track existed — and because the count is match.isNew, a face never
        # searched is a guest never counted. Raw boxes restore recall; the
        # dedupe stops a person covered by both a raw box and a track box from
        # being searched (and embedded) twice. See docs/planning/pipeline/
        # accuracy-and-tuning.md (finding D1).
        within = dedupe_boxes(
            [t["box"] for t in tracks] + list(boxes), iou_thr=0.6
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
            # report-only quality signals (accuracy R&D F1): recognition size is
            # honestly read in inter-eye distance, not box width.
            if "iedPx" in f:
                board.observe("face-detect", "faceIedPx", float(f["iedPx"]))
                sample["faceIedPx"] = float(f["iedPx"])
            if "frontality" in f:
                board.observe("face-detect", "frontality", float(f["frontality"]))
                sample["frontality"] = float(f["frontality"])
            samples.add("face-detect", t_ms, sample)

        # quality gate (local): floor 56 px; 56-79 px flagged sub-canon
        tq = time.perf_counter()
        kept = [f for f in faces if float(f["box"]["w"]) >= s.quality_min_px]
        board.frame("quality")
        board.observe("quality", "qualityMs", (time.perf_counter() - tq) * 1000.0)
        for f in kept:
            board.observe("quality", "faceBoxWPx", float(f["box"]["w"]))

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
                    "box": face.get("box"),
                }
            )
            with self._lock:
                st = self._status
                if m.get("isStaff"):
                    # Staff: tagged and counted separately, excluded from unique.
                    st["staffCrossings"] += 1
                else:
                    st["matches"] += 1
                    if m.get("subCanon"):
                        st["subCanonMatches"] += 1
                    if m.get("isNew"):
                        st["unique"] += 1
                    st["subCanonShare"] = (
                        st["subCanonMatches"] / st["matches"] if st["matches"] else 0.0
                    )

        self._remember(image_b64, frame.get("seq"), t_ms, boxes, tracks, faces, verdicts)
        self._count_stage()

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
        """Store the last frame's outputs so a debug tap can render them."""
        self._last = {
            "image_b64": image_b64,
            "seq": seq,
            "t_ms": t_ms,
            "boxes": boxes,
            "tracks": tracks,
            "faces": faces,
            "verdicts": verdicts,
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
        """
        if now - self._last_tap < self.s.tap_interval_s:
            return
        self._last_tap = now
        last = self._last
        if last is None:
            return
        st = self.status()
        payloads = taps.build_payloads(
            last, self.s.quality_min_px, self.s.quality_canon_px,
            st["unique"], st["staffCrossings"],
        )
        for stage, payload in payloads.items():
            self.planner.post_tap(stage, payload)

        try:
            img = decode_jpeg_b64(last["image_b64"])
        except Exception:  # noqa: BLE001 — opaque/stub frame: skip the image upload
            return
        for stage in annotate.STAGES:
            try:
                jpeg = annotate.render(
                    stage, img, last, self.s.quality_min_px, self.s.quality_canon_px
                )
            except Exception:  # noqa: BLE001 — one bad overlay must not stop the rest
                continue
            self.planner.post_frame(stage, jpeg)

    # ------------------------------------------------------ operator feedback

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

        A transient match-service error leaves the item open (not resolved) so
        it is retried on the next poll; the gallery corrections are idempotent,
        so a retry after a dropped status update never double-counts.
        """
        for item in items:
            if item.get("status", "open") != "open":
                continue
            fid = item.get("id")
            if fid in self._handled_feedback:
                continue
            applied = self._execute_action(plan_action(item))
            if applied is None:
                continue  # transient error — leave open, retry next poll
            if self.planner.resolve_feedback(fid, "applied" if applied else "rejected"):
                self._handled_feedback.add(fid)

    def _execute_action(self, action) -> bool | None:
        """Run one correction against the match service.

        Returns True (applied, gallery changed), False (rejected / no-op) or
        None (transient error — caller should retry).  Merge and mark-staff
        decrement the live unique count only when the gallery confirms a change.
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
                body = {"runId": run, "personKey": action.person_key}
                if site_id:
                    body["siteId"] = site_id
                if action.staff_id:
                    body["staffId"] = action.staff_id
                r = self._post(f"{self.s.match_url}/mark-staff", body)
                if r.get("moved", 0) > 0:
                    self._dec_unique()
                    return True
                return False
            # ack (missed / note / unknown): filed, gallery untouched.
            # invalid: reject without a gallery call.
            return action.kind == "ack"
        except Exception:  # noqa: BLE001 — match hiccup: retry on the next poll
            return None

    def _dec_unique(self) -> None:
        """Drop the live unique count by one after a confirmed correction."""
        with self._lock:
            st = self._status
            st["unique"] = max(0, st["unique"] - 1)
            st["feedbackApplied"] += 1

    # ------------------------------------------------------------ enrol mode

    def _run_enrol(self) -> None:
        """Capture a staff walk-through and write the best samples to the store.

        No pipeline_run and no counting: the deliverable is the site staff
        store plus a ``PUT /api/staff/:id`` reporting the sample count, so the
        operator can confirm the enrolment in the UI before the next person.
        """
        req = self.request
        site_id, staff_id = req["siteId"], req["staffId"]
        self._set(state="enrolling")
        self._post(f"{self.s.ingest_url}/open", req["source"])

        captured: list[tuple[float, list]] = []  # (quality, embedding)
        frames = 0
        faces_seen = 0

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
            kept = [f for f in faces if float(f["box"]["w"]) >= self.s.quality_min_px]
            if not kept:
                self._set(frames=frames)
                continue
            faces_seen += len(kept)
            embeddings = self._post(
                f"{self.s.embed_url}/embed", {"imageB64": image_b64, "faces": kept}
            ).get("embeddings", [])
            for face, emb in zip(kept, embeddings, strict=False):
                captured.append((float(face["box"]["w"]), emb))
            # Keep only the running best-N by quality to bound memory.
            captured.sort(key=lambda qe: qe[0], reverse=True)
            del captured[self.s.enrol_best_n :]
            self._set(frames=frames, facesSeen=faces_seen, sampleCount=len(captured))

        samples = [{"embedding": emb, "quality": q} for q, emb in captured[: self.s.enrol_best_n]]
        n = 0
        if samples:
            out = self._post(
                f"{self.s.match_url}/staff/enrol",
                {"siteId": site_id, "staffId": staff_id, "samples": samples},
            )
            n = int(out.get("sampleCount", len(samples)))
        # Report to the planner (retrying, but a planner outage must not fail
        # the enrolment itself — the samples are already stored).
        with contextlib.suppress(Exception):
            self.planner.report_enrolment(staff_id, _now_iso(), n)
        self._set(state="ended", sampleCount=n, facesSeen=faces_seen, frames=frames)

    # -------------------------------------------------------------- plumbing

    def _timed(self, stage: str, metric: str, fn):
        """Run fn, record its wall time under stage/metric, return its result."""
        t = time.perf_counter()
        out = fn()
        self._last_ms = (time.perf_counter() - t) * 1000.0
        self.board.observe(stage, metric, self._last_ms)
        return out

    def _flush(self, elapsed_s: float) -> None:
        """Push per-stage aggregates and the sample batch to the planner."""
        for body in self.board.snapshot(elapsed_s):
            self.planner.post_stats(
                body["stage"], frames=body["frames"], fps=body["fps"], metrics=body["metrics"]
            )
        rows = self.samples.drain()
        if rows:
            self.planner.post_samples([Sample(**row) for row in rows])
        self._set(samplesDropped=self.samples.dropped)
