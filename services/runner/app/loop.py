"""The run loop: one thread driving the whole pipeline for one run.

Per frame: ingest /frame -> persons /detect -> tracker /track -> faces /detect
(within the tracked boxes) -> local quality gate -> embed /embed -> match
/match per face -> unique count.  Every stage is timed; aggregates and sampled
raw rows are flushed to the planner (heco_common.planner.PlannerClient) every
`flush_interval_s` seconds; the run ends when the source ends, /stop is
called, or a stage errors.

End-of-source: ingest serves the LATEST frame with a monotonically increasing
``seq`` and never signals EOF explicitly — a **stalled seq** is the signal
(see services/ingest).  The loop also accepts a stub-friendly explicit end
(``{"ended": true}``, missing ``imageB64``, or HTTP 204/404/410).  RTSP
sources stall only on network loss; stop those via POST /runs/:id/stop.
"""

import contextlib
import threading
import time

import httpx
from heco_common.planner import PlannerClient, Transport
from heco_common.schemas import Sample

from .config import Settings
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
        self._status: dict = {
            "runId": run_id,
            "plannerRunId": None,
            "state": "starting",
            "frames": 0,
            "unique": 0,
            "matches": 0,
            "subCanonMatches": 0,
            "subCanonShare": 0.0,
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
        """Execute the whole run; returns (and stores) the final status."""
        try:
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
        frames = 0

        while not self._stop.is_set():
            frame = self._timed("ingest", "ingestMs", self._next_frame)
            if frame is None:
                break
            self.board.frame("ingest")
            frames += 1
            t_ms = int(frame.get("tMs", (time.monotonic() - t0) * 1000.0))

            self._pipeline_step(planner_run_id, frame["imageB64"], t_ms)
            self._set(frames=frames)

            now = time.monotonic()
            if now - last_flush >= self.s.flush_interval_s:
                self._flush(now - t0)
                last_flush = now

        self._flush(max(time.monotonic() - t0, 1e-9))
        st = self.status()
        notes = (
            f"unique={st['unique']} frames={frames} matches={st['matches']} "
            f"subCanonShare={st['subCanonShare']:.2f} "
            f"(POC geometry: 2.8mm @2.0m close-zone, faces ~64-85px, floor 56px)"
        )
        self.planner.end_run(status="ended", notes=notes)
        self._set(state="ended")

    def _pipeline_step(self, planner_run_id: str, image_b64: str, t_ms: int) -> None:
        """Run stages 2..8 for one frame, timing and measuring each."""
        s, board, samples = self.s, self.board, self.samples

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

        # face-detect (within tracked person boxes; raw boxes before the
        # tracker confirms any — min-hits means early frames have no tracks)
        within = [t["box"] for t in tracks] or boxes
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
            samples.add("face-detect", t_ms, {"faceBoxWPx": float(f["box"]["w"])})

        # quality gate (local): floor 56 px; 56-79 px flagged sub-canon
        tq = time.perf_counter()
        kept = [f for f in faces if float(f["box"]["w"]) >= s.quality_min_px]
        board.frame("quality")
        board.observe("quality", "qualityMs", (time.perf_counter() - tq) * 1000.0)
        for f in kept:
            board.observe("quality", "faceBoxWPx", float(f["box"]["w"]))

        if not kept:
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

        # match (one call per embedding; gallery keyed by the planner run id)
        for face, emb in zip(kept, embeddings, strict=False):
            w = float(face["box"]["w"])
            m = self._timed(
                "match",
                "matchMs",
                lambda e=emb, q=w: self._post(
                    f"{s.match_url}/match",
                    {"runId": planner_run_id, "embedding": e, "quality": q},
                ),
            )
            board.frame("match")
            if m.get("cosine") is not None:
                board.observe("match", "matchCosine", float(m["cosine"]))
                samples.add("match", t_ms, {"matchCosine": float(m["cosine"])})
            with self._lock:
                st = self._status
                st["matches"] += 1
                if m.get("subCanon"):
                    st["subCanonMatches"] += 1
                if m.get("isNew"):
                    st["unique"] += 1
                st["subCanonShare"] = st["subCanonMatches"] / st["matches"]

        self._count_stage()

    def _count_stage(self) -> None:
        """Record the count stage: the running unique total, once per frame."""
        self.board.frame("count")
        self.board.observe("count", "uniqueTotal", float(self.status()["unique"]))

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
