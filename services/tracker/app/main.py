"""tracker — FastAPI app on the contract port 7103.

Stateful per ``runId``: each run owns an independent SortLite instance with
its own id space. Endpoints (CONTRACTS.md + tracker brief):

* ``POST /reset`` {runId} — (re)create the run's tracker from current env.
* ``POST /track`` {runId, tMs, boxes} → {tracks:[{id, box, ageFrames,
  hits}]} — one frame's detections in, this frame's confirmed tracks out.
  An unknown runId is auto-created (first frame of a run needs no separate
  reset; /reset exists to *clear* state or pick up new env).
* ``GET /health`` → {ok, model, version}.

Note ``tMs`` is accepted for the wire contract but the tracker is
frame-based: max-age and min-hits count frames, not milliseconds — honest
POC simplification, documented in the README.
"""

from fastapi import FastAPI
from heco_common.config import env_float, env_int
from heco_common.schemas import (
    Box,
    Health,
    ResetRequest,
    Track,
    TrackRequest,
    TrackResponse,
)

from . import __version__
from .sort import SortLite

app = FastAPI(title="heco-tracker", version=__version__)

#: Per-run tracker instances; POC scale keeps this a plain in-process dict.
_runs: dict[str, SortLite] = {}


def _new_tracker() -> SortLite:
    """Build a SortLite from env (read per call so /reset picks up changes)."""
    return SortLite(
        max_age=env_int("TRACKER_MAX_AGE", 15),
        min_hits=env_int("TRACKER_MIN_HITS", 3),
        iou_min=env_float("TRACKER_IOU_MIN", 0.2),
        vel_smooth=env_float("TRACKER_VEL_SMOOTH", 0.5),
    )


@app.post("/reset")
def reset(body: ResetRequest) -> dict:
    """Create or clear the tracker state for a run."""
    _runs[body.runId] = _new_tracker()
    return {"ok": True, "runId": body.runId}


@app.post("/track")
def track(body: TrackRequest) -> TrackResponse:
    """Advance the run's tracker by one frame of detections."""
    tracker = _runs.get(body.runId)
    if tracker is None:
        tracker = _runs[body.runId] = _new_tracker()
    dets = [(b.x, b.y, b.w, b.h, b.conf) for b in body.boxes]
    out = tracker.step(dets)
    return TrackResponse(
        tracks=[
            Track(
                id=t.tid,
                box=Box(x=t.box()[0], y=t.box()[1], w=t.w, h=t.h, conf=t.conf),
                ageFrames=t.age_frames,
                hits=t.hits,
            )
            for t in out
        ]
    )


@app.get("/health")
def health() -> Health:
    """Liveness + identity; 'model' names the algorithm, not a DNN."""
    return Health(ok=True, model="sort-lite-iou-velocity", version=__version__)
