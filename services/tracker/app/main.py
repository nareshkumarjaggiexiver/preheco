"""tracker — FastAPI app on the contract port 7103.

Stateful per ``runId``: each run owns an independent SortLite instance with
its own id space. Endpoints (CONTRACTS.md + tracker brief):

* ``POST /reset``   {runId} — (re)create the run's tracker from current env.
* ``POST /release`` {runId} — forget the run's tracker entirely (end of run).
* ``POST /track``   {runId, tMs, boxes} → {tracks:[{id, box, ageFrames,
  hits}]} — one frame's detections in, this frame's confirmed tracks out.
  An unknown runId is auto-created (first frame of a run needs no separate
  reset; /reset exists to *clear* state or pick up new env).
* ``GET /health``   → {ok, model, version, runs} — ``runs`` is how many run
  trackers are resident, so a leak is visible without a debugger.

Per-run state has a LIFECYCLE.  Every run gets a fresh uuid-suffixed id, so
``_runs`` only ever grew: a season of events left one dead SortLite per run
resident until somebody restarted the process.  The runner now calls
``/release`` when a run ends, and every ``/track`` evicts entries untouched
for ``TRACKER_RUN_TTL_S`` (default 1 h) as a backstop for runs that died
without releasing.

Note ``tMs`` is accepted for the wire contract but the tracker is
frame-based: max-age and min-hits count frames, not milliseconds — honest
POC simplification, documented in the README.
"""

import time

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

#: Monotonic timestamp of the last /track (or /reset) per run, for TTL eviction.
_last_used: dict[str, float] = {}


#: Frames a track may coast unmatched before it is dropped.  30, not the
#: original 15, because of bench 6e1a5d (2026-08-06): ONE person — out of
#: frame, back in, then seated — produced SIX tracker ids (2, 5, 6, 7, 8, 12)
#: at 3.97 fps, where 15 frames is only 3.75 s of coasting and every seated
#: detector gap longer than that minted a fresh id.  30 frames is ~7.5 s.  See
#: SortLite.__init__ for the full measurement AND the caveat: this treats the
#: symptom (the disease is YOLOX-nano losing seated bodies), and a longer coast
#: widens the window in which a ghost track can latch onto a DIFFERENT person.
DEFAULT_MAX_AGE = 30


def _max_age() -> int:
    """Frames of coasting, from ``HECO_TRACKER_MAX_AGE`` (default 30).

    Two names are read on purpose.  ``HECO_TRACKER_MAX_AGE`` is the knob the
    pipeline's compose file plumbs through and the one to use; ``TRACKER_MAX_AGE``
    is the name this service shipped with and documented, so an existing
    deployment that set it by hand keeps working rather than having its
    explicit choice silently replaced by the new default — a knob that stops
    being obeyed without saying so is the worst way to change behaviour.  The
    HECO_-prefixed name wins when both are set.
    """
    legacy = env_int("TRACKER_MAX_AGE", DEFAULT_MAX_AGE)
    return env_int("HECO_TRACKER_MAX_AGE", legacy)


def _new_tracker() -> SortLite:
    """Build a SortLite from env (read per call so /reset picks up changes)."""
    return SortLite(
        max_age=_max_age(),
        min_hits=env_int("TRACKER_MIN_HITS", 3),
        iou_min=env_float("TRACKER_IOU_MIN", 0.2),
        vel_smooth=env_float("TRACKER_VEL_SMOOTH", 0.5),
    )


def _forget(run_id: str) -> bool:
    """Drop one run's tracker state; True if there was any."""
    _last_used.pop(run_id, None)
    return _runs.pop(run_id, None) is not None


def _evict_idle(now: float, keep: str) -> list[str]:
    """Forget runs untouched for the TTL; never the run being served."""
    ttl = env_float("TRACKER_RUN_TTL_S", 3600.0)
    if ttl <= 0:
        return []
    stale = [r for r, t in _last_used.items() if r != keep and now - t > ttl]
    for run_id in stale:
        _forget(run_id)
    return stale


@app.post("/reset")
def reset(body: ResetRequest) -> dict:
    """Create or clear the tracker state for a run."""
    _runs[body.runId] = _new_tracker()
    _last_used[body.runId] = time.monotonic()
    return {"ok": True, "runId": body.runId}


@app.post("/release")
def release(body: ResetRequest) -> dict:
    """Forget a finished run's tracker so its state stops occupying memory.

    Idempotent: releasing an unknown run is success (``released: false``) —
    the guarantee wanted is "nothing of that run remains", not "something was
    found", so a duplicate stop or a retry after a dropped reply is harmless.
    """
    return {"ok": True, "runId": body.runId, "released": _forget(body.runId)}


@app.post("/track")
def track(body: TrackRequest) -> TrackResponse:
    """Advance the run's tracker by one frame of detections."""
    now = time.monotonic()
    _evict_idle(now, keep=body.runId)
    tracker = _runs.get(body.runId)
    if tracker is None:
        tracker = _runs[body.runId] = _new_tracker()
    _last_used[body.runId] = now
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
def health() -> dict:
    """Liveness + identity; 'model' names the algorithm, not a DNN.

    ``runs`` reports how many run trackers are resident so an operator can see
    a leak (a number that only ever grows) without attaching a debugger.
    """
    return {
        **Health(ok=True, model="sort-lite-iou-velocity", version=__version__).model_dump(),
        "runs": len(_runs),
    }
