"""ingest — FastAPI app on the contract port 7101.

Endpoints (CONTRACTS.md):

* ``POST /open``  {url|path, loop, owner?, takeover?} — start capturing.
  The capture slot is EXCLUSIVE: see "one slot, one owner" below.
* ``POST /close`` {owner?, force?} — release the slot at end of run.
* ``GET /frame``  → {tMs, imageB64, w, h, seq, ended} — the LATEST frame only
  (drop-not-queue; see app.capture for the policy).
* ``GET /health`` → {ok, model, version, owner}.

One slot, one owner
-------------------
This service holds exactly ONE capture worker.  It used to let every /open
replace it unconditionally, which made a second run a silent thief: start a
staff enrolment while a count run is live at the gate (the documented
mid-event workflow) and the count run keeps polling /frame, gets the
enrolment's frames, and counts the walk-through — no error anywhere, just a
plausible-looking frame stream and a corrupted count.

So an /open that carries an ``owner`` CLAIMS the slot, and any later /open by
a different owner is refused with 409 naming the holder.  Three escapes, all
deliberate: the same owner may re-open (an idempotent restart), a slot whose
capture thread has died is not a live run and may be claimed, and
``takeover: true`` is the operator's explicit "I know, take it anyway".  An
/open with no owner keeps the old replace-anything behaviour so ad-hoc probes
and the smoke script are unaffected — but the runner always sends one.

Frame encoding happens here, on demand, per request — the capture thread
stores raw arrays so an idle pipeline costs no JPEG work.
"""

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from heco_common.config import env_int
from heco_common.imaging import encode_jpeg_b64
from heco_common.schemas import CloseSource, Frame, Health, OpenSource

from . import __version__
from .capture import CaptureError, CaptureWorker


class _State:
    """Holds the single active capture worker and the run that claimed it."""

    def __init__(self) -> None:
        """Start with an empty, unowned slot."""
        self.worker: CaptureWorker | None = None
        self.owner: str | None = None
        self.lock = threading.Lock()

    def swap(self, new: CaptureWorker | None, owner: str | None = None) -> None:
        """Install a new worker (or None), stopping the previous one."""
        old, self.worker, self.owner = self.worker, new, (owner if new else None)
        if old is not None:
            old.stop()

    def held_by_live_run(self) -> bool:
        """True when an owning run's capture thread is still alive.

        A dead thread means the owner crashed or its process went away, so the
        slot is not really held and refusing the next run would just wedge the
        pipeline until somebody restarts ingest.
        """
        return (
            self.worker is not None
            and self.owner is not None
            and self.worker.is_alive()
        )


state = _State()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Ensure the capture thread and device are released on shutdown."""
    yield
    state.swap(None)


app = FastAPI(title="heco-ingest", version=__version__, lifespan=_lifespan)


@app.post("/open")
def open_source(body: OpenSource) -> dict:
    """Open an RTSP url or a video file, claiming the exclusive capture slot.

    409 when a different, still-live owner holds the slot — the error names
    that run so the operator learns "the gate count is using this camera"
    instead of silently losing it.
    """
    if body.path is not None and not os.path.exists(body.path):
        raise HTTPException(status_code=400, detail=f"no such file: {body.path}")
    with state.lock:
        holder = state.owner
        if (
            body.owner
            and not body.takeover
            and state.held_by_live_run()
            and holder != body.owner
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"capture slot is held by run {holder!r}; stop that run or "
                    f"re-send with takeover=true to seize its camera"
                ),
            )
        source, is_file = (body.path, True) if body.path else (body.url, False)
        try:
            worker = CaptureWorker(source=source, is_file=is_file, loop=body.loop)
        except CaptureError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        worker.start()
        state.swap(worker, owner=body.owner)
    return {"ok": True, "source": source, "loop": body.loop, "owner": body.owner}


@app.post("/close")
def close_source(body: CloseSource) -> dict:
    """Release the capture slot so the next run can claim the camera.

    Owner-checked: a stale close from a run that already lost the slot must
    not stop the run that holds it now.  ``force`` is the operator override.
    Closing an already-free slot is success — release is idempotent.
    """
    with state.lock:
        if state.worker is None:
            return {"ok": True, "released": False, "owner": None}
        holder = state.owner
        if body.owner and holder and holder != body.owner and not body.force:
            raise HTTPException(
                status_code=409,
                detail=f"capture slot is held by run {holder!r}, not {body.owner!r}",
            )
        state.swap(None)
    return {"ok": True, "released": True, "owner": holder}


@app.get("/frame")
def get_frame() -> Frame:
    """Return the latest captured frame as base64 JPEG.

    409: no source open. 503: source open but no frame decoded yet (a live
    RTSP source can take a moment) — callers should retry shortly.
    """
    worker = state.worker
    if worker is None:
        raise HTTPException(status_code=409, detail="no source open — POST /open first")
    latest = worker.latest()
    if latest is None:
        raise HTTPException(status_code=503, detail="no frame captured yet — retry")
    seq, t_ms, img = latest
    h, w = img.shape[:2]
    quality = env_int("INGEST_JPEG_QUALITY", 85)
    # `ended` is the only signal that separates a played-out file from a camera
    # that blinked: both freeze `seq`. The worker knows which it is (it retries
    # a live stream forever and only sets ended for a finished file), and until
    # now it kept that to itself.
    return Frame(
        tMs=t_ms, imageB64=encode_jpeg_b64(img, quality=quality),
        w=w, h=h, seq=seq, ended=bool(getattr(worker, "ended", False)),
    )


@app.get("/health")
def health() -> dict:
    """Liveness + identity; 'model' names the capture backend, not a DNN.

    ``owner`` is included so an operator can see which run holds the camera
    without guessing from a 409.
    """
    return {
        **Health(ok=True, model="opencv-videocapture", version=__version__).model_dump(),
        "owner": state.owner,
    }
