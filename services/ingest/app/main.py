"""ingest — FastAPI app on the contract port 7101.

Endpoints (CONTRACTS.md):

* ``POST /open``  {url|path, loop} — start capturing; replaces any current
  source (the old worker is stopped first).
* ``GET /frame``  → {tMs, imageB64, w, h, seq} — the LATEST frame only
  (drop-not-queue; see app.capture for the policy).
* ``GET /health`` → {ok, model, version}.

Frame encoding happens here, on demand, per request — the capture thread
stores raw arrays so an idle pipeline costs no JPEG work.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from heco_common.config import env_int
from heco_common.imaging import encode_jpeg_b64
from heco_common.schemas import Frame, Health, OpenSource

from . import __version__
from .capture import CaptureError, CaptureWorker


class _State:
    """Holds the single active capture worker (one source per instance)."""

    worker: CaptureWorker | None = None

    def swap(self, new: CaptureWorker | None) -> None:
        """Install a new worker (or None), stopping the previous one."""
        old, self.worker = self.worker, new
        if old is not None:
            old.stop()


state = _State()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Ensure the capture thread and device are released on shutdown."""
    yield
    state.swap(None)


app = FastAPI(title="heco-ingest", version=__version__, lifespan=_lifespan)


@app.post("/open")
def open_source(body: OpenSource) -> dict:
    """Open an RTSP url or a video file; replaces the current source."""
    if body.path is not None and not os.path.exists(body.path):
        raise HTTPException(status_code=400, detail=f"no such file: {body.path}")
    source, is_file = (body.path, True) if body.path else (body.url, False)
    try:
        worker = CaptureWorker(source=source, is_file=is_file, loop=body.loop)
    except CaptureError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    worker.start()
    state.swap(worker)
    return {"ok": True, "source": source, "loop": body.loop}


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
    return Frame(tMs=t_ms, imageB64=encode_jpeg_b64(img, quality=quality), w=w, h=h, seq=seq)


@app.get("/health")
def health() -> Health:
    """Liveness + identity; 'model' names the capture backend, not a DNN."""
    return Health(ok=True, model="opencv-videocapture", version=__version__)
