"""FastAPI app for the persons service (heco-pipeline port 7102).

Contract (CONTRACTS.md):
    POST /detect {imageB64, confMin?} -> {boxes: [{x, y, w, h, conf}], inferMs}
    GET  /health -> {ok, model, version}
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .codec import b64_to_bgr
from .model import MODEL_PATH, PersonDetector

log = logging.getLogger("persons")

app = FastAPI(title="heco persons", version=__version__)

_detector: PersonDetector | None = None
_load_error: str | None = None


def _get_detector() -> PersonDetector | None:
    """Load the detector lazily so the app can boot (unhealthy) without weights."""
    global _detector, _load_error
    if _detector is None and _load_error is None:
        try:
            _detector = PersonDetector()
        except FileNotFoundError as exc:
            _load_error = str(exc)
            log.error("model load failed: %s", exc)
    return _detector


class DetectRequest(BaseModel):
    """POST /detect body: a base64 JPEG frame plus an optional threshold."""

    imageB64: str = Field(min_length=1)
    confMin: float | None = Field(default=None, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict:
    """Liveness + model identity; ok is false until the weights are loadable."""
    det = _get_detector()
    return {
        "ok": det is not None,
        "model": det.model_name if det else MODEL_PATH.name,
        "version": __version__,
    }


@app.post("/detect")
def detect(req: DetectRequest) -> dict:
    """Detect persons (COCO class 0 only) in one frame."""
    det = _get_detector()
    if det is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    try:
        img = b64_to_bgr(req.imageB64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    boxes, infer_ms = det.detect(img, conf_min=req.confMin)
    return {"boxes": boxes, "inferMs": infer_ms}
