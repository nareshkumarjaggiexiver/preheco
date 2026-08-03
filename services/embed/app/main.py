"""FastAPI app for the embed service (heco-pipeline port 7105).

Contract (CONTRACTS.md):
    POST /embed {imageB64, faces: [{box, landmarks, conf?}]}
        -> {embeddings: [[128 floats]], alignMs}
    GET  /health -> {ok, model, version}

Callers (the runner) apply the POC quality gate first — only faces at or above
the 56 px floor should reach embedding (sub-canon ones flagged upstream).
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .codec import b64_to_bgr
from .recognizer import MODEL_PATH, FaceEmbedder

log = logging.getLogger("embed")

app = FastAPI(title="heco embed", version=__version__)

_embedder: FaceEmbedder | None = None
_load_error: str | None = None


def _get_embedder() -> FaceEmbedder | None:
    """Load SFace lazily so the app can boot (unhealthy) without weights."""
    global _embedder, _load_error
    if _embedder is None and _load_error is None:
        try:
            _embedder = FaceEmbedder()
        except FileNotFoundError as exc:
            _load_error = str(exc)
            log.error("model load failed: %s", exc)
    return _embedder


class FaceIn(BaseModel):
    """One face from the faces service: box + YuNet-order landmarks."""

    box: dict
    landmarks: list[list[float]]
    conf: float | None = None


class EmbedRequest(BaseModel):
    """POST /embed body: the frame plus the faces to align and embed."""

    imageB64: str = Field(min_length=1)
    faces: list[FaceIn]


@app.get("/health")
def health() -> dict:
    """Liveness + model identity; ok is false until the weights are loadable."""
    emb = _get_embedder()
    return {
        "ok": emb is not None,
        "model": emb.model_name if emb else MODEL_PATH.name,
        "version": __version__,
    }


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    """Align (by landmarks) and embed each face; order matches the request."""
    emb = _get_embedder()
    if emb is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    try:
        img = b64_to_bgr(req.imageB64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        embeddings, align_ms = emb.embed(img, [f.model_dump() for f in req.faces])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"embeddings": embeddings, "alignMs": align_ms}
