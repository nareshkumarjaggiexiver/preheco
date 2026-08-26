"""FastAPI app for the embed service (heco-pipeline port 7105).

Contract (CONTRACTS.md):
    POST /embed {imageB64, faces: [{box, landmarks, conf?}]}
        -> {embeddings: [[128 floats]], alignMs}
    GET  /health -> {ok, model, version}

Callers (the runner) apply the POC quality gate first — only faces at or above
the 56 px floor should reach embedding (sub-canon ones flagged upstream).
"""

import logging
import threading

from fastapi import FastAPI, HTTPException
from heco_common.gate_auth import install_bearer_gate
from pydantic import BaseModel, Field

from . import __version__
from .codec import b64_to_bgr
from .recognizer import MODEL_PATH, FaceEmbedder, build_embedder

log = logging.getLogger("embed")

app = FastAPI(title="heco embed", version=__version__)
# Inbound auth (runbook step 8): armed by HECO_REQUIRE_AUTH=1, this refuses
# LAN callers without a bearer credential — an heco-auth token verified
# locally, or the legacy shared secret while it survives. /health stays
# open for the compose healthchecks. Unarmed, nothing changes.
install_bearer_gate(app)

_embedder: FaceEmbedder | None = None
_load_error: str | None = None
_load_lock = threading.Lock()


def _get_embedder() -> FaceEmbedder | None:
    """Load the configured family lazily so the app can boot (unhealthy)
    without weights. Any failure is caught, not just a missing file — the
    persons review rule: a truncated weight raises InvalidProtobuf, and an
    exception escaping /health is a 500 per probe."""
    global _embedder, _load_error
    if _embedder is not None:
        return _embedder
    with _load_lock:
        if _embedder is None and _load_error is None:
            try:
                _embedder = build_embedder()
            except Exception as exc:  # noqa: BLE001 — see the docstring
                _load_error = f"{type(exc).__name__}: {exc}"
                log.error("model load failed: %s", _load_error)
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
        # Family, device truth AND dimension: the runner cross-checks dim
        # against the deployment's HECO_EMBEDDING_DIM story, and the family
        # says which alignment produced these vectors.
        "device": {
            "requested": emb.device_requested,
            "active": emb.providers_active,
            "family": emb.family,
            "dim": emb.dim,
        } if emb else None,
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
