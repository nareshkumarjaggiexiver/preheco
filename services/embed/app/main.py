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
from .codec import frame_from
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
_failed_stat: tuple | None = None  # (mtime_ns, size) of the file that failed


def _get_embedder() -> FaceEmbedder | None:
    """Load the configured family lazily so the app can boot unhealthy without weights.

    The persons review rules apply here too:

      * ANY load failure is caught, not just a missing file — a truncated
        weight raises InvalidProtobuf, and an exception escaping /health
        turns the documented "boot unhealthy" contract into a 500 per probe;
      * the failure is sticky per FILE STATE, not per process: /health keeps
        answering ok:false cheaply, but when MODEL_PATH's bytes CHANGE on
        disk (the bind-mount delivering weights after boot, a re-fetch
        fixing a truncation) the next probe retries — "ok is false until
        the weights are loadable" now means what it says. This matters MORE
        here than in persons/faces: embed deliberately has no POST /model
        (an embedder change is a deployment — see recognizer.py), so a
        stat-gated retry is the ONLY recovery short of a manual restart;
      * one lock: two concurrent first-probes must not both build a session.
    """
    global _embedder, _load_error, _failed_stat
    if _embedder is not None:
        return _embedder
    with _load_lock:
        if _embedder is not None:
            return _embedder
        if _load_error is not None and _current_stat() == _failed_stat:
            return None  # same broken file — stay cheap, stay unhealthy
        # Stat BEFORE the attempt: a writer replacing the file DURING a
        # failed load must invalidate the memo, not be masked by a
        # post-failure stat of the new file (persons review finding).
        attempt_stat = _current_stat()
        try:
            _embedder = build_embedder()
            _load_error = None
            _failed_stat = None
        except Exception as exc:  # noqa: BLE001 — see the docstring
            _load_error = f"{type(exc).__name__}: {exc}"
            _failed_stat = attempt_stat
            log.error("model load failed: %s", _load_error)
    return _embedder


def _current_stat() -> tuple | None:
    """(mtime_ns, size) of the model file, or None while it is absent."""
    try:
        st = MODEL_PATH.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class FaceIn(BaseModel):
    """One face from the faces service: box + YuNet-order landmarks."""

    box: dict
    landmarks: list[list[float]]
    conf: float | None = None


class EmbedRequest(BaseModel):
    """POST /embed body: the frame plus the faces to align and embed."""

    imageB64: str = Field(min_length=1)
    #: See persons.DetectRequest.frameRef — same contract, same fallback.
    frameRef: str | None = None
    faces: list[FaceIn]


@app.get("/health")
def health() -> dict:
    """Liveness + model identity; ok is false until the weights are loadable."""
    emb = _get_embedder()
    return {
        "ok": emb is not None,
        "model": emb.model_name if emb else MODEL_PATH.name,
        "version": __version__,
        # While unhealthy, say WHY: an operator staring at a red healthcheck
        # needs the blocking error (missing file? truncated? bad dtype?)
        # without docker exec — the deploy-integrity lesson.
        "error": None if emb is not None else _load_error,
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
        img = frame_from(req.imageB64, req.frameRef)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        embeddings, align_ms = emb.embed(img, [f.model_dump() for f in req.faces])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"embeddings": embeddings, "alignMs": align_ms}
