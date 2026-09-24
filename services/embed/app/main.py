"""FastAPI app for the embed service (heco-pipeline port 7105).

Contract (CONTRACTS.md):
    POST /embed {imageB64, faces: [{box, landmarks, conf?}]}
        -> {embeddings: [[128 floats]], alignMs,
            norms: [float],                       # L2 norm of each RAW feature
            attributes: [{gender, genderP, age} | null] | null,
            attrMs: float | null}
    GET  /health -> {ok, model, version, device, attrModel, attrError}

`norms` and `attributes` are index-parallel to `embeddings`. `attributes`
is null as a WHOLE when no attribute model is configured (EMBED_ATTR_MODEL
— see attributes.py); null means "not measured", never "nobody".

Callers (the runner) apply the POC quality gate first — only faces at or above
the 56 px floor should reach embedding (sub-canon ones flagged upstream).
"""

import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from heco_common.gate_auth import install_bearer_gate
from heco_common.ort import is_trt
from pydantic import BaseModel, Field

from . import __version__
from .attributes import ATTR_MODEL_EXPLICIT, ATTR_MODEL_PATH, AttributeModel, build_attributes
from .codec import b64_to_bgr
from .recognizer import BATCH, BATCH_MAX, DEVICE, MODEL_PATH, FaceEmbedder, build_embedder

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
    return _stat_of(MODEL_PATH)


def _stat_of(path: Path | None) -> tuple | None:
    """(mtime_ns, size) of `path`, or None while it is absent (or unset)."""
    if path is None:
        return None
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


_attributes: AttributeModel | None = None
_attr_error: str | None = None
_attr_lock = threading.Lock()
_attr_failed_stat: tuple | None = None


def _get_attributes() -> AttributeModel | None:
    """Load the optional genderage model lazily, with the embedder's retry rules.

    Same sticky-per-file-state loop as `_get_embedder` — one failed load is
    memoized until the file's (mtime, size) changes, then retried once —
    with two differences that follow from the pass being OPTIONAL:

      * the DEFAULT file merely being absent is the pass being off, not an
        error: nothing is logged, /health says attrModel null and the
        next probe stats again (so a bind mount delivering the weights
        after boot switches the pass on without a restart);
      * a file the operator NAMED (EMBED_ATTR_MODEL) that is missing or
        refused IS an error — but it is served as /health attrError, not
        as ok:false. Embedding still works and the count must not stop for
        an advisory signal; what must not happen is the loss being silent,
        which is why the reason is served and logged rather than swallowed.
    """
    global _attributes, _attr_error, _attr_failed_stat
    if _attributes is not None:
        return _attributes
    if ATTR_MODEL_PATH is None:
        return None  # EMBED_ATTR_MODEL=off
    with _attr_lock:
        if _attributes is not None:
            return _attributes
        attempt_stat = _stat_of(ATTR_MODEL_PATH)
        if attempt_stat is None and not ATTR_MODEL_EXPLICIT:
            _attr_error = None
            return None  # default file absent: off, quietly
        if _attr_error is not None and attempt_stat == _attr_failed_stat:
            return None  # same broken file — stay cheap
        try:
            # Batching reaches the attribute pass only when switched on; off,
            # this is exactly the call it always was.
            _attributes = (
                build_attributes(ATTR_MODEL_PATH, DEVICE, BATCH, BATCH_MAX)
                if BATCH else build_attributes(ATTR_MODEL_PATH, DEVICE)
            )
            _attr_error = None
            _attr_failed_stat = None
        except Exception as exc:  # noqa: BLE001 — surfaced in /health, see above
            _attr_error = f"{type(exc).__name__}: {exc}"
            _attr_failed_stat = attempt_stat
            log.error("attribute model load failed: %s", _attr_error)
    return _attributes


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
    attrs = _get_attributes()
    return {
        "ok": emb is not None,
        "model": emb.model_name if emb else MODEL_PATH.name,
        "version": __version__,
        # The optional gender/age pass: its file name while loaded, null
        # while off — and, when a NAMED file will not load, the reason
        # (ok stays true: embedding works, the loss must just not be silent).
        "attrModel": attrs.model_name if attrs else None,
        "attrError": None if attrs is not None else _attr_error,
        # While unhealthy, say WHY: an operator staring at a red healthcheck
        # needs the blocking error (missing file? truncated? bad dtype?)
        # without docker exec — the deploy-integrity lesson.
        "error": None if emb is not None else _load_error,
        # Family, device truth AND dimension: the runner cross-checks dim
        # against the deployment's HECO_EMBEDDING_DIM story, and the family
        # says which alignment produced these vectors.
        "device": _device_block(emb) if emb else None,
        # EMBED_BATCH as asked and as it runs (a static-batch graph cannot).
        "knobs": _knobs(emb) if emb else None,
    }


def _knobs(emb) -> dict:
    """Build the /health knobs block: each switch as asked AND as it actually runs."""
    return {
        "batch": {
            "requested": bool(getattr(emb, "batch_requested", False)),
            "active": bool(getattr(emb, "batch_active", False)),
            "max": BATCH_MAX,
        },
    }


def _device_block(emb) -> dict:
    """Build the /health device truth; `trt` rides along only when TRT was asked.

    Keyed on the REQUEST, not on success: a TensorRT that failed to load
    answers "trt": null beside an `active` list without it.
    """
    block = {
        "requested": emb.device_requested,
        "active": emb.providers_active,
        "family": emb.family,
        "dim": emb.dim,
    }
    if is_trt(emb.device_requested):
        block["trt"] = getattr(emb, "trt", None)
    return block


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
        embeddings, norms, attributes, align_ms, attr_ms = emb.embed_faces(
            img, [f.model_dump() for f in req.faces], _get_attributes()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "embeddings": embeddings,
        "alignMs": align_ms,
        "norms": norms,
        "attributes": attributes,
        "attrMs": attr_ms,
    }
