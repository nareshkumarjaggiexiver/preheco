"""FastAPI app for the persons service (heco-pipeline port 7102).

Contract (CONTRACTS.md):
    POST /detect {imageB64, confMin?} -> {boxes: [{x, y, w, h, conf}], inferMs}
    GET  /health -> {ok, model, version}
"""

import logging
import threading

from fastapi import FastAPI, HTTPException
from heco_common.gate_auth import install_bearer_gate
from pydantic import BaseModel, Field

from . import __version__
from .codec import b64_to_bgr
from .model import DEFAULT_MODEL, MODEL_PATH, PersonDetector, persist_selection, spec_for

log = logging.getLogger("persons")

app = FastAPI(title="heco persons", version=__version__)
# Inbound auth (runbook step 8): armed by HECO_REQUIRE_AUTH=1, this refuses
# LAN callers without a bearer credential — an heco-auth token verified
# locally, or the legacy shared secret while it survives. /health stays
# open for the compose healthchecks. Unarmed, nothing changes.
install_bearer_gate(app)

_detector: PersonDetector | None = None
_load_error: str | None = None


_load_lock = threading.Lock()
_failed_stat: tuple | None = None  # (mtime_ns, size) of the file that failed


def _get_detector() -> PersonDetector | None:
    """Load the detector lazily so the app can boot (unhealthy) without weights.

    Three review findings live here (2026-08-14), each with its rule:
      * ANY load failure is caught, not just a missing file — a truncated
        weight raises InvalidProtobuf, an OpenVINO compile raises
        RuntimeError, and an exception escaping /health turns the documented
        "boot unhealthy" contract into a 500 per probe;
      * the failure is sticky per FILE STATE, not per process: /health keeps
        answering ok:false cheaply, but when the weights CHANGE on disk (the
        live bind-mount delivering them, or a re-fetch fixing a truncation)
        the next probe retries — "ok is false until the weights are
        loadable" now means what it says, without a manual restart;
      * one lock: two concurrent first-probes must not both build a
        multi-second session.
    """
    global _detector, _load_error, _failed_stat
    if _detector is not None:
        return _detector
    with _load_lock:
        if _detector is not None:
            return _detector
        if _load_error is not None and _current_stat() == _failed_stat:
            return None  # same broken file — stay cheap, stay unhealthy
        # Stat BEFORE the attempt: a writer replacing the file DURING a failed
        # load must invalidate the memo, not be masked by a post-failure stat
        # of the new file (review finding).
        attempt_stat = _current_stat()
        try:
            _detector = PersonDetector()
            _load_error = None
            _failed_stat = None
        except Exception as exc:  # noqa: BLE001 — see the docstring
            _load_error = f"{type(exc).__name__}: {exc}"
            _failed_stat = attempt_stat
            log.error("model load failed: %s", _load_error)
    return _detector


def _current_stat() -> tuple | None:
    """(mtime_ns, size) of the model file, or None while it is absent."""
    try:
        st = MODEL_PATH.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class ApplyModelRequest(BaseModel):
    """Body of POST /model — which installed weights to run, by filename."""

    file: str = Field(min_length=1, max_length=200)


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
        # The device TRUTH, not the request: both ORT accelerator EPs fall
        # back to CPU silently, so /health serves the session's ACTIVE
        # provider list — the benchmark-honesty line, queryable.
        "device": {
            "requested": det.device_requested,
            "active": det.providers_active,
            "family": det.family,
        } if det else None,
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


@app.post("/model")
def apply_model(req: ApplyModelRequest) -> dict:
    """Hot-swap the loaded model to another INSTALLED weight file.

    The planner's no-DevOps path (doc 15 addendum): validate-before-swap —
    the new session is fully constructed and graph-reconciled BEFORE the
    global moves, so a bad request leaves the old model serving untouched;
    the swap is atomic under the load lock; the selection persists in the
    bind-mounted models dir so a container restart keeps it. Weights must
    already be on the box (`make models` / models-restricted): this endpoint
    selects among installed files, it does not fetch — a model download is
    a provisioning step with its own verification, not a request handler.
    Auth rides the standard inbound gate (armed boxes require a token).
    """
    global _detector, _load_error, _failed_stat
    name = req.file.strip()
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="file must be a bare model filename")
    path = DEFAULT_MODEL.parent / name
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{name} is not installed on this box — fetch it with make models "
            "(or models-restricted) and verify-models first",
        )
    spec = spec_for(path)
    try:
        candidate = PersonDetector(path, spec["input"], spec["family"])
    except Exception as exc:  # noqa: BLE001 — the refusal IS the feature
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    with _load_lock:
        _detector = candidate
        _load_error = None
        _failed_stat = None
    persist_selection(name)
    return {
        "ok": True,
        "model": candidate.model_name,
        "family": candidate.family,
        "device": {
            "requested": candidate.device_requested,
            "active": candidate.providers_active,
        },
    }
