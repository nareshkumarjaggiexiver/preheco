"""FastAPI app for the faces service (heco-pipeline port 7104).

Contract (CONTRACTS.md):
    POST /detect {imageB64, within?: [{x, y, w, h, ...}]}
        -> {faces: [{box, landmarks: [5x[x, y]], conf, widthPx, quality,
                     iedPx?, frontality?, sharpness?, eyeSpanRatio?,
                     landmarksPlausible?}], inferMs}
    GET  /health -> {ok, model, version}

With `within`, detection runs per person-box crop and coordinates are mapped
back to frame space; `quality` is the POC flag defined in app/quality.py.  The
three optional signals are what the runner's composite quality gate is built
from — see :func:`_with_quality`.
"""

import logging
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from heco_common.gate_auth import install_bearer_gate
from heco_common.geometry import dedupe_boxes
from pydantic import BaseModel, Field

from . import __version__
from .codec import b64_to_bgr
from .detector import (
    DEFAULT_MODEL,
    MODEL_PATH,
    FaceDetector,
    build_detector,
    persist_selection,
)
from .mapping import clamp_box, offset_face
from .quality import classify_width, crop_sharpness, eye_span_ratio, landmarks_plausible

log = logging.getLogger("faces")

app = FastAPI(title="heco faces", version=__version__)
# Inbound auth (runbook step 8): armed by HECO_REQUIRE_AUTH=1, this refuses
# LAN callers without a bearer credential — an heco-auth token verified
# locally, or the legacy shared secret while it survives. /health stays
# open for the compose healthchecks. Unarmed, nothing changes.
install_bearer_gate(app)

_detector: FaceDetector | None = None
_load_error: str | None = None
_load_lock = threading.Lock()


def _get_detector() -> FaceDetector | None:
    """Load the configured family lazily so the app can boot (unhealthy)
    without weights. Any load failure is caught (the persons review rule:
    a truncated weight raises InvalidProtobuf, not FileNotFoundError, and
    an exception escaping /health is a 500 per probe)."""
    global _detector, _load_error
    if _detector is not None:
        return _detector
    with _load_lock:
        if _detector is None and _load_error is None:
            try:
                _detector = build_detector()
            except Exception as exc:  # noqa: BLE001 — see the docstring
                _load_error = f"{type(exc).__name__}: {exc}"
                log.error("model load failed: %s", _load_error)
    return _detector


class WithinBox(BaseModel):
    """A person box to scope detection to (extra keys like conf are ignored)."""

    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class DetectRequest(BaseModel):
    """POST /detect body: a base64 JPEG frame, optionally scoped to boxes."""

    imageB64: str = Field(min_length=1)
    within: list[WithinBox] | None = None


def _with_quality(face: dict, img: np.ndarray) -> dict:
    """Attach every measured quality signal to a detected face.

    ``widthPx`` and the POC ``quality`` flag are unchanged.  Beside them go the
    three signals the FR literature says actually predict match success, so the
    runner's gate can be composed from them and the pilot can price each floor
    against measured outcomes (accuracy R&D finding F1):

    - ``iedPx`` inter-eye distance from YuNet's two eye landmarks.  This is the
      size measure recognition standards use; our box-width floor of 56/80 px
      maps to only ~24/34 px IED, which is why size should be read in IED.
    - ``frontality`` 0..1 from how centred the nose sits between the eyes.
    - ``sharpness`` variance of the Laplacian over the face crop, normalised
      for crop size (see :func:`crop_sharpness`).  Size and pose can both be
      perfect while the crop is smeared by a walking guest, and no other signal
      here can see that.
    - ``eyeSpanRatio`` eye separation as a fraction of box width — the pose
      reading ``iedPx`` cannot give, because IED in pixels grows as a guest
      walks toward the lens while the ratio collapses in profile at any
      distance (:func:`eye_span_ratio`).
    - ``landmarksPlausible`` whether the 5 landmarks describe a face at all.
      A detection on clothing satisfies the detector's own confidence — the
      sibling pipeline measured striped shirts verifying at 70–91 % — but its
      landmarks are effectively random (:func:`landmarks_plausible`).

    ``img`` is the image the face's box coordinates refer to — the FULL frame
    on both paths, because the ``within`` path offsets crop-space boxes back to
    frame space before this is called.  Getting that pairing wrong would
    silently measure the sharpness of some other part of the picture, so the
    image is a required argument rather than an optional convenience.

    Signals are emitted whenever they can be measured and omitted when they
    cannot (no landmarks, a degenerate box).  Absence means UNKNOWN, and the
    runner's gate is required to read it that way: rejecting a face for a
    signal nobody managed to measure would drop a guest from an invoice.
    """
    width_px = face["box"]["w"]
    out = {**face, "widthPx": width_px, "quality": classify_width(width_px)}
    sharp = crop_sharpness(img, face["box"])
    if sharp is not None:
        out["sharpness"] = sharp
    lm = face.get("landmarks")
    if lm and len(lm) >= 3:
        # Only the nose's x matters for a yaw proxy; its y would speak to
        # pitch, which we do not gate on.
        (rex, rey), (lex, ley), (nx, _ny) = lm[0], lm[1], lm[2]
        ied = ((lex - rex) ** 2 + (ley - rey) ** 2) ** 0.5
        eye_mid_x = (rex + lex) / 2.0
        # frontality: nose offset from the eye midpoint, normalised by IED;
        # 0 offset -> 1.0 (dead frontal), one IED of offset -> 0.0.
        frontality = max(0.0, 1.0 - abs(nx - eye_mid_x) / ied) if ied > 0 else 0.0
        out["iedPx"] = round(ied, 1)
        out["frontality"] = round(frontality, 3)
        span = eye_span_ratio(lm, face["box"])
        if span is not None:
            out["eyeSpanRatio"] = span
    plausible = landmarks_plausible(lm, face["box"])
    if plausible is not None:
        out["landmarksPlausible"] = plausible
    return out


@app.get("/health")
def health() -> dict:
    """Liveness + model identity; ok is false until the weights are loadable."""
    det = _get_detector()
    return {
        "ok": det is not None,
        "model": det.model_name if det else MODEL_PATH.name,
        "version": __version__,
        # Family + device truth, same shape as persons: requested vs ACTIVE,
        # because accelerator EPs fall back to CPU silently.
        "device": {
            "requested": det.device_requested,
            "active": det.providers_active,
            "family": det.family,
        } if det else None,
    }


class ApplyModelRequest(BaseModel):
    """Body of POST /model — which installed weights to run, by filename."""

    file: str = Field(min_length=1, max_length=200)


@app.post("/model")
def apply_model(req: ApplyModelRequest) -> dict:
    """Hot-swap the loaded face detector to another INSTALLED weight file.

    Same contract as persons /model (the planner's no-DevOps path):
    validate-before-swap, atomic under the load lock, durable .selected in
    the bind-mounted models dir, bare filenames only, and this endpoint
    SELECTS among installed weights — it never fetches.
    """
    global _detector, _load_error
    name = req.file.strip()
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="file must be a bare model filename")
    path = DEFAULT_MODEL.parent / name
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{name} is not installed on this box — fetch it (make models / "
            "models-restricted) and verify-models first",
        )
    try:
        candidate = build_detector(path)
    except Exception as exc:  # noqa: BLE001 — the refusal IS the feature
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    with _load_lock:
        _detector = candidate
        _load_error = None
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


@app.post("/detect")
def detect(req: DetectRequest) -> dict:
    """Detect faces in one frame — whole-frame, or per person box via `within`."""
    det = _get_detector()
    if det is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    try:
        img = b64_to_bgr(req.imageB64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    faces: list[dict] = []
    if req.within is None:
        faces = [_with_quality(f, img) for f in det.detect(img)]
    else:
        for box in req.within:
            clamped = clamp_box(box.model_dump(), w, h)
            if clamped is None:
                continue
            cx, cy, cw, ch = clamped
            crop = img[cy : cy + ch, cx : cx + cw]
            for face in det.detect(crop):
                # offset FIRST: _with_quality measures sharpness against `img`,
                # so the box handed to it must already be in frame space.
                faces.append(_with_quality(offset_face(face, cx, cy), img))
        # Overlapping person crops (e.g. a raw box and its track box, or two
        # people cropped together) can surface the SAME physical face twice.
        # Collapse boxes that coincide, keeping the higher-confidence one, so a
        # single guest is embedded and matched once. The 0.6 threshold is high
        # enough that two adjacent faces (cheek to cheek) both survive.
        faces.sort(key=lambda f: f.get("conf", 0.0), reverse=True)
        faces = dedupe_boxes(faces, iou_thr=0.6, box_of=lambda f: f["box"])
    infer_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {"faces": faces, "inferMs": infer_ms}
