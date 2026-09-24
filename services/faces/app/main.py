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
from .codec import frame_from
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
#: (mtime_ns, size) of the configured weight file at the FAILED attempt —
#: the stat gate that keeps the error sticky per FILE STATE, not forever.
_failed_stat: tuple | None = None
#: Serialises persist+swap across concurrent POST /model requests, so the
#: durable .selected and the serving detector cannot end up naming
#: different models (see apply_model). The slow candidate build stays
#: outside it on purpose.
_apply_lock = threading.Lock()


def _get_detector() -> FaceDetector | None:
    """Load the configured family lazily so the app can boot (unhealthy) without weights.

    The persons review rules live here too (ported 2026-08-26):

      * ANY load failure is caught, not just a missing file — a truncated
        weight raises InvalidProtobuf, an exception escaping /health is a
        500 per probe;
      * the failure is sticky per FILE STATE, not per process: /health
        keeps answering ok:false cheaply on the same broken file, but when
        the weights CHANGE on disk (the bind-mount delivering them after a
        raced `make models`, a re-fetch fixing a truncation) the next probe
        retries — "ok is false until the weights are loadable" means what
        it says, without a manual restart or a knowing POST /model;
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
        # Stat BEFORE the attempt: a writer replacing the file DURING a
        # failed load must invalidate the memo, not be masked by a
        # post-failure stat of the new file (persons review finding).
        attempt_stat = _current_stat()
        try:
            _detector = build_detector()
            _load_error = None
            _failed_stat = None
        except Exception as exc:  # noqa: BLE001 — see the docstring
            _load_error = f"{type(exc).__name__}: {exc}"
            _failed_stat = attempt_stat
            log.error("model load failed: %s", _load_error)
    return _detector


def _current_stat() -> tuple | None:
    """(mtime_ns, size) of the configured weight file, or None while absent."""
    try:
        st = MODEL_PATH.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class WithinBox(BaseModel):
    """A person box to scope detection to (extra keys like conf are ignored)."""

    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class DetectRequest(BaseModel):
    """POST /detect body: a base64 JPEG frame, optionally scoped to boxes."""

    #: Empty is legal ONLY with a resolvable frameRef — the pixels come from
    #: the shared transport instead. The handler refuses (400) when neither
    #: yields a frame, which says WHICH half failed; a min_length here would
    #: refuse the ref-only request before anything could look at the ref.
    imageB64: str = ""
    #: See persons.DetectRequest.frameRef — same contract, same fallback.
    frameRef: str | None = None
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
    body = {
        "ok": det is not None,
        "model": det.model_name if det else MODEL_PATH.name,
        "version": __version__,
        # Family + device truth, same shape as persons: requested vs ACTIVE,
        # because accelerator EPs fall back to CPU silently. scoreMin rides
        # along because the families' scores are NOT commensurable — a
        # golden-replay ledger must record which operating point made it.
        "device": {
            "requested": det.device_requested,
            "active": det.providers_active,
            "family": det.family,
            "scoreMin": det.score_min,
        } if det else None,
    }
    if det is None and _load_error:
        # The cause, not just the fact: an operator staring at an unhealthy
        # probe needs to know WHAT blocks the load without docker exec.
        body["error"] = _load_error
    return body


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
    global _detector, _load_error, _failed_stat
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
    # persist + swap are ONE critical section: two concurrent applies must
    # not interleave so .selected names A while the process serves B — that
    # half-state would flip the model on the next random restart with no
    # log trail. The last _apply_lock holder wins BOTH the durable file and
    # the pointer. The slow part (build_detector, above) stays outside; the
    # 507 refusal still runs before anything moves.
    with _apply_lock:
        try:
            persist_selection(name)
        except RuntimeError as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        with _load_lock:
            _detector = candidate
            _load_error = None
            _failed_stat = None
    return {
        "ok": True,
        "model": candidate.model_name,
        "family": candidate.family,
        "device": {
            "requested": candidate.device_requested,
            "active": candidate.providers_active,
            "scoreMin": candidate.score_min,
        },
    }


@app.post("/detect")
def detect(req: DetectRequest) -> dict:
    """Detect faces in one frame — whole-frame, or per person box via `within`."""
    det = _get_detector()
    if det is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    try:
        img = frame_from(req.imageB64, req.frameRef)
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
    out = {"faces": faces, "inferMs": infer_ms}
    if req.within is None and det.family == "scrfd":
        # Additive field, whole-frame scrfd only: the letterbox downscale
        # can leave the 56 px POC floor unresolvable in network pixels, so
        # the caller must be able to tell "no faces present" from "faces
        # this small were invisible" (the `within` crop path never
        # downscales enough to care; YuNet runs at native resolution).
        out["minResolvableFacePx"] = det.min_resolvable_face_px(w, h)
    return out
