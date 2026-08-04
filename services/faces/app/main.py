"""FastAPI app for the faces service (heco-pipeline port 7104).

Contract (CONTRACTS.md):
    POST /detect {imageB64, within?: [{x, y, w, h, ...}]}
        -> {faces: [{box, landmarks: [5x[x, y]], conf, widthPx, quality}], inferMs}
    GET  /health -> {ok, model, version}

With `within`, detection runs per person-box crop and coordinates are mapped
back to frame space; `quality` is the POC flag defined in app/quality.py.
"""

import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .codec import b64_to_bgr
from .detector import MODEL_PATH, FaceDetector
from .mapping import clamp_box, offset_face
from heco_common.geometry import dedupe_boxes
from .quality import classify_width

log = logging.getLogger("faces")

app = FastAPI(title="heco faces", version=__version__)

_detector: FaceDetector | None = None
_load_error: str | None = None


def _get_detector() -> FaceDetector | None:
    """Load YuNet lazily so the app can boot (unhealthy) without weights."""
    global _detector, _load_error
    if _detector is None and _load_error is None:
        try:
            _detector = FaceDetector()
        except FileNotFoundError as exc:
            _load_error = str(exc)
            log.error("model load failed: %s", exc)
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


def _with_quality(face: dict) -> dict:
    """Attach quality signals to a detected face.

    ``widthPx`` and the POC ``quality`` flag remain the gate (unchanged). We
    ALSO emit, report-only, the signals the FR literature says actually predict
    match success, so the pilot can set an evidence-based floor and calibrate
    the match threshold (accuracy R&D finding F1):

    - ``iedPx``  inter-eye distance from YuNet's two eye landmarks. This is the
      size measure recognition standards use; our box-width floor of 56/80 px
      maps to only ~24/34 px IED, which is why size should be read in IED.
    - ``sharpness`` a relative focus proxy (variance of the landmark spread is
      unavailable here without the crop, so this is the eye-to-mouth span used
      as a rough scale-normalised proxy; the runner adds Laplacian sharpness
      where it still holds the crop).
    - ``frontality`` 0..1 from how centred the nose sits between the eyes.

    Nothing here changes a decision; the gate stays width-only until pilot data
    justifies moving it.
    """
    width_px = face["box"]["w"]
    out = {**face, "widthPx": width_px, "quality": classify_width(width_px)}
    lm = face.get("landmarks")
    if lm and len(lm) >= 3:
        (rex, rey), (lex, ley), (nx, ny) = lm[0], lm[1], lm[2]
        ied = ((lex - rex) ** 2 + (ley - rey) ** 2) ** 0.5
        eye_mid_x = (rex + lex) / 2.0
        # frontality: nose offset from the eye midpoint, normalised by IED;
        # 0 offset -> 1.0 (dead frontal), one IED of offset -> 0.0.
        frontality = max(0.0, 1.0 - abs(nx - eye_mid_x) / ied) if ied > 0 else 0.0
        out["iedPx"] = round(ied, 1)
        out["frontality"] = round(frontality, 3)
    return out


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
        faces = [_with_quality(f) for f in det.detect(img)]
    else:
        for box in req.within:
            clamped = clamp_box(box.model_dump(), w, h)
            if clamped is None:
                continue
            cx, cy, cw, ch = clamped
            crop = img[cy : cy + ch, cx : cx + cw]
            for face in det.detect(crop):
                faces.append(_with_quality(offset_face(face, cx, cy)))
        # Overlapping person crops (e.g. a raw box and its track box, or two
        # people cropped together) can surface the SAME physical face twice.
        # Collapse boxes that coincide, keeping the higher-confidence one, so a
        # single guest is embedded and matched once. The 0.6 threshold is high
        # enough that two adjacent faces (cheek to cheek) both survive.
        faces.sort(key=lambda f: f.get("conf", 0.0), reverse=True)
        faces = dedupe_boxes(faces, iou_thr=0.6, box_of=lambda f: f["box"])
    infer_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {"faces": faces, "inferMs": infer_ms}
