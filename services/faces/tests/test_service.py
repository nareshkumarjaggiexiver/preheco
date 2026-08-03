"""Endpoint tests for the faces service.

Model-dependent tests skip loudly when the weights are absent; run
`make models` in services/faces to enable them. No network — TestClient only.
Synthetic frames prove the endpoint contract (shape, quality flags, `within`
mapping path), not detector recall — that needs pilot footage.
"""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

requires_model = pytest.mark.skipif(
    not MODEL.is_file(),
    reason=(
        "face_detection_yunet_2023mar.onnx missing — run `make models` in "
        "services/faces to download it"
    ),
)


def _b64(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _face_frame(w=640, h=480):
    """Draw a crude frontal face; YuNet often (not always) fires on it."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(img, (cx, cy), (70, 95), 0, 0, 360, (140, 160, 200), -1)  # head
    for ex in (cx - 30, cx + 30):
        cv2.circle(img, (ex, cy - 25), 10, (40, 40, 40), -1)  # eyes
    cv2.ellipse(img, (cx, cy + 45), (28, 12), 0, 0, 180, (60, 60, 90), 4)  # mouth
    cv2.line(img, (cx, cy - 5), (cx, cy + 20), (90, 100, 140), 6)  # nose
    return img


@requires_model
def test_health_ok():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "face_detection_yunet_2023mar.onnx"


@requires_model
def test_detect_contract_shape_blank_frame():
    r = TestClient(app).post("/detect", json={"imageB64": _b64(np.zeros((240, 320, 3), np.uint8))})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"faces", "inferMs"}
    assert body["faces"] == []
    assert body["inferMs"] > 0


@requires_model
def test_detect_face_fields_when_any_found():
    r = TestClient(app).post("/detect", json={"imageB64": _b64(_face_frame())})
    assert r.status_code == 200
    for face in r.json()["faces"]:  # shape-checked only if the crude face fires
        assert set(face) == {"box", "landmarks", "conf", "widthPx", "quality"}
        assert len(face["landmarks"]) == 5
        assert face["quality"] in {"ok", "sub-canon", "reject"}
        assert face["widthPx"] == face["box"]["w"]


@requires_model
def test_within_path_returns_and_maps():
    img = _face_frame()
    within = [{"x": 100, "y": 50, "w": 440, "h": 380, "conf": 0.9}]
    r = TestClient(app).post("/detect", json={"imageB64": _b64(img), "within": within})
    assert r.status_code == 200
    for face in r.json()["faces"]:
        # mapped back to frame coordinates -> inside the person box
        assert face["box"]["x"] >= 100 and face["box"]["y"] >= 50


@requires_model
def test_within_degenerate_boxes_skipped():
    img = np.zeros((240, 320, 3), np.uint8)
    within = [{"x": 400, "y": 400, "w": 50, "h": 50}, {"x": 10, "y": 10, "w": 4, "h": 4}]
    r = TestClient(app).post("/detect", json={"imageB64": _b64(img), "within": within})
    assert r.status_code == 200
    assert r.json()["faces"] == []


@requires_model
def test_detect_rejects_bad_base64():
    r = TestClient(app).post("/detect", json={"imageB64": "!!nope!!"})
    assert r.status_code == 400
