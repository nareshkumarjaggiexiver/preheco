"""Endpoint tests for the embed service.

Model-dependent tests skip loudly when the weights are absent; run
`make models` in services/embed to enable them. No network — TestClient only.
SFace align+embed works on any pixels once given plausible landmarks, so
synthetic frames fully exercise the real model here.
"""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_recognition_sface_2021dec.onnx"

requires_model = pytest.mark.skipif(
    not MODEL.is_file(),
    reason=(
        "face_recognition_sface_2021dec.onnx missing — run `make models` in "
        "services/embed to download it"
    ),
)


def _b64(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _frame(seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def _face(cx=320.0, cy=240.0, w=80.0, h=100.0):
    return {
        "box": {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h},
        "landmarks": [
            [cx - 20, cy - 20],  # right eye
            [cx + 20, cy - 20],  # left eye
            [cx, cy + 2],  # nose tip
            [cx - 15, cy + 25],  # right mouth corner
            [cx + 15, cy + 25],  # left mouth corner
        ],
        "conf": 0.9,
    }


@requires_model
def test_health_ok():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "face_recognition_sface_2021dec.onnx"


@requires_model
def test_embed_returns_128_floats_per_face():
    r = TestClient(app).post(
        "/embed", json={"imageB64": _b64(_frame()), "faces": [_face(), _face(cx=160.0)]}
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"embeddings", "alignMs"}
    assert len(body["embeddings"]) == 2
    for emb in body["embeddings"]:
        assert len(emb) == 128
        assert all(isinstance(v, float) for v in emb)
        assert any(v != 0.0 for v in emb)
    assert body["alignMs"] > 0


@requires_model
def test_embed_is_deterministic_for_same_input():
    client = TestClient(app)
    payload = {"imageB64": _b64(_frame()), "faces": [_face()]}
    a = client.post("/embed", json=payload).json()["embeddings"][0]
    b = client.post("/embed", json=payload).json()["embeddings"][0]
    assert a == b


@requires_model
def test_embed_empty_faces_is_empty_list():
    r = TestClient(app).post("/embed", json={"imageB64": _b64(_frame()), "faces": []})
    assert r.status_code == 200
    assert r.json()["embeddings"] == []


@requires_model
def test_embed_rejects_bad_base64():
    r = TestClient(app).post("/embed", json={"imageB64": "@@@", "faces": []})
    assert r.status_code == 400


@requires_model
def test_embed_rejects_malformed_landmarks():
    face = _face()
    face["landmarks"] = face["landmarks"][:3]
    r = TestClient(app).post("/embed", json={"imageB64": _b64(_frame()), "faces": [face]})
    assert r.status_code == 400
