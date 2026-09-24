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
def test_embed_returns_128_floats_per_face(monkeypatch):
    """The day-one shape plus the additive fields: `norms` index-parallel
    to the embeddings (the raw SFace feature's L2 norm, always > 0), and
    `attributes`/`attrMs` null as a whole while no attribute model is
    configured — null, not [], because absent is not zero."""
    import app.main as app_main

    _attributes_off(monkeypatch, app_main)
    r = TestClient(app).post(
        "/embed", json={"imageB64": _b64(_frame()), "faces": [_face(), _face(cx=160.0)]}
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"embeddings", "alignMs", "norms", "attributes", "attrMs", "balance"}
    assert len(body["embeddings"]) == 2
    for emb, norm in zip(body["embeddings"], body["norms"], strict=True):
        assert len(emb) == 128
        assert all(isinstance(v, float) for v in emb)
        assert any(v != 0.0 for v in emb)
        assert norm == pytest.approx(float(np.linalg.norm(emb)), rel=1e-6)
        assert norm > 0.0
    assert body["alignMs"] > 0
    assert body["attributes"] is None
    assert body["attrMs"] is None


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


def _reset_loader(monkeypatch, app_main, path):
    """Point the lazy loader at `path` with a clean slate (monkeypatch
    restores the real globals afterwards, so the model-backed tests above
    are untouched whichever order pytest runs them in). The attribute pass
    is switched OFF unless a test arms it — a genderage.onnx in models/
    must not change what these tests see."""
    monkeypatch.setattr(app_main, "MODEL_PATH", path)
    monkeypatch.setattr(app_main, "_embedder", None)
    monkeypatch.setattr(app_main, "_load_error", None)
    monkeypatch.setattr(app_main, "_failed_stat", None)
    _attributes_off(monkeypatch, app_main)


def _attributes_off(monkeypatch, app_main):
    """EMBED_ATTR_MODEL=off, as the loader would resolve it, with a clean slate."""
    _attributes_at(monkeypatch, app_main, None, explicit=True)


def _attributes_at(monkeypatch, app_main, path, explicit):
    monkeypatch.setattr(app_main, "ATTR_MODEL_PATH", path)
    monkeypatch.setattr(app_main, "ATTR_MODEL_EXPLICIT", explicit)
    monkeypatch.setattr(app_main, "_attributes", None)
    monkeypatch.setattr(app_main, "_attr_error", None)
    monkeypatch.setattr(app_main, "_attr_failed_stat", None)


def test_health_reports_the_error_and_recovers_when_weights_appear(tmp_path, monkeypatch):
    """The sticky-load-error fix (persons' stat-gated retry, ported):

    * the first failed load is memoized — same file state, no retry storm;
    * /health says WHY it is unhealthy (the `error` field), not just false;
    * the moment the weights CHANGE on disk (`make models` finishing after
      boot), the next probe retries and the service self-heals — critical
      here because embed deliberately has no POST /model to clear the error.
    """
    import app.main as app_main

    path = tmp_path / "model.onnx"
    calls = []

    class DummyEmbedder:
        model_name = "model.onnx"
        device_requested = "CPU"
        providers_active = ["cv2"]
        family = "sface"
        dim = 128

    def fake_build():
        calls.append(1)
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — run `make models` in services/embed")
        return DummyEmbedder()

    _reset_loader(monkeypatch, app_main, path)
    monkeypatch.setattr(app_main, "build_embedder", fake_build)
    client = TestClient(app_main.app)

    first = client.get("/health").json()
    assert first["ok"] is False
    assert "FileNotFoundError" in first["error"]
    assert len(calls) == 1

    assert client.get("/health").json()["ok"] is False
    assert len(calls) == 1, "same absent file — memoized, no load storm"

    path.write_bytes(b"weights")  # `make models` delivers the file
    third = client.get("/health").json()
    assert third["ok"] is True
    assert third["error"] is None
    assert len(calls) == 2, "changed file state — exactly one retry"


def test_a_replaced_broken_weight_is_retried_without_restart(tmp_path, monkeypatch):
    """The truncation shape of the same fix: a broken file stays a cheap
    ok:false until its (mtime, size) changes, then one retry loads it."""
    import app.main as app_main

    path = tmp_path / "model.onnx"
    path.write_bytes(b"truncated")
    calls = []

    class DummyEmbedder:
        model_name = "model.onnx"
        device_requested = "CPU"
        providers_active = ["cv2"]
        family = "sface"
        dim = 128

    def fake_build():
        calls.append(1)
        if path.read_bytes() == b"truncated":
            raise RuntimeError("[ONNXRuntimeError] INVALID_PROTOBUF")
        return DummyEmbedder()

    _reset_loader(monkeypatch, app_main, path)
    monkeypatch.setattr(app_main, "build_embedder", fake_build)
    client = TestClient(app_main.app)

    assert client.get("/health").json()["ok"] is False
    assert client.get("/health").json()["ok"] is False
    assert len(calls) == 1, "same broken bytes — no retry"

    path.write_bytes(b"the full weights")  # re-fetch fixes the truncation
    assert client.get("/health").json()["ok"] is True
    assert len(calls) == 2


def test_arcface_family_serves_the_contract_end_to_end(tmp_path, monkeypatch):
    """POST /embed against a real (tiny) arcface graph: dim read from the
    graph into /health, embeddings served, and the two landmark refusals the
    review demanded — malformed nesting and degenerate points — both 400,
    exactly like the sface path."""
    from tiny_onnx import FLOAT, write_model

    import app.main as app_main
    from app.recognizer import ArcFaceEmbedder

    path = write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112))
    _reset_loader(monkeypatch, app_main, path)
    monkeypatch.setattr(app_main, "build_embedder", lambda: ArcFaceEmbedder(path, "CPU"))
    client = TestClient(app_main.app)

    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["device"]["family"] == "arcface"
    assert health["device"]["dim"] == 3 * 112 * 112

    ok = client.post("/embed", json={"imageB64": _b64(_frame()), "faces": [_face()]})
    assert ok.status_code == 200
    assert len(ok.json()["embeddings"][0]) == 3 * 112 * 112

    nested = _face()
    nested["landmarks"] = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
    r = client.post("/embed", json={"imageB64": _b64(_frame()), "faces": [nested]})
    assert r.status_code == 400
    assert "five [x, y] pairs" in r.json()["detail"]

    degenerate = _face()
    degenerate["landmarks"] = [[0.0, 0.0]] * 5
    r = client.post("/embed", json={"imageB64": _b64(_frame()), "faces": [degenerate]})
    assert r.status_code == 400
    assert "degenerate landmarks" in r.json()["detail"]


def test_embed_serves_norms_and_attributes_beside_the_embeddings(tmp_path, monkeypatch):
    """The additive contract on the wire, against two tiny graphs: the
    arcface Flatten (whose embedding IS the fed blob, so the norm is
    checkable to the digit) and the ReduceMean genderage stand-in (whose
    answer is the crop's RGB means, so gender/age follow the paint)."""
    from tiny_onnx import FLOAT, write_model, write_reduce_mean_model

    import app.main as app_main
    from app.attributes import AttributeModel
    from app.recognizer import ArcFaceEmbedder

    path = write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112))
    attr_path = write_reduce_mean_model(tmp_path, "tiny_genderage.onnx", FLOAT, (1, 3, 96, 96))
    _reset_loader(monkeypatch, app_main, path)
    monkeypatch.setattr(app_main, "build_embedder", lambda: ArcFaceEmbedder(path, "CPU"))
    _attributes_at(monkeypatch, app_main, attr_path, explicit=True)
    monkeypatch.setattr(
        app_main, "build_attributes", lambda p, d: AttributeModel(p, "CPU")
    )
    client = TestClient(app_main.app)

    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["attrModel"] == "tiny_genderage.onnx"
    assert health["attrError"] is None

    # Paint the frame so the attribute crop reads [R, G, B] = [10, 20, 45]:
    # G > R is "M", B is age 4500 — nonsense years, exact arithmetic.
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (45, 20, 10)
    faces = [_face(), _face(cx=160.0)]
    r = client.post("/embed", json={"imageB64": _b64(img), "faces": faces})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"embeddings", "alignMs", "norms", "attributes", "attrMs", "balance"}
    assert len(body["embeddings"]) == len(body["norms"]) == len(body["attributes"]) == 2
    for emb, norm in zip(body["embeddings"], body["norms"], strict=True):
        assert norm == pytest.approx(float(np.linalg.norm(np.asarray(emb, np.float64))), rel=1e-6)
    for attr in body["attributes"]:
        assert set(attr) == {"gender", "genderP", "age"}
        assert attr["gender"] == "M"
        assert 0.99 < attr["genderP"] <= 1.0
        assert attr["age"] == pytest.approx(4500.0, rel=1e-3)
    assert body["attrMs"] >= 0.0
    assert body["alignMs"] >= 0.0

    # No faces: every list empty, attributes an empty LIST (the pass is on).
    r = client.post("/embed", json={"imageB64": _b64(img), "faces": []})
    assert r.json()["attributes"] == [] and r.json()["norms"] == []

    # A malformed box is the same 400 as malformed landmarks, on this path too.
    bad = _face()
    bad["box"] = {"x": 1.0, "y": 2.0}
    r = client.post("/embed", json={"imageB64": _b64(img), "faces": [bad]})
    assert r.status_code == 400
    assert "x, y, w, h" in r.json()["detail"]


def test_a_named_attribute_model_that_will_not_load_is_served_not_swallowed(
    tmp_path, monkeypatch
):
    """EMBED_ATTR_MODEL names a file that is absent: embedding still works
    (ok:true — the count must not stop for an advisory signal) but /health
    carries attrError and /embed answers attributes:null; when the file
    appears the next probe loads it, exactly like the embedder's retry."""
    from tiny_onnx import FLOAT, write_model, write_reduce_mean_model

    import app.main as app_main
    from app.attributes import AttributeModel
    from app.recognizer import ArcFaceEmbedder

    path = write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112))
    attr_path = tmp_path / "named_genderage.onnx"
    _reset_loader(monkeypatch, app_main, path)
    monkeypatch.setattr(app_main, "build_embedder", lambda: ArcFaceEmbedder(path, "CPU"))
    _attributes_at(monkeypatch, app_main, attr_path, explicit=True)
    calls = []

    def build(p, d):
        calls.append(1)
        return AttributeModel(p, "CPU")

    monkeypatch.setattr(app_main, "build_attributes", build)
    client = TestClient(app_main.app)

    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["attrModel"] is None
    assert "FileNotFoundError" in health["attrError"]
    assert client.get("/health").json()["attrModel"] is None
    assert len(calls) == 1, "same absent file — memoized, no load storm"

    r = client.post("/embed", json={"imageB64": _b64(_frame()), "faces": [_face()]})
    assert r.status_code == 200
    assert r.json()["attributes"] is None and r.json()["attrMs"] is None
    assert len(r.json()["norms"]) == 1

    write_reduce_mean_model(tmp_path, "named_genderage.onnx", FLOAT, (1, 3, 96, 96))
    health = client.get("/health").json()
    assert health["attrModel"] == "named_genderage.onnx"
    assert health["attrError"] is None
    assert len(calls) == 2, "changed file state — exactly one retry"
    r = client.post("/embed", json={"imageB64": _b64(_frame()), "faces": [_face()]})
    assert len(r.json()["attributes"]) == 1


def test_the_default_attribute_file_being_absent_is_off_not_an_error(tmp_path, monkeypatch):
    from tiny_onnx import FLOAT, write_model

    import app.main as app_main
    from app.recognizer import ArcFaceEmbedder

    path = write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112))
    _reset_loader(monkeypatch, app_main, path)
    monkeypatch.setattr(app_main, "build_embedder", lambda: ArcFaceEmbedder(path, "CPU"))
    _attributes_at(monkeypatch, app_main, tmp_path / "genderage.onnx", explicit=False)
    calls = []
    monkeypatch.setattr(app_main, "build_attributes", lambda p, d: calls.append(1))
    client = TestClient(app_main.app)

    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["attrModel"] is None
    assert health["attrError"] is None
    assert calls == [], "an absent default file is never even attempted"
    r = client.post("/embed", json={"imageB64": _b64(_frame()), "faces": [_face()]})
    assert r.status_code == 200
    assert r.json()["attributes"] is None


def test_an_init_refused_model_is_unhealthy_with_the_reason(tmp_path, monkeypatch):
    """An unsupported graph must be /health ok:false WITH the reason — never
    ok:true followed by a 500 per /embed (the finding's failure shape). The
    double-dtype fixture stands in for any init-refused export."""
    from tiny_onnx import DOUBLE, write_model

    import app.main as app_main

    path = write_model(tmp_path, "tiny_arcface_double.onnx", DOUBLE, (1, 3, 112, 112))
    _reset_loader(monkeypatch, app_main, path)
    # The real factory: spec_for routes the name to the arcface family,
    # whose init refuses the dtype.
    from app.recognizer import build_embedder as real_build
    monkeypatch.setattr(app_main, "build_embedder", lambda: real_build(path, "CPU"))
    client = TestClient(app_main.app)

    health = client.get("/health").json()
    assert health["ok"] is False
    assert "tensor(double)" in health["error"]
    r = client.post("/embed", json={"imageB64": _b64(_frame()), "faces": []})
    assert r.status_code == 503, "unloadable model refuses requests, never 500s them"


def test_inbound_auth_gate_refuses_the_open_lan_when_armed(monkeypatch):
    """HECO_REQUIRE_AUTH=1 turns the LAN door off (runbook step 8).

    No credential -> 401 with the machine-readable code; a token signed by the
    auth service passes; /health stays open for the compose healthcheck. The
    gate sits ahead of routing, so an unknown path proves both halves: 401
    without a credential, 404 — the router's own answer — with one. Every
    other test in this file runs unarmed and is untouched.

    The key set is PINNED here (HECO_JWKS_JSON) so the check is hermetic: no
    network, no Worker, no clock beyond the token's own expiry.
    """
    import base64
    import json as _json
    import time as _time

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from fastapi.testclient import TestClient

    from app.main import app

    def b64(raw):
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    key = Ed25519PrivateKey.generate()
    jwks = {"keys": [{
        "kty": "OKP", "crv": "Ed25519", "kid": "k-test", "alg": "EdDSA", "use": "sig",
        "x": b64(key.public_key().public_bytes_raw()),
    }]}
    head = b64(_json.dumps({"alg": "EdDSA", "kid": "k-test", "typ": "JWT"}).encode())
    body = b64(_json.dumps({
        "iss": "https://auth.test", "aud": "heco-planner",
        "iat": int(_time.time()), "exp": int(_time.time()) + 3600,
    }).encode())
    token = f"{head}.{body}.{b64(key.sign(f'{head}.{body}'.encode()))}"

    monkeypatch.setenv("HECO_REQUIRE_AUTH", "1")
    monkeypatch.setenv("HECO_JWKS_JSON", _json.dumps(jwks))
    monkeypatch.setenv("HECO_AUTH_ISSUER", "https://auth.test")
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    refused = client.get("/gate-probe")
    assert refused.status_code == 401
    assert refused.json()["code"] == "auth"

    allowed = client.get("/gate-probe", headers={"Authorization": f"Bearer {token}"})
    assert allowed.status_code == 404, "a valid credential reaches the router itself"


def test_health_device_and_knobs_blocks():
    """device keeps its four keys off TRT (trt only rides a TRT request), and
    knobs shows EMBED_BATCH as asked AND as it runs — off by default."""
    from types import SimpleNamespace

    from app.main import _device_block, _knobs

    emb = SimpleNamespace(device_requested="CUDA", family="arcface", dim=512, trt=None,
                          providers_active=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert set(_device_block(emb)) == {"requested", "active", "family", "dim"}
    emb.device_requested = "TENSORRT"
    assert _device_block(emb)["trt"] is None
    assert _knobs(emb) == {"batch": {"requested": False, "active": False, "max": 16}}
    emb.batch_requested, emb.batch_active = True, False
    assert _knobs(emb)["batch"] == {"requested": True, "active": False, "max": 16}
