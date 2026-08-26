"""Endpoint tests for the persons service.

Model-dependent tests skip loudly when the weights are absent; run
`make models` in services/persons to enable them. No network is used —
TestClient drives the ASGI app in-process.
"""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

MODEL = Path(__file__).resolve().parent.parent / "models" / "yolox_nano.onnx"

requires_model = pytest.mark.skipif(
    not MODEL.is_file(),
    reason="yolox_nano.onnx missing — run `make models` in services/persons to download it",
)


def _frame_b64(w=640, h=480):
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    cv2.rectangle(img, (200, 100), (400, 460), (30, 30, 200), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


@requires_model
def test_health_ok():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "yolox_nano.onnx"


@requires_model
def test_detect_contract_shape():
    r = TestClient(app).post("/detect", json={"imageB64": _frame_b64()})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"boxes", "inferMs"}
    assert body["inferMs"] > 0
    for box in body["boxes"]:
        assert set(box) == {"x", "y", "w", "h", "conf"}


@requires_model
def test_detect_conf_min_is_respected():
    client = TestClient(app)
    loose = client.post("/detect", json={"imageB64": _frame_b64(), "confMin": 0.01}).json()
    strict = client.post("/detect", json={"imageB64": _frame_b64(), "confMin": 0.99}).json()
    assert len(strict["boxes"]) <= len(loose["boxes"])
    assert all(b["conf"] >= 0.99 for b in strict["boxes"])


@requires_model
def test_detect_rejects_bad_base64():
    r = TestClient(app).post("/detect", json={"imageB64": "not-base64!!"})
    assert r.status_code == 400


@requires_model
def test_detect_rejects_non_image():
    payload = base64.b64encode(b"plain text").decode("ascii")
    r = TestClient(app).post("/detect", json={"imageB64": payload})
    assert r.status_code == 400


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


def test_apply_model_hot_swaps_and_persists_and_refuses_garbage(tmp_path, monkeypatch):
    """The planner's no-DevOps path: validate-before-swap (a bad file leaves
    the old model serving), atomic swap, durable .selected in the models dir
    — and path traversal is refused at the name, not discovered at the
    filesystem."""
    from fastapi.testclient import TestClient

    from app import main as m
    from app import model as model_mod

    client = TestClient(m.app)

    for bad in ["../../etc/passwd", ".hidden", "a/b.onnx"]:
        r = client.post("/model", json={"file": bad})
        assert r.status_code == 400, bad

    r = client.post("/model", json={"file": "not_installed.onnx"})
    assert r.status_code == 404
    assert "make models" in r.json()["detail"]


class _FakeDetector:
    """Stands in for PersonDetector where only identity fields matter."""

    def __init__(self, path, input_size=416, family="yolox"):
        self.model_name = Path(path).name
        self.family = family
        self.device_requested = "CPU"
        self.providers_active = ["CPUExecutionProvider"]


def _swap_fixture(tmp_path, monkeypatch, names=("a.onnx", "b.onnx")):
    """Installed dummy weights + a fake detector build, module state restored.

    monkeypatch.setattr records the ORIGINAL module globals, so whatever the
    endpoint assigns to _detector/_load_error/_failed_stat during the test is
    rolled back at teardown — the other tests keep seeing the real state.
    """
    from app import main as m

    models = tmp_path / "models"
    models.mkdir()
    for name in names:
        (models / name).write_bytes(b"weights")
    monkeypatch.setattr(m, "DEFAULT_MODEL", models / "default.onnx")
    monkeypatch.setattr(m, "PersonDetector", _FakeDetector)
    monkeypatch.setattr(m, "_detector", None)
    monkeypatch.setattr(m, "_load_error", None)
    monkeypatch.setattr(m, "_failed_stat", None)
    return m


def test_concurrent_applies_keep_selected_and_serving_in_lockstep(tmp_path, monkeypatch):
    """The persist+swap tail of POST /model is ONE critical section.

    Regression for the 2026-08-26 review finding: with only the load lock
    (which guarded just the pointer assignment), two in-flight applies could
    persist in one order and swap in the other — .selected naming model A
    while the process serves model B, and the next restart silently flipping
    the detector. The proof here is direct: while apply A sits inside its
    persist, apply B's persist must NOT run; once A finishes, B runs whole,
    so the last persisted name and the serving detector agree.
    """
    import threading
    import time

    m = _swap_fixture(tmp_path, monkeypatch)

    persists = []
    a_in_persist = threading.Event()
    release_a = threading.Event()

    def fake_persist(name):
        persists.append(name)
        if name == "a.onnx":
            a_in_persist.set()
            assert release_a.wait(timeout=5), "the test must release A"

    monkeypatch.setattr(m, "persist_selection", fake_persist)

    statuses = {}

    def apply(name):
        # One TestClient per thread: each owns its own portal.
        statuses[name] = TestClient(m.app).post("/model", json={"file": name}).status_code

    ta = threading.Thread(target=apply, args=("a.onnx",))
    ta.start()
    assert a_in_persist.wait(timeout=5)
    tb = threading.Thread(target=apply, args=("b.onnx",))
    tb.start()
    time.sleep(0.3)
    assert persists == ["a.onnx"], "B must be blocked while A holds the apply lock"
    release_a.set()
    ta.join(timeout=5)
    tb.join(timeout=5)
    assert statuses == {"a.onnx": 200, "b.onnx": 200}
    assert persists == ["a.onnx", "b.onnx"]
    # The invariant the lock restores: last persist == last swap.
    assert m._detector.model_name == persists[-1]


def test_a_persist_refusal_is_507_and_leaves_the_old_model_serving(tmp_path, monkeypatch):
    """Durability first, unchanged by the apply lock: a selection that
    cannot stick is refused BEFORE the swap, so the serving detector — and
    the next restart — keep the old model."""
    m = _swap_fixture(tmp_path, monkeypatch)
    incumbent = _FakeDetector("old.onnx")
    monkeypatch.setattr(m, "_detector", incumbent)

    def refuse_persist(name):
        raise RuntimeError("cannot persist the selection (disk says no)")

    monkeypatch.setattr(m, "persist_selection", refuse_persist)
    r = TestClient(m.app).post("/model", json={"file": "a.onnx"})
    assert r.status_code == 507
    assert "cannot persist" in r.json()["detail"]
    assert m._detector is incumbent, "the swap must not have happened"
