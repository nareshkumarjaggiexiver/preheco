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
