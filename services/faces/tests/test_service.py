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
        # The four always-present keys, plus the measured signals, which are
        # emitted only when measurable (see _with_quality).
        assert {"box", "landmarks", "conf", "widthPx", "quality"} <= set(face)
        assert set(face) <= {
            "box", "landmarks", "conf", "widthPx", "quality",
            "iedPx", "frontality", "sharpness",
        }
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


def test_ied_and_frontality_emitted():
    """F1: a face with landmarks reports inter-eye distance and frontality."""
    from app.main import _with_quality
    img = _face_frame(320, 240)
    face = {
        "box": {"x": 0, "y": 0, "w": 100, "h": 120},
        "landmarks": [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]],
        "conf": 0.9,
    }
    out = _with_quality(face, img)
    assert out["iedPx"] == 40.0            # |70-30| horizontally
    assert out["frontality"] == 1.0        # nose dead-centre between the eyes
    # a face with no landmarks still gets width + quality, no IED keys
    bare = _with_quality({"box": {"x": 0, "y": 0, "w": 60, "h": 70}, "conf": 0.8}, img)
    assert bare["widthPx"] == 60 and "iedPx" not in bare


def test_sharpness_emitted_beside_the_other_signals():
    """The docstring used to promise `sharpness` and nothing computed it.

    Regression for a claim, not a crash: a reader ticked "IED + Laplacian
    sharpness" off as shipped, and the gate that was meant to use it had only
    two of its signals to compose from.
    """
    from app.main import _with_quality
    box = {"x": 40, "y": 30, "w": 100, "h": 120}
    face = {"box": box, "conf": 0.9}

    sharp = _with_quality(face, _face_frame(320, 240))
    blurred = _with_quality(face, cv2.GaussianBlur(_face_frame(320, 240), (11, 11), 0))
    assert sharp["sharpness"] > blurred["sharpness"]

    # Unmeasurable stays absent, so the gate can tell "blurred" from "unknown".
    off_frame = _with_quality({"box": {"x": 900, "y": 900, "w": 40, "h": 40}}, _face_frame())
    assert "sharpness" not in off_frame


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


def test_health_serves_family_and_device_truth():
    """The persons convention, now at faces: requested vs ACTIVE, plus which
    family answered — the yunet family honestly reports cv2, because OpenCV
    ignores acceleration targets (measured on the iGPU bench). The active
    scoreMin rides along: the families' scores are not commensurable, so a
    golden-replay ledger must record which operating point produced it."""
    from fastapi.testclient import TestClient

    from app import main as m

    body = TestClient(m.app).get("/health").json()
    assert body["ok"] is True
    assert body["device"]["family"] == "yunet"
    assert body["device"]["active"] == ["cv2"]
    assert body["device"]["scoreMin"] == 0.8  # FACES_SCORE_MIN, the yunet knob
    assert "error" not in body, "the error field is for unhealthy answers only"


def test_apply_model_refuses_traversal_and_uninstalled_files():
    from fastapi.testclient import TestClient

    from app import main as m

    client = TestClient(m.app)
    for bad in ["../x.onnx", ".selected", "a/b.onnx"]:
        assert client.post("/model", json={"file": bad}).status_code == 400, bad
    # A name no box will ever hold: scrfd_2.5g_kps.onnx is a real, fetched
    # weight on the CUDA box and in dev checkouts now, and a present file
    # answers 400/200, not 404 — the test must not depend on what is fetched.
    r = client.post("/model", json={"file": "scrfd_never_installed_kps.onnx"})
    assert r.status_code == 404
    assert "models-restricted" in r.json()["detail"]


def _fake_family(name, family="scrfd", score_min=0.5):
    """A stand-in detector carrying every attribute /health and /model serve."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model_name=name,
        family=family,
        device_requested="CPU",
        providers_active=["CPUExecutionProvider"] if family == "scrfd" else ["cv2"],
        score_min=score_min,
        detect=lambda img: [],
        min_resolvable_face_px=lambda w, h: 96,
    )


def test_load_error_unsticks_when_the_weight_file_changes(monkeypatch, tmp_path):
    """The persons stat-gated retry, ported: the first failure is cached per
    FILE STATE, so /health stays cheap on the same broken file but heals on
    the very next probe once the weights appear — no manual restart, no
    knowing POST /model. The cause is served in the unhealthy body."""
    from fastapi.testclient import TestClient

    from app import main as m

    weight = tmp_path / "w.onnx"  # absent at the first attempt
    monkeypatch.setattr(m, "MODEL_PATH", weight)
    monkeypatch.setattr(m, "_detector", None)
    monkeypatch.setattr(m, "_load_error", None)
    monkeypatch.setattr(m, "_failed_stat", None)
    calls = []

    def failing():
        calls.append(1)
        raise FileNotFoundError(f"{weight} missing")

    monkeypatch.setattr(m, "build_detector", failing)
    client = TestClient(m.app)

    body = client.get("/health").json()
    assert body["ok"] is False
    assert "missing" in body["error"], "the unhealthy body names the blocker"
    # Same broken state: the memo holds, no second multi-second build.
    assert client.get("/health").json()["ok"] is False
    assert len(calls) == 1

    # The weights appear — the next probe retries and heals.
    weight.write_bytes(b"weights")
    monkeypatch.setattr(m, "build_detector", lambda: _fake_family("w.onnx", "yunet", 0.8))
    body = client.get("/health").json()
    assert body["ok"] is True
    assert "error" not in body


def test_whole_frame_scrfd_reply_carries_the_resolvable_floor(monkeypatch):
    """Additive field on the whole-frame scrfd path only: `faces: []` on a
    big letterboxed frame must not be readable as "no faces present"."""
    from fastapi.testclient import TestClient

    from app import main as m

    monkeypatch.setattr(m, "_detector", _fake_family("scrfd_test_kps.onnx"))
    client = TestClient(m.app)
    frame = _b64(np.zeros((240, 320, 3), np.uint8))

    whole = client.post("/detect", json={"imageB64": frame}).json()
    assert whole["minResolvableFacePx"] == 96

    # The `within` crop path never downscales enough to care — no field.
    within = client.post(
        "/detect",
        json={"imageB64": frame, "within": [{"x": 0, "y": 0, "w": 100, "h": 100}]},
    ).json()
    assert "minResolvableFacePx" not in within
    # The yunet family never carries it either — pinned by
    # test_detect_contract_shape_blank_frame's exact key-set assertion.


def test_apply_persists_and_swaps_under_one_lock(monkeypatch, tmp_path):
    """persist + swap are one critical section: the durable .selected and
    the serving detector must be written by the same lock holder."""
    from fastapi.testclient import TestClient

    from app import main as m

    (tmp_path / "cand.onnx").write_bytes(b"x")
    cand = _fake_family("cand.onnx")
    monkeypatch.setattr(m, "DEFAULT_MODEL", tmp_path / "default.onnx")
    monkeypatch.setattr(m, "build_detector", lambda p: cand)
    seen = {}
    monkeypatch.setattr(
        m, "persist_selection", lambda name: seen.__setitem__("locked", m._apply_lock.locked())
    )
    monkeypatch.setattr(m, "_detector", None)

    r = TestClient(m.app).post("/model", json={"file": "cand.onnx"})
    assert r.status_code == 200
    assert seen["locked"] is True, "persist runs INSIDE the apply lock"
    assert m._detector is cand
    assert r.json()["device"]["scoreMin"] == 0.5, "the reply records the operating point"


def test_apply_refuses_with_507_when_persist_cannot_stick(monkeypatch, tmp_path):
    """A selection that cannot be made durable is refused BEFORE the swap
    (the .94 incident contract) — and the 507 path releases the lock."""
    from fastapi.testclient import TestClient

    from app import main as m

    (tmp_path / "cand.onnx").write_bytes(b"x")
    monkeypatch.setattr(m, "DEFAULT_MODEL", tmp_path / "default.onnx")
    monkeypatch.setattr(m, "build_detector", lambda p: _fake_family("cand.onnx"))

    def refuse(name):
        raise RuntimeError("cannot persist the selection (EACCES)")

    monkeypatch.setattr(m, "persist_selection", refuse)
    before = object()
    monkeypatch.setattr(m, "_detector", before)

    r = TestClient(m.app).post("/model", json={"file": "cand.onnx"})
    assert r.status_code == 507
    assert m._detector is before, "a selection that cannot stick is not applied"
    assert not m._apply_lock.locked(), "the refusal releases the apply lock"


def test_concurrent_applies_cannot_interleave_persist_and_swap(monkeypatch, tmp_path):
    """Two racing POST /model requests must serialise persist+swap, so the
    last lock holder wins BOTH the durable file and the pointer — never
    .selected naming A while the process serves B (the silent model flip
    on the next restart)."""
    import threading as th
    import time as _time

    from fastapi.testclient import TestClient

    from app import main as m

    for n in ("a.onnx", "b.onnx"):
        (tmp_path / n).write_bytes(b"x")
    dets = {n: _fake_family(n) for n in ("a.onnx", "b.onnx")}
    monkeypatch.setattr(m, "DEFAULT_MODEL", tmp_path / "default.onnx")
    monkeypatch.setattr(m, "build_detector", lambda p: dets[p.name])
    monkeypatch.setattr(m, "_detector", None)

    events: list[str] = []
    release_a = th.Event()

    def persist(name):
        events.append(f"persist:{name}:enter")
        if name == "a.onnx":
            release_a.wait(timeout=5)
        events.append(f"persist:{name}:exit")

    monkeypatch.setattr(m, "persist_selection", persist)

    def apply(name):
        TestClient(m.app).post("/model", json={"file": name})

    ta = th.Thread(target=apply, args=("a.onnx",))
    tb = th.Thread(target=apply, args=("b.onnx",))
    ta.start()
    deadline = _time.time() + 5
    while "persist:a.onnx:enter" not in events and _time.time() < deadline:
        _time.sleep(0.005)
    assert "persist:a.onnx:enter" in events, "A never reached persist"
    tb.start()
    _time.sleep(0.15)  # give B every chance to (wrongly) enter the section
    assert "persist:b.onnx:enter" not in events, "B must wait out A's persist+swap"
    release_a.set()
    ta.join(timeout=5)
    tb.join(timeout=5)
    assert events == [
        "persist:a.onnx:enter",
        "persist:a.onnx:exit",
        "persist:b.onnx:enter",
        "persist:b.onnx:exit",
    ]
    assert m._detector is dets["b.onnx"], "the last persist and the last swap agree"
