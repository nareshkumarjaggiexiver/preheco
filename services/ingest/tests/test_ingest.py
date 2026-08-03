"""Ingest service tests: open → frames advance → dimensions correct.

All in-process (TestClient) against a synthetic MJPG clip — no network.
"""

import time

import numpy as np
import pytest
from heco_common.imaging import decode_jpeg_b64

from .conftest import VID_H, VID_W


def _wait_frame(client, timeout_s: float = 3.0) -> dict:
    """Poll GET /frame until the capture thread has produced one."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        res = client.get("/frame")
        if res.status_code == 200:
            return res.json()
        assert res.status_code == 503, res.text  # open but not ready is the only excuse
        time.sleep(0.01)
    pytest.fail("no frame within timeout")


def test_health(client):
    """Health reports ok + backend identity + version."""
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["model"] == "opencv-videocapture"
    assert body["version"]


def test_frame_before_open_conflicts(client):
    """GET /frame without a source is a 409, not a hang or a 500."""
    assert client.get("/frame").status_code == 409


def test_open_rejects_missing_file(client):
    """A nonexistent path fails fast with 400."""
    res = client.post("/open", json={"path": "/nope/missing.avi"})
    assert res.status_code == 400


def test_open_rejects_url_and_path_together(client, synthetic_video):
    """The contract is url XOR path — both at once is a validation error."""
    res = client.post("/open", json={"url": "rtsp://x/1", "path": synthetic_video})
    assert res.status_code == 422


def test_open_then_frames_advance(client, synthetic_video):
    """Frames arrive, seq/tMs advance, dimensions and pixels are correct."""
    res = client.post("/open", json={"path": synthetic_video, "loop": True})
    assert res.status_code == 200 and res.json()["ok"] is True

    first = _wait_frame(client)
    assert first["w"] == VID_W and first["h"] == VID_H
    img1 = decode_jpeg_b64(first["imageB64"])
    assert img1.shape == (VID_H, VID_W, 3)

    # 100 fps pacing → a newer frame occupies the slot within ~10 ms.
    deadline = time.monotonic() + 3.0
    second = client.get("/frame").json()
    while second["seq"] <= first["seq"] and time.monotonic() < deadline:
        time.sleep(0.01)
        second = client.get("/frame").json()
    assert second["seq"] > first["seq"]
    assert second["tMs"] >= first["tMs"]

    # The rectangle moves, so sufficiently-spaced frames must differ.
    img2 = decode_jpeg_b64(second["imageB64"])
    if second["seq"] - first["seq"] >= 2:
        assert int(np.abs(img2.astype(int) - img1.astype(int)).sum()) > 0


def test_loop_wraps_past_clip_length(client, synthetic_video):
    """With loop=True, seq climbs beyond the clip's 40 frames."""
    client.post("/open", json={"path": synthetic_video, "loop": True})
    deadline = time.monotonic() + 5.0
    seq = _wait_frame(client)["seq"]
    while seq <= 45 and time.monotonic() < deadline:
        time.sleep(0.02)
        seq = client.get("/frame").json()["seq"]
    assert seq > 45


def test_shutdown_stops_capture_thread(synthetic_video):
    """Leaving the app context joins the worker and releases the capture."""
    from app.main import app, state
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        c.post("/open", json={"path": synthetic_video, "loop": True})
        _wait_frame(c)
        worker = state.worker
        assert worker is not None and worker.is_alive()
    assert not worker.is_alive()
    assert state.worker is None
