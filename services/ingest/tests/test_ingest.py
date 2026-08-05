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


# ---------------------------------------------- regressions (2026-08-05 review)


def test_second_run_cannot_silently_steal_a_live_runs_camera(client, synthetic_video):
    """A claimed capture slot is refused to anyone else, by name.

    Regression: /open unconditionally swapped THE one capture worker and
    /frame had no run affinity, so starting a staff enrolment during a live
    gate count replaced the count run's source — the count run then counted
    the enrolment walk-through. Nothing errored; the only trace was a
    plausible-looking frame stream and a corrupted event total.
    """
    first = client.post("/open", json={"path": synthetic_video, "loop": True, "owner": "run-gate1"})
    assert first.status_code == 200 and first.json()["owner"] == "run-gate1"
    _wait_frame(client)

    clash = client.post("/open", json={"path": synthetic_video, "owner": "run-enrol"})
    assert clash.status_code == 409
    assert "run-gate1" in clash.json()["detail"], "the operator must learn WHO holds it"
    assert client.get("/health").json()["owner"] == "run-gate1"
    # And the live run still has its own source.
    assert _wait_frame(client)["w"] == VID_W


def test_same_owner_may_reopen_and_takeover_is_explicit(client, synthetic_video):
    """Re-opening your own slot is idempotent; seizing another's is deliberate."""
    client.post("/open", json={"path": synthetic_video, "loop": True, "owner": "run-a"})
    _wait_frame(client)

    same = client.post("/open", json={"path": synthetic_video, "loop": True, "owner": "run-a"})
    assert same.status_code == 200, "a run may restart its own capture"

    seize = client.post(
        "/open", json={"path": synthetic_video, "loop": True, "owner": "run-b", "takeover": True}
    )
    assert seize.status_code == 200
    assert client.get("/health").json()["owner"] == "run-b"


def test_close_releases_the_slot_for_the_next_run(client, synthetic_video):
    """End of run hands the camera back; a stale close cannot steal it."""
    client.post("/open", json={"path": synthetic_video, "loop": True, "owner": "run-a"})
    _wait_frame(client)

    stale = client.post("/close", json={"owner": "run-zzz"})
    assert stale.status_code == 409, "a finished run must not stop the live one"

    res = client.post("/close", json={"owner": "run-a"})
    assert res.status_code == 200 and res.json()["released"] is True
    assert client.get("/frame").status_code == 409  # slot really is empty
    assert client.post("/close", json={"owner": "run-a"}).json()["released"] is False

    nxt = client.post("/open", json={"path": synthetic_video, "loop": True, "owner": "run-b"})
    assert nxt.status_code == 200


def test_unowned_open_keeps_the_old_replace_anything_behaviour(client, synthetic_video):
    """Ad-hoc probes (no owner) are unaffected by the exclusivity rule."""
    client.post("/open", json={"path": synthetic_video, "loop": True})
    _wait_frame(client)
    assert client.post("/open", json={"path": synthetic_video, "loop": True}).status_code == 200


def test_a_failed_open_never_quotes_the_camera_password(monkeypatch):
    """The message from a failed open travels a long way.

    ingest -> the runner's StageError -> the planner's PERMANENT run notes ->
    the browser -> the export. So it must not carry rtsp://user:pass@.
    """
    import cv2
    from app.capture import CaptureError, CaptureWorker

    class Unopenable:
        """A VideoCapture that refuses to open, without touching the network."""

        def isOpened(self):  # noqa: N802 — mirrors cv2's API
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: Unopenable())
    # The constructor opens the source, so the failure happens right here.
    with pytest.raises(CaptureError) as caught:
        CaptureWorker(
            source="rtsp://admin:Hunter2@192.168.1.64:554/media/video1", is_file=False,
        )

    assert "Hunter2" not in str(caught.value), "a failed open leaked the camera password"
    assert "admin" not in str(caught.value)
    # ...while still naming the camera, or the message would be useless.
    assert "192.168.1.64" in str(caught.value)


def test_downscale_is_off_by_default_and_exact_when_armed(monkeypatch):
    """INGEST_MAX_WIDTH must change nothing until an operator sets it.

    Face pixels scale with the frame, so a silent downscale would quietly move
    every quality threshold the site survey was planned against.
    """
    import cv2
    import numpy as np
    from app.capture import CaptureWorker

    class Stub:
        """Enough of cv2.VideoCapture for construction; a file source probes FPS."""

        def isOpened(self): return True          # noqa: N802, E704
        def release(self): pass                  # noqa: E704
        def get(self, _prop): return 25.0        # noqa: E704

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: Stub())
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    monkeypatch.delenv("INGEST_MAX_WIDTH", raising=False)
    off = CaptureWorker(source="/x.mp4", is_file=True)
    assert off._fit(frame).shape == (2160, 3840, 3), "default must not resize"

    monkeypatch.setenv("INGEST_MAX_WIDTH", "1920")
    on = CaptureWorker(source="/x.mp4", is_file=True)
    out = on._fit(frame)
    assert out.shape == (1080, 1920, 3), "aspect ratio is preserved"

    # A frame already smaller than the cap is left alone, not upscaled.
    small = np.zeros((576, 704, 3), dtype=np.uint8)
    assert on._fit(small).shape == (576, 704, 3)


def test_an_unread_frame_is_grabbed_not_retrieved(monkeypatch):
    """The expensive half of read() is retrieve(); skip it for frames the
    drop-not-queue slot is going to discard anyway."""
    import cv2
    import numpy as np
    from app.capture import CaptureWorker

    calls = {"grab": 0, "read": 0}

    class Cap:
        def isOpened(self): return True                                  # noqa: N802, E704
        def release(self): pass                                          # noqa: E704
        def get(self, _prop): return 25.0                                # noqa: E704
        def grab(self):
            calls["grab"] += 1
            return True
        def read(self):
            calls["read"] += 1
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: Cap())
    w = CaptureWorker(source="/x.mp4", is_file=True)

    # Nothing in the slot yet -> must retrieve, so there is something to serve.
    assert not w._slot_unread()

    # Once a frame is sitting unread, the loop's choice is grab.
    w._latest = (1, 0, np.zeros((8, 8, 3), dtype=np.uint8))
    w._unread = True
    assert w._slot_unread()

    # ...and taking the frame clears the flag, so the next one is retrieved.
    w.latest()
    assert not w._slot_unread()
