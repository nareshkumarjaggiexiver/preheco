"""L6 — decoding in an ffmpeg subprocess, and falling back loudly when it cannot.

Most tests run tests/fake_ffmpeg.py in place of the binary, so every failure
mode (cannot load, hangs, dies mid-file, floods stderr, never ends) is
deterministic and needs no GPU. One test runs the REAL ffmpeg in software
mode on the synthetic clip when a binary is installed, which checks the
stderr parsing and the NV12 conversion against a genuine decoder.
"""

import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from app import ffmpeg_source as fs
from app.capture import CaptureWorker
from app.config import Levers

from .conftest import VID_FRAMES, VID_H, VID_W, textured

FAKE = str(Path(__file__).with_name("fake_ffmpeg.py"))


@pytest.fixture(autouse=True)
def _fake_ffmpeg(monkeypatch):
    """Every test decodes with the fake unless it says otherwise."""
    monkeypatch.setattr(fs, "FFMPEG", [sys.executable, FAKE])
    monkeypatch.setattr(fs, "_VERSION", None)
    monkeypatch.setenv("INGEST_FILE_PACE", "0")
    for key in ("FAKE_FFMPEG_MODE", "FAKE_FRAMES", "FAKE_PERIOD", "FAKE_STATIC"):
        monkeypatch.delenv(key, raising=False)


def reaped(pid: int) -> bool:
    """True when the child is gone AND collected — no zombie left behind."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


def drain(worker, budget_s=10.0):
    """Take every fresh frame until ended (or the budget runs out)."""
    seen, deadline = [], time.monotonic() + budget_s
    while time.monotonic() < deadline:
        got = worker.take()
        if got is None:
            time.sleep(0.002)
            continue
        if got.ended:
            return seen, True
        if not seen or got.seq != seen[-1].seq:
            seen.append(got)
        else:
            time.sleep(0.002)
    return seen, False


# ----------------------------------------------------------- the command


def test_nvdec_keeps_frames_on_the_gpu_so_a_cpu_fallback_fails(monkeypatch):
    """hwdownload refuses a software frame: ffmpeg's own quiet fallback to CPU
    decode becomes an error the worker can see, not a slow success."""
    monkeypatch.setattr(fs, "_VERSION", (7, 1))
    cmd = " ".join(fs.build_command("/c.mp4", "nvdec"))
    assert "-hwaccel cuda -hwaccel_output_format cuda" in cmd
    assert "-vf hwdownload,format=nv12" in cmd
    assert "-fps_mode passthrough" in cmd, "every frame as decoded, none duplicated"
    assert cmd.endswith("-pix_fmt nv12 -f rawvideo pipe:1")
    assert "-stream_loop" not in cmd and "-rtsp_transport" not in cmd


def test_vaapi_loop_and_software_commands(monkeypatch):
    """VA-API names its render node; loop is ffmpeg's own; software adds nothing."""
    monkeypatch.setattr(fs, "_VERSION", (7, 1))
    va = fs.build_command("/c.mp4", "vaapi", loop=True)
    assert va[va.index("-hwaccel_device") + 1] == fs.VAAPI_DEVICE
    assert va.index("-stream_loop") < va.index("-i"), "an input option"
    sw = " ".join(fs.build_command("/c.mp4", "ffmpeg"))
    assert "-hwaccel" not in sw and "hwdownload" not in sw


def test_rtsp_options_follow_the_binarys_version(monkeypatch):
    """-timeout is the socket timeout from 5.0; in 4.x it meant LISTEN mode."""
    monkeypatch.setattr(fs, "_VERSION", (7, 1))
    new = fs.build_command("rtsp://u:p@cam/1", "nvdec")
    assert new[new.index("-rtsp_transport") + 1] == "tcp"
    assert "-timeout" in new and "-stimeout" not in new
    monkeypatch.setattr(fs, "_VERSION", (4, 4))
    old = fs.build_command("rtsp://u:p@cam/1", "nvdec", rtsp_tcp=False)
    assert "-stimeout" in old and "-timeout" not in old and "-rtsp_transport" not in old
    assert "-vsync" in old and "-fps_mode" not in old


# ------------------------------------------------------ frames and gate


def test_decoded_frames_reach_the_consumer_as_bgr():
    """NV12 off the pipe, BGR out of take(), every frame in order, then ended."""
    w = CaptureWorker("/c.mp4", is_file=True, lockstep=True,
                      levers=Levers(decoder="nvdec"))
    assert w.decoder == {"requested": "nvdec", "active": "nvdec", "error": None}
    assert w.levered is True
    w.start()
    try:
        seen, ended = drain(w)
    finally:
        w.stop()
    assert ended and [s.seq for s in seen] == list(range(1, 11))
    assert seen[0].image.shape == (48, 64, 3)
    assert seen[0].skipped is None and seen[-1].captured == 10


def test_the_gate_reads_the_y_plane(monkeypatch):
    """A still NV12 stream is skipped down to keepalives; no BGR is made for it."""
    monkeypatch.setenv("FAKE_STATIC", "1")
    monkeypatch.setenv("FAKE_FRAMES", "30")
    w = CaptureWorker("/c.mp4", is_file=True, lockstep=True,
                      levers=Levers(motion_gate=True, decoder="nvdec"))
    w._fps = 10.0  # the cv2 probe cannot read a fake path; footage clock at 10 fps
    w.start()
    try:
        seen, ended = drain(w)
    finally:
        w.stop()
    assert ended and [s.seq for s in seen] == [1, 11, 21]
    assert w.describe()["counters"]["skipped"] == 27


# ------------------------------------------------------ loud fallback


def test_a_decoder_that_cannot_start_falls_back_to_cpu_loudly(monkeypatch, synthetic, capfd):
    """No libnvcuvid: the camera still works (cv2), and everybody is told."""
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "fail")
    synthetic(lambda i: textured(), n_frames=5, fps=10.0)
    w = CaptureWorker("/c.mp4", is_file=True, levers=Levers(decoder="nvdec"))
    try:
        assert w.decoder["active"] == "cpu" and "libnvcuvid" in w.decoder["error"]
        assert w.levered is False, "nothing else armed: this IS today's worker"
        err = capfd.readouterr().err
        assert "requested=nvdec active=['cpu']" in err
        assert "fell back to cpu" in err
    finally:
        w.stop()


def test_a_missing_binary_falls_back_too(monkeypatch, synthetic):
    """No ffmpeg in the image at all: same loud fallback, not a dead ingest."""
    monkeypatch.setattr(fs, "FFMPEG", ["/nonexistent/ffmpeg"])
    synthetic(lambda i: textured(), n_frames=5, fps=10.0)
    w = CaptureWorker("/c.mp4", is_file=True,
                      levers=Levers(decoder="nvdec", motion_gate=True))
    try:
        assert w.decoder["active"] == "cpu" and "could not start" in w.decoder["error"]
        assert w.levered is True, "the gate still runs, on the cv2 decoder"
    finally:
        w.stop()


def test_a_hung_decoder_times_out_is_reaped_and_falls_back(monkeypatch, synthetic):
    """A decoder that never answers costs OPEN_TIMEOUT_S, not the run."""
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "hang")
    monkeypatch.setattr(fs, "OPEN_TIMEOUT_S", 0.5)
    synthetic(lambda i: textured(), n_frames=5, fps=10.0)
    t0 = time.monotonic()
    w = CaptureWorker("/c.mp4", is_file=True, levers=Levers(decoder="nvdec"))
    try:
        assert time.monotonic() - t0 < 3
        assert w.decoder["active"] == "cpu" and "timed out" in w.decoder["error"]
    finally:
        w.stop()


def test_health_serves_the_device_truth(client, monkeypatch, synthetic):
    """/health device: requested vs active, and the reason when they differ."""
    monkeypatch.setenv("INGEST_DECODER", "nvdec")
    body = client.get("/health").json()
    assert body["device"] == {"requested": "nvdec", "active": None, "error": None}
    assert body["knobs"]["decoder"] == "nvdec"
    client.post("/open", json={"path": __file__})
    dev = client.get("/health").json()["device"]
    assert dev == {"requested": "nvdec", "active": "nvdec", "error": None}
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "fail")
    synthetic(lambda i: textured(), n_frames=None, fps=10.0, period_s=0.01)
    client.post("/open", json={"path": __file__})
    dev = client.get("/health").json()["device"]
    assert dev["requested"] == "nvdec" and dev["active"] == "cpu" and dev["error"]
    monkeypatch.setenv("INGEST_DECODER", "nvedc")
    assert client.get("/health").json()["ok"] is False, "a typo refuses, loudly"


# ------------------------------------------------- subprocess lifecycle


def test_stop_kills_and_reaps_a_live_decoder(monkeypatch):
    """A camera never ends on its own: stop() must end it, and collect it."""
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "forever")
    monkeypatch.setenv("FAKE_PERIOD", "0.01")
    w = CaptureWorker("rtsp://cam/1", is_file=False, levers=Levers(decoder="nvdec"))
    pid = w._ff.pid
    w.start()
    deadline = time.monotonic() + 3
    while w.take() is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    t0 = time.monotonic()
    w.stop()
    assert not w.is_alive() and time.monotonic() - t0 < 2
    assert reaped(pid), "the decoder is a zombie"


def test_stop_reaps_a_decoder_whose_worker_never_started(monkeypatch):
    """/open can open and then be replaced before the thread runs."""
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "forever")
    w = CaptureWorker("rtsp://cam/1", is_file=False, levers=Levers(decoder="nvdec"))
    pid = w._ff.pid
    w.stop()
    assert reaped(pid)


def test_a_flood_on_stderr_never_blocks_the_frames(monkeypatch):
    """~2 MB of warnings per frame would fill a 64 KiB stderr pipe at once;
    drained by its own thread, every frame still arrives."""
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "noisy")
    w = CaptureWorker("/c.mp4", is_file=True, lockstep=True, levers=Levers(decoder="nvdec"))
    w.start()
    try:
        seen, ended = drain(w, budget_s=20)
    finally:
        w.stop()
    assert ended and len(seen) == 10


def test_a_decoder_that_dies_mid_file_is_never_an_end(monkeypatch):
    """Exit 1 after 4 frames is a failure, not EOF: those 4 frames are served,
    `ended` never comes (the runner stalls and fails the run, gallery kept),
    and /health says why."""
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "die")
    monkeypatch.setenv("FAKE_FRAMES", "4")
    w = CaptureWorker("/c.mp4", is_file=True, lockstep=True, levers=Levers(decoder="nvdec"))
    pid = w._ff.pid
    w.start()
    try:
        seen, ended = drain(w, budget_s=1.5)
    finally:
        w.stop()
    assert [s.seq for s in seen] == [1, 2, 3, 4] and ended is False
    assert "exited 1 mid-file" in w.decoder["error"]
    assert reaped(pid)


def test_a_live_decoder_that_exits_is_restarted(monkeypatch):
    """A camera that drops comes back on the same decoder; seq keeps counting."""
    monkeypatch.setenv("FAKE_FRAMES", "3")
    w = CaptureWorker("rtsp://cam/1", is_file=False,
                      levers=Levers(decoder="nvdec", buffer_s=30.0))
    first_pid = w._ff.pid
    w.start()
    deadline = time.monotonic() + 5
    try:
        while w.describe()["counters"]["captured"] < 5:
            assert time.monotonic() < deadline, "the decoder was not restarted"
            time.sleep(0.05)
    finally:
        w.stop()
    assert w.decoder["active"] == "nvdec"
    assert reaped(first_pid)
    assert [w.take().seq for _ in range(5)] == [1, 2, 3, 4, 5]


# ------------------------------------------------------- the real binary


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg binary")
def test_real_ffmpeg_software_decode_matches_cv2(monkeypatch, synthetic_video):
    """The genuine decoder: its stderr parses, its NV12 converts to what cv2
    decodes itself (within rounding), and a file plays out to `ended`."""
    monkeypatch.setattr(fs, "FFMPEG", ["ffmpeg"])
    monkeypatch.setattr(fs, "_VERSION", None)
    w = CaptureWorker(synthetic_video, is_file=True, lockstep=True,
                      levers=Levers(decoder="ffmpeg"))
    assert w.decoder["active"] == "ffmpeg", w.decoder
    w.start()
    try:
        seen, ended = drain(w)
    finally:
        w.stop()
    assert ended and [s.seq for s in seen] == list(range(1, VID_FRAMES + 1))
    cap = cv2.VideoCapture(synthetic_video)
    ok, ref = cap.read()
    cap.release()
    assert ok and seen[0].image.shape == (VID_H, VID_W, 3)
    diff = np.abs(seen[0].image.astype(np.int16) - ref.astype(np.int16))
    assert diff.mean() < 4, f"NV12 path drifted from cv2's own decode: {diff.mean():.2f}"
