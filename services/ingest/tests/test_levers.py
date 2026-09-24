"""L1 levers: the motion gate and the bounded live buffer.

Every source here is synthetic — numpy frames behind the cv2.VideoCapture
seam (conftest.synthetic) — so each test controls every pixel of every frame
and nothing depends on a codec. The first block pins what OFF means: today's
worker, byte for byte.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from app.capture import CaptureWorker
from app.config import ENV_KNOBS, Levers, levers_from_env
from app.frames import FrameStore, Item, RawFrame
from app.motion import MotionGate, small_luma

from .conftest import textured

W, H = 320, 240
LEVER_ENV = ENV_KNOBS
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No lever leaks in from the shell; files run unpaced unless a test says."""
    for key in LEVER_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("INGEST_FILE_PACE", "0")


def blob(i, step=8, size=40, base=None):
    """The static scene with a bright square marching ``step`` px per frame."""
    img = (textured() if base is None else base).copy()
    x = 10 + (i * step) % (W - size - 20)
    img[100:100 + size, x:x + size] = 250
    return img


def lit(img, gain=1.2, offset=8):
    """The same scene under a global lighting change (no clipping, see textured)."""
    return np.clip(img.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)


def drain(worker, budget_s=10.0, work_s=0.0):
    """Take every fresh frame until the worker says ended; return them."""
    seen = []
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        got = worker.take()
        if got is None:
            time.sleep(0.002)
            continue
        if got.ended:
            return seen
        if not seen or got.seq != seen[-1].seq:
            seen.append(got)
            if work_s:
                time.sleep(work_s)
        else:
            time.sleep(0.002)
    pytest.fail(f"worker never ended; saw {[s.seq for s in seen]}")


def gated(**kw):
    """Levers with the gate on and today's defaults for everything else."""
    return Levers(motion_gate=True, **kw)


# ------------------------------------------------------------- OFF is today


def test_every_lever_defaults_off_and_off_is_not_armed(monkeypatch):
    """Unset, empty (compose's ${X-}) and explicit 0 all mean today's worker."""
    assert Levers().armed is False
    assert levers_from_env().armed is False
    for key in LEVER_ENV:
        monkeypatch.setenv(key, "")
    assert levers_from_env() == Levers(), "an empty knob is an unchosen knob"
    monkeypatch.setenv("INGEST_MOTION_GATE", "0")
    monkeypatch.setenv("INGEST_BUFFER_S", "0")
    assert levers_from_env().armed is False


def test_off_runs_todays_loop_grab_not_retrieve(synthetic):
    """With no lever the worker is today's: one retrieve, then grabs while
    the slot is unread. Armed with the gate, every frame is retrieved — the
    gate has to look at a frame to skip it."""
    opened = synthetic(lambda i: textured(), n_frames=30)
    off = CaptureWorker(source="/x.mp4", is_file=True)
    assert off.levered is False
    off.start()
    off.join(timeout=5)
    cap = opened[-1]
    assert cap.calls == {"read": 1, "grab": 30}, "OFF retrieved a frame nobody would read"
    assert off.ended is True

    opened = synthetic(lambda i: textured(), n_frames=30)
    on = CaptureWorker(source="/x.mp4", is_file=True, levers=gated())
    assert on.levered is True
    on.start()
    on.join(timeout=5)
    assert opened[-1].calls == {"read": 31, "grab": 0}


def test_off_frame_body_is_byte_for_byte_the_old_six_fields(client, synthetic_video):
    """GET /frame with every lever off: the same six keys, same order, same
    bytes — nothing downstream can tell the levers exist."""
    client.post("/open", json={"path": synthetic_video, "loop": True})
    for _ in range(300):
        res = client.get("/frame")
        if res.status_code == 200:
            break
        time.sleep(0.01)
    body = res.json()
    assert list(body) == ["tMs", "imageB64", "w", "h", "seq", "ended"]
    expected = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    assert res.content == expected


def test_open_without_overrides_keeps_the_old_worker(client, synthetic_video):
    """A runner that sends neither override gets exactly the worker it always got."""
    from app.main import state

    client.post("/open", json={"path": synthetic_video, "loop": True})
    assert state.worker.levered is False
    client.post("/open", json={"path": synthetic_video, "loop": True,
                               "motionGate": None, "bufferS": None})
    assert state.worker.levered is False


# ------------------------------------------------------------ the gate, unit


def test_gate_skips_a_still_scene_and_keeps_a_heartbeat():
    """No motion is skipped, but a frame is published every keepalive."""
    gate = MotionGate(min_frac=0.002, pixel_thr=0.08, keepalive_s=1.0)
    small = small_luma(textured())
    published = [i for i in range(50) if gate.decide(small, clock_s=i / 10.0)[0]]
    assert published == [0, 10, 20, 30, 40]


def test_gate_first_frame_is_published_with_motion_unmeasured():
    """Nothing to compare with: publish, and say motion=None rather than 0."""
    gate = MotionGate(0.002, 0.08, 1.0)
    assert gate.decide(small_luma(textured()), 0.0) == (True, None)
    gate.reset()
    assert gate.decide(small_luma(textured()), 0.1) == (True, None), "a reconnect starts over"


def test_gate_a_global_lighting_change_is_not_motion():
    """An exposure step or a DJ flash moves every pixel through one affine
    map; normalising each small frame by its own mean and std removes it."""
    gate = MotionGate(0.002, 0.08, keepalive_s=100.0)
    scene = textured()
    gate.decide(small_luma(scene), 0.0)
    for gain, offset in ((1.2, 8), (0.7, -5), (1.3, 0)):
        publish, motion = gate.decide(small_luma(lit(scene, gain, offset)), 0.1)
        assert publish is False, f"x{gain}{offset:+} read as motion ({motion})"
        assert motion < 0.002


def test_gate_a_slow_walker_accumulates_against_the_last_published_frame():
    """A faint figure creeping 1 px a frame changes too little from one frame
    to the next to count; against the last PUBLISHED frame the change builds
    until it crosses the threshold, so he is published anyway."""
    base = textured()

    def faint(i):  # a low-contrast square, 1 px further right each frame
        img = base.astype(np.int16)
        img[100:140, 40 + i:80 + i] += 12
        return np.clip(img, 0, 255).astype(np.uint8)

    smalls = [small_luma(faint(i)) for i in range(40)]
    # The premise: frame to frame, the gate's own measure sees nothing.
    per_frame = MotionGate(0.002, 0.08, keepalive_s=100.0)
    for i, small in enumerate(smalls):
        per_frame.reset()
        per_frame.decide(smalls[i - 1] if i else small, 0.0)
        publish, motion = per_frame.decide(small, 0.1)
        assert publish is False, f"frame {i} alone moved {motion} — the premise is broken"
    # The gate as built: publishes him regularly.
    gate = MotionGate(0.002, 0.08, keepalive_s=100.0)
    published = [i for i, small in enumerate(smalls) if gate.decide(small, i / 10.0)[0]]
    assert published[0] == 0 and len(published) >= 5, published
    assert max(b - a for a, b in zip(published, published[1:], strict=False)) <= 8


# ------------------------------------------------------ the gate, in a worker


def test_still_frames_are_skipped_but_a_keepalive_is_published(synthetic):
    """Footage clock: a keepalive every 10 frames at 10 fps, all else skipped."""
    synthetic(lambda i: textured(), n_frames=50, fps=10.0)
    w = CaptureWorker(source="/x.mp4", is_file=True, lockstep=True, levers=gated())
    w.start()
    try:
        served = drain(w)
    finally:
        w.stop()
    assert [s.seq for s in served] == [1, 11, 21, 31, 41]
    assert served[0].motion is None and all(s.motion < 0.002 for s in served[1:])
    last = w.describe()["counters"]
    assert last["captured"] == 50 and last["skipped"] == 45 and last["dropped"] == 0


def test_a_moving_blob_is_always_published(synthetic):
    """Every frame with a guest walking through it reaches the pipeline."""
    synthetic(lambda i: blob(i), n_frames=30, fps=10.0)
    w = CaptureWorker(source="/x.mp4", is_file=True, lockstep=True, levers=gated())
    w.start()
    try:
        served = drain(w)
    finally:
        w.stop()
    assert [s.seq for s in served] == list(range(1, 31))
    assert all(s.motion >= 0.002 for s in served[1:])
    assert served[-1].skipped == 0


def test_a_lighting_step_and_a_flash_alone_publish_nothing(synthetic):
    """Frame 15 steps the exposure, 25 flashes for one frame: only keepalives."""
    scene = textured()

    def frame(i):
        if i == 25:
            return lit(scene, 1.3, 0)  # the flash
        return lit(scene) if i >= 15 else scene

    synthetic(frame, n_frames=50, fps=10.0)
    w = CaptureWorker(source="/x.mp4", is_file=True, lockstep=True, levers=gated())
    w.start()
    try:
        served = drain(w)
    finally:
        w.stop()
    assert [s.seq for s in served] == [1, 11, 21, 31, 41]


def test_live_keepalive_runs_on_the_wall_clock(synthetic):
    """A camera's heartbeat is wall time: ~1 + 1.0/0.25 frames in a second."""
    synthetic(lambda i: textured(), n_frames=None, fps=0.0, period_s=0.01)
    w = CaptureWorker(source="rtsp://cam/1", is_file=False,
                      levers=gated(motion_keepalive_s=0.25, buffer_s=5.0))
    w.start()
    time.sleep(1.1)
    w.stop()
    c = w.describe()["counters"]
    assert 4 <= c["published"] <= 6, c
    assert c["skipped"] >= 50, "a still camera at 100 fps should be mostly skipped"


def test_lockstep_with_the_gate_waits_and_never_drops(synthetic):
    """Lockstep still hands out every PUBLISHED frame to a slow consumer:
    the gate removes still frames, the lockstep keeps the rest."""
    base = textured()

    def frame(i):  # still for 20 frames, then a walker for 20
        return base if i < 20 else blob(i - 20, base=base)

    synthetic(frame, n_frames=40, fps=10.0)
    w = CaptureWorker(source="/x.mp4", is_file=True, lockstep=True, levers=gated())
    w.start()
    try:
        served = drain(w, work_s=0.01)
    finally:
        w.stop()
    seqs = [s.seq for s in served]
    assert seqs[:2] == [1, 11]
    assert seqs[-20:] == list(range(21, 41)), "every walking frame, in order"
    c = w.describe()["counters"]
    assert c["dropped"] == 0 and c["backlogMax"] == 1


# ------------------------------------------------------------ the FIFO, unit


def _item(seq, clock_s, nbytes=100):
    frame = RawFrame.from_bgr(np.zeros((1, nbytes // 3, 3), np.uint8))
    return Item(seq=seq, t_ms=0, clock_s=clock_s, frame=frame, motion=None)


def test_store_is_a_fifo_and_take_dequeues():
    """Oldest first, each frame once, None when empty."""
    store = FrameStore(buffer_s=10.0, cap_bytes=10**9)
    for seq in range(1, 6):
        assert store.put(_item(seq, seq * 0.1)) == 0
    assert [store.take().seq for _ in range(5)] == [1, 2, 3, 4, 5]
    assert store.take() is None and store.bytes == 0


def test_store_age_overflow_drops_the_oldest_and_counts_it():
    """0.5 s of frames at 10 fps is six frames; the rest are dropped oldest-first."""
    store = FrameStore(buffer_s=0.5, cap_bytes=10**9)
    dropped = sum(store.put(_item(seq, seq * 0.1)) for seq in range(1, 11))
    assert dropped == 4
    assert [store.take().seq for _ in range(store.pending())] == [5, 6, 7, 8, 9, 10]


def test_store_byte_cap_holds():
    """Never more bytes than the cap, and never the newest frame dropped."""
    store = FrameStore(buffer_s=1000.0, cap_bytes=350)
    for seq in range(1, 20):
        store.put(_item(seq, seq * 0.1, nbytes=99))
        assert store.bytes <= 350 and store.pending() <= 3
    assert [i.seq for i in (store.take(), store.take(), store.take())] == [17, 18, 19]
    tiny = FrameStore(buffer_s=1000.0, cap_bytes=10)
    tiny.put(_item(1, 0.1, nbytes=99))
    assert tiny.put(_item(2, 0.2, nbytes=99)) == 1 and tiny.take().seq == 2


def test_slot_mode_overwrites_and_counts_the_unread_frame():
    """buffer_s = 0 is the newest-frame slot: an overwrite is a drop."""
    slot = FrameStore(buffer_s=0.0, cap_bytes=10**9)
    assert slot.put(_item(1, 0.1)) == 0
    assert slot.put(_item(2, 0.2)) == 1
    assert slot.take().seq == 2 and slot.pending() == 0


# ---------------------------------------------------------- the FIFO, worker


def test_buffered_frames_come_out_oldest_first_with_seq_intact(client, synthetic):
    """GET /frame dequeues the oldest unread; an empty queue repeats the last."""
    synthetic(lambda i: blob(i), n_frames=12, fps=10.0)
    res = client.post("/open", json={"path": __file__, "bufferS": 30})
    assert res.status_code == 200
    from app.main import state

    deadline = time.monotonic() + 5
    while not state.worker.describe()["counters"]["exhausted"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    bodies = [client.get("/frame").json() for _ in range(12)]
    assert [b["seq"] for b in bodies] == list(range(1, 13))
    assert [b["backlog"] for b in bodies] == list(range(11, -1, -1))
    assert all(b["ended"] is False for b in bodies), "a fresh frame never rides with ended"
    assert bodies[0]["skipped"] is None, "gate off: skipped is unmeasured, not 0"
    assert bodies[0]["motion"] is None
    again = client.get("/frame").json()
    assert again["seq"] == 12 and again["ended"] is True


def test_ended_waits_for_the_queue_to_drain(synthetic):
    """A file read to its end with frames still queued is NOT ended."""
    synthetic(lambda i: blob(i), n_frames=8, fps=10.0)
    w = CaptureWorker(source="/x.mp4", is_file=True, levers=Levers(buffer_s=30.0))
    w.start()
    w.join(timeout=5)
    assert w.describe()["counters"]["exhausted"] is True
    for seq in range(1, 9):
        got = w.take()
        assert got.seq == seq and got.ended is False
    assert w.take().ended is True
    assert w.ended is True


def test_live_overflow_drops_the_oldest_and_counts_it(synthetic):
    """A camera outrunning its consumer: the queue holds bufferS seconds of the
    NEWEST frames and every eviction is counted."""
    synthetic(lambda i: blob(i), n_frames=None, fps=0.0, period_s=0.005)
    w = CaptureWorker(source="rtsp://cam/1", is_file=False, levers=Levers(buffer_s=0.2))
    w.start()
    time.sleep(0.6)
    w.stop()
    c = w.describe()["counters"]
    assert c["dropped"] > 0
    assert c["captured"] == c["published"] == c["dropped"] + c["pending"]
    first = w.take()
    assert first.seq > 1, "the oldest frames are the ones dropped"
    assert first.dropped == c["dropped"]


def test_the_byte_cap_holds_in_a_running_worker(synthetic, monkeypatch):
    """INGEST_BUFFER_MB is a hard cap: 1 MiB holds four 230 KB frames, never five."""
    synthetic(lambda i: blob(i), n_frames=40, fps=10.0)
    w = CaptureWorker(source="/x.mp4", is_file=True,
                      levers=Levers(buffer_s=1000.0, buffer_mb=1))
    w.start()
    w.join(timeout=5)
    c = w.describe()["counters"]
    assert c["pending"] == 4 and c["bufferBytes"] <= 1024 * 1024
    assert c["dropped"] == 36 and c["backlogMax"] == 5 - 1


def test_counters_are_monotonic_and_every_frame_is_accounted_for(client, synthetic):
    """captured == skipped + published; published == served + dropped + pending."""
    base = textured()

    def frame(i):  # alternate still stretches and walking stretches
        return blob(i, base=base) if (i // 10) % 2 else base

    synthetic(frame, n_frames=60, fps=20.0)
    client.post("/open", json={"path": __file__, "motionGate": True, "bufferS": 1.0})
    from app.main import state

    prev = {"captured": 0, "skipped": 0, "dropped": 0}
    for _ in range(400):
        res = client.get("/frame")
        if res.status_code == 503:
            time.sleep(0.005)
            continue
        body = res.json()
        for key in prev:
            assert body[key] >= prev[key], f"{key} went backwards"
            prev[key] = body[key]
        if body["ended"]:
            break
        time.sleep(0.003)
    c = state.worker.describe()["counters"]
    assert c["captured"] == 60
    assert c["captured"] == c["skipped"] + c["published"]
    assert c["published"] == c["served"] + c["dropped"] + c["pending"]


# -------------------------------------------------- knobs: open, env, health


def test_open_overrides_win_over_the_env(client, synthetic, monkeypatch):
    """motionGate/bufferS on /open override the service env for that run."""
    synthetic(lambda i: textured(), n_frames=None, fps=10.0, period_s=0.01)
    from app.main import state

    monkeypatch.setenv("INGEST_MOTION_GATE", "1")
    client.post("/open", json={"url": "rtsp://cam/1"})
    assert state.worker.levers.motion_gate is True
    client.post("/open", json={"url": "rtsp://cam/1", "motionGate": False, "bufferS": 2.5})
    k = state.worker.levers
    assert k.motion_gate is False and k.buffer_s == 2.5 and state.worker.levered is True
    assert client.post("/open", json={"url": "rtsp://cam/1", "bufferS": -1}).status_code == 422


def test_health_shows_the_knobs_and_the_open_workers_counters(client, synthetic, monkeypatch):
    """Every knob is visible on /health; a malformed one turns ok false."""
    body = client.get("/health").json()
    assert body["knobs"] == Levers().knobs() and body["capture"] is None
    synthetic(lambda i: textured(), n_frames=None, fps=10.0, period_s=0.01)
    client.post("/open", json={"url": "rtsp://cam/1", "motionGate": True})
    cap = client.get("/health").json()["capture"]
    assert cap["levered"] is True and cap["knobs"]["motionGate"] is True
    assert set(cap["counters"]) >= {"captured", "skipped", "dropped", "backlogMax", "bufferBytes"}
    monkeypatch.setenv("INGEST_MOTION_MIN_FRAC", "lots")
    bad = client.get("/health").json()
    assert bad["ok"] is False and "INGEST_MOTION_MIN_FRAC" in bad["knobs"]["error"]
    assert client.post("/open", json={"url": "rtsp://cam/1"}).status_code == 500


def test_every_knob_reaches_the_container():
    """Each env knob app.config reads is passed through docker-compose.yml.

    Three knobs in this repo once looked set on the host and changed nothing,
    because the container never received them.
    """
    compose = (REPO / "docker-compose.yml").read_text()
    for name in ENV_KNOBS:
        assert f"{name}: ${{{name}-}}" in compose, f"{name} has no compose passthrough"
