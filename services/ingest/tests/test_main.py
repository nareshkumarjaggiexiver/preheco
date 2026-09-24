"""The shared-frame transport as ingest serves it: refs, with and without the levers.

What these pin:

* a frame is written to tmpfs once however often it is polled, and the
  cache is keyed by (worker generation, seq) — seq restarts at 1 on every
  /open, and seq alone handed a new run the previous run's frame;
* the ref's number is ingest's WRITE number, not the capture seq, so
  HECO_FRAMES_KEEP means "the last N frames handed out": under the motion
  gate a still room publishes one frame per keepalive, 10+ seqs apart, and a
  capture-seq sweep retired the frame the runner was still embedding;
* with a lever armed the frame is written when it is TAKEN — the buffer's
  backlog stays in ingest's memory, tmpfs holds at most KEEP frames, and a
  frame the gate skipped never touches it; ``jpeg=0`` is honoured there too;
* a run's frames are cleared when its source is closed or replaced.
"""

import time
from types import SimpleNamespace

import pytest

from .conftest import textured


def _frames_on(directory):
    """The frame files (not temp parts) currently on the shared mount."""
    return sorted(p.name for p in directory.iterdir() if not p.name.startswith("."))


@pytest.fixture()
def shared(monkeypatch, tmp_path):
    """A shared frames dir, and the ref cache emptied."""
    from app import main as m

    monkeypatch.setenv("HECO_FRAMES_DIR", str(tmp_path))
    monkeypatch.setattr(m, "_ref_last", None)
    return tmp_path


def test_a_frame_is_written_to_the_shared_transport_once_per_frame(monkeypatch):
    """REGRESSION (2026-09-24). The runner polls /frame at 50 Hz waiting for
    the seq to advance, and every poll returned the same frame — so writing
    on each one put 24 MB into tmpfs fifty times a second for ONE frame.

    The JPEG encode had been hiding it: at 13.9 ms it throttled the polling.
    Remove the encode and the waste ran free, and the 'faster' ref-only path
    measured 2.32 fps against the 5.14 of the JPEG it replaced.
    """
    from app import main as m

    monkeypatch.setattr(m, "_ref_last", None)
    writes = []

    def _spy(img, n, directory=None):
        writes.append(n)
        return f"f{n}_1x1.bgr"

    monkeypatch.setattr(m.frameref, "write_frame", _spy)
    run = SimpleNamespace(generation=1)
    img = object()
    first = m._cached_ref(run, 7, img)
    for _ in range(50):                    # the poll storm
        assert m._cached_ref(run, 7, img) == first
    assert len(writes) == 1, "one write for one frame, however many times it is asked for"
    m._cached_ref(run, 8, img)
    assert len(writes) == 2, "and a new seq is a new frame"
    # A NEW RUN starts its seqs again at 1: its seq 8 is not the old run's.
    m._cached_ref(SimpleNamespace(generation=2), 8, img)
    assert len(writes) == 3, "seq alone would hand the new run the old run's picture"
    assert writes == sorted(set(writes)), "write numbers only ever grow"


def _open_gated_still_room(client, synthetic):
    """A still room under the gate: one frame per 1 s keepalive, 10 seqs apart."""
    synthetic(lambda i: textured(), n_frames=45, fps=10.0)
    client.post("/open", json={"path": __file__, "motionGate": True, "bufferS": 30})


def _take_fresh(client, n, jpeg=True, budget_s=5.0):
    """GET /frame until ``n`` distinct seqs were handed out; their bodies."""
    got, deadline = [], time.monotonic() + budget_s
    while len(got) < n and time.monotonic() < deadline:
        res = client.get("/frame" if jpeg else "/frame?jpeg=0")
        if res.status_code == 503:
            time.sleep(0.005)
            continue
        body = res.json()
        if body["ended"]:
            break
        if not got or body["seq"] != got[-1]["seq"]:
            got.append(body)
        else:
            time.sleep(0.005)
    return got


def test_under_the_gate_keep_counts_frames_handed_out_not_capture_seqs(
    client, synthetic, shared, monkeypatch
):
    """Seqs 1, 11, 21, 31, 41 are published; with KEEP=2 the frame before the
    newest must survive, as the runner may still be embedding it."""
    monkeypatch.setenv("HECO_FRAMES_KEEP", "2")
    _open_gated_still_room(client, synthetic)
    got = _take_fresh(client, 5)
    assert [b["seq"] for b in got] == [1, 11, 21, 31, 41], "the keepalive, 10 seqs apart"
    refs = [b["frameRef"] for b in got]
    numbers = [int(r[1:].split("_")[0]) for r in refs]
    assert numbers == list(range(numbers[0], numbers[0] + 5)), "numbered by write"
    on_disk = _frames_on(shared)
    assert refs[-1] in on_disk and refs[-2] in on_disk, "the one before the newest survives"
    assert len(on_disk) <= 3, "bounded: the newest KEEP (+1), whatever the backlog"


def test_a_levered_frame_is_written_when_taken_and_jpeg_0_is_honoured(
    client, synthetic, shared
):
    """With a live buffer the backlog stays in memory; only served frames are
    written, each once, and ``jpeg=0`` drops the encode when a ref exists."""
    synthetic(lambda i: textured(), n_frames=20, fps=10.0)
    client.post("/open", json={"path": __file__, "bufferS": 30})
    from app.main import state

    deadline = time.monotonic() + 5
    while state.worker.describe()["counters"]["pending"] < 20 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _frames_on(shared) == [], "20 frames queued, none written: nobody took them"
    first = client.get("/frame?jpeg=0").json()
    assert first["seq"] == 1 and first["imageB64"] == "" and first["frameRef"]
    again = client.get("/frame?jpeg=0").json()
    assert again["seq"] == 2 and again["frameRef"] != first["frameRef"]
    assert _frames_on(shared) == sorted([first["frameRef"], again["frameRef"]])
    with_jpeg = client.get("/frame").json()
    assert with_jpeg["imageB64"] and with_jpeg["frameRef"], "the JPEG unless declined"


def test_a_skipped_frame_never_touches_tmpfs(client, synthetic, shared):
    """The gate withheld 40 of 45 frames: 5 writes, not 45."""
    _open_gated_still_room(client, synthetic)
    got = _take_fresh(client, 5)
    assert len(got) == 5
    numbers = sorted(int(r[1:].split("_")[0]) for r in _frames_on(shared))
    assert numbers[-1] - numbers[0] <= 4, "only the 5 served frames were ever written"


def test_closing_or_replacing_the_source_clears_its_frames(client, synthetic, shared):
    """A finished run's frames do not sit on tmpfs until the next run's writes
    happen to retire them — nor forever after a restart renumbers the writes."""
    synthetic(lambda i: textured(), n_frames=None, fps=10.0, period_s=0.005)
    client.post("/open", json={"path": __file__, "bufferS": 5, "owner": "run-a"})
    assert _take_fresh(client, 3)
    assert _frames_on(shared)
    (shared / "not-ours.txt").write_text("kept")
    client.post("/close", json={"owner": "run-a"})
    assert _frames_on(shared) == ["not-ours.txt"], "only the frames this module wrote"

    client.post("/open", json={"path": __file__, "bufferS": 5, "owner": "run-b"})
    assert _take_fresh(client, 2)
    before = set(_frames_on(shared)) - {"not-ours.txt"}
    client.post("/open", json={"path": __file__, "owner": "run-c", "takeover": True})
    assert not before & set(_frames_on(shared)), "a replaced run's frames go with it"


def test_without_a_mount_nothing_changes(client, synthetic, monkeypatch):
    """No HECO_FRAMES_DIR: frameRef null, the JPEG always, jpeg=0 ignored."""
    monkeypatch.delenv("HECO_FRAMES_DIR", raising=False)
    synthetic(lambda i: textured(), n_frames=None, fps=10.0, period_s=0.005)
    client.post("/open", json={"path": __file__, "bufferS": 5})
    (body,) = _take_fresh(client, 1, jpeg=False)
    assert body["frameRef"] is None and body["imageB64"], "no ref, so the JPEG is kept"
