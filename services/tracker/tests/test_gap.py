"""A break in the source clears the tracks (HECO_TRACKER_MAX_GAP_MS).

THE CASE (2026-09-25, live test on the Sharon CP Plus camera): the camera
dropped off the network for about two minutes and the run survived it — but
max_age counts FRAMES, so to the tracker the first frame after the outage was
one frame after the last one before it, and a detection where a track had
been could continue that track onto whoever stood there now.
"""

import pytest
from app.main import _last_used, _runs, app
from app.sort import SortLite
from fastapi.testclient import TestClient

BOX = (100.0, 100.0, 40.0, 80.0, 0.9)


def ids(tracks):
    """The reported track ids of one step."""
    return [t.tid for t in tracks]


def walk(tracker, t0_ms, frames, step_ms=67):
    """``frames`` frames of one standing person at 15 fps from ``t0_ms``; the last ids."""
    out = []
    for i in range(frames):
        out = ids(tracker.step([BOX], t_ms=t0_ms + i * step_ms))
    return out


def test_an_ordinary_frame_gap_keeps_the_track():
    """Two seconds without a frame (a slow patch) is not a break."""
    t = SortLite(max_gap_ms=10_000)
    before = walk(t, 0, 5)
    after = ids(t.step([BOX], t_ms=5 * 67 + 2_000))
    assert after == before == [1] and t.gap_resets == 0


def test_an_outage_longer_than_the_gap_starts_fresh_ids():
    """Two minutes of silence: the person standing there is a NEW track."""
    t = SortLite(max_gap_ms=10_000)
    assert walk(t, 0, 5) == [1]
    back = 5 * 67 + 120_000
    assert ids(t.step([BOX], t_ms=back)) == [], "a new track is unconfirmed on its first frame"
    assert walk(t, back + 67, 3) == [2], "and when confirmed it is not track 1"
    assert t.gap_resets == 1


def test_a_clock_that_goes_backwards_is_a_restarted_source():
    """A reopened source restarts its frame clock: that is a break too."""
    t = SortLite(max_gap_ms=10_000)
    walk(t, 50_000, 5)
    t.step([BOX], t_ms=100)
    assert t.gap_resets == 1 and all(tr.tid != 1 for tr in t.tracks)


def test_zero_is_off_and_frames_alone_decide():
    """max_gap_ms 0: the outage coasts through, as before 2026-09-25."""
    t = SortLite(max_gap_ms=0)
    walk(t, 0, 5)
    assert ids(t.step([BOX], t_ms=500_000)) == [1] and t.gap_resets == 0


def test_steps_without_a_frame_time_never_reset():
    """No t_ms (an older caller): nothing to judge a gap by."""
    t = SortLite(max_gap_ms=10_000)
    for _ in range(5):
        t.step([BOX])
    assert ids(t.step([BOX])) == [1] and t.gap_resets == 0


@pytest.fixture()
def client(monkeypatch):
    """In-process client with a clean run table and the default gap."""
    monkeypatch.delenv("HECO_TRACKER_MAX_GAP_MS", raising=False)
    _runs.clear()
    _last_used.clear()
    with TestClient(app) as c:
        yield c
    _runs.clear()
    _last_used.clear()


def post(client, t_ms):
    """One frame with one person box at ``t_ms``; the reported ids."""
    res = client.post("/track", json={
        "runId": "live", "tMs": t_ms,
        "boxes": [{"x": BOX[0], "y": BOX[1], "w": BOX[2], "h": BOX[3], "conf": BOX[4]}],
    })
    assert res.status_code == 200, res.text
    return [t["id"] for t in res.json()["tracks"]]


def test_the_service_reads_tms_and_says_so_on_health(client):
    """The wire's tMs reaches the tracker; /health counts the breaks."""
    for i in range(5):
        post(client, i * 67)
    for i in range(3):
        last = post(client, 130_000 + i * 67)
    assert last == [2]
    body = client.get("/health").json()
    assert body["gapResets"] == 1 and body["maxGapMs"] == 10_000


def test_the_knob_turns_it_off(client, monkeypatch):
    """HECO_TRACKER_MAX_GAP_MS=0: a new run's tracker coasts the outage."""
    monkeypatch.setenv("HECO_TRACKER_MAX_GAP_MS", "0")
    client.post("/reset", json={"runId": "live"})
    for i in range(5):
        post(client, i * 67)
    assert post(client, 130_000) == [1]
    assert client.get("/health").json()["gapResets"] == 0
