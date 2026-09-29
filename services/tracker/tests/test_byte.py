"""Byte mode (app/sort.py, "BYTE MODE"): low-score boxes keep tracks alive in
a group, and association is optimal rather than greedy.

Every test states the behaviour a group at an entrance needs, and where it
matters, runs the same frames through sort mode to show the difference.
"""

import pytest
from app.main import _last_used, _runs, app
from app.sort import SortLite
from fastapi.testclient import TestClient


def _box(x: float, y: float = 50.0, w: float = 20.0, h: float = 40.0, conf: float = 0.9):
    """Detection tuple helper."""
    return (x, y, w, h, conf)


def _low(x: float, y: float = 50.0, w: float = 20.0, h: float = 40.0):
    """A low-score detection: the half-hidden guest."""
    return (x, y, w, h, 0.15)


def _walk_then_hide(trk: SortLite):
    """Five confident frames walking right, then four where only a low box
    sees the guest; returns the ids reported on each hidden frame."""
    for i in range(5):
        trk.step([_box(10 + i * 4)])
    seen = []
    for i in range(5, 9):
        out = trk.step([], low=[_low(10 + i * 4)])
        seen.append([t.tid for t in out])
    return seen


def test_a_half_hidden_guest_keeps_their_track_in_byte_mode():
    """Four frames seen only by a low box: byte keeps and reports the id; sort coasts blind."""
    byte = SortLite(min_hits=3, mode="byte")
    assert _walk_then_hide(byte) == [[1], [1], [1], [1]], "reported every frame, same id"
    assert byte.low_matches == 4
    track = byte.tracks[0]
    assert track.misses == 0 and track.hits == 9
    assert track.cx == pytest.approx(10 + 8 * 4 + 10), "anchored on the low box, not a blind coast"

    sort = SortLite(min_hits=3)
    assert _walk_then_hide(sort) == [[], [], [], []], "sort mode coasts blind and reports nothing"
    assert sort.low_matches == 0


def test_a_low_box_never_starts_a_track():
    """Unmatched low boxes vanish; they are never born as a person."""
    trk = SortLite(min_hits=1, mode="byte")
    for _ in range(5):
        assert trk.step([], low=[_low(100)]) == []
    assert trk.tracks == []


def test_a_low_box_never_revives_a_lost_track():
    """Only a track matched on the previous frame may take a low box."""
    trk = SortLite(min_hits=3, max_age=30, mode="byte")
    for _ in range(4):
        trk.step([_box(10)])
    trk.step([])  # lost for a frame: misses 1
    trk.step([])  # misses 2 — no longer "matched on the previous frame"
    assert trk.step([], low=[_low(10)]) == [], "a weak box must not reach a lost track"
    assert trk.low_matches == 0
    # A confident box still re-associates through the coast, as in sort mode.
    assert [t.tid for t in trk.step([_box(10)])] == [1]


def test_a_low_box_never_props_up_an_unconfirmed_track():
    """A one-hit track past warm-up gets no help from low boxes."""
    trk = SortLite(min_hits=3, mode="byte")
    trk.step([_box(300)])            # warm-up frame
    trk.step([_box(300)])
    trk.step([_box(300)])
    trk.step([_box(10)])             # a new, unconfirmed track (1 hit) past warm-up
    trk.step([_box(300)], low=[_low(10)])
    young = next(t for t in trk.tracks if t.tid == 2)
    assert young.hits == 1, "a one-frame ghost cannot be kept alive by weak boxes"
    assert trk.low_matches == 0


def test_optimal_assignment_keeps_both_guests_where_greedy_drops_one():
    """Two guests close together (a group).  Greedy takes the single best
    pair (track 1 -> the right box, IoU 0.60) and leaves track 2 with nothing,
    so the left box is born as a THIRD person.  The assignment pairs track 1
    with the left box (0.43) and track 2 with the right (0.48)."""
    def run(mode):
        trk = SortLite(min_hits=1, mode=mode)
        for _ in range(3):  # two stationary guests, 6 px apart
            trk.step([(0, 0, 10, 10, 0.9), (6, 0, 10, 10, 0.9)])
        out = trk.step([(-4, 0, 10, 10, 0.9), (2.5, 0, 10, 10, 0.9)])
        return sorted(t.tid for t in out), trk._next_id - 1

    assert run("byte") == ([1, 2], 2), "both guests keep their ids; nobody new"
    ids, minted = run("sort")
    assert minted == 3 and 3 in ids, "greedy mints a third person"


def test_sort_mode_ignores_low_boxes_entirely():
    """Sort mode is unchanged: low boxes are not even looked at."""
    trk = SortLite(min_hits=1)
    for _ in range(3):
        trk.step([_box(10)], low=[_low(200)])
    assert [t.tid for t in trk.tracks] == [1]
    assert trk.low_matches == 0


def test_an_unknown_mode_is_refused():
    """A misspelt mode fails loudly rather than running sort silently."""
    with pytest.raises(ValueError, match="tracker mode"):
        SortLite(mode="bytetrack")


# ---------------------------------------------------------------- the API


@pytest.fixture
def byte_client(monkeypatch):
    """In-process client whose trackers are built in byte mode."""
    monkeypatch.setenv("HECO_TRACKER_MODE", "byte")
    _runs.clear()
    _last_used.clear()
    with TestClient(app) as c:
        yield c
    _runs.clear()
    _last_used.clear()


def _post(client, run_id, boxes, low=()):
    """POST one frame of confident and low boxes; return the reply body."""
    body = {
        "runId": run_id,
        "tMs": 0,
        "boxes": [{"x": x, "y": 50, "w": 20, "h": 40, "conf": 0.9} for x in boxes],
        "lowBoxes": [{"x": x, "y": 50, "w": 20, "h": 40, "conf": 0.15} for x in low],
    }
    res = client.post("/track", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_the_api_carries_low_boxes_and_reports_what_they_held(byte_client):
    """/track passes lowBoxes through and counts the ones that held a track."""
    assert byte_client.get("/health").json()["mode"] == "byte"
    for i in range(4):
        _post(byte_client, "r1", [10 + i * 4])
    held = _post(byte_client, "r1", [], low=[26])
    assert [t["id"] for t in held["tracks"]] == [1]
    assert held["lowMatched"] == 1
    assert held["tracks"][0]["box"]["conf"] == 0.15, "the box that held it, as detected"


def test_a_runner_that_sends_no_low_boxes_gets_the_old_reply_shape(client_sort):
    """No lowBoxes in, lowMatched 0 out, and the default mode is sort."""
    body = {"runId": "r2", "tMs": 0, "boxes": [{"x": 10, "y": 50, "w": 20, "h": 40, "conf": 0.9}]}
    res = client_sort.post("/track", json=body)
    assert res.status_code == 200
    assert res.json()["lowMatched"] == 0
    assert client_sort.get("/health").json()["mode"] == "sort"


@pytest.fixture
def client_sort(monkeypatch):
    """In-process client with the mode knob unset (sort)."""
    monkeypatch.delenv("HECO_TRACKER_MODE", raising=False)
    _runs.clear()
    _last_used.clear()
    with TestClient(app) as c:
        yield c
    _runs.clear()
    _last_used.clear()
