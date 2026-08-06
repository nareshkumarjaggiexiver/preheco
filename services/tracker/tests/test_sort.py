"""Unit tests for SortLite on synthetic box sequences.

The three behaviours the brief demands proof of:
1. a straight pass keeps one id;
2. two crossing boxes hold their ids (documented: constant-velocity
   prediction separates them; only exact coincidence is ambiguous);
3. a dropout shorter than max-age re-associates to the same id.
"""

from app.sort import SortLite, iou


def _box(x: float, y: float, w: float = 20.0, h: float = 40.0, conf: float = 0.9):
    """Detection tuple helper."""
    return (x, y, w, h, conf)


def test_iou_basics():
    """IoU: identity is 1, disjoint is 0, half-overlap is computed right."""
    a = (0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, (20, 20, 10, 10)) == 0.0
    assert abs(iou(a, (5, 0, 10, 10)) - (50 / 150)) < 1e-9
    assert iou((0, 0, 0, 0), a) == 0.0  # degenerate box


def test_straight_pass_keeps_one_id():
    """One box marching right stays one track with one id, hits climbing."""
    trk = SortLite(max_age=5, min_hits=3)
    ids = set()
    last = None
    for i in range(12):
        out = trk.step([_box(10 + i * 8, 50)])
        if i >= 2:  # confirmed past warm-up regardless
            assert len(out) == 1
            ids.add(out[0].tid)
            last = out[0]
    assert ids == {1}
    assert last.hits == 12
    assert last.age_frames == 12


def test_min_hits_suppresses_one_frame_ghost():
    """After warm-up, a detection must persist min_hits frames to report."""
    trk = SortLite(max_age=5, min_hits=3)
    for i in range(6):  # established track, past the warm-up window
        trk.step([_box(10 + i * 8, 50)])
    # A one-frame ghost appears far away, then vanishes: never reported.
    out = trk.step([_box(58, 50), _box(400, 200)])
    assert [t.tid for t in out] == [1]
    out = trk.step([_box(66, 50)])
    assert [t.tid for t in out] == [1]


def test_two_crossing_boxes_hold_ids():
    """Documented behaviour: distinct lanes → ids hold through the cross.

    Two boxes approach on slightly offset y-lanes and pass through each
    other. Constant-velocity prediction keeps each track glued to its own
    detection, so ids must not swap. (Exact coincidence of both detections
    would be ambiguous and MAY swap — accepted, gallery repairs identity.)
    """
    trk = SortLite(max_age=5, min_hits=3)
    lane_a, lane_b = 50.0, 62.0  # offset lanes, boxes h=40 so heavy overlap
    n = 26
    id_by_lane: dict[str, set[int]] = {"a": set(), "b": set()}
    for i in range(n):
        ax = 10 + i * 8.0          # a: left -> right
        bx = 10 + (n - 1 - i) * 8.0  # b: right -> left
        out = trk.step([_box(ax, lane_a), _box(bx, lane_b)])
        for t in out:
            lane = "a" if abs((t.cy - 20) - lane_a) < 6 else "b"
            id_by_lane[lane].add(t.tid)
    # Each lane saw exactly one id, and they differ: no swap, no ghosts.
    assert len(id_by_lane["a"]) == 1
    assert len(id_by_lane["b"]) == 1
    assert id_by_lane["a"] != id_by_lane["b"]


def test_dropout_within_max_age_reassociates():
    """A track missing < max_age frames resumes with the SAME id."""
    trk = SortLite(max_age=6, min_hits=3)
    for i in range(8):
        out = trk.step([_box(10 + i * 8, 50)])
    tid = out[0].tid
    for _ in range(4):  # detector drops the person for 4 frames (< max_age)
        assert trk.step([]) == []  # coasting tracks are not reported
    out = trk.step([_box(10 + 12 * 8, 50)])  # reappears where motion put them
    assert [t.tid for t in out] == [tid]
    assert out[0].age_frames == 13  # age kept counting through the gap


def test_dropout_beyond_max_age_becomes_new_id():
    """Past max-age the track is pruned; reappearance is a new id."""
    trk = SortLite(max_age=3, min_hits=1)
    for i in range(5):
        out = trk.step([_box(10 + i * 8, 50)])
    tid = out[0].tid
    for _ in range(5):  # > max_age
        trk.step([])
    out = trk.step([_box(10 + 10 * 8, 50)])
    assert len(out) == 1 and out[0].tid != tid


def test_ids_are_stable_and_monotonic():
    """New tracks take fresh, increasing ids."""
    trk = SortLite(min_hits=1)
    out = trk.step([_box(0, 0), _box(300, 300)])
    assert sorted(t.tid for t in out) == [1, 2]
    out = trk.step([_box(2, 0), _box(302, 300), _box(600, 600)])
    assert sorted(t.tid for t in out) == [1, 2, 3]


def test_default_max_age_coasts_the_seated_gap_that_shattered_one_person():
    """The default is 30 frames — ~7.5 s at the bench's 3.97 fps, not 3.75 s.

    Bench 6e1a5d (2026-08-06, ONE person out of frame, back in, then seated)
    produced SIX tracker ids: 2 (53 frames), 5 (24), 6 (8), 7 (6), 8 (6), 12
    (7). At 3.97 fps the old 15-frame max-age died after 3.75 s unmatched, and
    the seated subject's detector gaps were longer than that — so tracks
    6/7/8/12 are the same person, four times. A 20-frame gap (~5 s, squarely
    inside what killed those ids) must now re-associate to the SAME id.
    """
    seated = _box(100, 50)  # sitting still: the detector's gaps, not motion
    assert SortLite().max_age == 30

    trk = SortLite(min_hits=3)  # default max_age
    for _ in range(8):
        out = trk.step([seated])
    tid = out[0].tid
    for _ in range(20):  # ~5 s the detector cannot see the seated body
        assert trk.step([]) == []
    out = trk.step([seated])
    assert [t.tid for t in out] == [tid], "a ~5 s seated gap must not mint a new id"

    # ...and the same gap under the OLD 15-frame max-age is where the extra
    # ids came from: the track is pruned mid-gap and the guest is minted anew.
    old = SortLite(min_hits=1, max_age=15)  # min_hits=1: report the re-birth at once
    for _ in range(8):
        out = old.step([seated])
    was = out[0].tid
    for _ in range(20):
        old.step([])
    assert old.step([seated])[0].tid != was, "15 frames is what shattered track 6/7/8/12"


def test_a_new_arrival_is_not_reported_until_confirmed():
    """The invariant behind runner finding D1.

    After warm-up the tracker reports a person only once they have min_hits
    frames, so a guest who has just walked into shot is a DETECTION with no
    reported track. The runner must therefore search faces in the raw
    detection boxes, never in the reported tracks alone — a face never
    searched is a guest never counted. If this test ever fails because new
    arrivals ARE reported, the runner's face-search region can be simplified;
    until then, loop.py must keep raw boxes in `within`.
    """
    t = SortLite(min_hits=3, max_age=15, iou_min=0.3)
    steady = (100.0, 100.0, 50.0, 120.0, 0.9)
    for _ in range(6):                       # past warm-up
        reported = t.step([steady])
    assert len(reported) == 1

    arrival = (400.0, 100.0, 50.0, 120.0, 0.9)
    reported = t.step([steady, arrival])
    assert len(reported) == 1, "an unconfirmed arrival must not be reported yet"

    # ...and every reported box IS a detection box (why the union's track half
    # is redundant today, and why raw boxes must come first in the dedupe).
    x, y, w, h = reported[0].box()
    assert (round(x), round(y), round(w), round(h)) == (100, 100, 50, 120)
