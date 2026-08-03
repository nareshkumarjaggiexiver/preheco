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
