"""The face-search cadence rule, as pure functions (lever L4)."""

from heco_counting import face_search as fs

A = {"x": 10, "y": 20, "w": 60, "h": 100}
B = {"x": 100, "y": 20, "w": 60, "h": 100}
A_NUDGED = {"x": 14, "y": 22, "w": 60, "h": 100}    # IoU ~0.84 with A
A_HALF = {"x": 45, "y": 20, "w": 60, "h": 100}      # IoU ~0.26 with A


def track(tid, box):
    """One tracker row."""
    return {"id": tid, "box": box}


def test_settled_means_a_fresh_lock_and_nothing_less():
    """A lock refreshed inside the interval settles; stale, absent or future does not."""
    tracks = [track(1, A), track(2, B), track(3, A), track(4, B)]
    lock_at = {1: 9.5, 2: 7.0, 4: 10.5}   # 3 holds no lock
    settled = fs.settled_tracks(tracks, lock_at, now_s=10.0, fresh_s=2.0, window_s=20.0)
    assert [t["id"] for t in settled] == [1], "only the fresh lock; 4's is in the future"


def test_an_interval_of_zero_settles_nobody():
    """faceReverifyIntervalS 0 = verify every frame = a cadence that never skips."""
    assert fs.settled_tracks([track(1, A)], {1: 10.0}, 10.0, fresh_s=0.0, window_s=20.0) == []


def test_a_lock_past_the_heal_window_is_not_evidence():
    """The lock expires on the heal window; an interval longer than it cannot revive it."""
    tracks = [track(1, A)]
    assert fs.settled_tracks(tracks, {1: 0.0}, 25.0, fresh_s=60.0, window_s=20.0) == []
    assert fs.settled_tracks(tracks, {1: 6.0}, 25.0, fresh_s=60.0, window_s=20.0) == tracks


def test_the_search_is_skipped_only_when_every_body_is_settled_and_recent():
    """The one path to None."""
    settled = [track(1, A), track(2, B)]
    assert fs.search_due([A_NUDGED, B], settled, 10.0, 9.8, 1.0) is None


def test_every_other_case_searches_and_says_why():
    """Newcomer, no bodies, first frame, gap, clock trouble: all search."""
    settled = [track(1, A)]
    assert fs.search_due([A, B], settled, 10.0, 9.8, 1.0) == "unsettled", "B is a newcomer"
    assert fs.search_due([A_HALF], settled, 10.0, 9.8, 1.0) == "unsettled", "IoU 0.26 < 0.5"
    assert fs.search_due([], settled, 10.0, 9.8, 1.0) == "no-bodies"
    assert fs.search_due([A], settled, 10.0, None, 1.0) == "first"
    assert fs.search_due([A], settled, 10.0, 9.0, 1.0) == "gap", "exactly max_gap_s searches"
    assert fs.search_due([A], settled, 10.0, 10.5, 1.0) == "gap", "a clock gone backwards"
    assert fs.search_due([A], settled, None, 9.8, 1.0) == "no-clock"
    assert fs.search_due([A], [], 10.0, 9.8, 1.0) == "unsettled"


def test_two_people_inside_one_settled_box_are_not_both_covered():
    """One to one: a settled track vouches for ONE body."""
    settled = [track(1, A)]
    assert fs.search_due([A, A_NUDGED], settled, 10.0, 9.8, 1.0) == "unsettled"
    assert fs.search_due([A, A_NUDGED], [track(1, A), track(2, A_NUDGED)], 10.0, 9.8, 1.0) is None


def test_the_matching_is_maximal_not_greedy():
    """A body whose only partner a neighbour could take is still matched.

    Body X overlaps both settled boxes (0.92 with S1, 0.72 with S2); body Y
    overlaps only S1 (0.67; 0.43 with S2).  Greedy by IoU gives S1 to X and
    strands Y; the right answer is X-S2, Y-S1, and every body is covered.
    """
    s1 = {"x": 0, "y": 0, "w": 100, "h": 100}
    s2 = {"x": 20, "y": 0, "w": 100, "h": 100}
    x = {"x": 4, "y": 0, "w": 100, "h": 100}
    y = {"x": -20, "y": 0, "w": 100, "h": 100}
    assert fs.covered([x, y], [s1, s2])
    assert not fs.covered([x, y, y], [s1, s2]), "three bodies, two tracks"


def test_may_skip_is_false_whenever_the_search_is_certain():
    """Decidable before persons returns: no clock, first frame, gap, nobody settled."""
    assert fs.may_skip(2, 10.0, 9.8, 1.0) is True
    assert fs.may_skip(0, 10.0, 9.8, 1.0) is False
    assert fs.may_skip(2, 10.0, None, 1.0) is False
    assert fs.may_skip(2, 10.0, 9.0, 1.0) is False
    assert fs.may_skip(2, None, 9.8, 1.0) is False


# ------------------------------------------------ the face search region (L7)


def test_a_region_becomes_whole_pixels_inside_the_frame():
    """The 4K bench frame's central 1920x1080."""
    region = {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}
    assert fs.region_px(region, 3840, 2160) == {"x": 960, "y": 540, "w": 1920, "h": 1080}


def test_a_region_is_clamped_and_an_unplaceable_one_is_none():
    """Past the edge is clamped; no size, no region, nothing left: None."""
    assert fs.region_px({"x": 0.9, "y": 0.0, "w": 0.2, "h": 1.0}, 100, 50) == {
        "x": 90, "y": 0, "w": 10, "h": 50,
    }
    assert fs.region_px({"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}, None, 50) is None
    assert fs.region_px(None, 100, 50) is None
    assert fs.region_px({"x": 1.0, "y": 0.0, "w": 0.1, "h": 0.1}, 100, 50) is None
    assert fs.region_px({"x": "a", "y": 0, "w": 1, "h": 1}, 100, 50) is None


def test_only_bodies_touching_the_region_count_for_the_cadence():
    """A guest across the hall cannot hold a doorway's search open."""
    region = {"x": 0, "y": 0, "w": 100, "h": 100}
    inside = {"x": 10, "y": 10, "w": 20, "h": 40}
    straddling = {"x": 90, "y": 50, "w": 40, "h": 80}
    outside = {"x": 150, "y": 10, "w": 20, "h": 40}
    assert fs.bodies_in([inside, straddling, outside], region) == [inside, straddling]
    assert fs.bodies_in([inside, outside], None) == [inside, outside]
