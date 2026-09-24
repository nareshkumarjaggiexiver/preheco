"""The runner's face search: where it looks, and when it may not look at all.

Runner-side tests for the face-search levers of 2026-09-24 — the cadence
(HECO_FACE_CADENCE) and the search region (faceRegion) — and for the
re-verify gate they sit beside.  The pure rules live in
``heco_counting.face_search`` and are unit-tested there; these drive the
real loop against the scripted scene fake, so what is pinned is what a run
actually does.
"""

import json

import httpx
import pytest

from tests.test_loop_v1 import make_loop
from tests.test_presence import FA, RUN, A, Scene
from tests.test_reverify_fixtures import SETTLING, settling_frames


class Searches(Scene):
    """The scene fake, recording every face search it is asked for."""

    def __init__(self, frames, script):
        super().__init__(frames, script)
        self.face_bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Record a /detect body on faces, then serve the scene."""
        if request.url.host == "faces" and request.url.path == "/detect":
            self.face_bodies.append(json.loads(request.content))
        return super().handler(request)


def test_a_whole_frame_search_skips_no_crops_so_it_counts_none():
    """faceSearchesSkipped claimed crops the whole-frame search never made.

    With the re-verify interval armed and the frame searched whole, a
    settled track's crop used to be counted as skipped on every frame while
    the whole frame — that track included — was searched anyway.  The saving
    exists only on the crop path; here it must read zero, and every search
    must still be the whole frame.
    """
    fake = Searches(settling_frames(6), SETTLING)
    final = make_loop(
        fake, RUN, faces_whole_frame=True, face_reverify_interval_s=3600.0,
        frame_prefetch=False,
    ).run()
    assert final["faceSearchesSkipped"] == 0
    assert len(fake.face_bodies) == 6
    assert all(b["within"] is None for b in fake.face_bodies)


def test_the_crop_path_still_takes_its_saving():
    """The same footage on crops: the settled track's crop IS skipped."""
    fake = Searches(settling_frames(6), SETTLING)
    final = make_loop(
        fake, RUN, face_reverify_interval_s=3600.0, frame_prefetch=False,
    ).run()
    assert final["faceSearchesSkipped"] >= 1
    assert fake.face_bodies[-1]["within"] == []


# ------------------------------------------------ the face-search cadence (L4)

#: The cadence settings the tests share: a 3 s settledness window and a
#: max gap of 0.25 s of footage — frames here are 0.1 s apart, so a room of
#: settled guests is searched every third frame.
CADENCE = {"face_cadence": True, "face_reverify_interval_s": 3.0,
           "face_cadence_max_gap_s": 0.25}


def searched_frames(fake: Searches) -> list[int]:
    """Which scenes the faces service was asked to search, in order."""
    from tests.test_presence import scene_index

    return [scene_index(b["imageB64"]) for b in fake.face_bodies]


def records(fake) -> dict:
    """seq -> the frame's ledger record."""
    return {r["seq"]: r for r in fake.frame_records}


@pytest.mark.parametrize("whole_frame", [False, True])
def test_a_settled_room_is_searched_only_at_the_max_gap(whole_frame):
    """One guest, settled from frame 1: searched at 0, 1, then every 0.3 s.

    Frame 0 is the first search and mints her; frame 1 still searches (no
    lock until its verdict) and locks her track.  From then on every body is
    settled, so the search runs only when 0.25 s of footage has gone by —
    frames 4, 7, 10 — and each of those refreshes the lock.  The count does
    not move: one guest, whatever was skipped.
    """
    n = 11
    fake = Searches(settling_frames(n), SETTLING)
    final = make_loop(
        fake, RUN, faces_whole_frame=whole_frame, frame_prefetch=False, **CADENCE,
    ).run()
    assert searched_frames(fake) == [0, 1, 4, 7, 10]
    assert final["faceDetectSkippedSettled"] == n - 5
    assert final["unique"] == 1
    assert final["frames"] == n, "a skipped frame is still a processed frame"
    events = records(fake)
    assert "face search skipped: 1 settled" in events[2]["events"]
    assert not any(e.startswith("face search") for e in events[4]["events"])
    # On the permanent record, beside unique.
    assert fake.run_ended["results"]["faceDetectSkippedSettled"] == n - 5
    assert f"faceDetectSkippedSettled={n - 5} " in fake.run_ended["notes"]
    lever = fake.run_created["config"]["levers"]["faceCadence"]
    assert lever == {"maxGapS": 0.25, "reverifyIntervalS": 3.0}


def test_a_newcomer_is_searched_every_frame_until_settled():
    """Two bodies, one unknown: nothing about the known one saves the search."""
    from tests.test_loop_v1 import scripted_verdict
    from tests.test_presence import FB, B

    frames = settling_frames(3) + [{"boxes": [A, B], "faces": [FA, FB]}] * 4
    # Frames 0 and 1 are searched (A minted, then locked) and frame 2 is
    # skipped, so the script holds two verdicts for A alone.  B only ever
    # matches in impostor range, so B's track never locks.
    script = SETTLING[:2] + [
        v
        for _ in range(4)
        for v in (scripted_verdict("p00001", False, 0.8),
                  scripted_verdict("p00002", False, 0.3))
    ]
    fake = Searches(frames, script)
    final = make_loop(fake, RUN, faces_whole_frame=True, frame_prefetch=False, **CADENCE).run()
    assert searched_frames(fake)[-4:] == [3, 4, 5, 6], "every frame B is in is searched"
    assert final["faceDetectSkippedSettled"] == 1, "only frame 2, when A stood alone"


def test_an_empty_frame_is_always_searched():
    """No person box is not "everyone settled": the detector can miss a body."""
    frames = settling_frames(3) + [{"boxes": [], "faces": [FA]}] * 2
    fake = Searches(frames, SETTLING)
    make_loop(fake, RUN, faces_whole_frame=True, frame_prefetch=False, **CADENCE).run()
    assert searched_frames(fake)[-2:] == [3, 4]


def test_skipped_frames_still_assert_track_presence():
    """A skipped search still runs the tracker AND the track-presence door.

    The pair's split is refused (503) twice while the frames are searched;
    the third attempt is on a frame the cadence skipped — both guests
    settled, no face searched — and it lands on the tracks' word alone.
    """
    from tests.test_loop_v1 import scripted_verdict
    from tests.test_presence import FB, B

    class RefusesTwice(Searches):
        def handler(self, request):
            if request.url.host == "match" and request.url.path == "/split":
                self.split_status = 503 if len(self.splits) < 2 else None
            return super().handler(request)

    frames = [
        {"boxes": [A], "faces": [FA]},        # 0: p00001 minted
        {"boxes": [A], "faces": [FA]},        # 1: p00001 locked + bound
        {"boxes": [A, B], "faces": [FB]},     # 2: p00002 minted — split 503
        {"boxes": [A, B], "faces": [FB]},     # 3: p00002 locked + bound — split 503
        {"boxes": [A, B], "faces": [FA, FB]}, # 4: both settled: SKIPPED — split lands
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),
        scripted_verdict("p00002", False, 0.70),
    ]
    fake = RefusesTwice(frames, script)
    final = make_loop(
        fake, RUN, faces_whole_frame=True, frame_prefetch=False,
        **{**CADENCE, "face_cadence_max_gap_s": 1.0},
    ).run()
    assert searched_frames(fake) == [0, 1, 2, 3], "frame 4 was not searched"
    assert final["faceDetectSkippedSettled"] == 1
    pair = {"runId": "prun-1", "a": "p00001", "b": "p00002"}
    assert fake.splits == [pair, pair, pair]
    assert final["trackPresenceSplits"] == 1
    assert records(fake)[4]["events"] == [
        "face search skipped: 2 settled", "track-presence p00001 != p00002",
    ]


def test_with_the_overlap_the_worker_rules_and_the_ruling_is_fixed():
    """Detect worker + cadence: skips happen, the count holds, and it repeats.

    The worker rules on frame k+1 from the evidence the loop held when k was
    handed over (after frame k-1), so its pattern lags the serial one by a
    frame — searched at 0, 1, 2, then every 0.3 s — and, because that
    evidence is taken on the loop thread, it is the same pattern every run.
    """
    patterns = []
    for _ in range(3):
        fake = Searches(settling_frames(11), SETTLING)
        final = make_loop(
            fake, RUN, faces_whole_frame=True, pipeline_overlap=True, **CADENCE,
        ).run()
        assert final["unique"] == 1 and final["frames"] == 11
        patterns.append((searched_frames(fake), final["faceDetectSkippedSettled"]))
    assert patterns[0] == ([0, 1, 2, 5, 8], 6)
    assert patterns[0] == patterns[1] == patterns[2]


def test_the_worker_searches_a_body_that_moved_off_its_track():
    """Overlap: a settled track only covers a body that is still ON it.

    From frame 3 the guest's box jumps 0.4 box-widths a frame — her box on
    frame k+1 overlaps her track from frame k-1 under IoU 0.5 — so the
    worker's evidence does not cover her and every such frame is searched.
    """
    frames = settling_frames(3) + [
        {"boxes": [dict(A, x=10 + 48 * i)], "faces": [dict(FA, x=12 + 48 * i)]}
        for i in range(1, 5)
    ]
    fake = Searches(frames, SETTLING)
    make_loop(fake, RUN, faces_whole_frame=True, pipeline_overlap=True, **CADENCE).run()
    assert searched_frames(fake)[-4:] == [3, 4, 5, 6]


@pytest.mark.parametrize("overlap", [False, True])
def test_parallel_detect_waits_for_persons_only_when_the_rule_could_skip(overlap):
    """Side by side while a search is certain; persons first once it might not be.

    Same footage, same skips as the rule without parallel detect: the
    parallel lever may change when faces is ISSUED, never whether.
    """
    runs = {}
    for parallel in (False, True):
        fake = Searches(settling_frames(11), SETTLING)
        final = make_loop(
            fake, RUN, faces_whole_frame=True, pipeline_overlap=overlap,
            parallel_detect=parallel, frame_prefetch=False, **CADENCE,
        ).run()
        runs[parallel] = (searched_frames(fake), final["faceDetectSkippedSettled"])
    if overlap:
        assert runs[True] == runs[False]
    else:
        # Serial: the plain rule reads THIS frame's tracks, the parallel one
        # the previous frame's — a still guest's box is the same either way.
        assert runs[True][1] == runs[False][1] == 6


def test_an_interval_of_zero_makes_the_cadence_skip_nothing_and_say_so(caplog):
    """The dependency, stated where an operator will see it."""
    fake = Searches(settling_frames(6), SETTLING)
    final = make_loop(
        fake, RUN, faces_whole_frame=True, frame_prefetch=False,
        **{**CADENCE, "face_reverify_interval_s": 0.0},
    ).run()
    assert final["faceDetectSkippedSettled"] == 0
    assert len(fake.face_bodies) == 6
    assert any("skip nothing" in r.getMessage() for r in caplog.records)


def test_the_cadence_off_measures_nothing():
    """Off: no counter (None, not 0), nothing in the run record."""
    fake = Searches(settling_frames(4), SETTLING)
    final = make_loop(fake, RUN, faces_whole_frame=True, frame_prefetch=False).run()
    assert final["faceDetectSkippedSettled"] is None
    assert "faceDetectSkippedSettled" not in fake.run_ended["results"]
    assert "faceDetectSkippedSettled" not in fake.run_ended["notes"]
    assert "levers" not in fake.run_created["config"]
