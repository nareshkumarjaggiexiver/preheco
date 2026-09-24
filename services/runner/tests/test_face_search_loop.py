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


# ------------------------------------------------ the face search region (L7)

#: The central half of the 160x120 scene: 40..120 x 30..90 in pixels.
REGION = {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}
REGION_PX = {"x": 40, "y": 30, "w": 80, "h": 60}


def region_run(**kw) -> dict:
    """The request body of a count run with the region set."""
    return {**RUN, "faceRegion": dict(REGION), **kw}


@pytest.mark.parametrize("whole_frame", [False, True])
def test_the_region_is_searched_as_one_native_crop_whatever_the_switch(whole_frame):
    """within = [the region in pixels], every frame; persons still see it all."""
    fake = Searches(settling_frames(4), SETTLING)
    final = make_loop(
        fake, region_run(), faces_whole_frame=whole_frame, frame_prefetch=False,
    ).run()
    assert [b["within"] for b in fake.face_bodies] == [[REGION_PX]] * 4
    assert final["faceRegionUnplaced"] == 0
    assert final["unique"] == 1
    assert fake.run_created["config"]["faceRegion"] == REGION
    assert fake.run_ended["results"]["faceRegionUnplaced"] == 0
    assert "faceRegionUnplaced=0 " in fake.run_ended["notes"]


@pytest.mark.parametrize("parallel", [False, True])
def test_the_detect_worker_searches_the_region_and_reasons_the_same(parallel):
    """Overlap: the region needs no tracks, so the worker searches it."""
    runs = []
    for overlap in (False, True):
        fake = Searches(settling_frames(5), SETTLING)
        final = make_loop(
            fake, region_run(), pipeline_overlap=overlap, parallel_detect=parallel,
            frame_prefetch=False,
        ).run()
        runs.append(([b["within"] for b in fake.face_bodies], final["unique"],
                     final["matches"], final["faceRegionUnplaced"]))
    assert runs[0] == runs[1] == ([[REGION_PX]] * 5, 1, 5, 0)


class Sizeless(Searches):
    """Frames that state no pixel size (and carry no decodable JPEG)."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Strip w/h from every frame ingest serves."""
        out = super().handler(request)
        if request.url.host == "ingest" and request.url.path == "/frame":
            body = json.loads(out.content)
            body.pop("w", None)
            body.pop("h", None)
            return httpx.Response(200, json=body)
        return out


@pytest.mark.parametrize("overlap", [False, True])
def test_a_frame_with_no_size_is_searched_as_without_a_region_and_counted(overlap):
    """The region cannot be placed: look anyway (crops, here), and say so."""
    fake = Sizeless(settling_frames(3), SETTLING)
    final = make_loop(
        fake, region_run(), pipeline_overlap=overlap, frame_prefetch=False,
    ).run()
    assert final["faceRegionUnplaced"] == 3, "once per frame, never twice"
    assert all(isinstance(b["within"], list) and b["within"] != [REGION_PX]
               for b in fake.face_bodies), "the crop search, as configured"
    assert final["unique"] == 1


def test_a_body_outside_the_region_does_not_hold_the_cadence_open():
    """A guest across the hall is never searched for by a doorway region.

    B stands wholly outside the region and never settles.  Without a region
    she keeps every frame searched; with one she is not the search's
    business, and the settled guest inside it lets the cadence skip.
    """
    from tests.test_loop_v1 import scripted_verdict

    far = {"x": 130, "y": 20, "w": 25, "h": 60}   # right of x=120: outside
    frames = [{"boxes": [A, far], "faces": [FA]} for _ in range(7)]
    script = [scripted_verdict("p00001", True, None)] + [
        scripted_verdict("p00001", False, 0.8) for _ in range(10)
    ]
    inside = dict(REGION, x=0.0, w=0.75)          # 0..120: holds A, not `far`
    skipped = {}
    for request in (RUN, {**RUN, "faceRegion": inside}):
        fake = Searches(frames, list(script))
        final = make_loop(
            fake, request, faces_whole_frame=True, frame_prefetch=False, **CADENCE,
        ).run()
        skipped["region" if "faceRegion" in request else "none"] = (
            final["faceDetectSkippedSettled"]
        )
    assert skipped["none"] == 0, "an unsettled body anywhere keeps the search running"
    assert skipped["region"] > 0, "outside the region, she cannot"


def test_malformed_regions_are_refused_with_a_readable_422():
    """Outside the frame, pixels for fractions, or a sliver: refused at the edge."""
    from app.main import app as runner_app
    from fastapi.testclient import TestClient

    client = TestClient(runner_app)
    base = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    for region, needle in (
        ({"x": 0.8, "y": 0.1, "w": 0.4, "h": 0.5}, "inside the frame"),
        ({"x": 10, "y": 10, "w": 800, "h": 600}, "faceRegion"),
        ({"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.5}, "at least 5%"),
        ({"x": -0.1, "y": 0.1, "w": 0.5, "h": 0.5}, "faceRegion"),
    ):
        r = client.post("/runs", json={**base, "faceRegion": region})
        assert r.status_code == 422, region
        assert needle in r.text, (region, r.text[:300])


def test_the_manifest_declares_the_region():
    """The console forwards a region only to a pipeline that says it takes one."""
    from app import main

    assert main._manifest()["capabilities"]["faceRegion"] is True


# ------------------------------------ the cadence against zones and the crops


class IdScene(Scene):
    """A scene whose gallery knows WHO each face is (by its box x, carried as
    the embedding), so a verdict does not depend on how many /match calls a
    lever skipped — the count is only right if every guest was searched."""

    def __init__(self, frames, who: dict):
        super().__init__(frames, [])
        self.who, self.seen = who, set()
        self.face_bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Faces recorded; embed carries the face's x; match knows its owner."""
        host, path = request.url.host, request.url.path
        if host == "faces" and path == "/detect":
            self.face_bodies.append(json.loads(request.content))
        if host == "embed" and path == "/embed":
            faces = json.loads(request.content)["faces"]
            return httpx.Response(200, json={
                "embeddings": [[float(f["box"]["x"])] * 128 for f in faces]})
        if host == "match" and path == "/match":
            body = json.loads(request.content)
            self.match_bodies.append(body)
            key = self.who[int(body["embedding"][0])]
            new = key not in self.seen
            self.seen.add(key)
            self.guest_n += new
            return httpx.Response(200, json={
                "personKey": key, "isNew": new, "cosine": None if new else 0.8,
                "subCanon": False, "isStaff": False, "staffId": None,
                "templateN": 1, "templateAdded": False, "galleryN": self.guest_n})
        return super().handler(request)


#: A detections zone over the lower half of B's box: B's BODY centre (130, 70)
#: is inside it, B's FACE centre (132, 45) is not — so B's face is countable.
TV_UNDER_B = {"label": "tv", "mode": "detections",
              "points": [[0.70, 0.50], [0.95, 0.50], [0.95, 0.90], [0.70, 0.90]]}
#: ...and one over B's head too: then no face on that body can count.
TV_OVER_B = {"label": "tv", "mode": "detections",
             "points": [[0.60, 0.10], [0.99, 0.10], [0.99, 0.95], [0.60, 0.95]]}


@pytest.mark.parametrize("overlap", [False, True])
def test_a_guest_in_front_of_a_detections_zone_holds_the_search_open(overlap):
    """A settles alone; then B joins, body centre in a detections zone, face
    not. The zone drops B's box before the tracker, so B never settles — and
    must therefore keep the whole-frame search running until B is counted.

    Verified on the cadence as it was: unique 1 (B never searched, the ledger
    saying "face search skipped: 1 settled" with two bodies in frame), where
    the cadence off counted 2.
    """
    from tests.test_presence import FB, B

    frames = [{"boxes": [A], "faces": [FA]}] * 4 + [{"boxes": [A, B], "faces": [FA, FB]}] * 5
    who = {FA["x"]: "p00001", FB["x"]: "p00002"}
    cadence = {"face_cadence": True, "face_reverify_interval_s": 3.0,
               "face_cadence_max_gap_s": 1.0}
    for zone, unique in ((TV_UNDER_B, 2), (TV_OVER_B, 1)):
        fake = IdScene(frames, who)
        final = make_loop(fake, {**RUN, "exclusionZones": [zone]}, faces_whole_frame=True,
                          frame_prefetch=False, pipeline_overlap=overlap, **cadence).run()
        assert final["unique"] == unique, (zone["points"], final)
        searched = searched_frames(fake)
        if unique == 2:
            assert 4 in searched, "B's first frame was searched"
        else:
            # B's head is in the zone too: its face would be excluded, so the
            # TV does not keep the search running.
            assert searched[-1] < 8, searched


def test_under_the_cadence_a_crop_search_is_a_full_pass():
    """Crop path: the re-verify gate emptied the cadence's max-gap searches.

    One still, settled guest, re-verify interval 2 s, max gap 0.25 s, 30
    frames 0.1 s apart. The gate dropped her crop from every search inside
    the interval, so "searches" went out with within=[] — nobody searched —
    and each still reset the gap clock: her face was really searched at 0,
    1, 4 and 25 only. Now every search the cadence lets through carries her
    crop, and there is one every 0.3 s.
    """
    fake = IdScene([{"boxes": [A], "faces": [FA]}] * 30, {FA["x"]: "p00001"})
    final = make_loop(fake, RUN, frame_prefetch=False, face_cadence=True,
                      face_reverify_interval_s=2.0, face_cadence_max_gap_s=0.25).run()
    assert fake.face_bodies, "nothing searched at all"
    assert all(b["within"] for b in fake.face_bodies), "a cadence search searched nobody"
    assert searched_frames(fake) == [0, 1, *range(4, 30, 3)]
    assert final["unique"] == 1
