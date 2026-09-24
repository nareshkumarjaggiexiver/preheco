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

from tests.test_loop_v1 import make_loop
from tests.test_presence import RUN, Scene
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
