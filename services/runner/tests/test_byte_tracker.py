"""The runner's half of byte mode (Settings.tracker_low_conf_min).

Low-score person boxes are asked for only when the knob is on, reach the
tracker ALONE — zone-filtered, silently — and the run's record says how many
went and how many kept a track alive.  Off, nothing about a run changes.
"""

import json

import httpx

from tests.test_loop import FakePipeline, _zone, make_loop

#: The half-hidden guest: well away from the confident box at (10, 20).
LOW = {"x": 200, "y": 20, "w": 40, "h": 100, "conf": 0.15}


class LowBoxPipeline(FakePipeline):
    """FakePipeline whose persons stage also answers lowBoxes when asked."""

    def __init__(self, **kw):
        """Record every persons body and every tracker lowBoxes list."""
        super().__init__(**kw)
        self.persons_bodies: list[dict] = []
        self.tracked_low: list[list] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Intercept persons /detect and tracker /track; the rest as before."""
        host, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else {}
        if host == "persons" and path == "/detect":
            self.persons_bodies.append({k: v for k, v in body.items() if k != "imageB64"})
            reply = json.loads(super().handler(request).content)
            if "lowConfMin" in body:
                reply["lowBoxes"] = [dict(LOW)]
            return httpx.Response(200, json=reply)
        if host == "tracker" and path == "/track":
            self.tracked_low.append(body.get("lowBoxes"))
            reply = json.loads(super().handler(request).content)
            reply["lowMatched"] = len(body.get("lowBoxes") or [])
            return httpx.Response(200, json=reply)
        return super().handler(request)


def test_off_by_default_the_run_is_exactly_as_before():
    """No lowConfMin to persons, no lowBoxes to the tracker, no new counters."""
    fake = LowBoxPipeline(n_frames=2)
    make_loop(fake).run()
    assert fake.persons_bodies == [{}, {}]
    assert fake.tracked_low == [None, None]
    results = fake.run_ended["results"]
    assert "trackerLowBoxes" not in results and "trackerLowHeld" not in results
    assert "trackerLow" not in fake.run_ended["notes"]


def test_on_the_low_boxes_reach_the_tracker_and_the_record_counts_them():
    """Asked for at the knob's score, handed to the tracker, summed per run."""
    fake = LowBoxPipeline(n_frames=3)
    make_loop(fake, tracker_low_conf_min=0.1).run()
    assert fake.persons_bodies == [{"lowConfMin": 0.1}] * 3
    assert fake.tracked_low == [[LOW]] * 3
    assert all(len(b) == 1 for b in fake.tracked_boxes), "the confident list is untouched"
    results = fake.run_ended["results"]
    assert results["trackerLowBoxes"] == 3
    assert results["trackerLowHeld"] == 3
    assert "trackerLowBoxes=3 trackerLowHeld=3" in fake.run_ended["notes"]


def test_the_low_boxes_reach_nothing_but_the_tracker():
    """Faces search, the frame record and the person counts never see them."""
    fake = LowBoxPipeline(n_frames=2)
    loop = make_loop(fake, tracker_low_conf_min=0.1)
    loop.run()
    assert all(b.get("conf") != 0.15 for b in loop._last["boxes"]), "not in the loop's boxes"


def test_a_detections_zone_drops_low_boxes_too_without_counting_them():
    """A wall TV's weak boxes must not keep its phantom alive either; the zone
    counter still counts only the boxes the loop works with."""
    fake = LowBoxPipeline(n_frames=2)
    loop = make_loop(fake, tracker_low_conf_min=0.1)
    # A zone over the LOW box only (x 200..240 of 320): the confident box at
    # x 10..50 stays trackable.
    loop.zones = [{"label": "tv", "mode": "detections",
                   "points": [[0.55, 0.0], [1.0, 0.0], [1.0, 1.0], [0.55, 1.0]]}]
    final = loop.run()
    assert fake.tracked_low == [None, None], "zoned away before the tracker"
    assert not final.get("personsZoned"), "and not counted as a zoned person"
    assert fake.run_ended["results"]["trackerLowBoxes"] == 0, "on, and zero — kept"


def test_the_zone_helper_itself_is_unchanged():
    """The existing detections-zone behaviour for confident boxes still holds."""
    fake = LowBoxPipeline(n_frames=2)
    loop = make_loop(fake, tracker_low_conf_min=0.1)
    loop.zones = [_zone("detections")]
    final = loop.run()
    assert final["personsZoned"] == 2
    assert fake.tracked_boxes == [[], []]
