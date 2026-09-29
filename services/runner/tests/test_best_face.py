"""Best-face minting and best-face templates (Settings.mint_min_px,
best_face_anchor): the runner's half.

A scripted WALK-IN drives each test: per frame, the person boxes (with the
tracker's ids) and the faces on them, whose width grows as the guest nears the
camera. A small fake gallery answers /match the way match 0.20.0 does — a probe
(``mint: false``) that matches nobody is ``deferred`` and writes nothing, a
``minCosine`` raises the bar — so each test can say who was created, from
which face, and when.
"""

import json

import httpx
import numpy as np
import pytest
from app import loop as loop_mod

from tests.test_loop import FakePipeline, frame_index, make_loop

THRESHOLD = 0.363


def _vec(person: int, tilt: float = 0.0, axis: int = 60) -> list[float]:
    """Person ``person``'s face: their own axis, optionally tilted towards a
    shared one so two views of one guest can sit at a chosen cosine."""
    v = np.zeros(64)
    v[person] = 1.0
    if tilt:
        v = np.cos(tilt) * v
        v[axis] = np.sin(tilt)
    return [float(x) for x in v / np.linalg.norm(v)]


def _person(tid: int, x: float, width: float, vec: list[float], frontality=0.9, sharpness=100.0):
    """One tracked body with one face of ``width`` px at column ``x``."""
    face_h = width * 1.2
    return {
        "track": tid,
        "box": {"x": x, "y": 10.0, "w": width * 1.6, "h": 220.0, "conf": 0.9},
        "face": {
            "box": {"x": x + width * 0.3, "y": 20.0, "w": width, "h": face_h},
            "landmarks": [[1, 1]] * 5,
            "conf": 0.9,
            "widthPx": width,
            "frontality": frontality,
            "sharpness": sharpness,
        },
        "vec": vec,
    }


class WalkIn(FakePipeline):
    """FakePipeline driven by a per-frame script of people, with a fake gallery."""

    def __init__(self, script: list[list[dict]], known: dict | None = None):
        """``script[i]`` is frame i's people; ``known`` pre-seeds the gallery."""
        super().__init__(n_frames=len(script))
        self.script = script
        self.gallery: dict[str, list[np.ndarray]] = {
            k: [np.array(v)] for k, v in (known or {}).items()
        }
        self.minted = len(self.gallery)
        self.match_bodies: list[dict] = []
        self.anchors: list[dict] = []
        self.split_pairs: list[tuple] = []

    def _frame(self, body) -> list[dict]:
        return self.script[min(frame_index(body["imageB64"]), len(self.script) - 1)]

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Script persons/tracker/faces/embed; simulate the gallery."""
        host, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else {}
        if host == "persons" and path == "/detect":
            self._last_people = self._frame(body)
            return httpx.Response(200, json={"boxes": [p["box"] for p in self._last_people]})
        if host == "tracker" and path == "/track":
            tracks = [{"id": p["track"], "box": p["box"], "ageFrames": 5, "hits": 5}
                      for p in self._last_people]
            return httpx.Response(200, json={"tracks": tracks})
        if host == "faces" and path == "/detect":
            return httpx.Response(200, json={
                "faces": [p["face"] for p in self._frame(body)], "inferMs": 1.0,
            })
        if host == "embed" and path == "/embed":
            vecs = [p["vec"] for p in self._last_people][: len(body["faces"])]
            return httpx.Response(200, json={"embeddings": vecs, "alignMs": 0.5})
        if host == "match" and path == "/match":
            return httpx.Response(200, json=self._match(body))
        if host == "match" and path == "/template/anchor":
            self.anchors.append(body)
            self.gallery[body["personKey"]].append(np.array(body["embedding"]))
            return httpx.Response(200, json={"ok": True, "templateN": 1})
        if host == "match" and path == "/split":
            self.split_pairs.append(tuple(sorted((body["a"], body["b"]))))
            return httpx.Response(200, json={"ok": True, "galleryN": len(self.gallery)})
        return super().handler(request)

    def _match(self, body: dict) -> dict:
        self.match_bodies.append({k: v for k, v in body.items() if k != "embedding"})
        q = np.array(body["embedding"])
        best_key, best = None, -1.0
        for k, templates in self.gallery.items():
            c = max(float(t @ q) for t in templates)
            if c > best:
                best_key, best = k, c
        bar = max(THRESHOLD, body.get("minCosine") or 0.0)
        if best_key is not None and best >= bar:
            return {"personKey": best_key, "deferred": False, "isNew": False, "cosine": best,
                    "galleryN": len(self.gallery), "subCanon": False, "templateN": 1}
        if body.get("mint") is False:
            return {"personKey": None, "deferred": True, "isNew": False,
                    "cosine": best if best_key else None, "galleryN": len(self.gallery),
                    "subCanon": False, "templateN": 0}
        self.minted += 1
        key = f"p{self.minted:05d}"
        self.gallery[key] = [q]
        return {"personKey": key, "deferred": False, "isNew": True,
                "cosine": best if best_key else None, "galleryN": len(self.gallery),
                "subCanon": False, "templateN": 1}


@pytest.fixture(autouse=True)
def pixels(monkeypatch):
    """Give the loop a real (blank) 320x240 picture, so crops and cards run."""
    monkeypatch.setattr(loop_mod, "decode_jpeg_b64", lambda _b64: np.zeros((240, 320, 3), np.uint8))


HOLD = {"mint_min_px": 112.0, "quality_min_px": 80.0, "hold_flush_frames": 2}


def _counters(fake):
    return fake.run_ended["results"]


def test_off_by_default_nothing_is_held_and_no_new_field_is_sent():
    """Without mint_min_px the first face creates the guest, exactly as before."""
    fake = WalkIn([[_person(1, 20, w, _vec(1))] for w in (85, 95, 120)])
    make_loop(fake, quality_min_px=80.0).run()
    assert _counters(fake)["unique"] == 1
    assert fake.match_bodies[0]["quality"] == 85, "created from the first, smallest face"
    assert all("mint" not in b and "minCosine" not in b for b in fake.match_bodies)
    assert "facesHeld" not in _counters(fake)


def test_small_faces_are_held_and_the_guest_is_created_from_the_first_big_face():
    """85, 95, 105 px are held; the 120 px face creates the guest."""
    fake = WalkIn([[_person(1, 20, w, _vec(1))] for w in (85, 95, 105, 120, 125)])
    make_loop(fake, **HOLD).run()
    res = _counters(fake)
    assert res["unique"] == 1
    probes = [b for b in fake.match_bodies if b.get("mint") is False]
    assert [b["quality"] for b in probes] == [85, 95, 105], "only the small faces probe"
    minting = [b for b in fake.match_bodies if b.get("mint") is not False]
    assert minting[0]["quality"] == 120, "the guest is created from the first big face"
    assert res["facesHeld"] == 3 and res["heldResolved"] == 1
    assert "heldMinted" not in res, "nothing was left to create from the hold"


def test_a_guest_who_never_comes_close_is_created_from_their_best_held_face():
    """No face reaches 112 px; when the track leaves, the best held face creates
    the guest — front-on first: the 100 px front-on face beats a 104 px turned
    one, and it is clearly (over 8%) wider than the first 90 px look."""
    frames = [
        [_person(1, 20, 90, _vec(1), frontality=0.9)],
        [_person(1, 20, 104, _vec(1), frontality=0.6)],   # wider but turned
        [_person(1, 20, 100, _vec(1), frontality=0.95)],  # front-on: the best
        [], [], [], [],                                    # the track has gone
    ]
    fake = WalkIn(frames)
    make_loop(fake, **HOLD).run()
    res = _counters(fake)
    assert res["unique"] == 1 and res["heldMinted"] == 1
    created = [b for b in fake.match_bodies if b.get("mint") is not False]
    assert [b["quality"] for b in created] == [100]


def test_the_end_of_the_run_creates_everyone_still_held():
    """A track still in view at end of file is created from its best face."""
    fake = WalkIn([[_person(1, 20, w, _vec(1))] for w in (85, 100)])
    make_loop(fake, **HOLD).run()
    assert _counters(fake)["unique"] == 1 and _counters(fake)["heldMinted"] == 1


def test_a_small_face_of_a_known_guest_claims_them_and_creates_nobody():
    """The probe finds the guest already in the gallery: an ordinary match."""
    fake = WalkIn([[_person(2, 20, 90, _vec(1, tilt=0.3))]], known={"p00001": _vec(1)})
    make_loop(fake, **HOLD).run()
    res = _counters(fake)
    assert res["unique"] == 0 and res["matches"] == 1
    assert "facesHeld" not in res


def test_the_stricter_bar_holds_a_doubtful_small_face_and_it_still_resolves():
    """At cosine ~0.40 a small face may not claim the known guest under a 0.45
    bar, so it is held — and at the end its best face, asked at the ordinary
    threshold, turns out to be that guest: matched, not counted twice."""
    known = {"p00001": _vec(1)}
    tilt = float(np.arccos(0.40))
    fake = WalkIn([[_person(2, 20, 90, _vec(1, tilt=tilt))]], known=known)
    make_loop(fake, small_match_min_cosine=0.45, **HOLD).run()
    res = _counters(fake)
    assert fake.match_bodies[0]["minCosine"] == 0.45
    assert res["facesHeld"] == 1 and res["heldMatched"] == 1
    assert res["unique"] == 0, "no second guest"


def test_two_held_bodies_seen_together_are_kept_apart_once_created():
    """Two small-faced guests walking side by side, never close: both created
    from their best faces when they leave, and asserted as two people."""
    frames = [
        [_person(1, 10, 90, _vec(1)), _person(2, 170, 88, _vec(2))],
        [_person(1, 10, 95, _vec(1)), _person(2, 170, 92, _vec(2))],
        [], [], [], [],
    ]
    fake = WalkIn(frames)
    make_loop(fake, **HOLD).run()
    res = _counters(fake)
    assert res["unique"] == 2 and res["heldMinted"] == 2
    assert ("p00001", "p00002") in fake.split_pairs, "co-present: cannot be one person"


def test_the_best_face_follows_the_guest_and_the_same_image_is_their_template():
    """With best_face_anchor, each clearly better face replaces the card AND
    the best-face template; a merely similar face replaces neither, and a
    stranger's better face never takes over the guest."""
    frames = [
        [_person(1, 20, w, _vec(1))] for w in (120, 125, 135, 150)
    ] + [[_person(1, 20, 170, _vec(9))]]  # a different face on the same body
    fake = WalkIn(frames)
    lp = make_loop(fake, best_face_anchor=True, quality_min_px=80.0)
    lp.run()
    res = _counters(fake)
    assert res["unique"] == 2, "the stranger's face is its own guest"
    widths = [a["quality"] for a in fake.anchors]
    assert widths == [135, 150], "125 is under 8% better; 135 and 150 are not"
    assert all(a["personKey"] == "p00001" for a in fake.anchors)
    assert lp._card_rank["p00001"][1] == 150


def test_front_on_beats_width_then_width_then_sharpness():
    """The operator's order, as the rank comparison implements it."""
    better = loop_mod.RunLoop._rank_better
    assert better((2, 90.0, 50.0), (1, 200.0, 500.0)), "front-on beats a wider turned face"
    assert not better((1, 200.0, 500.0), (2, 90.0, 50.0))
    assert better((2, 110.0, 50.0), (2, 100.0, 50.0)), "10% wider"
    assert not better((2, 105.0, 50.0), (2, 100.0, 50.0)), "5% wider is not clearly better"
    assert better((2, 99.0, 130.0), (2, 100.0, 100.0)), "as wide and 30% sharper"
    assert not better((2, 99.0, 110.0), (2, 100.0, 100.0))
    assert better((0, 50.0, 1.0), None), "anything beats nothing"


def test_a_track_whose_best_face_is_turned_may_claim_a_guest_but_never_creates_one():
    """Both Sharon doubles came from a held track created off a turned face:
    such a track probes (it may be a known guest), and otherwise is not counted."""
    turned = [[_person(1, 20, 95, _vec(1), frontality=0.4)], [], [], [], []]
    fake = WalkIn(turned)
    make_loop(fake, **HOLD).run()
    res = _counters(fake)
    assert res["unique"] == 0 and res["heldTurned"] == 1
    assert "heldMinted" not in res
    assert all(b.get("mint") is False for b in fake.match_bodies), "every ask was a probe"

    # A known guest is still claimed — on the first frame, as an ordinary
    # small-face probe: never held, never turned away.
    known = WalkIn(turned, known={"p00001": _vec(1)})
    make_loop(known, **HOLD).run()
    res = _counters(known)
    assert res["unique"] == 0 and res["matches"] == 1
    assert "heldTurned" not in res and "facesHeld" not in res
