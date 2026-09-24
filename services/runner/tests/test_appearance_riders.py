"""The runner's appearance riders: white balance, head and beard.

White balance (HECO_APPEARANCE_WB) is off by default, and off is today's
loop exactly: frame_gains is never called and the torso descriptor is read
without gains.  On, the frame's gains are estimated ONCE per decoded frame
and handed to every descriptor read off it.  The knob reaches GET /health
as knobs.HECO_APPEARANCE_WB.

Head and beard ride /match for every kept face that has landmarks, and ride
the same-frame re-ask too, so a re-resolved sighting is not logged blind.
"""

import httpx
import pytest
from app import config as cfg
from app import loop as loop_mod
from app.config import Settings
from fastapi.testclient import TestClient

from .test_loop_v1 import (
    FA8FC3_SCRIPT,
    MEN_FACES,
    RED_BGR,
    TorsoFrames,
    TwoMen,
    make_loop,
    solid_jpeg_b64,
)

REQUEST = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}


def test_wb_defaults_off_and_reads_the_environment(monkeypatch):
    """Default off; 1 turns it on; empty (compose's unset) is the default."""
    assert Settings().appearance_wb is False
    assert cfg.from_env().appearance_wb is False
    monkeypatch.setenv("HECO_APPEARANCE_WB", "1")
    assert cfg.from_env().appearance_wb is True
    monkeypatch.setenv("HECO_APPEARANCE_WB", "")
    assert cfg.from_env().appearance_wb is False


def test_wb_off_never_estimates_and_reads_torsos_without_gains(monkeypatch):
    """Off is the loop as it was: no estimate, no gains argument."""
    seen = []
    real = loop_mod.appearance.torso_descriptor

    def spy(img, face_box, person_box, gains=None):
        seen.append(gains)
        return real(img, face_box, person_box, gains)

    def boom(_img):
        raise AssertionError("white balance is off; nothing may estimate it")

    monkeypatch.setattr(loop_mod.appearance, "torso_descriptor", spy)
    monkeypatch.setattr(loop_mod.appearance, "frame_gains", boom)
    fake = TorsoFrames(images=[solid_jpeg_b64(RED_BGR)] * 2)
    make_loop(fake, REQUEST).run()
    assert seen and all(g is None for g in seen)
    assert all(len(b["appearance"]) == 64 for b in fake.match_bodies)


def test_wb_on_estimates_once_per_frame_and_hands_the_gains_on(monkeypatch):
    """One estimate per decoded frame; every descriptor on it gets those gains."""
    estimates, seen = [], []
    real = loop_mod.appearance.torso_descriptor

    def gains(img):
        estimates.append(img.shape)
        return (1.2, 1.0, 0.8)

    def spy(img, face_box, person_box, g=None):
        seen.append(g)
        return real(img, face_box, person_box, g)

    monkeypatch.setattr(loop_mod.appearance, "frame_gains", gains)
    monkeypatch.setattr(loop_mod.appearance, "torso_descriptor", spy)
    fake = TorsoFrames(images=[solid_jpeg_b64(RED_BGR)] * 3)
    make_loop(fake, REQUEST, appearance_wb=True).run()
    assert len(estimates) == len(fake.match_bodies) == 3, "once per frame, one face each"
    assert seen == [(1.2, 1.0, 0.8)] * 3


def test_wb_reaches_health_as_a_knob(monkeypatch):
    """The switch is read back from the process that runs it."""
    from app import main

    with TestClient(main.app) as client:
        assert client.get("/health").json()["knobs"]["HECO_APPEARANCE_WB"] is False
        monkeypatch.setattr(
            main.manager, "settings", Settings(appearance_wb=True), raising=False
        )
        assert client.get("/health").json()["knobs"]["HECO_APPEARANCE_WB"] is True


# ------------------------------------------------------------ head + beard

#: Plausible landmarks for TorsoFrames' face {x 12, y 22, w 60, h 78}:
#: eyes 30 px apart, nose under them, mouth corners under the nose.
FACE_LANDMARKS = [[27, 50], [57, 50], [42, 65], [32, 82], [52, 82]]


class LandmarkTorsoFrames(TorsoFrames):
    """TorsoFrames whose face carries landmarks a face could have."""

    def handler(self, request):
        """Serve the face with real landmarks; everything else as TorsoFrames."""
        if request.url.host == "faces" and request.url.path == "/detect":
            self.calls.append("faces /detect")
            return httpx.Response(200, json={"faces": [{
                "box": {"x": 12, "y": 22, "w": 60, "h": 78},
                "landmarks": FACE_LANDMARKS, "conf": 0.9,
            }], "inferMs": 1.0})
        return super().handler(request)


def test_head_and_beard_ride_the_match_when_the_face_has_landmarks():
    """A red frame: the head reads red hue bins, the chin reads as its cheek."""
    fake = LandmarkTorsoFrames(images=[solid_jpeg_b64(RED_BGR)])
    make_loop(fake, REQUEST).run()
    (body,) = fake.match_bodies
    head, beard = body["head"], body["beard"]
    assert len(head) == 40 and sum(head) == pytest.approx(1.0)
    assert sum(head[:24]) == pytest.approx(1.0), "all chromatic: a red head"
    assert beard == pytest.approx([1.0, 0.0, 0.0, 0.0]), "the chin is the cheek"


def test_a_face_without_usable_landmarks_sends_neither():
    """Degenerate landmarks (all at one point): no head key, no beard key."""
    fake = TorsoFrames(images=[solid_jpeg_b64(RED_BGR)])
    make_loop(fake, REQUEST).run()
    assert fake.match_bodies
    assert all("head" not in b and "beard" not in b for b in fake.match_bodies)


def test_an_opaque_frame_sends_neither():
    """No decodable frame, no readings — absent, never zeros."""
    fake = LandmarkTorsoFrames(images=["ZmFrZS1qcGVn"])
    make_loop(fake, REQUEST).run()
    assert fake.match_bodies
    assert all("head" not in b and "beard" not in b for b in fake.match_bodies)


class TwoMenWithLandmarks(TwoMen):
    """kf-577's two men, each face with landmarks a face could have."""

    def handler(self, request):
        """Serve the two faces with real landmarks; the rest as TwoMen."""
        if request.url.host == "faces" and request.url.path == "/detect":
            self.calls.append("faces /detect")
            faces = []
            for f in MEN_FACES:
                x = f["box"]["x"]
                lm = [[x + 15, 45], [x + 45, 45], [x + 30, 58], [x + 19, 72], [x + 41, 72]]
                faces.append({**f, "landmarks": lm})
            return httpx.Response(200, json={"faces": faces})
        return super().handler(request)


def test_the_same_frame_reask_carries_the_faces_head_and_beard():
    """The re-resolved sighting is logged with the readings the first ask had."""
    fake = TwoMenWithLandmarks(n_frames=1, match_script=list(FA8FC3_SCRIPT))
    make_loop(fake, REQUEST).run()
    first, second, reask = fake.match_bodies[:3]
    assert reask.get("excludeKeys") == ["p00001"]
    assert reask["head"] == second["head"] and reask["beard"] == second["beard"]
    assert first["head"] != second["head"], "red man and blue man read differently"
