"""The runner's appearance riders: white balance (HECO_APPEARANCE_WB).

Off by default, and off is today's loop exactly: frame_gains is never
called and the torso descriptor is read without gains.  On, the frame's
gains are estimated ONCE per decoded frame and handed to every descriptor
read off it.  The knob reaches GET /health as knobs.appearanceWb.
"""

from app import config as cfg
from app import loop as loop_mod
from app.config import Settings
from fastapi.testclient import TestClient

from .test_loop_v1 import RED_BGR, TorsoFrames, make_loop, solid_jpeg_b64

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
        assert client.get("/health").json()["knobs"] == {"appearanceWb": False}
        monkeypatch.setattr(
            main.manager, "settings", Settings(appearance_wb=True), raising=False
        )
        assert client.get("/health").json()["knobs"]["appearanceWb"] is True
