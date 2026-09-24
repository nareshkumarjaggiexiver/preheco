"""Shared fixtures: a synthetic video file (moving rectangle) and a client.

The video is written with cv2.VideoWriter using MJPG/.avi — the one
codec/container combination opencv-python-headless can always write without
system codecs. No network is touched anywhere in these tests.

The lever tests use ``synthetic`` instead: numpy frames served through the
cv2.VideoCapture seam, so a test controls every pixel of every frame.
"""

import time

import cv2
import numpy as np
import pytest
from app.main import app, state
from fastapi.testclient import TestClient

VID_W, VID_H, VID_FPS, VID_FRAMES = 64, 48, 100.0, 40


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> str:
    """Write a tiny clip of a green rectangle marching across the frame."""
    path = str(tmp_path_factory.mktemp("vid") / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), VID_FPS, (VID_W, VID_H))
    assert vw.isOpened(), "cv2.VideoWriter could not open MJPG/.avi"
    for i in range(VID_FRAMES):
        img = np.zeros((VID_H, VID_W, 3), np.uint8)
        x = (i * 3) % (VID_W - 10)
        cv2.rectangle(img, (x, 10), (x + 8, 26), (0, 255, 0), -1)
        vw.write(img)
    vw.release()
    return path


@pytest.fixture()
def client():
    """In-process TestClient with lifespan; leaves no worker running."""
    with TestClient(app) as c:
        yield c
    state.swap(None)  # belt over lifespan braces: never leak a capture thread


# ------------------------------------------------ synthetic source (L1 levers)


class SyntheticCapture:
    """A cv2.VideoCapture stand-in that serves numpy frames: no codec, no disk.

    ``frame_fn(i)`` makes frame i (0-based); ``n_frames`` None never ends.
    ``period_s`` sleeps before each frame, the way a camera makes you wait.
    ``calls`` counts read() vs grab(), which is how the tests see whether a
    frame was retrieved (converted) or only decoded.
    """

    def __init__(self, frame_fn, n_frames=None, fps=10.0, period_s=0.0):
        """A fresh stream positioned at frame 0."""
        self.frame_fn = frame_fn
        self.n_frames = n_frames
        self.fps = fps
        self.period_s = period_s
        self.i = 0
        self.calls = {"read": 0, "grab": 0}

    def isOpened(self):  # noqa: N802 — mirrors cv2's API
        """Always open."""
        return True

    def get(self, prop):
        """Only FPS is asked for."""
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop, value):
        """Rewind support for loop=True."""
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.i = int(value)
        return True

    def _next(self):
        if self.period_s:
            time.sleep(self.period_s)
        if self.n_frames is not None and self.i >= self.n_frames:
            return None
        frame = self.frame_fn(self.i)
        self.i += 1
        return frame

    def grab(self):
        """Decode without retrieving."""
        self.calls["grab"] += 1
        return self._next() is not None

    def read(self):
        """Decode and retrieve."""
        self.calls["read"] += 1
        frame = self._next()
        return frame is not None, frame

    def release(self):
        """Nothing to free."""


@pytest.fixture()
def synthetic(monkeypatch):
    """Patch cv2.VideoCapture to serve SyntheticCapture streams.

    Returns ``make(frame_fn, **kw)``; every VideoCapture the worker opens (the
    FPS probe included) is a fresh stream built from the same arguments, and
    the list of streams opened is returned so a test can inspect the one the
    worker read from (the last).
    """
    opened: list[SyntheticCapture] = []

    def make(frame_fn, **kw):
        def factory(*_a, **_k):
            cap = SyntheticCapture(frame_fn, **kw)
            opened.append(cap)
            return cap

        monkeypatch.setattr(cv2, "VideoCapture", factory)
        return opened

    return make


def textured(w=320, h=240, seed=7):
    """A static scene with structure at 1/8 scale, kept clear of 0 and 255.

    Values stay in [30, 190] so a x1.2 + 8 lighting step never clips — the
    gate's normalisation removes an affine change exactly, and clipping is
    the one thing that would make a global change look local.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(30, 190, size=(h // 16 + 1, w // 16 + 1), dtype=np.uint8)
    img = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.merge([img, np.roll(img, 7, axis=1), np.roll(img, 13, axis=0)])
