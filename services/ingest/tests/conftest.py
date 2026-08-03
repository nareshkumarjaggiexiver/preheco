"""Shared fixtures: a synthetic video file (moving rectangle) and a client.

The video is written with cv2.VideoWriter using MJPG/.avi — the one
codec/container combination opencv-python-headless can always write without
system codecs. No network is touched anywhere in these tests.
"""

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
