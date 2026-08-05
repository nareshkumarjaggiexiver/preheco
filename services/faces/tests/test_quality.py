"""Boundary tests for the POC quality flag (80 / 56 px per CONTRACTS.md) and
for the Laplacian sharpness proxy that sits beside it."""

import cv2
import numpy as np
import pytest

from app.quality import classify_width, crop_sharpness


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (120, "ok"),
        (80, "ok"),  # canon boundary inclusive
        (79.9, "sub-canon"),
        (79, "sub-canon"),
        (64, "sub-canon"),  # POC expected range 64-85 px lands here or in ok
        (56, "sub-canon"),  # floor boundary inclusive
        (55.9, "reject"),
        (10, "reject"),
    ],
)
def test_default_thresholds(width, expected):
    assert classify_width(width) == expected


def test_env_style_overrides():
    assert classify_width(70, canon_px=60, floor_px=40) == "ok"
    assert classify_width(50, canon_px=60, floor_px=40) == "sub-canon"
    assert classify_width(30, canon_px=60, floor_px=40) == "reject"


# ------------------------------------------------------- sharpness (point #6)


def _textured(w=200, h=200, seed=7):
    """A high-frequency patch: the sharpest thing the measure can see."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_sharpness_ranks_a_blurred_crop_below_its_sharp_original():
    """The whole point of the signal: same content, only focus differs.

    Regression guard for the claim the docstring used to make and the code did
    not: this must be an ORDERING over focus, not a restatement of size.
    """
    img = _textured()
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    box = {"x": 50, "y": 50, "w": 100, "h": 100}
    assert crop_sharpness(img, box) > crop_sharpness(blurred, box)


def test_sharpness_is_normalised_for_crop_size():
    """A near face and a far face of the SAME picture must score alike.

    Without the resize, Laplacian variance mostly measures how many pixels the
    crop has — which is what iedPx already measures honestly, so an unnormalised
    value would gate on size twice under a name that promised focus.
    """
    img = _textured(400, 400)
    big = crop_sharpness(img, {"x": 0, "y": 0, "w": 400, "h": 400})
    small = crop_sharpness(
        cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA),
        {"x": 0, "y": 0, "w": 100, "h": 100},
    )
    # Same scene, 16x fewer pixels: the scores stay the same order of magnitude
    # (a raw Laplacian variance would not).
    assert 0.4 < small / big < 2.5


def test_sharpness_is_none_when_it_cannot_be_measured():
    """Unmeasurable must be None, never 0.0 — the gate reads None as UNKNOWN
    and keeps the face, because dropping a guest off an invoice for a signal
    nobody measured is the worse error."""
    img = _textured(50, 50)
    assert crop_sharpness(img, {"x": 0, "y": 0, "w": 1, "h": 1}) is None
    assert crop_sharpness(img, {"x": 80, "y": 80, "w": 20, "h": 20}) is None
    assert crop_sharpness(img, {"x": -50, "y": -50, "w": 10, "h": 10}) is None


def test_sharpness_clamps_a_box_running_off_the_edge():
    """A face at the frame edge is still measurable from the part that is in."""
    img = _textured(100, 100)
    assert crop_sharpness(img, {"x": 60, "y": 60, "w": 80, "h": 80}) is not None
