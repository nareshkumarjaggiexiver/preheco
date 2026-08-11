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


# ------------------------------------------- landmark topology and eye span
# Ported from the sibling face-detection pipeline, which found that YuNet
# verifies striped shirts as faces at 70-91% confidence. Their landmarks are
# effectively random, and that is what these two signals see.

from app.quality import eye_span_ratio, landmarks_plausible  # noqa: E402

BOX = {"x": 0, "y": 0, "w": 100, "h": 130}


def _face_landmarks(*, eye_y=30.0, nose_y=55.0, mouth_y=80.0, rex=35.0, lex=65.0, nx=50.0):
    """Five landmarks in YuNet order: right eye, left eye, nose, mouths."""
    return [[rex, eye_y], [lex, eye_y], [nx, nose_y], [40.0, mouth_y], [60.0, mouth_y]]


def test_a_real_face_layout_is_plausible():
    assert landmarks_plausible(_face_landmarks(), BOX) is True


def test_eyes_below_the_nose_are_not_a_face():
    """The one fact true of every human face at every yaw: eyes, nose, mouth,
    in that vertical order. Random landmarks on clothing break it."""
    assert landmarks_plausible(_face_landmarks(eye_y=60.0, nose_y=40.0), BOX) is False


def test_the_nose_below_the_mouth_is_not_a_face():
    assert landmarks_plausible(_face_landmarks(nose_y=90.0, mouth_y=80.0), BOX) is False


def test_collapsed_and_impossible_eye_spans_are_not_faces():
    """Under 15% of box width means both eyes landed on one point; over 85%
    means they landed on opposite edges of something that is not a head."""
    assert landmarks_plausible(_face_landmarks(rex=49.0, lex=51.0), BOX) is False
    assert landmarks_plausible(_face_landmarks(rex=2.0, lex=98.0), BOX) is False


def test_a_nose_far_outside_the_eye_span_is_not_a_face():
    assert landmarks_plausible(_face_landmarks(nx=99.0), BOX) is False


def test_a_three_quarter_view_survives_the_nose_margin():
    """The margin is a fraction of BOX WIDTH, not of eye distance, precisely
    so a yawed face — eyes compressed to an 18% span, nose projecting well
    past the far one — stays plausible. Judging whether that pose is good
    enough to embed is frontality's job and eyeSpanRatio's, not this one's:
    this test only asks "is it a face", and a turned head is."""
    turned = _face_landmarks(rex=40.0, lex=58.0, nx=75.0)
    assert landmarks_plausible(turned, BOX) is True
    # ...and the pose signals DO see it, which is the division of labour.
    assert eye_span_ratio(turned, BOX) == 0.18


def test_unmeasurable_landmarks_are_unknown_not_implausible():
    """None, never False: the runner's gate reads None as UNKNOWN and keeps
    the face, because dropping a guest for a signal nobody measured is worse."""
    assert landmarks_plausible(None, BOX) is None
    assert landmarks_plausible([[1, 2], [3, 4]], BOX) is None
    assert landmarks_plausible(_face_landmarks(), {"w": 0}) is None
    assert landmarks_plausible([["x", "y"]] * 5, BOX) is None


def test_eye_span_ratio_is_pose_not_size():
    """The signal iedPx cannot give: the same face at two distances has two
    IEDs in pixels but ONE eye-span ratio, while turning to profile collapses
    the ratio at any distance."""
    near = eye_span_ratio(_face_landmarks(rex=35.0, lex=65.0), {"w": 100})
    far = eye_span_ratio(_face_landmarks(rex=17.5, lex=32.5), {"w": 50})
    assert near == far == 0.3
    profile = eye_span_ratio(_face_landmarks(rex=45.0, lex=55.0), {"w": 100})
    assert profile < near


def test_eye_span_ratio_is_none_when_unmeasurable():
    assert eye_span_ratio(None, BOX) is None
    assert eye_span_ratio(_face_landmarks(), {"w": 0}) is None
