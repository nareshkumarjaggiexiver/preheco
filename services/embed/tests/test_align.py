"""The alignment cv2 kept private, now ours — proven against known truth."""

import numpy as np
import pytest

from app.align import (
    ARCFACE_TEMPLATE,
    MIN_LANDMARK_VAR,
    TEMPLATE_SIZE,
    align_face,
    similarity_transform,
)


def test_the_template_maps_onto_itself_with_identity():
    m = similarity_transform(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE)
    assert np.allclose(m[:, :2], np.eye(2), atol=1e-5)
    assert np.allclose(m[:, 2], 0, atol=1e-4)


def test_scale_rotation_translation_are_recovered_exactly():
    """Points made from the template by a known similarity must map back —
    the closed form's whole promise, and what makes alignment deterministic
    enough to test at all."""
    theta = np.deg2rad(20)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = (ARCFACE_TEMPLATE @ rot.T) * 1.7 + [40, -12]
    m = similarity_transform(src, ARCFACE_TEMPLATE)
    mapped = src @ m[:, :2].T + m[:, 2]
    assert np.allclose(mapped, ARCFACE_TEMPLATE, atol=1e-3)


def test_a_reflection_is_never_produced():
    """A mirrored face is a different face: the transform must fix the SVD's
    reflection case rather than happily flipping the crop."""
    src = ARCFACE_TEMPLATE.copy()
    src[:, 0] = -src[:, 0]  # mirrored landmarks
    m = similarity_transform(src, ARCFACE_TEMPLATE)
    assert np.linalg.det(m[:, :2]) > 0, "rotation, not reflection"


def test_align_face_returns_the_template_frame():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = align_face(img, ARCFACE_TEMPLATE * 3 + 50)
    assert out.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE, 3)


def test_all_zero_landmarks_are_refused():
    """The all-zeros 'no landmarks' sentinel used to fall back to scale=1.0
    and warp the frame's top-left corner into a confident, matchable 'face'
    embedding — degenerate input is a ValueError, not a fallback."""
    with pytest.raises(ValueError, match="degenerate landmarks"):
        similarity_transform(np.zeros((5, 2)), ARCFACE_TEMPLATE)


def test_coincident_landmarks_are_refused_anywhere_in_the_frame():
    with pytest.raises(ValueError, match="degenerate landmarks"):
        similarity_transform(np.full((5, 2), 87.3), ARCFACE_TEMPLATE)


def test_near_coincident_landmarks_below_the_variance_floor_are_refused():
    """The floor is a band, not an exact-zero check: sub-pixel jitter around
    one point (variance well under 1 px^2) is the huge-scale sibling of the
    coincident case and must be refused the same way."""
    jitter = np.array([[0.1, -0.1], [-0.1, 0.1], [0.0, 0.1], [0.1, 0.0], [-0.1, -0.1]])
    src = np.full((5, 2), 40.0) + jitter
    assert ((src - src.mean(axis=0)) ** 2).sum() / 5 < MIN_LANDMARK_VAR
    with pytest.raises(ValueError, match="degenerate landmarks"):
        similarity_transform(src, ARCFACE_TEMPLATE)


def test_align_face_refuses_the_all_zero_sentinel():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="cannot align"):
        align_face(img, [[0.0, 0.0]] * 5)


def test_the_smallest_gate_passing_face_clears_the_variance_floor():
    """A 56 px POC-floor face is roughly half template scale — its landmark
    variance sits two orders of magnitude above MIN_LANDMARK_VAR, so the
    degeneracy guard can never bite a real detection."""
    src = ARCFACE_TEMPLATE * 0.5 + [200, 100]
    m = similarity_transform(src, ARCFACE_TEMPLATE)
    mapped = src @ m[:, :2].T + m[:, 2]
    assert np.allclose(mapped, ARCFACE_TEMPLATE, atol=1e-3)
