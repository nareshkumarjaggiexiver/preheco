"""The alignment cv2 kept private, now ours — proven against known truth."""

import numpy as np

from app.align import ARCFACE_TEMPLATE, TEMPLATE_SIZE, align_face, similarity_transform


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
