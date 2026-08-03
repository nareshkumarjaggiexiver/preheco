"""Tests for the pure face-JSON -> FaceRecognizerSF row conversion."""

import numpy as np
import pytest

from app.face_row import face_to_row


def _face():
    return {
        "box": {"x": 10.0, "y": 20.0, "w": 64.0, "h": 80.0},
        "landmarks": [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]],
        "conf": 0.93,
    }


def test_golden_layout():
    row = face_to_row(_face())
    assert row.shape == (15,)
    assert row.dtype == np.float32
    np.testing.assert_allclose(
        row,
        [10, 20, 64, 80, 30, 40, 70, 40, 50, 60, 35, 80, 65, 80, 0.93],
        rtol=1e-6,
    )


def test_conf_defaults_to_one():
    face = _face()
    del face["conf"]
    assert face_to_row(face)[14] == 1.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda f: f.pop("box"),
        lambda f: f["box"].pop("w"),
        lambda f: f.pop("landmarks"),
        lambda f: f.__setitem__("landmarks", f["landmarks"][:4]),  # only 4 points
        lambda f: f.__setitem__("landmarks", [[1, 2, 3]] * 5),  # not (x, y) pairs
    ],
)
def test_malformed_faces_raise(mutation):
    face = _face()
    mutation(face)
    with pytest.raises(ValueError):
        face_to_row(face)
