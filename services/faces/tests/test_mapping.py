"""Tests for the pure crop-mapping helpers."""

from app.mapping import clamp_box, offset_face


def test_clamp_inside_untouched():
    assert clamp_box({"x": 10, "y": 20, "w": 30, "h": 40}, 100, 100) == (10, 20, 30, 40)


def test_clamp_negative_origin():
    assert clamp_box({"x": -10, "y": -5, "w": 30, "h": 30}, 100, 100) == (0, 0, 20, 25)


def test_clamp_overflow_right_bottom():
    assert clamp_box({"x": 90, "y": 95, "w": 30, "h": 30}, 100, 100) == (90, 95, 10, 5)


def test_clamp_fully_outside_is_none():
    assert clamp_box({"x": 200, "y": 10, "w": 30, "h": 30}, 100, 100) is None
    assert clamp_box({"x": -50, "y": 10, "w": 30, "h": 30}, 100, 100) is None


def test_clamp_rounds_fractional_pixels():
    assert clamp_box({"x": 1.6, "y": 2.4, "w": 10.0, "h": 10.0}, 100, 100) == (2, 2, 10, 10)


def test_offset_face_shifts_box_and_all_landmarks():
    face = {
        "box": {"x": 5.0, "y": 6.0, "w": 20.0, "h": 22.0},
        "landmarks": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        "conf": 0.9,
    }
    out = offset_face(face, 100, 200)
    assert out["box"] == {"x": 105.0, "y": 206.0, "w": 20.0, "h": 22.0}
    assert out["landmarks"] == [[101, 202], [103, 204], [105, 206], [107, 208], [109, 210]]
    assert out["conf"] == 0.9
    # original untouched (pure function)
    assert face["box"]["x"] == 5.0 and face["landmarks"][0] == [1.0, 2.0]
