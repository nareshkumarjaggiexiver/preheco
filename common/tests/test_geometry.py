"""Tests for the shared box geometry (IoU + dedupe + point-in-polygon)."""

import pytest
from heco_common.geometry import dedupe_boxes, iou_xywh, point_in_polygon


def test_iou_basic():
    """IoU is 1 for identical boxes, 0 when disjoint, and exact on a half overlap."""
    a = {"x": 0, "y": 0, "w": 10, "h": 10}
    assert iou_xywh(a, a) == 1.0
    assert iou_xywh(a, {"x": 20, "y": 20, "w": 5, "h": 5}) == 0.0
    # half-overlap along x: intersection 5x10=50, union 100+100-50=150
    half = iou_xywh(a, {"x": 5, "y": 0, "w": 10, "h": 10})
    assert abs(half - 50 / 150) < 1e-9


def test_iou_zero_area():
    """A degenerate box overlaps nothing — no division by a zero union."""
    assert iou_xywh({"x": 0, "y": 0, "w": 0, "h": 5}, {"x": 0, "y": 0, "w": 5, "h": 5}) == 0.0


def test_dedupe_keeps_first_of_duplicates():
    """Coincident boxes collapse to the first, which callers order deliberately."""
    # two near-identical boxes + one distinct -> 2 survive, first duplicate kept
    dup_a = {"x": 0, "y": 0, "w": 10, "h": 10, "tag": "a"}
    dup_b = {"x": 1, "y": 0, "w": 10, "h": 10, "tag": "b"}   # IoU ~0.82 with a
    far = {"x": 40, "y": 40, "w": 10, "h": 10, "tag": "c"}
    out = dedupe_boxes([dup_a, dup_b, far], iou_thr=0.6)
    assert [b["tag"] for b in out] == ["a", "c"]


def test_dedupe_keeps_adjacent_people():
    """Two people cheek to cheek must both survive — the threshold is not a merge."""
    # cheek-to-cheek: touching but low IoU -> both kept
    left = {"x": 0, "y": 0, "w": 10, "h": 20}
    right = {"x": 9, "y": 0, "w": 10, "h": 20}   # IoU ~0.05
    assert len(dedupe_boxes([left, right], iou_thr=0.6)) == 2


# ------------------------------------------------------------ point_in_polygon


def test_point_in_polygon_square():
    """Inside a unit square is inside; well outside is outside."""
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert point_in_polygon(0.5, 0.5, square) is True
    assert point_in_polygon(1.5, 0.5, square) is False
    assert point_in_polygon(-0.1, 0.5, square) is False
    assert point_in_polygon(0.5, 2.0, square) is False


def test_point_in_polygon_triangle_and_vertex_order():
    """Works for a triangle, and the answer ignores winding direction."""
    tri = [[0, 0], [1, 0], [0.5, 1]]
    assert point_in_polygon(0.5, 0.4, tri) is True
    assert point_in_polygon(0.05, 0.9, tri) is False  # outside the slanted edge
    assert point_in_polygon(0.5, 0.4, list(reversed(tri))) is True


def test_point_in_polygon_concave():
    """A concave notch is genuinely outside — the operator's freehand shape.

    An exclusion zone traced around a frosted partition will not be convex,
    so the test that decides whether a face is counted must not quietly
    convex-hull the polygon.
    """
    # A "C" shape: the notch [2..4] x [2..8] on the right side is outside.
    c_shape = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 8], [4, 8], [4, 10], [0, 10]]
    assert point_in_polygon(1.0, 5.0, c_shape) is True  # in the spine
    assert point_in_polygon(3.0, 5.0, c_shape) is False  # in the notch
    assert point_in_polygon(3.0, 1.0, c_shape) is True  # in the lower arm


def test_point_in_polygon_normalized_coordinates():
    """The exclusion-zone use: normalized 0..1 vertices, fractional points."""
    left_half = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    assert point_in_polygon(0.26, 0.51, left_half) is True
    assert point_in_polygon(0.74, 0.51, left_half) is False


def test_point_in_polygon_rejects_degenerate():
    """Fewer than 3 vertices encloses nothing; that is a caller bug, not False."""
    with pytest.raises(ValueError):
        point_in_polygon(0.5, 0.5, [[0, 0], [1, 1]])
