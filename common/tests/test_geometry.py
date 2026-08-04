"""Tests for the shared box geometry (IoU + dedupe)."""

from heco_common.geometry import dedupe_boxes, iou_xywh


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
