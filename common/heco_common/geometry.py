"""Box geometry shared across services.

Detection, tracking and face services all speak the same axis-aligned box in
image pixels, expressed as ``{"x", "y", "w", "h"}`` (top-left origin, width and
height). This module holds the operations several of them need — IoU, a greedy
IoU de-duplication, and a point-in-polygon test — in one tested place, so the
runner's region-of-interest union and the face service's cross-crop
de-duplication cannot drift apart in how they decide "these two boxes are the
same face", and the runner's exclusion zones share one polygon test with
whatever draws them.
"""

from __future__ import annotations

from collections.abc import Iterable


def iou_xywh(a: dict, b: dict) -> float:
    """Intersection-over-union of two ``{x, y, w, h}`` boxes.

    Returns 0.0 when either box has non-positive area or they do not overlap.
    """
    ax, ay, aw, ah = a["x"], a["y"], a["w"], a["h"]
    bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def dedupe_boxes(boxes: Iterable[dict], iou_thr: float = 0.6, box_of=None) -> list[dict]:
    """Collapse near-duplicate boxes, keeping the first of each overlapping set.

    Greedy: walk the input in order; keep an item unless its box overlaps an
    already-kept item's box by more than ``iou_thr``. Order therefore decides
    which survivor is kept — callers that care (e.g. keep the higher-confidence
    detection) should sort the input first. The threshold is deliberately high
    so only genuinely coincident boxes merge; two adjacent people (cheek to
    cheek) sit well below it and both survive.

    ``box_of`` extracts the ``{x, y, w, h}`` box from each item; the default
    treats each item AS the box (the runner passes flat person/track boxes),
    while the face service passes ``box_of=lambda f: f["box"]`` for its nested
    face records so the whole record — landmarks, quality and all — is kept.
    """
    get = box_of or (lambda item: item)
    kept: list[dict] = []
    for item in boxes:
        b = get(item)
        if all(iou_xywh(b, get(k)) <= iou_thr for k in kept):
            kept.append(item)
    return kept


def point_in_polygon(x: float, y: float, points: list) -> bool:
    """Is point ``(x, y)`` inside the polygon given by ``points``?

    ``points`` is an ordered list of ``[x, y]`` vertices (length >= 3, the
    closing edge back to the first vertex is implicit). Ray casting: cast a
    ray from the point toward +x and count edge crossings — odd means inside.
    Works for concave and self-intersecting polygons (even-odd rule), which is
    what an operator freehand-drawing an exclusion zone over a frosted
    partition will actually produce.

    Boundary behaviour: a point exactly ON an edge or vertex is decided by
    floating-point luck (the half-open ``>`` / ``<=`` comparisons make the
    result consistent but not meaningful).  That is fine for the one caller
    this exists for — a face box centre landing within a pixel of a zone edge
    is not a case any count depends on — and documenting it beats pretending
    exactness the arithmetic does not have.

    Fewer than 3 vertices cannot enclose anything and raises ValueError: the
    runner validates zone shapes at the API edge, so a short polygon reaching
    here is a programming error, not operator input.
    """
    n = len(points)
    if n < 3:
        raise ValueError(f"a polygon needs at least 3 vertices, got {n}")
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(points[i][0]), float(points[i][1])
        xj, yj = float(points[j][0]), float(points[j][1])
        if (yi > y) != (yj > y):
            # x where the edge crosses the horizontal line through the point.
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside
