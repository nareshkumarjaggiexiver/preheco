"""Pure coordinate helpers for running YuNet inside person boxes.

When /detect receives `within` person boxes, each box is clamped to the frame,
cropped, detected on, and every face box + landmark is offset back into
frame coordinates. All functions here are numpy/scalar-pure and unit-tested.
"""


def clamp_box(box: dict, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Clamp a contract box {x, y, w, h} to image bounds as int pixels.

    Returns `(x, y, w, h)` or None when nothing of the box lies inside the
    image (callers skip such crops).
    """
    x0 = max(0, int(round(box["x"])))
    y0 = max(0, int(round(box["y"])))
    x1 = min(img_w, int(round(box["x"] + box["w"])))
    y1 = min(img_h, int(round(box["y"] + box["h"])))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def offset_face(face: dict, dx: float, dy: float) -> dict:
    """Translate a face dict (box + 5x2 landmarks) by (dx, dy), e.g. crop->frame."""
    box = face["box"]
    return {
        **face,
        "box": {"x": box["x"] + dx, "y": box["y"] + dy, "w": box["w"], "h": box["h"]},
        "landmarks": [[lx + dx, ly + dy] for lx, ly in face["landmarks"]],
    }
