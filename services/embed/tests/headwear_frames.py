"""Synthetic frames and face boxes for the head-covering reader's parity tests.

Integer arithmetic and cv2 integer drawing only, so every pixel is identical
on every platform: the reader's preprocessing is pinned against the verbatim
reference reader on these, and by a digest computed with the ORIGINAL
reference (numpy 2.5.3, opencv 5.0.0) when this file was written — an
OpenCV upgrade that moves ``cv2.INTER_AREA`` fails the pin, and a different
resize moved 16 of 461 calls in the evaluation.

The boxes cover what the geometry has to get right: an interior face, faces
whose regions clamp at each frame edge, non-integer boxes, a box whose edges
round at exactly .5 (Python's round is half-to-even), a tall and a wide one.
"""

import cv2
import numpy as np


def frame(height: int, width: int, k: int) -> np.ndarray:
    """A BGR uint8 frame: three integer patterns plus a few filled shapes."""
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.stack(
        [(xx * (7 + k) + yy * 3) % 256, (xx * 5 + yy * (11 + k) + 17 * k) % 256,
         ((xx // 3) ^ (yy // (5 + k))) % 256],
        axis=2,
    ).astype(np.uint8)
    cv2.circle(img, (width // 3, height // 4), max(4, height // 6), (40, 70 + 20 * k, 200), -1)
    cv2.rectangle(img, (width // 2, height // 2), (width // 2 + width // 5, height - 5),
                  (230, 30 * k, 60), -1)
    cv2.ellipse(img, (2 * width // 3, height // 3), (width // 9, height // 7), 30, 0, 360,
                (10, 10, 10), -1)
    return img


#: (frame height, frame width, pattern k, [face boxes]) — boxes in frame pixels.
CASES = [
    (360, 480, 0, [
        {"x": 200.0, "y": 150.0, "w": 80.0, "h": 104.0},      # interior
        {"x": 5.0, "y": 60.0, "w": 70.0, "h": 90.0},          # clamps left and top
    ]),
    (300, 400, 1, [
        {"x": 340.0, "y": 210.0, "w": 58.0, "h": 80.0},       # clamps right and bottom
        {"x": 100.25, "y": 120.75, "w": 61.3, "h": 79.9},     # non-integer
    ]),
    (320, 320, 2, [
        {"x": 100.0, "y": 150.0, "w": 125.0, "h": 150.0},     # x - 0.3w = 62.5: half-to-even
        {"x": 30.0, "y": 200.0, "w": 40.0, "h": 110.0},       # tall
        {"x": 150.0, "y": 60.0, "w": 150.0, "h": 50.0},       # wide
    ]),
]
