"""YOLOX letterbox preprocessing as a pure, unit-testable function.

Mirrors the reference `preproc` in Megvii's YOLOX ONNXRuntime demo: resize
keeping aspect ratio, pad bottom/right with grey (114), transpose to CHW
float32 — and, per the post-2021 YOLOX exports, NO mean/std normalisation
(the released ONNX graphs expect raw 0–255 pixel values).
"""

import cv2
import numpy as np

#: Padding intensity used by YOLOX letterboxing (all three channels).
PAD_VALUE = 114


def letterbox(img: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Letterbox `img` (HWC BGR uint8) into an `input_size` (h, w) canvas.

    Returns `(blob, ratio)` where `blob` is a contiguous float32 CHW array
    (no batch dimension) and `ratio` is the applied scale factor — divide
    network-space coordinates by it to get back to source-image pixels.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("letterbox expects an HWC 3-channel image")
    h, w = input_size
    padded = np.full((h, w, 3), PAD_VALUE, dtype=np.uint8)
    ratio = min(h / img.shape[0], w / img.shape[1])
    rw = int(img.shape[1] * ratio)
    rh = int(img.shape[0] * ratio)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded[:rh, :rw] = resized
    blob = padded.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(blob), ratio
