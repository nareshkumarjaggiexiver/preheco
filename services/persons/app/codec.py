"""Base64-JPEG frame transport helpers (POC frame format per CONTRACTS.md).

Frames travel as base64-encoded JPEG inside JSON at POC scale; this module is
the single place that turns them into BGR numpy images.
"""

import base64
import binascii

import cv2
import numpy as np


def b64_to_bgr(image_b64: str) -> np.ndarray:
    """Decode a base64-encoded JPEG/PNG string into a BGR uint8 image.

    Raises ValueError when the payload is not valid base64 or does not decode
    to an image, so the HTTP layer can map it to a 400.
    """
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("imageB64 is not valid base64") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("imageB64 does not decode to an image")
    return img
