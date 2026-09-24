"""Base64-JPEG frame transport helpers (POC frame format per CONTRACTS.md).

Duplicated per service on purpose: services are isolated (one venv each, one
container each) and 20 lines of codec is cheaper than a shared package.
"""

import base64
import binascii

import cv2
import numpy as np
from heco_common import frameref


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


def frame_from(image_b64: str | None, frame_ref: str | None = None) -> np.ndarray:
    """A frame from the shared transport if it is there, else from base64.

    The ref is an OPTIMISATION and never a dependency: no mount, a retired
    frame, a producer one version behind — each falls through to the JPEG the
    caller always sends. See heco_common.frameref for why that fallback can
    be relied on and why a ref can never be a torn read.
    """
    if frame_ref:
        img = frameref.read_frame(frame_ref)
        if img is not None:
            return img
    if not image_b64:
        raise ValueError("neither frameRef nor imageB64 yielded a frame")
    return b64_to_bgr(image_b64)
