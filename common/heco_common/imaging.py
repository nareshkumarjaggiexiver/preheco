"""Base64-JPEG helpers and aspect-preserving resize.

Frames travel between services as base64 JPEG inside JSON (POC-scale
decision, CONTRACTS.md): simple, debuggable, language-neutral. Shared memory
arrives only if profiling shows encode/decode is the bottleneck.

Images are numpy BGR uint8 arrays — OpenCV's native layout — everywhere.
"""

import base64

import cv2
import numpy as np


def encode_jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR uint8 image to a base64 JPEG string.

    ``quality`` is the JPEG quality (1-100); 85 keeps ~64-85 px faces
    legible without bloating payloads.
    """
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed (empty or invalid image?)")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_jpeg_b64(data: str) -> np.ndarray:
    """Decode a base64 JPEG string back to a BGR uint8 image.

    Raises ValueError on malformed base64 or undecodable JPEG bytes.
    """
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid base64: {exc}") from exc
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bytes did not decode as an image")
    return img


def resize_keep_aspect(
    img: np.ndarray,
    max_w: int | None = None,
    max_h: int | None = None,
    upscale: bool = False,
) -> np.ndarray:
    """Resize so the image fits within (max_w, max_h), preserving aspect.

    Either bound may be None (unbounded on that axis). By default the image
    is never upscaled — pass ``upscale=True`` to allow growing small images
    up to the bounds. Returns the input unchanged when no resize is needed.
    """
    if max_w is None and max_h is None:
        return img
    h, w = img.shape[:2]
    scale = min(
        (max_w / w) if max_w else float("inf"),
        (max_h / h) if max_h else float("inf"),
    )
    if not upscale:
        scale = min(scale, 1.0)
    if scale == 1.0:
        return img
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)
