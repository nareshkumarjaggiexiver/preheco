"""Unit tests for base64-JPEG helpers and aspect-preserving resize."""

import numpy as np
import pytest
from heco_common import imaging


def _gradient(w: int = 64, h: int = 48) -> np.ndarray:
    """A smooth BGR gradient — JPEG-friendly so roundtrips stay close."""
    x = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = x[None, :, None]
    return img


def test_encode_decode_roundtrip():
    """Encode→decode preserves shape/dtype and approximate content."""
    img = _gradient()
    out = imaging.decode_jpeg_b64(imaging.encode_jpeg_b64(img, quality=95))
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert float(np.abs(out.astype(int) - img.astype(int)).mean()) < 5.0


def test_decode_rejects_garbage():
    """Bad base64 and non-image bytes both raise ValueError."""
    with pytest.raises(ValueError):
        imaging.decode_jpeg_b64("not-base64!!!")
    with pytest.raises(ValueError):
        imaging.decode_jpeg_b64("aGVsbG8gd29ybGQ=")  # valid b64, not a JPEG


def test_resize_fits_bounds_keeps_aspect():
    """Downscale fits within both bounds and preserves the aspect ratio."""
    img = _gradient(w=200, h=100)
    out = imaging.resize_keep_aspect(img, max_w=50, max_h=50)
    assert out.shape == (25, 50, 3)  # 200x100 → 50x25, aspect 2:1 held


def test_resize_never_upscales_by_default():
    """Small images pass through untouched unless upscale=True."""
    img = _gradient(w=40, h=30)
    assert imaging.resize_keep_aspect(img, max_w=400, max_h=300) is img
    up = imaging.resize_keep_aspect(img, max_w=400, max_h=300, upscale=True)
    assert up.shape == (300, 400, 3)


def test_resize_single_bound():
    """One-sided bounds work: only max_w given."""
    img = _gradient(w=100, h=80)
    out = imaging.resize_keep_aspect(img, max_w=50)
    assert out.shape == (40, 50, 3)
