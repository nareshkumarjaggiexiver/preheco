"""Golden tests for the pure YOLOX letterbox preprocessing."""

import numpy as np
import pytest

from app.preprocess import PAD_VALUE, letterbox


def test_upscale_constant_image_pads_bottom_right():
    img = np.full((2, 4, 3), 10, dtype=np.uint8)
    blob, ratio = letterbox(img, (8, 8))
    assert ratio == 2.0
    assert blob.shape == (3, 8, 8)
    assert blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]
    # resized content occupies the top-left 4x8 region, all channels
    assert (blob[:, :4, :8] == 10).all()
    # everything below is letterbox padding
    assert (blob[:, 4:, :] == PAD_VALUE).all()


def test_downscale_keeps_aspect_ratio():
    img = np.full((10, 20, 3), 200, dtype=np.uint8)
    blob, ratio = letterbox(img, (8, 8))
    assert ratio == pytest.approx(0.4)
    # 10x20 * 0.4 -> 4x8 content region
    assert (blob[:, :4, :8] == 200).all()
    assert (blob[:, 4:, :] == PAD_VALUE).all()


def test_no_normalisation_raw_pixel_range():
    """Post-2021 YOLOX ONNX exports expect raw 0-255 floats, not mean/std."""
    img = np.full((4, 4, 3), 255, dtype=np.uint8)
    blob, _ = letterbox(img, (4, 4))
    assert blob.max() == 255.0


def test_rejects_non_3channel():
    with pytest.raises(ValueError):
        letterbox(np.zeros((4, 4), dtype=np.uint8), (8, 8))
