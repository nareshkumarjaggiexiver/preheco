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


def test_rtdetr_blob_is_rgb_unit_range_chw():
    """The RT-DETR contract differs from YOLOX in all three ways that silently
    break accuracy if crossed: RGB not BGR, 0-1 not 0-255, stretch not
    letterbox. Pin each."""
    import numpy as np
    from app.preprocess import rtdetr_blob

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # pure blue in BGR
    blob = rtdetr_blob(img, (640, 640))
    assert blob.shape == (3, 640, 640), "stretched to the square, no letterbox"
    assert blob.max() <= 1.0 and blob.min() >= 0.0, "unit range"
    assert blob[2].mean() > 0.99 and blob[0].mean() < 0.01, "BGR became RGB"
