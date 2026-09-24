"""The optional gender/age pass (app/attributes.py) against real (tiny) graphs.

tiny_onnx's ReduceMean fixture answers [mean R, mean G, mean B] of the exact
crop AttributeModel fed, so every preprocessing promise is a value-level
assertion: the InsightFace box-centred 1.5x warp, BGR->RGB, no mean/std,
NCHW vs NHWC. Painting the frame steers the "prediction": R vs G decides
gender, B sets age — which pins the postprocess (argmax, softmax of the
reported gender, age = out[2] * 100) end to end. The refusal tests pin the
init-time guards: an export this pass could never feed must raise (the
lazy loader then serves it as /health attrError) instead of building a
session that 500s every /embed behind a green healthcheck.

The real genderage.onnx, when present in models/, is exercised for shape
and range only — synthetic frames carry no gender to assert on; the face
cards of run f0bfc5 were the field check (see the README).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from tiny_onnx import DOUBLE, FLOAT, FLOAT16, write_model, write_reduce_mean_model

from app.attributes import (
    DEFAULT_ATTR_MODEL,
    INPUT_SIZE,
    AttributeModel,
    attr_model_from_env,
    attribute_transform,
    box_geometry,
    crop_face,
    postprocess,
)

REAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "genderage.onnx"


def _painted_frame(inner_bgr, outer_bgr=(0, 0, 0), box=(200.0, 100.0, 80.0, 100.0)):
    """A 480x640 frame: `outer_bgr` everywhere, `inner_bgr` over 1.5x the box —
    the whole region the attribute crop sees — so the crop is one flat colour
    and its per-channel means are exact."""
    img = np.empty((480, 640, 3), dtype=np.uint8)
    img[:] = outer_bgr
    x, y, w, h = box
    side = max(w, h) * 1.5
    cx, cy = x + w / 2, y + h / 2
    x0, x1 = int(cx - side / 2) - 2, int(cx + side / 2) + 3
    y0, y1 = int(cy - side / 2) - 2, int(cy + side / 2) + 3
    img[y0:y1, x0:x1] = inner_bgr
    return img, {"x": x, "y": y, "w": w, "h": h}


def _nchw(tmp_path, name="tiny_genderage.onnx", elem=FLOAT):
    return AttributeModel(write_reduce_mean_model(tmp_path, name, elem, (1, 3, 96, 96)), "CPU")


def test_transform_is_insightfaces_box_centred_scale_with_no_rotation():
    """face_align.transform(img, centre, 96, 96 / (1.5 * max(w, h)), 0) —
    the matrix InsightFace's Attribute feeds warpAffine, value for value."""
    box = {"x": 100.0, "y": 50.0, "w": 80.0, "h": 120.0}
    m = attribute_transform(box)
    scale = 96 / (120.0 * 1.5)
    cx, cy = 140.0, 110.0
    np.testing.assert_allclose(
        m, [[scale, 0.0, 48.0 - cx * scale], [0.0, scale, 48.0 - cy * scale]], rtol=1e-6
    )
    # The box centre lands on the crop centre; the longer side spans 2/3 of it.
    centre = m @ np.array([cx, cy, 1.0])
    np.testing.assert_allclose(centre, [48.0, 48.0], atol=1e-5)
    top = m @ np.array([cx, 50.0, 1.0])
    bottom = m @ np.array([cx, 170.0, 1.0])
    assert bottom[1] - top[1] == pytest.approx(64.0)


def test_crop_is_96_square_and_centred_on_the_box():
    img, box = _painted_frame((10, 20, 30))
    crop = crop_face(img, box)
    assert crop.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert crop.dtype == np.uint8
    assert tuple(crop[48, 48]) == (10, 20, 30)


def test_box_geometry_refuses_the_boxes_that_would_become_a_nan_scale():
    with pytest.raises(ValueError, match="x, y, w, h"):
        box_geometry({"x": 1.0, "y": 2.0, "w": 3.0})
    with pytest.raises(ValueError, match="x, y, w, h"):
        box_geometry(None)
    with pytest.raises(ValueError, match="numbers"):
        box_geometry({"x": "a", "y": 2.0, "w": 3.0, "h": 4.0})
    with pytest.raises(ValueError, match="positive"):
        box_geometry({"x": 1.0, "y": 2.0, "w": 0.0, "h": 0.0})
    with pytest.raises(ValueError, match="finite"):
        box_geometry({"x": 1.0, "y": 2.0, "w": float("nan"), "h": 4.0})
    # A thin-but-real box is fine: only max(w, h) must be positive.
    assert box_geometry({"x": 0, "y": 0, "w": 0, "h": 10}) == (0.0, 5.0, 10.0)


def test_postprocess_reads_insightfaces_output_order():
    """[F logit, M logit, age/100]: argmax picks the gender, genderP is the
    softmax mass OF THAT gender (never below 0.5), age is out[2] * 100."""
    male = postprocess([1.0, 3.0, 0.415])
    assert male["gender"] == "M"
    assert male["genderP"] == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert male["age"] == pytest.approx(41.5)
    female = postprocess(np.array([[2.5, -0.5, 0.63]], dtype=np.float32))
    assert female["gender"] == "F"
    assert female["genderP"] == pytest.approx(1 / (1 + np.exp(-3.0)), rel=1e-5)
    assert female["age"] == pytest.approx(63.0, rel=1e-5)
    tie = postprocess([0.0, 0.0, 0.0])
    assert tie["genderP"] == pytest.approx(0.5)
    assert all(isinstance(v, float) for k, v in male.items() if k != "gender")


def test_postprocess_softmax_survives_huge_logits():
    """A graph is free to answer in the hundreds; exp() must not overflow into NaN."""
    out = postprocess([1000.0, -1000.0, 0.2])
    assert out["gender"] == "F"
    assert out["genderP"] == pytest.approx(1.0)
    assert out["age"] == pytest.approx(20.0)


def test_postprocess_refuses_the_wrong_width():
    with pytest.raises(ValueError, match="expected 3"):
        postprocess([0.1, 0.2])


def test_nchw_graph_receives_rgb_0_255_unnormalised(tmp_path):
    """The blob is RGB in 0..255 with NO mean/std (the graph normalises
    itself): a BGR-painted crop must come back as its RGB channel means."""
    model = _nchw(tmp_path)
    assert model.providers_active == ["CPUExecutionProvider"]
    assert model.device_requested == "CPU"
    img, box = _painted_frame((30, 20, 10))  # BGR -> RGB means [10, 20, 30]
    blob = model.blob(img, box)
    assert blob.shape == (1, 3, 96, 96) and blob.dtype == np.float32
    np.testing.assert_allclose(blob[0].mean(axis=(1, 2)), [10.0, 20.0, 30.0], atol=1e-6)
    pred = model._session.run(None, {model._input_name: blob})[0]
    np.testing.assert_allclose(pred, [[10.0, 20.0, 30.0]], atol=1e-4)


def test_nhwc_graph_takes_the_untransposed_branch(tmp_path):
    path = write_reduce_mean_model(tmp_path, "tiny_nhwc.onnx", FLOAT, (1, 96, 96, 3), (1, 2))
    model = AttributeModel(path, "CPU")
    img, box = _painted_frame((30, 20, 10))
    blob = model.blob(img, box)
    assert blob.shape == (1, 96, 96, 3)
    np.testing.assert_allclose(model.predict(img, box)["age"], 30.0 * 100.0, atol=1e-3)


def test_fp16_graph_receives_a_fp16_blob(tmp_path):
    model = _nchw(tmp_path, "tiny_fp16.onnx", FLOAT16)
    img, box = _painted_frame((30, 20, 10))
    assert model.blob(img, box).dtype == np.float16
    out = model.predict(img, box)
    assert out["age"] == pytest.approx(3000.0, rel=1e-3)


def test_predict_steers_gender_by_channel_and_age_by_blue(tmp_path):
    """End to end through a real session: G > R (in RGB) reads male, R > G
    female, and B is the age — pinning warp, swap, argmax and softmax at once."""
    model = _nchw(tmp_path)
    img, box = _painted_frame((45, 20, 10))  # RGB means [10, 20, 45]
    male = model.predict(img, box)
    assert male["gender"] == "M"
    assert male["genderP"] == pytest.approx(1 / (1 + np.exp(-10.0)), rel=1e-4)
    assert male["age"] == pytest.approx(4500.0, rel=1e-4)
    img, box = _painted_frame((45, 10, 20))  # RGB means [20, 10, 45]
    female = model.predict(img, box)
    assert female["gender"] == "F"
    assert female["genderP"] == pytest.approx(1 / (1 + np.exp(-10.0)), rel=1e-4)


def test_predict_uses_the_box_not_the_whole_frame(tmp_path):
    """Outside the 1.5x box region the frame is a different colour; the
    crop must not see it (the warp is box-centred, box-scaled)."""
    model = _nchw(tmp_path)
    img, box = _painted_frame((30, 20, 10), outer_bgr=(255, 255, 255))
    np.testing.assert_allclose(model.predict(img, box)["age"], 3000.0, rtol=1e-3)


def test_predict_refuses_a_malformed_box_like_the_landmark_guard(tmp_path):
    model = _nchw(tmp_path)
    img, _ = _painted_frame((30, 20, 10))
    with pytest.raises(ValueError, match="x, y, w, h"):
        model.predict(img, {"x": 1, "y": 2})
    with pytest.raises(ValueError, match="positive"):
        model.predict(img, {"x": 1, "y": 2, "w": 0, "h": -3})


def test_missing_file_is_a_file_not_found_naming_the_env(tmp_path):
    with pytest.raises(FileNotFoundError, match="EMBED_ATTR_MODEL"):
        AttributeModel(tmp_path / "absent.onnx", "CPU")


def test_an_unsupported_input_dtype_is_refused_at_init(tmp_path):
    path = write_reduce_mean_model(tmp_path, "tiny_double.onnx", DOUBLE, (1, 3, 96, 96))
    with pytest.raises(ValueError, match=r"tensor\(double\)"):
        AttributeModel(path, "CPU")


def test_a_grayscale_graph_is_refused_at_init(tmp_path):
    path = write_reduce_mean_model(tmp_path, "tiny_gray.onnx", FLOAT, (1, 1, 96, 96))
    with pytest.raises(ValueError, match="3-channel RGB"):
        AttributeModel(path, "CPU")


def test_wrong_static_spatial_dims_are_refused_at_init(tmp_path):
    """A 112x112 export (the embedder's size, an easy mix-up) would build a
    session and then 500 every /embed — ORT checks dims at run() time."""
    path = write_reduce_mean_model(tmp_path, "tiny_112.onnx", FLOAT, (1, 3, 112, 112))
    with pytest.raises(ValueError, match="112x112"):
        AttributeModel(path, "CPU")


def test_a_graph_that_is_not_genderage_shaped_is_refused_at_init(tmp_path):
    """The arcface Flatten fixture has the right input and 27648 outputs:
    an embedder dropped into EMBED_ATTR_MODEL must be refused by name."""
    path = write_model(tmp_path, "tiny_not_genderage.onnx", FLOAT, (1, 3, 96, 96))
    with pytest.raises(ValueError, match="27648 values per face"):
        AttributeModel(path, "CPU")


def test_attr_model_from_env_resolves_off_default_and_named():
    """Unset/empty -> the default file, NOT explicit (absent = off, quietly);
    a path -> that file, explicit (absent = an error worth serving);
    `off` -> no model at all, even when the default file exists."""
    assert attr_model_from_env(None) == (DEFAULT_ATTR_MODEL, False)
    assert attr_model_from_env("") == (DEFAULT_ATTR_MODEL, False)  # compose's ${VAR-}
    assert attr_model_from_env("  ") == (DEFAULT_ATTR_MODEL, False)
    assert attr_model_from_env("/srv/weights/ga.onnx") == (Path("/srv/weights/ga.onnx"), True)
    assert attr_model_from_env("off") == (None, True)
    assert attr_model_from_env("OFF") == (None, True)


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="models/genderage.onnx absent")
def test_the_real_genderage_graph_loads_and_answers_in_range():
    """Shape and range only — a synthetic face has no gender to assert."""
    model = AttributeModel(REAL_MODEL, "CPU")
    assert model.model_name == "genderage.onnx"
    rng = np.random.default_rng(11)
    img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    out = model.predict(img, {"x": 280.0, "y": 190.0, "w": 80.0, "h": 100.0})
    assert out["gender"] in ("M", "F")
    assert 0.5 <= out["genderP"] <= 1.0
    assert np.isfinite(out["age"])


def test_crop_matches_cv2_warp_with_the_documented_matrix():
    """crop_face IS cv2.warpAffine(img, attribute_transform(box), (96, 96))
    with a black border — a real image, checked pixel for pixel."""
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    box = {"x": 250.0, "y": 150.0, "w": 60.0, "h": 90.0}  # hangs off the edge
    expected = cv2.warpAffine(img, attribute_transform(box), (96, 96), borderValue=0.0)
    np.testing.assert_array_equal(crop_face(img, box), expected)
