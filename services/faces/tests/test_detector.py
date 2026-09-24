"""Family-layer tests for detector.py — a fake ORT session, no weights.

The scrfd family's build-time proof (static-shape reconcile + one-frame
dry run), its per-family operating point, and the whole-frame downscale
guard are all testable without onnxruntime installed: ScrfdDetector defers
the import, so a fake module in sys.modules stands in for the runtime —
the same no-weights philosophy as test_scrfd.py.
"""

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app import detector as d


class _FakeInput:
    def __init__(self, shape):
        self.shape = shape
        self.name = "input.1"


class _FakeSession:
    """Mimics ort.InferenceSession for exactly the calls __init__ makes."""

    def __init__(self, shape=None, outputs=9, run=None):
        self._shape = shape or [1, 3, "?", "?"]  # dynamic H/W by default
        self._outputs = outputs
        self._run = run

    def get_inputs(self):
        return [_FakeInput(self._shape)]

    def get_outputs(self):
        return list(range(self._outputs))

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _names, feeds):
        if self._run is not None:
            return self._run(feeds)
        return _zero_tensors()


def _zero_tensors(input_hw=(640, 640)):
    """Nine zero tensors of the correct *_kps shapes (see test_scrfd.py)."""
    outs = []
    for group in range(3):  # scores, boxes, kps
        for stride in (8, 16, 32):
            n = (input_hw[0] // stride) * (input_hw[1] // stride) * 2
            width = {0: 1, 1: 4, 2: 10}[group]
            outs.append(np.zeros((n, width), dtype=np.float32))
    return outs


@pytest.fixture
def fake_ort(monkeypatch, tmp_path):
    """A fake onnxruntime in sys.modules plus a dummy installed weight.

    Tests mutate cfg["session"] BEFORE constructing a detector to steer
    what the next InferenceSession pretends to be.
    """
    cfg = {"session": _FakeSession()}
    mod = types.ModuleType("onnxruntime")
    mod.InferenceSession = lambda *a, **k: cfg["session"]
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    weight = tmp_path / "scrfd_test_kps.onnx"
    weight.write_bytes(b"not-a-real-graph")
    cfg["weight"] = weight
    return cfg


def test_a_fixed_shape_export_at_the_wrong_size_is_refused_at_build(fake_ort):
    """ORT checks input dims at run() time, not session creation — a fixed
    320 export configured 640 must be a build-time ValueError naming both,
    never an ok:true swap that bricks every subsequent /detect."""
    fake_ort["session"] = _FakeSession(shape=[1, 3, 320, 320])
    with pytest.raises(ValueError) as err:
        d.ScrfdDetector(fake_ort["weight"], 640, "CPU")
    assert "320x320" in str(err.value) and "640" in str(err.value)


def test_a_loadable_but_unrunnable_graph_is_refused_by_the_dry_run(fake_ort):
    """A dynamic-shaped graph that still refuses our blob at run() time must
    raise out of __init__ (apply_model then answers 400 and the old model
    keeps serving; at boot /health goes honestly unhealthy)."""

    def _raise(_feeds):
        raise RuntimeError("Got invalid dimensions for input 'input.1'")

    fake_ort["session"] = _FakeSession(run=_raise)
    with pytest.raises(RuntimeError, match="invalid dimensions"):
        d.ScrfdDetector(fake_ort["weight"], 640, "CPU")


def test_nine_outputs_with_a_wrong_tensor_contract_fail_the_dry_run(fake_ort):
    """Nine tensors that do not reshape into the *_kps decode contract are
    a refusal too — the dry run pushes the outputs through select_faces."""
    fake_ort["session"] = _FakeSession(run=lambda feeds: [np.zeros((7,), np.float32)] * 9)
    with pytest.raises(ValueError):
        d.ScrfdDetector(fake_ort["weight"], 640, "CPU")


def test_a_runnable_graph_builds_and_carries_the_operating_point(fake_ort):
    det = d.ScrfdDetector(fake_ort["weight"], 640, "CPU")
    assert det.family == "scrfd"
    assert det.score_min == 0.5  # InsightFace's det_thresh, NOT yunet's 0.8


def test_scrfd_specs_carry_the_family_operating_point():
    """Both listed rows and the inferred branch declare score_min 0.5;
    yunet's spec stays bare (FACES_SCORE_MIN is its knob)."""
    assert d.MODEL_SPECS["scrfd_2.5g_kps.onnx"]["score_min"] == 0.5
    assert d.MODEL_SPECS["scrfd_10g_kps.onnx"]["score_min"] == 0.5
    assert d.spec_for(Path("scrfd_500m_kps.onnx"))["score_min"] == 0.5
    assert "score_min" not in d.spec_for(Path("face_detection_yunet_2023mar.onnx"))


def test_env_override_beats_the_spec_row(fake_ort, monkeypatch):
    """FACES_SCRFD_SCORE_MIN must reach LISTED models too, or the operator
    knob is dead exactly where it matters."""
    monkeypatch.setattr(d, "SCRFD_SCORE_MIN", 0.35)
    monkeypatch.setattr(d, "_SCRFD_SCORE_MIN_OVERRIDDEN", True)
    det = d.build_detector(fake_ort["weight"])
    assert det.score_min == 0.35


def test_build_detector_uses_the_spec_operating_point_by_default(fake_ort):
    det = d.build_detector(fake_ort["weight"])
    assert det.score_min == 0.5


def test_whole_frame_downscale_guard_warns_once_per_frame_size(fake_ort, caplog):
    """8MP whole-frame letterboxing makes the 56 px POC floor ~9 net px —
    below stride-8 resolution. The guard warns once per distinct frame
    size and the resolvable floor is exposed as pure arithmetic."""
    det = d.ScrfdDetector(fake_ort["weight"], 640, "CPU")
    assert det.min_resolvable_face_px(3840, 2160) == 96  # ratio 1/6
    frame = np.zeros((2160, 3840, 3), np.uint8)
    with caplog.at_level(logging.WARNING, logger="faces"):
        assert det.detect(frame) == []
        det.detect(frame)
    warned = [r for r in caplog.records if "unresolvable" in r.getMessage()]
    assert len(warned) == 1, "once per distinct frame size, not per frame"


def test_person_crops_do_not_trip_the_downscale_guard(fake_ort, caplog):
    """The production `within` path upscales person crops into the input —
    the guard must stay silent there or it would warn every frame."""
    det = d.ScrfdDetector(fake_ort["weight"], 640, "CPU")
    with caplog.at_level(logging.WARNING, logger="faces"):
        det.detect(np.zeros((300, 200, 3), np.uint8))
    assert not [r for r in caplog.records if "unresolvable" in r.getMessage()]


def test_scrfd_blob_is_byte_identical_to_the_arithmetic_it_replaced():
    """The LUT blob must be the SAME numbers as (rgb - 127.5) / 128 in CHW —
    every uint8 value included — or a speed-up becomes a model change."""
    import cv2

    rng = np.random.default_rng(3)
    canvas = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    canvas[0, :256 // 3 + 1, :] = np.arange(0, 256, 3, dtype=np.uint8)[:, None]
    canvas[1, :86, 0] = np.arange(170, 256, dtype=np.uint8)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    legacy = np.ascontiguousarray(
        ((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None])
    blob = d.scrfd_blob(canvas)
    assert blob.shape == (1, 3, 64, 96) and blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]
    assert blob.tobytes() == legacy.tobytes()
    assert set(np.unique(d._SCRFD_NORM_LUT[np.arange(256)] * 128.0 + 127.5)) == set(range(256))


def test_detect_feeds_the_session_the_legacy_blob(fake_ort):
    """End to end through detect(): the tensor the session receives is the
    one the pre-LUT code built, byte for byte, letterbox included."""
    import cv2

    seen = {}

    def _capture(feeds):
        seen["blob"] = next(iter(feeds.values()))
        return _zero_tensors((96, 160))

    fake_ort["session"] = _FakeSession(run=_capture)
    det = d.ScrfdDetector(fake_ort["weight"], (96, 160), "CPU")
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    det.detect(img)
    ratio = min(96 / 90, 160 / 120)
    rw, rh = int(120 * ratio), int(90 * ratio)
    canvas = np.zeros((96, 160, 3), dtype=np.uint8)
    canvas[:rh, :rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    legacy = np.ascontiguousarray(
        ((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None])
    assert seen["blob"].tobytes() == legacy.tobytes()


def test_trt_truth_is_absent_off_trt(fake_ort):
    """Off TensorRT the detector carries trt=None and /health's device block
    (main._device_block) keeps today's four keys exactly."""
    from app.main import _device_block

    det = d.ScrfdDetector(fake_ort["weight"], 640, "CPU")
    assert det.trt is None
    assert set(_device_block(det)) == {"requested", "active", "family", "scoreMin"}
    det.device_requested = "TRT"
    assert _device_block(det)["trt"] is None, "asked for TRT, did not get it: null, shown"


def test_trt_loads_the_dims_renamed_graph_and_other_devices_the_file(fake_ort, monkeypatch):
    """Under TRT the session gets the in-memory graph with H and W renamed
    apart (TensorRT otherwise fails 1472x832 and ORT drops to CUDA); every
    other device keeps loading the file by path, exactly as before."""
    seen = []
    mod = sys.modules["onnxruntime"]
    monkeypatch.setattr(mod, "InferenceSession",
                        lambda model, **k: seen.append(model) or fake_ort["session"])
    monkeypatch.setattr(d, "distinct_input_dims", lambda raw: b"renamed-graph")
    d.ScrfdDetector(fake_ort["weight"], 640, "CPU")
    d.ScrfdDetector(fake_ort["weight"], 640, "CUDA")
    d.ScrfdDetector(fake_ort["weight"], 640, "TRT")
    assert seen == [str(fake_ort["weight"]), str(fake_ort["weight"]), b"renamed-graph"]
    monkeypatch.setattr(d, "distinct_input_dims", lambda raw: None)
    d.ScrfdDetector(fake_ort["weight"], 640, "TRT")
    assert seen[-1] == str(fake_ort["weight"]), "nothing to rename: the file, as ever"


class _TrtSession(_FakeSession):
    """TensorRT in the session until the first run(); with ``falls_back``,
    ORT's silent rebuild on [CUDA, CPU] after a failed engine build — what
    the box did with the original SCRFD bytes at 1472x832 (\"Falling back to
    [CUDAExecutionProvider, CPUExecutionProvider] and retrying\")."""

    _TRT = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, falls_back, options):
        super().__init__()
        self.falls_back, self.options, self.ran = falls_back, options, False

    def get_providers(self):
        if self.ran and self.falls_back:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return list(self._TRT)

    def get_provider_options(self):
        return {"TensorrtExecutionProvider": self.options}

    def run(self, _names, feeds):
        self.ran = True
        return _zero_tensors()


@pytest.mark.parametrize("falls_back", [True, False])
def test_trt_device_truth_is_read_after_the_dry_run(fake_ort, monkeypatch, tmp_path, falls_back):
    """The providers ORT reports at construction are NOT the truth under TRT:
    a failed engine build inside the first run() drops the session to CUDA
    without raising. /health must then say CUDA and trt=null — and, when the
    engine did build, carry TensorRT's own read-back of its options."""
    monkeypatch.setenv("HECO_TRT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(d, "distinct_input_dims", lambda raw: None)
    mod = sys.modules["onnxruntime"]
    monkeypatch.setattr(mod, "InferenceSession", lambda model, providers, provider_options: (
        _TrtSession(falls_back, provider_options[0])))
    det = d.ScrfdDetector(fake_ort["weight"], 640, "TRT")
    if falls_back:
        assert det.providers_active == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert det.trt is None
    else:
        assert det.providers_active[0] == "TensorrtExecutionProvider"
        assert det.trt["fp16"] is True
        assert det.trt["engineCache"].startswith(str(tmp_path / "cache"))


def test_trt_engine_dir_is_keyed_on_the_graph_handed_to_ort(fake_ort, monkeypatch, tmp_path):
    """Loaded from BYTES, the EP's own engine name carries no file name: any
    same-architecture weight ran the cached engine (a bias shifted by +2.0
    answered the original's scores). The engine directory is the hash of
    exactly what ORT gets, so two weights under one name build apart."""
    from heco_common.ort import model_key

    monkeypatch.setenv("HECO_TRT_CACHE", str(tmp_path / "cache"))
    seen = []
    mod = sys.modules["onnxruntime"]
    monkeypatch.setattr(mod, "InferenceSession", lambda model, providers, provider_options: (
        seen.append((model, provider_options[0])) or fake_ort["session"]))
    monkeypatch.setattr(d, "distinct_input_dims", lambda raw: b"renamed:" + raw)
    other = tmp_path / "other" / fake_ort["weight"].name
    other.parent.mkdir()
    other.write_bytes(b"same-architecture-other-weights")
    d.ScrfdDetector(fake_ort["weight"], 640, "TRT")
    d.ScrfdDetector(other, 640, "TRT")
    (m1, o1), (m2, o2) = seen
    assert o1["trt_engine_cache_path"] == str(tmp_path / "cache" / model_key(m1))
    assert o2["trt_engine_cache_path"] == str(tmp_path / "cache" / model_key(m2))
    assert o1["trt_engine_cache_path"] != o2["trt_engine_cache_path"]
    # Nothing to rename: the path is loaded and its bytes are what key it.
    monkeypatch.setattr(d, "distinct_input_dims", lambda raw: None)
    d.ScrfdDetector(other, 640, "TRT")
    assert seen[-1][1]["trt_engine_cache_path"] == str(tmp_path / "cache" / model_key(other))
