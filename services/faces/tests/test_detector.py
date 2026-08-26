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
