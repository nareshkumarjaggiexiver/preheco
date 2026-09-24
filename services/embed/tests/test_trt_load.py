"""The TensorRT load path of both embed graphs, against a fake onnxruntime.

No GPU and no TensorRT here: what these pin is what the SERVICE does with
what ORT reports. Two facts from the .94 box drive them:

* a TensorRT engine that fails to build does so inside the first run() and
  ORT silently rebuilds the session on [CUDA, CPU] — so the device truth
  must be read AFTER the warm-up run, or /health claims a TensorRT that is
  not running (it did, before fc98f87);
* the EP names a cached engine without the weights in the name, so a weight
  re-fetched in place ran the OLD engine — engines now live under the
  weights' hash (heco_common.ort.model_key).
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from heco_common.ort import model_key

from app.attributes import AttributeModel
from app.main import _device_block
from app.recognizer import ArcFaceEmbedder

_TRT = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


class _Session:
    """TensorRT until the first run(); then [CUDA, CPU] when ``falls_back``."""

    def __init__(self, falls_back, options, size, out_dim):
        self.falls_back, self.options = falls_back, options
        self.size, self.out_dim, self.ran = size, out_dim, False

    def get_providers(self):
        if self.ran and self.falls_back:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return list(_TRT)

    def get_provider_options(self):
        return {"TensorrtExecutionProvider": self.options}

    def get_inputs(self):
        return [SimpleNamespace(name="data", type="tensor(float)",
                                shape=["N", 3, self.size, self.size])]

    def get_outputs(self):
        return [SimpleNamespace(shape=["N", self.out_dim])]

    def run(self, _names, feeds):
        self.ran = True
        n = next(iter(feeds.values())).shape[0]
        return [np.zeros((n, self.out_dim), dtype=np.float32)]


@pytest.fixture
def fake_ort(monkeypatch, tmp_path):
    """A fake onnxruntime; ``cfg`` steers it, ``cfg['seen']`` records each load."""
    cfg = {"falls_back": False, "size": 112, "out_dim": 512, "seen": []}
    mod = types.ModuleType("onnxruntime")
    mod.SessionOptions = lambda: SimpleNamespace()

    def session(model, *_opts, providers=None, provider_options=None):
        cfg["seen"].append((model, provider_options[0]))
        return _Session(cfg["falls_back"], provider_options[0], cfg["size"], cfg["out_dim"])

    mod.InferenceSession = session
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    monkeypatch.setenv("HECO_TRT_CACHE", str(tmp_path / "cache"))
    return cfg


def _weight(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize("falls_back", [True, False])
def test_embedder_reads_its_device_truth_after_the_warm_up(fake_ort, tmp_path, falls_back):
    """A failed engine build shows as CUDA with trt null, never as TensorRT."""
    fake_ort["falls_back"] = falls_back
    emb = ArcFaceEmbedder(_weight(tmp_path, "w600k_r50.onnx", b"arcface"), "TRT")
    if falls_back:
        assert emb.providers_active == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert emb.trt is None
    else:
        assert emb.providers_active == _TRT
        assert emb.trt["engineCache"] == str(tmp_path / "cache" / model_key(b"arcface"))


@pytest.mark.parametrize("falls_back", [True, False])
def test_attribute_pass_reads_its_device_truth_after_the_warm_up(fake_ort, tmp_path, falls_back):
    """Same rule for the genderage engine, and /health now shows it."""
    fake_ort.update(falls_back=falls_back, size=96, out_dim=3)
    attrs = AttributeModel(_weight(tmp_path, "genderage.onnx", b"genderage"), "TRT")
    if falls_back:
        assert attrs.providers_active == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert attrs.trt is None
    else:
        assert attrs.providers_active == _TRT
        assert attrs.trt["fp16"] is True
    emb = SimpleNamespace(device_requested="TRT", family="arcface", dim=512, trt=None,
                          providers_active=["CUDAExecutionProvider", "CPUExecutionProvider"])
    block = _device_block(emb, attrs)
    assert block["attributes"] == {
        "requested": "TRT", "active": attrs.providers_active, "trt": attrs.trt}


def test_off_trt_the_attribute_pass_carries_no_trt_and_health_keeps_four_keys(fake_ort, tmp_path):
    """OFF is today: a CUDA load runs no warm-up, trt is None, and the device
    block keeps exactly its four keys with the attribute model loaded."""
    fake_ort.update(size=96, out_dim=3)
    attrs = AttributeModel(_weight(tmp_path, "genderage.onnx", b"genderage"), "CUDA")
    assert attrs.trt is None
    emb = SimpleNamespace(device_requested="CUDA", family="arcface", dim=512, trt=None,
                          providers_active=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert set(_device_block(emb, attrs)) == {"requested", "active", "family", "dim"}


def test_both_graphs_key_their_engines_on_the_weights(fake_ort, tmp_path):
    """Loaded by path, the EP's engine name carries the file name and not the
    weights: re-fetched in place under the same name, the old engine answered.
    Each load's engine directory is the hash of the bytes it loaded."""
    path = _weight(tmp_path, "w600k_r50.onnx", b"arcface-v1")
    ArcFaceEmbedder(path, "TRT")
    path.write_bytes(b"arcface-v2")
    ArcFaceEmbedder(path, "TRT")
    fake_ort.update(size=96, out_dim=3)
    AttributeModel(_weight(tmp_path, "genderage.onnx", b"genderage"), "TRT")
    dirs = [opts["trt_engine_cache_path"] for _, opts in fake_ort["seen"]]
    assert dirs == [str(tmp_path / "cache" / model_key(p))
                    for p in (b"arcface-v1", b"arcface-v2", b"genderage")]
