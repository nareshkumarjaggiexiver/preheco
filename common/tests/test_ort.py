"""heco_common.ort — the one device table every model service reads.

Pure tests, no onnxruntime: the provider lists and option dicts are plain
data, and the TensorRT read-back takes any object shaped like a session.
"""

import pytest
from heco_common import ort


def test_cpu_cuda_and_openvino_rows_are_unchanged():
    """OFF is today: the TRT row is additive and moves none of the others."""
    assert ort.providers_for("CPU") == (["CPUExecutionProvider"], [{}])
    assert ort.providers_for(None) == (["CPUExecutionProvider"], [{}])
    assert ort.providers_for("") == (["CPUExecutionProvider"], [{}])
    assert ort.providers_for("cuda") == (
        ["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}])
    assert ort.providers_for("GPU") == (
        ["OpenVINOExecutionProvider", "CPUExecutionProvider"], [{"device_type": "GPU"}, {}])


@pytest.mark.parametrize("name", ["TRT", "trt", "TENSORRT", "TensorRT"])
def test_trt_is_tensorrt_then_cuda_then_cpu(name, tmp_path, monkeypatch):
    """CPU stays last (a missing accelerator degrades, never dies) and CUDA
    sits between: nodes TensorRT cannot take run on the GPU, not the CPU."""
    monkeypatch.setenv("HECO_TRT_CACHE", str(tmp_path / "cache"))
    providers, options = ort.providers_for(name)
    assert providers == [
        "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert options[1:] == [{}, {}]
    trt = options[0]
    assert trt["trt_fp16_enable"] == "True"
    assert trt["trt_engine_cache_enable"] == "True"
    assert trt["trt_timing_cache_enable"] == "True"
    assert trt["trt_engine_cache_path"] == str(tmp_path / "cache")
    assert trt["trt_timing_cache_path"] == str(tmp_path / "cache")
    # The contract's ceiling: three stacks already hold VRAM on the 8 GB card.
    assert int(trt["trt_max_workspace_size"]) == 2 * 1024**3
    assert all(isinstance(v, str) for v in trt.values())


def test_trt_cache_env_is_empty_string_safe_and_created(tmp_path, monkeypatch):
    """Compose renders an unset ${HECO_TRT_CACHE-} as "": that is the default,
    not the working directory. A nested cache dir is created up front — the
    EP only makes the last component, and a throw there costs all of TRT."""
    monkeypatch.setenv("HECO_TRT_CACHE", "")
    assert ort.trt_cache_dir() == "/srv/trt-cache"
    nested = tmp_path / "vol" / "faces"
    opts = ort.trt_options(str(nested))
    assert nested.is_dir()
    assert opts["trt_engine_cache_path"] == str(nested)


def test_an_uncreatable_cache_dir_does_not_raise(tmp_path):
    """The EP's own error and /health's active list report it; building the
    provider table must not be the thing that crashes the load."""
    blocker = tmp_path / "file"
    blocker.write_text("x")
    opts = ort.trt_options(str(blocker / "sub"))
    assert opts["trt_engine_cache_path"] == str(blocker / "sub")


def test_is_trt():
    """TRT and TENSORRT, any case; nothing else."""
    assert ort.is_trt("TRT") and ort.is_trt("tensorrt")
    assert not ort.is_trt("CUDA") and not ort.is_trt(None) and not ort.is_trt("")


class _Session:
    def __init__(self, providers, options=None, raises=False):
        self._providers, self._options, self._raises = providers, options or {}, raises

    def get_providers(self):
        if self._raises:
            raise RuntimeError("boom")
        return self._providers

    def get_provider_options(self):
        return self._options


def test_trt_truth_reads_back_what_the_ep_holds():
    """fp16, cache and workspace as the EP parsed them, not as we sent them."""
    s = _Session(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        {"TensorrtExecutionProvider": {
            "trt_fp16_enable": "1", "trt_engine_cache_path": "/srv/trt-cache/faces",
            "trt_max_workspace_size": "2147483648"}},
    )
    assert ort.trt_truth(s) == {
        "fp16": True, "engineCache": "/srv/trt-cache/faces", "workspaceBytes": 2147483648}


def test_trt_truth_is_none_when_tensorrt_did_not_load():
    """requested=TRT, active=[CUDA, CPU]: the fallback is shown as null,
    never dressed up as a TensorRT block."""
    assert ort.trt_truth(_Session(["CUDAExecutionProvider", "CPUExecutionProvider"])) is None
    assert ort.trt_truth(_Session([], raises=True)) is None
