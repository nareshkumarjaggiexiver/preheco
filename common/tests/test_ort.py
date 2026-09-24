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


# ------------------------------------------------- distinct_input_dims
# A hand-encoded ModelProto (field numbers from onnx/onnx.proto), just enough
# graph for the walker: inputs and outputs with named symbolic dims.

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        n, low = n >> 7, n & 0x7F
        out.append(low | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _blob(field: int, payload: bytes) -> bytes:
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _int(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _vi(name: str, dims) -> bytes:
    shape = b"".join(
        _blob(1, _blob(2, d.encode()) if isinstance(d, str) else _int(1, d)) for d in dims)
    tensor = _int(1, 1) + _blob(2, shape)
    return _blob(1, name.encode()) + _blob(2, _blob(1, tensor))


def _model(inputs, outputs=()) -> bytes:
    graph = _blob(2, b"g")
    graph += b"".join(_blob(11, _vi(n, d)) for n, d in inputs)
    graph += b"".join(_blob(12, _vi(n, d)) for n, d in outputs)
    return _int(1, 8) + _blob(7, graph) + _blob(8, _int(2, 13))


def _input_dim_params(model: bytes) -> list[list[str]]:
    """Read back each graph input's dim_params with the module's own walker."""
    out = []
    for gs, ge in ort._sub(model, 0, len(model), 7):
        for vs, ve in ort._sub(model, gs, ge, 11):
            names = []
            for ts, te in ort._sub(model, vs, ve, 2):
                for tts, tte in ort._sub(model, ts, te, 1):
                    for ss, se in ort._sub(model, tts, tte, 2):
                        for ds, de in ort._sub(model, ss, se, 1):
                            names += [model[s:e].decode() for s, e in ort._sub(model, ds, de, 2)]
            out.append(names)
    return out


def test_scrfd_style_shared_question_marks_are_renamed_apart():
    """InsightFace names H and W both "?" — TensorRT reads that as ONE dim
    and a 1472x832 input fails its engine build. Renamed apart, same byte
    length, outputs untouched, everything else byte-identical."""
    model = _model([("input.1", [1, 3, "?", "?"])], [("score_8", ["?", 1])])
    fixed = ort.distinct_input_dims(model)
    assert fixed is not None and len(fixed) == len(model)
    names = _input_dim_params(fixed)[0]
    assert names[0] == "?" and names[1] != "?" and len(names[1]) == 1
    tail = model.index(b"score_8")
    assert fixed[tail:] == model[tail:], "the output value_info is not touched"
    diffs = [i for i, (a, b) in enumerate(zip(model, fixed, strict=True)) if a != b]
    assert len(diffs) == 1, "exactly one byte renamed"


def test_nothing_repeated_means_nothing_to_do():
    """Distinct names, static shapes, or a dynamic batch alone: None, so the
    caller loads the file itself (and its engine cache keeps its key)."""
    assert ort.distinct_input_dims(_model([("x", [1, 3, "h", "w"])])) is None
    assert ort.distinct_input_dims(_model([("x", [1, 3, 640, 640])])) is None
    assert ort.distinct_input_dims(_model([("x", ["N", 3, 112, 112])])) is None


def test_a_rename_never_collides_with_a_name_already_in_use():
    """A new name is one no graph input already uses (TensorRT unifies by name)."""
    model = _model([("a", ["H", "?", "?"]), ("b", ["N", "W"])])
    fixed = ort.distinct_input_dims(model)
    first, second = _input_dim_params(fixed)
    assert len(set(first)) == 3 and not set(first) & {"W", "N"}
    assert second == ["N", "W"]


def test_a_file_that_is_not_a_model_is_left_for_ort_to_refuse():
    """Truncated or foreign bytes: None (load the file as-is), never a crash
    of our own in front of ORT's real error message."""
    assert ort.distinct_input_dims(b"not-a-real-graph") is None
    assert ort.distinct_input_dims(_model([("x", [1, 3, "?", "?"])])[:-9]) is None
