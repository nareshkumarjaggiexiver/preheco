

def test_thread_count_respects_the_cpuset_and_caps_itself(monkeypatch):
    """Pinning the container to one NUMA node broke ORT's own thread pinning.

    Left implicit, ORT counts the machine's cores (64 on the dual-socket
    T440), spawns that many threads and pins each to a chosen CPU — including
    CPUs on the socket the container no longer owns, which fails EINVAL and
    spams the log on every start. The number must come from what the process
    may actually use, and be capped so a small graph is not oversubscribed.
    """
    import os

    from app.model import _default_threads

    monkeypatch.delenv("PERSONS_THREADS", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(32)))
    assert _default_threads() == 8, "capped, not one-per-core"

    # A tighter cpuset wins over the cap — never ask for CPUs we do not have.
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 2})
    assert _default_threads() == 2

    # And an operator can still say.
    monkeypatch.setenv("PERSONS_THREADS", "16")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(32)))
    assert _default_threads() == 16


# ------------------------------------------------- specs and providers

def test_known_models_have_specs_and_unknown_names_infer():
    from pathlib import Path

    from app.model import MODEL_SPECS, spec_for

    assert MODEL_SPECS["yolox_nano.onnx"] == {"family": "yolox", "input": 416}
    assert MODEL_SPECS["yolox_s.onnx"] == {"family": "yolox", "input": 640}
    assert MODEL_SPECS["rtdetrv2_s.onnx"] == {"family": "rtdetr", "input": 640}
    # An experiment outside the table still loads: family from the name.
    assert spec_for(Path("rtdetrv2_m_exported.onnx"))["family"] == "rtdetr"
    assert spec_for(Path("yolox_custom.onnx"))["family"] == "yolox"


def test_providers_always_end_in_cpu_and_cuda_maps_cleanly():
    """CPU is the last provider ON PURPOSE (a missing accelerator degrades to
    a working detector, logged) — and CPU-only stays a single-entry list so
    the default box never even loads an accelerator EP."""
    from app.model import providers_for

    assert providers_for("CPU") == (["CPUExecutionProvider"], [{}])
    cuda, _ = providers_for("CUDA")
    assert cuda == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ov, opts = providers_for("GPU")
    assert ov == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert opts[0] == {"device_type": "GPU"}
    assert providers_for(None)[0] == ["CPUExecutionProvider"], "unset means CPU"


def test_persons_model_env_speaks_filenames_like_everything_else():
    """The lock, the specs and the catalog all say `yolox_s.onnx`; the env
    must map that same word to the models dir — a bare name resolved against
    the working directory booted a container that could not find weights it
    was mounted with (bitten live on the .94 CUDA arm)."""
    from pathlib import Path

    from app.model import DEFAULT_MODEL, _model_path

    assert _model_path(None) == DEFAULT_MODEL
    assert _model_path("") == DEFAULT_MODEL
    assert _model_path("yolox_s.onnx") == DEFAULT_MODEL.parent / "yolox_s.onnx"
    assert _model_path("/tmp/exp/custom.onnx") == Path("/tmp/exp/custom.onnx")


# ------------------------------------------------- selection durability

def test_persist_selection_names_the_root_owned_dir_case(tmp_path, monkeypatch):
    """The 507's sentence must send the operator to the REAL fix.

    On default deployments the mount exists, so the old "mount it writable"
    advice pointed at a fix already in place while the actual cause was
    ownership: a state dir absent from the checkout is created root-owned by
    the docker daemon, and a non-root container uid (userns-remap, host-run
    service) cannot write it. The error now says so, and names pre-creation
    with the right owner as the remedy.
    """
    import os

    import pytest

    from app import model as model_mod

    if os.geteuid() == 0:
        pytest.skip("root ignores permission bits; the write below would succeed")
    state = tmp_path / "state"
    state.mkdir()
    state.chmod(0o500)  # exists, but this uid cannot write it — the root-owned shape
    monkeypatch.setattr(model_mod, "STATE_DIR", state)
    monkeypatch.setattr(model_mod, "SELECTED_FILE", state / "selected")
    try:
        with pytest.raises(RuntimeError) as exc:
            model_mod.persist_selection("yolox_s.onnx")
    finally:
        state.chmod(0o700)  # let tmp_path cleanup work
    msg = str(exc.value)
    assert "root-owned by the docker daemon" in msg
    assert "pre-create services/persons/state" in msg
    assert "deployment env" in msg


def test_the_state_dir_ships_in_the_checkout():
    """The bind-mount source must exist OWNED BY THE CHECKOUT USER before
    the docker daemon can create it root-owned (and it must travel in
    tar-copy deploys): state/.gitkeep is that guarantee."""
    from pathlib import Path

    from app import model as model_mod

    keep = Path(model_mod.__file__).resolve().parent.parent / "state" / ".gitkeep"
    assert keep.is_file(), (
        "services/persons/state/.gitkeep is missing — without it the first "
        "`compose up` creates the state dir root-owned and every planner "
        "apply 507s on non-root-uid setups"
    )


def test_health_device_block_is_todays_off_trt_and_honest_on_it():
    """Off TRT the block keeps its three keys byte-for-byte; asked for TRT it
    carries `trt` — null when TensorRT did not come up (the fallback shown,
    never hidden), the EP's own read-back when it did."""
    from types import SimpleNamespace

    from app.main import _device_block

    det = SimpleNamespace(device_requested="CUDA", family="yolox", trt=None,
                          providers_active=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert _device_block(det) == {
        "requested": "CUDA", "active": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "family": "yolox"}
    det.device_requested = "TRT"
    assert _device_block(det)["trt"] is None
    det.trt = {"fp16": True, "engineCache": "/srv/trt-cache/persons", "workspaceBytes": 1}
    assert _device_block(det)["trt"]["fp16"] is True


def test_a_cpu_detector_never_warms_up_or_reads_trt(monkeypatch):
    """The TRT warm-up and read-back are TRT-only: a CPU load runs no extra
    inference and carries trt=None (today's load, exactly)."""
    import app.model as m

    if not m.DEFAULT_MODEL.is_file():
        import pytest

        pytest.skip("yolox_nano.onnx missing")
    calls = []
    real = m.PersonDetector.detect
    monkeypatch.setattr(m.PersonDetector, "detect",
                        lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    det = m.PersonDetector(m.DEFAULT_MODEL, 416, "yolox", "CPU")
    assert calls == [] and det.trt is None


# ------------------------------------------------- TensorRT load (fake ORT)

_TRT_LIST = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


class _TrtSession:
    """TensorRT in the session until the first run(); with ``falls_back``,
    ORT's silent rebuild on [CUDA, CPU] after a failed engine build (measured
    on the box: no exception, only a changed provider list)."""

    def __init__(self, falls_back: bool, options: dict, size: int):
        self.falls_back, self.options, self.size, self.ran = falls_back, options, size, False

    def get_providers(self):
        if self.ran and self.falls_back:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return list(_TRT_LIST)

    def get_provider_options(self):
        return {"TensorrtExecutionProvider": self.options}

    def get_inputs(self):
        from types import SimpleNamespace

        return [SimpleNamespace(name="images", shape=[1, 3, self.size, self.size])]

    def run(self, _names, _feeds):
        import numpy as np

        self.ran = True
        anchors = sum((self.size // s) ** 2 for s in (8, 16, 32))
        return [np.zeros((1, anchors, 85), dtype=np.float32)]


def _fake_ort(monkeypatch, falls_back: bool, size: int = 64) -> list:
    """Install a fake onnxruntime; returns the (model, options) each session got."""
    import sys
    import types

    seen: list = []
    mod = types.ModuleType("onnxruntime")
    mod.SessionOptions = lambda: types.SimpleNamespace()

    def session(model, sess_options=None, providers=None, provider_options=None):
        seen.append((model, provider_options[0]))
        return _TrtSession(falls_back, provider_options[0], size)

    mod.InferenceSession = session
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return seen


def test_trt_engines_are_keyed_on_the_weights_not_the_file_name(monkeypatch, tmp_path):
    """persons loads by PATH, and the EP's engine name carries the file name
    but not the weights: re-fetched in place, the old engine answered. The
    engine directory is the weights' hash, so new bytes build a new engine."""
    from heco_common.ort import model_key

    import app.model as m

    monkeypatch.setenv("HECO_TRT_CACHE", str(tmp_path / "cache"))
    weight = tmp_path / "yolox_s.onnx"
    seen = _fake_ort(monkeypatch, falls_back=False)
    weight.write_bytes(b"yolox-weights-v1")
    m.PersonDetector(weight, 64, "yolox", "TRT")
    weight.write_bytes(b"yolox-weights-v2")
    m.PersonDetector(weight, 64, "yolox", "TRT")
    (_, o1), (_, o2) = seen
    assert o1["trt_engine_cache_path"] == str(tmp_path / "cache" / model_key(b"yolox-weights-v1"))
    assert o2["trt_engine_cache_path"] == str(tmp_path / "cache" / model_key(b"yolox-weights-v2"))
