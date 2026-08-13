

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
