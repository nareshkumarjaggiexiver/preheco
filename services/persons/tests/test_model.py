

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
