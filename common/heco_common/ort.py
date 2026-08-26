"""ONNX Runtime device selection — one implementation for every service.

Grew up in the persons service (the first stage with a family layer); lifted
here the day faces and embed grew theirs, because three hand-copied
provider tables is how one box's "CUDA" quietly means another box's "CPU".

The two rules every consumer inherits:

* CPU is ALWAYS the last provider — a missing accelerator must degrade to a
  working service, never a dead one;
* the degradation is LOGGED at load and served from /health, never silent —
  both accelerator EPs fall back to CPU without a word, and a silent
  fallback is exactly how a "GPU" benchmark becomes a CPU benchmark
  (measured on the iGPU bench, bitten again on the .94 CUDA bring-up).
"""

from __future__ import annotations

import sys


def providers_for(device: str | None) -> tuple[list[str], list[dict]]:
    """ORT (providers, provider_options) for a HECO_DEVICE string.

    CPU → CPU only (the default box never even loads an accelerator EP);
    CUDA → CUDA with CPU fallback; anything else is handed to OpenVINO as a
    device string (GPU = iGPU, NPU, …).
    """
    dev = (device or "CPU").upper()
    if dev == "CPU":
        return ["CPUExecutionProvider"], [{}]
    if dev == "CUDA":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}]
    return (
        ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        [{"device_type": dev}, {}],
    )


def announce_device(service: str, requested: str, active: list[str]) -> None:
    """The one line that keeps a benchmark honest, same shape everywhere."""
    sys.stderr.write(
        f"[heco-device] {service} requested={requested} active={active}\n"
    )


def device_truth(requested: str | None, session) -> dict:
    """The /health `device` block: what was asked vs what actually runs."""
    return {
        "requested": (requested or "CPU").upper(),
        "active": list(session.get_providers()),
    }
