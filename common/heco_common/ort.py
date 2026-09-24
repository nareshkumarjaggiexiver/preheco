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

TRT (alias TENSORRT) is the fp16 arm: TensorRT first, then CUDA, then CPU.
Measured 2026-09-24 on the 4060, eight real 4K frames of the Sharon bench
clip, session.run median fp32 CUDA -> fp16 TensorRT: YOLOX-s 640 7.1 ->
3.3 ms, SCRFD-10G 1472x832 20.0 -> 5.4 ms, ArcFace-R50 23.3 -> 8.6 ms per
5.5-face frame. Parity, re-measured on 40 frames through the services' own
classes: no person or face gained or lost at score 0.5 or 0.7 (268 persons,
243 faces), confidence within 0.005, no gender flips, age within 0.17 y;
person IoU p1 0.987 (min 0.93: a 110x341 partial person), face IoU min
0.987, landmarks within 0.30 px, ArcFace cosine min 0.9997 (0.9999 for
faces >= 56 px), feature norm within 1.1% (p99 0.40%). The tails are small
or half-hidden subjects.

Nodes TensorRT cannot take fall to CUDA, and a TensorRT that will not load
at all (libnvinfer missing, a wrong soname) drops the session to CUDA+CPU —
which /health then shows as requested=TRT, active without TensorRT, the
same honesty rule as every other fallback. A TensorRT that loads but cannot
BUILD an engine fails later, inside the first run(), where ORT rebuilds the
session on CUDA without raising — so every service reads its device truth
after its first (warm-up) run, not after construction. Engines are built
once per model CONTENT, input shape and GPU and kept under HECO_TRT_CACHE,
in a directory named by the weights' own hash (:func:`model_key`): 35-160 s
cold, under 1 s from the cache (docker-compose.trt.yml keeps it on a volume
and gives the healthcheck the start period a cold build needs).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys

#: Where TensorRT keeps its built engines and its timing cache. Only useful
#: if it outlives the container — docker-compose.trt.yml puts it on a volume.
TRT_CACHE_ENV = "HECO_TRT_CACHE"
DEFAULT_TRT_CACHE = "/srv/trt-cache"

#: The builder's scratch ceiling: 2 GiB. The 4060 has 8 GB and three stacks
#: already hold ~3 GB of it; the builder takes whatever it is allowed, so
#: the cap is what keeps an engine build from starving a live stack.
TRT_WORKSPACE_BYTES = 2 * 1024**3

_TRT_NAMES = ("TRT", "TENSORRT")


def is_trt(device: str | None) -> bool:
    """True when a HECO_DEVICE string asks for the TensorRT arm."""
    return (device or "").upper() in _TRT_NAMES


def trt_cache_dir() -> str:
    """HECO_TRT_CACHE, empty-string-safe like every compose passthrough."""
    return os.environ.get(TRT_CACHE_ENV) or DEFAULT_TRT_CACHE


#: Hex digits of the weights' sha256 that name an engine directory: 48 bits,
#: for the handful of weights one cache volume ever holds.
MODEL_KEY_CHARS = 12


def model_key(model: bytes | bytearray | str | os.PathLike) -> str:
    """The first MODEL_KEY_CHARS hex digits of the sha256 of a model's BYTES.

    ``model`` is the graph ORT is handed: bytes as they are, a path read
    (in 1 MiB chunks — ArcFace-R50 is 166 MB).

    WHY THE ENGINE CACHE NEEDS IT. ORT's TensorRT EP names a cached engine
    by a hash of the GRAPH — nodes, shapes and, for a model loaded from a
    path, its file name — never of the weights. Measured 2026-09-25 on the
    4060 (ORT 1.30.0, TensorRT 10.16.1): a SCRFD-10G copy with one bias
    shifted by +2.0 (8 bytes; its stride-8 score mean 0.0096 -> 0.0643 on
    CUDA) "built" in 0.08 s against the original's 32.9 s and returned the
    ORIGINAL's scores (0.00962 both), under the very engine hash the faces
    service had cached; a persons or embed weight re-fetched in place under
    its old name did the same. So engines live in a directory named by the
    content (:func:`trt_options`): a new weight is a new engine, whatever
    it is called.
    """
    digest = hashlib.sha256()
    if isinstance(model, bytes | bytearray | memoryview):
        digest.update(model)
    else:
        with open(model, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:MODEL_KEY_CHARS]


def trt_options(cache_dir: str | None = None, model=None) -> dict:
    """TensorrtExecutionProvider options: fp16, engine + timing cache, 2 GiB.

    ``model`` (the bytes or path ORT is handed) puts the ENGINES under
    ``<cache>/<model_key(model)>`` — see :func:`model_key` for the stale
    engine that keying on anything else served. The timing cache stays at
    ``<cache>``: it holds per-layer kernel timings for this GPU and TensorRT
    version, valid for any weights, and sharing it only speeds a rebuild.

    The cache directory is created here, best-effort: the EP creates only
    the LAST path component and a session whose TensorRT EP throws falls
    back to CUDA — correct, but a per-service subdirectory of a fresh volume
    would otherwise cost the whole TensorRT arm for a missing mkdir.
    Values are strings because that is what ORT hands the EP either way.
    """
    cache = cache_dir or trt_cache_dir()
    engines = cache if model is None else os.path.join(cache, model_key(model))
    # A failure here is not ours to report: the EP's own error (and /health's
    # active list) will say so.
    with contextlib.suppress(OSError):
        os.makedirs(engines, exist_ok=True)
    return {
        "trt_fp16_enable": "True",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": engines,
        "trt_timing_cache_enable": "True",
        "trt_timing_cache_path": cache,
        "trt_max_workspace_size": str(TRT_WORKSPACE_BYTES),
    }


def trt_batch_profile(input_name: str, sample: tuple[int, ...], max_batch: int) -> dict:
    """TensorRT optimisation-profile options for a graph with a dynamic batch.

    Without an explicit profile the EP builds an engine for the first batch
    size it sees and REBUILDS whenever a larger one arrives — tens of
    seconds inside a live request. 1..max_batch, optimised for half of max.
    """
    tail = "x".join(str(int(d)) for d in sample)
    opt = max(1, max_batch // 2)
    return {
        "trt_profile_min_shapes": f"{input_name}:1x{tail}",
        "trt_profile_opt_shapes": f"{input_name}:{opt}x{tail}",
        "trt_profile_max_shapes": f"{input_name}:{max_batch}x{tail}",
    }


def providers_for(
    device: str | None, trt_extra: dict | None = None, model=None
) -> tuple[list[str], list[dict]]:
    """ORT (providers, provider_options) for a HECO_DEVICE string.

    CPU → CPU only (the default box never even loads an accelerator EP);
    CUDA → CUDA with CPU fallback; TRT/TENSORRT → TensorRT fp16, then CUDA,
    then CPU; anything else is handed to OpenVINO as a device string
    (GPU = iGPU, NPU, …). `trt_extra` adds TensorRT options (a batch
    profile) and `model` (the bytes or path the session loads) keys the
    engine directory on the weights (:func:`model_key`); both are ignored —
    `model` never even read — on every other device.
    """
    dev = (device or "CPU").upper()
    if dev == "CPU":
        return ["CPUExecutionProvider"], [{}]
    if dev == "CUDA":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}]
    if dev in _TRT_NAMES:
        return (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            [{**trt_options(model=model), **(trt_extra or {})}, {}, {}],
        )
    return (
        ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        [{"device_type": dev}, {}],
    )


def _varint(buf: bytes | bytearray, i: int) -> tuple[int, int]:
    """Decode one protobuf varint at `i`; (value, next index)."""
    value, shift = 0, 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


def _fields(buf: bytes | bytearray, start: int, end: int):
    """(field, wire type, value start, value end) for each field of one message."""
    i = start
    while i < end:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            _, j = _varint(buf, i)
        elif wire == 1:
            j = i + 8
        elif wire == 2:
            n, i = _varint(buf, i)
            j = i + n
        elif wire == 5:
            j = i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, i, j
        i = j


def _sub(buf, start: int, end: int, field: int):
    """Spans of every length-delimited `field` directly inside a message."""
    return [(s, e) for f, w, s, e in _fields(buf, start, end) if f == field and w == 2]


def distinct_input_dims(model: bytes) -> bytes | None:
    """The ONNX model with repeated symbolic input dims renamed apart, or None.

    WHY TensorRT NEEDS THIS. TensorRT treats two input dimensions that carry
    the same symbolic NAME as one dimension. The InsightFace SCRFD exports
    name both H and W "?", so a frame-shaped 1472x832 input is a
    contradiction and the engine build fails ("Input dimensions with this
    name have different constant values", measured 2026-09-24 on the
    4060) — after which ORT quietly re-runs the session on CUDA. A square
    640 input hides it. Renaming the second "?" gives TensorRT two free
    dimensions; ONNX Runtime itself never cared about the names.

    Only the ModelProto.graph.input shapes are touched, and each rename
    keeps its byte length, so no length prefix anywhere in the file moves.
    None when nothing repeats: the caller then loads the file unchanged.
    """
    try:
        return _distinct_input_dims(bytearray(model))
    except (IndexError, ValueError):
        return None  # not a parseable ModelProto: let ORT say what is wrong


def _distinct_input_dims(buf: bytearray) -> bytes | None:
    """distinct_input_dims on a mutable copy; raises on a malformed buffer."""
    spans: list[tuple[int, int]] = []  # dim_param spans, per input shape
    groups: list[list[tuple[int, int]]] = []
    for gs, ge in _sub(buf, 0, len(buf), 7):  # ModelProto.graph
        for vs, ve in _sub(buf, gs, ge, 11):  # GraphProto.input
            for ts, te in _sub(buf, vs, ve, 2):  # ValueInfoProto.type
                for tts, tte in _sub(buf, ts, te, 1):  # TypeProto.tensor_type
                    for ss, se in _sub(buf, tts, tte, 2):  # Tensor.shape
                        group = []
                        for ds, de in _sub(buf, ss, se, 1):  # TensorShapeProto.dim
                            group += _sub(buf, ds, de, 2)  # Dimension.dim_param
                        groups.append(group)
                        spans += group
    used = {bytes(buf[s:e]) for s, e in spans}
    changed = False
    for group in groups:
        seen: set[bytes] = set()
        for s, e in group:
            name = bytes(buf[s:e])
            if name in seen and e > s:
                for c in b"HWDCN0123456789abcdefghijklmnopqrstuvwxyz":
                    candidate = name[:-1] + bytes([c])
                    if candidate not in used:
                        buf[s:e] = candidate
                        used.add(candidate)
                        name = candidate
                        changed = True
                        break
            seen.add(name)
    return bytes(buf) if changed else None


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


def trt_truth(session) -> dict | None:
    """What TensorRT is ACTUALLY running with, read back from the session.

    None when the session has no TensorRT EP (it never loaded, or was not
    asked for) — absent, not a guess. Read from get_provider_options(), the
    EP's own parse of what it was given, so a flag the EP silently ignored
    shows up here as the value it really holds.
    """
    try:
        if "TensorrtExecutionProvider" not in session.get_providers():
            return None
        opts = session.get_provider_options().get("TensorrtExecutionProvider") or {}
    except Exception:  # noqa: BLE001 — a health read must never raise
        return None
    workspace = opts.get("trt_max_workspace_size")
    return {
        "fp16": str(opts.get("trt_fp16_enable", "")).lower() in ("1", "true"),
        "engineCache": opts.get("trt_engine_cache_path") or None,
        "workspaceBytes": int(workspace) if str(workspace or "").isdigit() else None,
    }
