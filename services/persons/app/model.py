"""ONNX Runtime person detection — a model FAMILY behind one contract.

Model choice for the POC was YOLOX-nano (0.91 M params, ~1.1 GFLOPs at
416x416): the POC geometry (CONTRACTS.md: subjects 2–3 m from a 2.0 m mount)
yields large, unoccluded person boxes, so the nano's recall is sufficient and
the CPU budget matters more. The open-catalog plan
(docs/planning/15-model-configurations.md M2) turns "a stronger detector is a
swap away" from a comment into machinery:

* MODEL_SPECS names the known models, their input sizes and their FAMILY —
  the pre/post pair they speak. Same-family swaps (nano -> s) are a
  models.lock row + PERSONS_MODEL; a new family (rtdetr) is a new pre/post
  pair behind the SAME /detect contract, and nothing upstream can tell.
* HECO_DEVICE picks the execution provider: CPU (default), CUDA (the .94
  box), anything else is handed to OpenVINO as a device string (GPU = the
  iGPU arm — the recipe proven on thinkcenter002, folded in from the
  box-local patch). CPU stays in the provider list as a fallback ON PURPOSE,
  and the ACTIVE list is logged and served from /health, because a silent
  fallback is exactly how a "GPU" number quietly becomes a CPU number.
"""

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .postprocess import decode_predictions, select_persons, select_persons_rtdetr
from .preprocess import letterbox, rtdetr_blob

#: Default model location — populated by `make models`, never committed.
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "yolox_nano.onnx"

#: The models this service knows how to run: input size + pre/post family.
#: A filename outside this table still loads (family inferred from its name,
#: size from PERSONS_INPUT_SIZE) so an experiment is never blocked by a table.
MODEL_SPECS = {
    "yolox_nano.onnx": {"family": "yolox", "input": 416},
    "yolox_tiny.onnx": {"family": "yolox", "input": 416},
    "yolox_s.onnx": {"family": "yolox", "input": 640},
    "rtdetrv2_s.onnx": {"family": "rtdetr", "input": 640},
}


def spec_for(model_path: Path) -> dict:
    """The spec for a model file: table first, name inference as fallback."""
    known = MODEL_SPECS.get(model_path.name)
    if known:
        return dict(known)
    family = "rtdetr" if "rtdetr" in model_path.name.lower() else "yolox"
    return {"family": family, "input": 640 if family == "rtdetr" else 416}


def providers_for(device: str) -> tuple[list[str], list[dict]]:
    """ORT (providers, provider_options) for a HECO_DEVICE string.

    CPU is always the last entry: a missing accelerator must degrade to a
    working detector, not a dead service — but the degradation is LOGGED and
    served from /health, never silent.
    """
    dev = (device or "CPU").upper()
    if dev == "CPU":
        return ["CPUExecutionProvider"], [{}]
    if dev == "CUDA":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}]
    # Anything else is an OpenVINO device string: GPU (iGPU), NPU, ...
    return (
        ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        [{"device_type": dev}, {}],
    )


# `or` rather than a default argument throughout: compose passthroughs render
# an unset variable as the EMPTY STRING, and Path("")/int("") would turn an
# unset knob into a boot failure. Review finding, 2026-08-14.
MODEL_PATH = Path(os.environ.get("PERSONS_MODEL") or str(DEFAULT_MODEL))
_SPEC = spec_for(MODEL_PATH)
# For models the SPECS table knows, the table is authoritative — a lingering
# env var tuned for the previous model must not misconfigure the next one
# (a 416 blob into the static-640 yolox_s graph fails EVERY /detect while
# /health says healthy — the exact incident class the healthcheck exists to
# stop). The env override applies only to models outside the table, which is
# what it was for: experiments.
_KNOWN = MODEL_PATH.name in MODEL_SPECS
INPUT_SIZE = (_SPEC["input"] if _KNOWN
              else int(os.environ.get("PERSONS_INPUT_SIZE") or _SPEC["input"]))
FAMILY = _SPEC["family"] if _KNOWN else (os.environ.get("PERSONS_FAMILY") or _SPEC["family"])
DEVICE = os.environ.get("HECO_DEVICE") or "CPU"
CONF_MIN = float(os.environ.get("PERSONS_CONF_MIN") or "0.30")
NMS_IOU = float(os.environ.get("PERSONS_NMS_IOU") or "0.45")

def _default_threads() -> int:
    """How many intra-op threads onnxruntime should use.

    Left to itself, ORT counts the machine's cores from /proc — 64 on the
    T440 — spawns a thread pool that size and PINS each thread to a chosen
    CPU. Once the container is confined to one NUMA node, half those CPUs are
    outside its cpuset and every pin fails:

        pthread_setaffinity_np failed ... mask: {1, 33, }, error code: 22
        Specify the number of threads explicitly so the affinity is not set.

    `sched_getaffinity` reports what this process may ACTUALLY use, so it
    respects the cpuset rather than the hardware. The cap is deliberate on top
    of that: YOLOX-nano at 416x416 is a small graph, and past a handful of
    threads the synchronisation costs more than the parallelism returns — the
    measured run used about 2.8 cores while ORT had spawned thirty.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        available = os.cpu_count() or 1
    return max(1, min(available, int(os.environ.get("PERSONS_THREADS", "8"))))


class PersonDetector:
    """Loads one detection model once and serves thread-safe detection."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        input_size: int = INPUT_SIZE,
        family: str = FAMILY,
        device: str = DEVICE,
    ):
        """Create the onnxruntime session; raises FileNotFoundError if absent."""
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — run `make models` in services/persons"
            )
        import onnxruntime as ort  # deferred so pure-function tests never need it

        self.model_name = model_path.name
        self.family = family
        self.input_size = (input_size, input_size)
        # Explicit thread counts: stops ORT pinning threads to CPUs the
        # container does not own, and stops it oversubscribing a small graph.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = _default_threads()
        # One image at a time through one graph — there is nothing to run in
        # parallel BETWEEN nodes, so an inter-op pool is pure overhead.
        opts.inter_op_num_threads = 1
        providers, provider_options = providers_for(device)
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=providers,
            provider_options=provider_options,
        )
        self.device_requested = (device or "CPU").upper()
        self.providers_active = list(self._session.get_providers())
        # The one line that keeps a benchmark honest: both ORT accelerator
        # EPs fall back to CPU silently, so the ACTIVE list is stated at load
        # (and served from /health) rather than trusted from the request.
        sys.stderr.write(
            f"[heco-device] persons requested={self.device_requested} "
            f"active={self.providers_active}\n"
        )
        # THE GRAPH IS THE GROUND TRUTH. A static export knows its own input
        # size and arity; configuration that disagrees would load cleanly and
        # then fail every /detect while /health reported healthy. Reconcile
        # here, loudly, so a mismatch is a boot failure the healthcheck sees.
        inputs = self._session.get_inputs()
        shape = inputs[0].shape
        static_hw = [d for d in shape[-2:] if isinstance(d, int)]
        if static_hw and any(d != input_size for d in static_hw):
            raise ValueError(
                f"{model_path.name} expects a {'x'.join(str(d) for d in static_hw)} "
                f"input, configured {input_size} — check PERSONS_INPUT_SIZE against MODEL_SPECS"
            )
        # Family by input NAME, not arity: the rtdetr contract is the named
        # second input `orig_target_sizes`; arity alone would misread any
        # future multi-input export (review finding).
        has_sizes_input = any(i.name == "orig_target_sizes" for i in inputs)
        graph_family = "rtdetr" if has_sizes_input else "yolox"
        if family != graph_family:
            raise ValueError(
                f"{model_path.name} is a {graph_family}-shaped graph "
                f"(inputs: {[i.name for i in inputs]}), configured family `{family}` — "
                "check PERSONS_FAMILY against MODEL_SPECS"
            )
        self._input_name = inputs[0].name
        self._lock = threading.Lock()

    def detect(
        self,
        img: np.ndarray,
        conf_min: float | None = None,
        nms_iou: float | None = None,
    ) -> tuple[list[dict], float]:
        """Detect persons in a BGR image.

        Returns `(boxes, infer_ms)` where boxes are contract dicts in source
        pixels and `infer_ms` covers letterbox + ONNX run + decode/NMS (JPEG
        decode excluded — that belongs to transport, not the model).
        """
        conf = CONF_MIN if conf_min is None else conf_min
        iou = NMS_IOU if nms_iou is None else nms_iou
        t0 = time.perf_counter()
        if self.family == "rtdetr":
            # Two-input contract: the graph takes the original size and does
            # its own geometry, so there is no ratio to undo and no NMS —
            # DETR set prediction is one box per object by construction.
            blob = rtdetr_blob(img, self.input_size)
            sizes = np.array([[img.shape[1], img.shape[0]]], dtype=np.int64)
            with self._lock:
                labels, out_boxes, scores = self._session.run(
                    None, {self._input_name: blob[None, :], "orig_target_sizes": sizes}
                )
            boxes = select_persons_rtdetr(
                labels, out_boxes, scores, conf, img.shape[1], img.shape[0]
            )
        else:
            blob, ratio = letterbox(img, self.input_size)
            with self._lock:
                raw = self._session.run(None, {self._input_name: blob[None, :]})[0]
            decoded = decode_predictions(raw, self.input_size)
            boxes = select_persons(decoded, ratio, conf, iou, img.shape[1], img.shape[0])
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return boxes, round(infer_ms, 2)
