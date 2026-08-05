"""ONNX Runtime wrapper around YOLOX-nano for person detection (CPU-only POC).

Model choice: YOLOX-nano (0.91 M params, ~1.1 GFLOPs at 416x416) over
YOLOX-tiny (~5.1 GFLOPs). The POC geometry (CONTRACTS.md: subjects 2–3 m from
a 2.0 m mount) yields large, unoccluded person boxes, so the nano's recall is
sufficient and the CPU budget matters more; tiny is a models.lock + env swap
away (same decode path) if pilot footage says otherwise.
"""

import os
import threading
import time
from pathlib import Path

import numpy as np

from .postprocess import decode_predictions, select_persons
from .preprocess import letterbox

#: Default model location — populated by `make models`, never committed.
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "yolox_nano.onnx"

MODEL_PATH = Path(os.environ.get("PERSONS_MODEL", str(DEFAULT_MODEL)))
INPUT_SIZE = int(os.environ.get("PERSONS_INPUT_SIZE", "416"))
CONF_MIN = float(os.environ.get("PERSONS_CONF_MIN", "0.30"))
NMS_IOU = float(os.environ.get("PERSONS_NMS_IOU", "0.45"))

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
    """Loads the YOLOX-nano ONNX graph once and serves thread-safe detection."""

    def __init__(self, model_path: Path = MODEL_PATH, input_size: int = INPUT_SIZE):
        """Create the onnxruntime CPU session; raises FileNotFoundError if absent."""
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — run `make models` in services/persons"
            )
        import onnxruntime as ort  # deferred so pure-function tests never need it

        self.model_name = model_path.name
        self.input_size = (input_size, input_size)
        # Explicit thread counts: stops ORT pinning threads to CPUs the
        # container does not own, and stops it oversubscribing a small graph.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = _default_threads()
        # One image at a time through one graph — there is nothing to run in
        # parallel BETWEEN nodes, so an inter-op pool is pure overhead.
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
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
        blob, ratio = letterbox(img, self.input_size)
        with self._lock:
            raw = self._session.run(None, {self._input_name: blob[None, :]})[0]
        decoded = decode_predictions(raw, self.input_size)
        boxes = select_persons(decoded, ratio, conf, iou, img.shape[1], img.shape[0])
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return boxes, round(infer_ms, 2)
