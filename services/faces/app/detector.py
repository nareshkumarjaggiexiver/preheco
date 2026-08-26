"""Face detection — a model FAMILY behind one contract (doc 15).

Two families behind one detect() shape:

  yunet  — cv2.FaceDetectorYN, the default since day one. One row per face:
           [x, y, w, h, five (x, y) landmarks (right eye, left eye, nose
           tip, right mouth corner, left mouth corner), score]. CPU via
           OpenCV; the iGPU bench measured OpenCV 5 ignoring acceleration
           targets, so no device plumbing is pretended here.
  scrfd  — generic ONNX Runtime + the pure decode in scrfd.py, HECO_DEVICE
           capable. Restricted tier (InsightFace, grant `contact`): the
           adapter ships ahead of weights, like rtdetr did at persons.

Both emit the SAME face dicts — box, five landmarks IN THE YUNET ORDER,
conf — because the embed alignment consumes that order verbatim and the
quality gate computes IED/frontality from it: the landmark contract is the
load-bearing wall (the review pinned it), and a family that cannot provide
it does not belong in this service.
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from heco_common.ort import announce_device, providers_for

from . import scrfd

#: Default model location — populated by `make models`, never committed.
DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
)

#: The models this service knows how to run. Unknown names infer scrfd when
#: they say so, else yunet — an experiment is never blocked by a table.
MODEL_SPECS = {
    "face_detection_yunet_2023mar.onnx": {"family": "yunet"},
    "scrfd_2.5g_kps.onnx": {"family": "scrfd", "input": 640},
    "scrfd_10g_kps.onnx": {"family": "scrfd", "input": 640},
}


def spec_for(model_path: Path) -> dict:
    known = MODEL_SPECS.get(model_path.name)
    if known:
        return dict(known)
    family = "scrfd" if "scrfd" in model_path.name.lower() else "yunet"
    return {"family": family, **({"input": 640} if family == "scrfd" else {})}


#: Planner-applied selection, persisted beside the weights (bind-mounted, so
#: it survives restarts). Precedence: .selected > env > default — the file
#: is the operator's LATEST intent through the planner; same rule as persons.
SELECTED_FILE = DEFAULT_MODEL.parent / ".selected"


def selected_model() -> str | None:
    try:
        name = SELECTED_FILE.read_text().strip()
    except OSError:
        return None
    if not name or "/" in name or not (DEFAULT_MODEL.parent / name).is_file():
        return None
    return name


def persist_selection(name: str) -> None:
    tmp = SELECTED_FILE.with_suffix(".tmp")
    tmp.write_text(name + "\n")
    tmp.replace(SELECTED_FILE)


def _model_path(value: str | None) -> Path:
    if not value:
        return DEFAULT_MODEL
    return Path(value) if "/" in value else DEFAULT_MODEL.parent / value


MODEL_PATH = _model_path(selected_model() or (os.environ.get("FACES_MODEL") or None))
SCORE_MIN = float(os.environ.get("FACES_SCORE_MIN") or "0.8")
NMS_IOU = float(os.environ.get("FACES_NMS_IOU") or "0.3")
TOP_K = int(os.environ.get("FACES_TOP_K") or "5000")
DEVICE = os.environ.get("HECO_DEVICE") or "CPU"

#: Crops smaller than this per side are skipped — YuNet cannot resolve them
#: and sizes this small are far below the 56 px POC face floor anyway.
MIN_CROP_SIDE = 16


def build_detector(model_path: Path | None = None, device: str | None = None):
    """The family factory: one detect() contract, family chosen by the file."""
    path = model_path or MODEL_PATH
    spec = spec_for(path)
    if spec["family"] == "scrfd":
        return ScrfdDetector(path, spec.get("input", 640), device or DEVICE)
    return FaceDetector(path)


class FaceDetector:
    """The yunet family: owns one FaceDetectorYN; a lock serialises calls."""

    family = "yunet"
    #: cv2 runs this family; the honest device answer is the one OpenCV
    #: gives, not the one HECO_DEVICE asks for (iGPU bench: targets ignored).
    device_requested = "CPU"
    providers_active = ["cv2"]

    def __init__(self, model_path: Path = None):
        """Create the YuNet detector; raises FileNotFoundError when weights absent."""
        model_path = model_path or MODEL_PATH
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — run `make models` in services/faces"
            )
        self.model_name = model_path.name
        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), SCORE_MIN, NMS_IOU, TOP_K
        )
        self._lock = threading.Lock()

    def detect(self, img: np.ndarray) -> list[dict]:
        """Detect faces in a BGR image; returns contract face dicts (no quality)."""
        h, w = img.shape[:2]
        if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE:
            return []
        with self._lock:
            self._det.setInputSize((w, h))
            _, rows = self._det.detect(img)
        if rows is None:
            return []
        faces = []
        for row in rows:
            faces.append(
                {
                    "box": {
                        "x": round(float(row[0]), 1),
                        "y": round(float(row[1]), 1),
                        "w": round(float(row[2]), 1),
                        "h": round(float(row[3]), 1),
                    },
                    "landmarks": [
                        [round(float(row[4 + 2 * i]), 1), round(float(row[5 + 2 * i]), 1)]
                        for i in range(5)
                    ],
                    "conf": round(float(row[14]), 4),
                }
            )
        return faces


class ScrfdDetector:
    """The scrfd family: generic ONNX Runtime + the pure decode in scrfd.py.

    Same detect() contract as FaceDetector — box, five landmarks in the
    yunet order, conf — so nothing downstream can tell the families apart.
    """

    family = "scrfd"

    def __init__(self, model_path: Path, input_size: int = 640, device: str | None = None):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — fetch it (restricted tier: make models-restricted) first"
            )
        import onnxruntime as ort  # deferred, like every family loader

        self.model_name = model_path.name
        self.input_size = (input_size, input_size)
        providers, provider_options = providers_for(device)
        self._session = ort.InferenceSession(
            str(model_path), providers=providers, provider_options=provider_options
        )
        self.device_requested = (device or "CPU").upper()
        self.providers_active = list(self._session.get_providers())
        announce_device("faces", self.device_requested, self.providers_active)
        if len(self._session.get_outputs()) != 9:
            raise ValueError(
                f"{model_path.name} has {len(self._session.get_outputs())} outputs — "
                "the scrfd family speaks the 9-tensor *_kps export (3 strides x "
                "score/box/kps); a different export needs its own family"
            )
        self._input_name = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def detect(self, img: np.ndarray) -> list[dict]:
        h, w = img.shape[:2]
        if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE:
            return []
        # Letterbox keep-ratio into the square input, zeros pad — the decode
        # divides by the same ratio on the way out.
        ih, iw = self.input_size
        ratio = min(ih / h, iw / w)
        rw, rh = int(w * ratio), int(h * ratio)
        canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
        canvas[:rh, :rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = ((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        with self._lock:
            outputs = self._session.run(None, {self._input_name: np.ascontiguousarray(blob)})
        return scrfd.select_faces(
            outputs, self.input_size, ratio, SCORE_MIN, NMS_IOU, w, h
        )
