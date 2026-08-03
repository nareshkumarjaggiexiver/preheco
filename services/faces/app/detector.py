"""cv2.FaceDetectorYN (YuNet) wrapper: load once, detect thread-safely.

YuNet returns one row per face: [x, y, w, h, five (x, y) landmarks
(right eye, left eye, nose tip, right mouth corner, left mouth corner), score].
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np

#: Default model location — populated by `make models`, never committed.
DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
)

MODEL_PATH = Path(os.environ.get("FACES_MODEL", str(DEFAULT_MODEL)))
SCORE_MIN = float(os.environ.get("FACES_SCORE_MIN", "0.8"))
NMS_IOU = float(os.environ.get("FACES_NMS_IOU", "0.3"))
TOP_K = int(os.environ.get("FACES_TOP_K", "5000"))

#: Crops smaller than this per side are skipped — YuNet cannot resolve them
#: and sizes this small are far below the 56 px POC face floor anyway.
MIN_CROP_SIDE = 16


class FaceDetector:
    """Owns one FaceDetectorYN instance; a lock serialises detect calls."""

    def __init__(self, model_path: Path = MODEL_PATH):
        """Create the YuNet detector; raises FileNotFoundError when weights absent."""
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
