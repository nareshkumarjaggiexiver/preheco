"""cv2.FaceRecognizerSF (SFace) wrapper: align by landmarks, then embed.

`alignCrop` warps the face to the 112x112 template using the five landmarks,
`feature` produces the 128-D embedding. Embeddings are returned raw (not
L2-normalised); the match service computes cosine similarity, which is
scale-invariant, so normalisation is its choice.
"""

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .face_row import face_to_row

#: Default model location — populated by `make models`, never committed.
DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "face_recognition_sface_2021dec.onnx"
)

MODEL_PATH = Path(os.environ.get("EMBED_MODEL", str(DEFAULT_MODEL)))


class FaceEmbedder:
    """Owns one FaceRecognizerSF instance; a lock serialises embed calls."""

    def __init__(self, model_path: Path = MODEL_PATH):
        """Create the SFace recognizer; raises FileNotFoundError when weights absent."""
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — run `make models` in services/embed"
            )
        self.model_name = model_path.name
        self._rec = cv2.FaceRecognizerSF.create(str(model_path), "")
        self._lock = threading.Lock()

    def embed(self, img: np.ndarray, faces: list[dict]) -> tuple[list[list[float]], float]:
        """Align + embed every face against `img` (BGR frame).

        Returns `(embeddings, align_ms)`: one 128-float list per input face,
        order preserved; `align_ms` is the wall time of the full align+embed
        loop (the contract's alignMs).
        """
        t0 = time.perf_counter()
        embeddings: list[list[float]] = []
        with self._lock:
            for face in faces:
                row = face_to_row(face)
                aligned = self._rec.alignCrop(img, row)
                feat = self._rec.feature(aligned)
                embeddings.append([float(v) for v in np.asarray(feat).ravel()])
        align_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return embeddings, align_ms
