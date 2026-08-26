"""Face embedding — a model FAMILY behind one contract (doc 15).

Two families behind one embed() shape:

  sface   — cv2.FaceRecognizerSF, the default since day one: `alignCrop`
            warps to the 112x112 template with the five landmarks, `feature`
            produces the 128-D embedding.
  arcface — the generic path doc 06 said an embedder swap requires: OUR
            alignment (align.py — the same 112 template, closed-form
            similarity transform) + a generic ONNX Runtime session,
            HECO_DEVICE capable, dimension read from the graph. Restricted
            tier in practice (the strong candidates are InsightFace-lineage,
            grant `contact`), and UNCALIBRATED until eval/sweep.py writes it
            a pack from our labels — the profile checker enforces both.

Embeddings are returned raw (not L2-normalised); the match service computes
cosine similarity, which is scale-invariant, so normalisation is its choice.

DELIBERATELY NO HOT-SWAP HERE. persons and faces gained POST /model; embed
did not, and it must not: an embedder change renames the per-site staff
store, invalidates every threshold in the match service, and re-teaches
what every cosine means (doc 15 M3). That is a DEPLOYMENT — env + stack
restart with HECO_EMBEDDER_ID and HECO_EMBEDDING_DIM travelling together —
never a request handler. The absence of the endpoint IS the guard.
"""

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from heco_common.ort import announce_device, providers_for

from .align import TEMPLATE_SIZE, align_face
from .face_row import face_to_row, validate_landmarks

#: Default model location — populated by `make models`, never committed.
DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "face_recognition_sface_2021dec.onnx"
)

#: The models this service knows how to run. Unknown names infer arcface
#: unless they say sface — the generic path is the right guess for a new
#: export, and the sface family only fits the one cv2 zoo model anyway.
MODEL_SPECS = {
    "face_recognition_sface_2021dec.onnx": {"family": "sface", "dim": 128},
}


def spec_for(model_path: Path) -> dict:
    """Resolve the family/dim spec for a model file (unknown names infer arcface)."""
    known = MODEL_SPECS.get(model_path.name)
    if known:
        return dict(known)
    family = "sface" if "sface" in model_path.name.lower() else "arcface"
    return {"family": family, "dim": None}  # arcface dim is read from the graph


# `or`, not a default argument: compose renders ${EMBED_MODEL-} as an empty
# string, and Path("") silently resolves to the working directory — the
# service then boots ok:false with model "" (bitten live on the T440 the
# night the passthrough shipped; review finding made flesh).
MODEL_PATH = Path(os.environ.get("EMBED_MODEL") or str(DEFAULT_MODEL))
DEVICE = os.environ.get("HECO_DEVICE") or "CPU"


def build_embedder(model_path: Path | None = None, device: str | None = None):
    """Build the configured family — one embed() contract, family chosen by the file."""
    path = model_path or MODEL_PATH
    spec = spec_for(path)
    if spec["family"] == "arcface":
        return ArcFaceEmbedder(path, device or DEVICE)
    return FaceEmbedder(path)


class FaceEmbedder:
    """The sface family: owns one FaceRecognizerSF; a lock serialises calls."""

    family = "sface"
    #: cv2 runs this family; the honest device answer is cv2's, not
    #: HECO_DEVICE's (the iGPU bench measured OpenCV ignoring targets).
    device_requested = "CPU"
    providers_active = ["cv2"]
    dim = 128

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


class ArcFaceEmbedder:
    """The arcface family: our alignment + a generic ORT session.

    Same embed() contract as FaceEmbedder — one raw embedding per face, in
    order — so the runner and match cannot tell the families apart on the
    wire (the DIMENSION difference is deliberate and travels as deployment
    config: HECO_EMBEDDING_DIM beside HECO_EMBEDDER_ID).
    """

    family = "arcface"

    def __init__(self, model_path: Path, device: str | None = None):
        """Build the ORT session and read dtype/layout/dim from the graph.

        Raises (ValueError) on any graph embed() could never feed — wrong
        input dtype, 1-channel input — because ORT only checks those at
        run() time: accepted here, they would be a green /health and a 500
        on every request.
        """
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — fetch it (restricted tier: make models-restricted) first"
            )
        import onnxruntime as ort  # deferred, like every family loader

        self.model_name = model_path.name
        providers, provider_options = providers_for(device)
        self._session = ort.InferenceSession(
            str(model_path), providers=providers, provider_options=provider_options
        )
        self.device_requested = (device or "CPU").upper()
        self.providers_active = list(self._session.get_providers())
        announce_device("embed", self.device_requested, self.providers_active)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        # The graph's input DTYPE read from the graph, like the layout below:
        # ORT checks dtype at run() time, not at session build, so a float16
        # export accepted here without this check would be green /health and
        # a 500 on every POST /embed — the healthy-but-dead shape the deploy
        # integrity guards exist to ban. Refusing at init instead rides the
        # lazy loader out as /health ok:false with the reason.
        itype = inp.type
        if itype == "tensor(float16)":
            self._np_dtype = np.float16
        elif itype == "tensor(float)":
            self._np_dtype = np.float32
        else:
            raise ValueError(
                f"{model_path.name} declares input dtype {itype} — this family feeds "
                "tensor(float) or tensor(float16); re-export the model with a float input"
            )
        # NCHW vs NHWC read from the graph, not guessed from the filename:
        # both layouts exist in the wild and a silent transpose error embeds
        # noise that COSINES happily compare.
        shape = list(inp.shape)
        self._nchw = len(shape) == 4 and shape[1] in (1, 3)
        # A 1-channel graph builds a session fine but can never accept the
        # 3-channel RGB blob embed() feeds — same green-health/always-500
        # trap as the dtype hole, refused the same way. Both layouts get the
        # channel check, and static spatial dims must match the aligned crop
        # (ORT checks dimensions at run() time, not session creation, so a
        # 224x224 export would pass init and then 500 every request).
        channels = shape[1] if self._nchw else (shape[3] if len(shape) == 4 else None)
        if isinstance(channels, int) and channels != 3:
            raise ValueError(
                f"{model_path.name} expects {channels}-channel input; embed feeds "
                "3-channel RGB — deploy a 3-channel export"
            )
        spatial = shape[2:4] if self._nchw else shape[1:3]
        static = [d for d in spatial if isinstance(d, int)]
        if static and any(d != TEMPLATE_SIZE for d in static):
            raise ValueError(
                f"{model_path.name} declares a {'x'.join(str(d) for d in spatial)} input; "
                f"alignment produces {TEMPLATE_SIZE}x{TEMPLATE_SIZE} crops — "
                "deploy a matching export"
            )
        out_shape = self._session.get_outputs()[0].shape
        self.dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else None
        self._lock = threading.Lock()

    def embed(self, img: np.ndarray, faces: list[dict]) -> tuple[list[list[float]], float]:
        """Align (our transform) + embed every face; order preserved."""
        t0 = time.perf_counter()
        embeddings: list[list[float]] = []
        with self._lock:
            for face in faces:
                # Same landmark guard as the sface path (face_to_row):
                # np.reshape would happily re-pair ANY nesting totalling ten
                # floats into wrong (x, y) points and embed the garbage crop
                # with 200 OK — malformed landmarks must be the same
                # ValueError -> 400 on both families.
                aligned = align_face(img, validate_landmarks(face))
                rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                blob = (rgb.astype(np.float32) - 127.5) / 127.5
                # Cast to the graph's declared dtype (validated at init):
                # ORT refuses a float32 blob on a float16 graph at run().
                blob = blob.astype(self._np_dtype, copy=False)
                blob = blob.transpose(2, 0, 1)[None] if self._nchw else blob[None]
                feat = self._session.run(
                    None, {self._input_name: np.ascontiguousarray(blob)}
                )[0]
                embeddings.append([float(v) for v in np.asarray(feat).ravel()])
        align_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return embeddings, align_ms
