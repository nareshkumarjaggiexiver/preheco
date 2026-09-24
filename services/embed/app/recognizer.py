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
The raw vector's L2 NORM rides beside it ("norms", index-parallel): for
both families the feature magnitude tracks crop quality — a blurred,
occluded or averted face embeds short — and the runner can floor on it
(HECO_QUALITY_MIN_FEAT_NORM) once the vector is normalised away. Both
families expose a raw vector: cv2's SFace `feature` is unnormalised too
(its `match` normalises internally), so the norm is meaningful there as
well.

The optional ATTRIBUTE PASS (attributes.py — gender + age) runs in the same
loop under the same lock, one extra session run per face; the lock already
serialises the embedder, so a second lock would only add a place to
deadlock.

BATCHING (EMBED_BATCH=1, default off). With whole-frame SCRFD on a 4K
wedding frame a request carries several faces at once, and per-face runs
pay the launch and host<->device copies once per face. On an arcface graph
with a DYNAMIC batch dimension the switch runs a request's faces through
one session.run (chunks of BATCH_MAX), and the attribute pass likewise;
order is preserved and every list stays index-parallel. A static-batch
graph cannot, so it keeps the per-face loop and /health says so
(knobs.batch.active false). Off, the loop below is today's, byte for byte.

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
from heco_common.ort import (
    announce_device,
    is_trt,
    providers_for,
    trt_batch_profile,
    trt_truth,
)

from .align import TEMPLATE_SIZE, align_face
from .attributes import AttributeModel
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
#: EMBED_BATCH=1: one session.run per request instead of one per face (see
#: the module docstring). Anything but "1" is off — today's loop.
BATCH = (os.environ.get("EMBED_BATCH") or "").strip() == "1"
#: Faces per run when batching; a request with more runs in chunks. Also the
#: TensorRT optimisation profile's ceiling, so a batch never forces a rebuild.
BATCH_MAX = 16


def build_embedder(model_path: Path | None = None, device: str | None = None):
    """Build the configured family — one embed() contract, family chosen by the file."""
    path = model_path or MODEL_PATH
    spec = spec_for(path)
    if spec["family"] == "arcface":
        return ArcFaceEmbedder(path, device or DEVICE)
    return FaceEmbedder(path)


class _Embedder:
    """The loop both families share: one lock, request order, norms, attributes.

    A family supplies `_feature(img, face)` — the raw vector for one face —
    and inherits the two public shapes: `embed()` (the day-one contract,
    embeddings + alignMs) and `embed_faces()` (the same loop also answering
    norms and, given an AttributeModel, gender/age per face).
    """

    family = "unknown"
    _lock: threading.Lock
    #: Batching as asked (EMBED_BATCH) and as it can actually run on this
    #: graph. The cv2 sface family never batches.
    batch_requested = False
    batch_active = False

    def _feature(self, img: np.ndarray, face: dict) -> np.ndarray:
        raise NotImplementedError

    def _features(self, img: np.ndarray, faces: list[dict]) -> list[np.ndarray]:
        """Every face's raw vector, in order — batched families override."""
        return [self._feature(img, face) for face in faces]

    def embed(self, img: np.ndarray, faces: list[dict]) -> tuple[list[list[float]], float]:
        """Align + embed every face; `(embeddings, align_ms)`, order preserved."""
        embeddings, _norms, _attrs, align_ms, _attr_ms = self.embed_faces(img, faces)
        return embeddings, align_ms

    def embed_faces(
        self, img: np.ndarray, faces: list[dict], attributes: AttributeModel | None = None
    ) -> tuple[list[list[float]], list[float], list[dict | None] | None, float, float | None]:
        """Run the full /embed loop: `(embeddings, norms, attributes, align_ms, attr_ms)`.

        Every list is index-parallel to `faces`. `attributes` is None (not
        an empty list) when no model was given — absent is not zero, and
        the wire answer "attributes": null must mean "not measured", never
        "no faces". `align_ms` stays the align+embed time alone, as the
        contract always defined it; the attribute pass is timed apart
        (`attr_ms`, None when off) so the new cost is visible on its own
        and the old number keeps meaning what it meant.
        """
        t0 = time.perf_counter()
        embeddings: list[list[float]] = []
        norms: list[float] = []
        attrs: list[dict | None] = []
        attr_s = 0.0
        if self.batch_active and faces:
            with self._lock:
                for feat in self._features(img, faces):
                    feat = np.asarray(feat).ravel()
                    embeddings.append([float(v) for v in feat])
                    norms.append(float(np.linalg.norm(feat.astype(np.float64))))
                if attributes is not None:
                    ta = time.perf_counter()
                    attrs = attributes.predict_many(img, [face.get("box") for face in faces])
                    attr_s = time.perf_counter() - ta
            total_ms = (time.perf_counter() - t0) * 1000.0
            align_ms = round(total_ms - attr_s * 1000.0, 2)
            attr_ms = round(attr_s * 1000.0, 2) if attributes is not None else None
            return embeddings, norms, (attrs if attributes is not None else None), align_ms, attr_ms
        with self._lock:
            for face in faces:
                feat = np.asarray(self._feature(img, face)).ravel()
                embeddings.append([float(v) for v in feat])
                norms.append(float(np.linalg.norm(feat.astype(np.float64))))
                if attributes is not None:
                    ta = time.perf_counter()
                    attrs.append(attributes.predict(img, face.get("box")))
                    attr_s += time.perf_counter() - ta
        total_ms = (time.perf_counter() - t0) * 1000.0
        align_ms = round(total_ms - attr_s * 1000.0, 2)
        attr_ms = round(attr_s * 1000.0, 2) if attributes is not None else None
        return embeddings, norms, (attrs if attributes is not None else None), align_ms, attr_ms


class FaceEmbedder(_Embedder):
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

    def _feature(self, img: np.ndarray, face: dict) -> np.ndarray:
        """cv2's alignCrop (its 112 template, the five landmarks) then `feature`."""
        aligned = self._rec.alignCrop(img, face_to_row(face))
        return np.asarray(self._rec.feature(aligned))


class ArcFaceEmbedder(_Embedder):
    """The arcface family: our alignment + a generic ORT session.

    Same embed() contract as FaceEmbedder — one raw embedding per face, in
    order — so the runner and match cannot tell the families apart on the
    wire (the DIMENSION difference is deliberate and travels as deployment
    config: HECO_EMBEDDING_DIM beside HECO_EMBEDDER_ID).
    """

    family = "arcface"

    def __init__(self, model_path: Path, device: str | None = None, batch: bool | None = None):
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
        if not is_trt(device):
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
        # Batching needs a DYNAMIC batch dimension; a static one (an int)
        # keeps the per-face loop, and /health's knobs block says so.
        self.batch_requested = BATCH if batch is None else bool(batch)
        self.batch_active = self.batch_requested and not isinstance(shape[0], int)
        if is_trt(device) and self.batch_active and "TensorrtExecutionProvider" in (
            self.providers_active
        ):
            # TensorRT builds an engine per input-shape RANGE: without an
            # explicit 1..BATCH_MAX profile every new batch size seen in a
            # live count would rebuild the engine (tens of seconds, inside a
            # request). Rebuilt here with the profile, once, at load.
            sample = (3, TEMPLATE_SIZE, TEMPLATE_SIZE) if self._nchw else (
                TEMPLATE_SIZE, TEMPLATE_SIZE, 3)
            providers, provider_options = providers_for(
                device, trt_batch_profile(inp.name, sample, BATCH_MAX))
            self._session = ort.InferenceSession(
                str(model_path), providers=providers, provider_options=provider_options
            )
            self.providers_active = list(self._session.get_providers())
        self.trt = None
        # TensorRT builds (or deserialises) its engine at the first run of a
        # shape: pay it HERE, inside the load /health waits on, never in the
        # first /embed of a live count (the runner's stage timeout is 30 s).
        if is_trt(device):
            zeros = np.zeros(
                (1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE) if self._nchw
                else (1, TEMPLATE_SIZE, TEMPLATE_SIZE, 3),
                dtype=self._np_dtype,
            )
            self._session.run(None, {self._input_name: zeros})
            # After, not before: a failed engine build inside run() makes
            # ORT rebuild the session on CUDA without raising.
            self.providers_active = list(self._session.get_providers())
            self.trt = trt_truth(self._session)
            announce_device("embed", self.device_requested, self.providers_active)

    def _feature(self, img: np.ndarray, face: dict) -> np.ndarray:
        """Our alignment, the normalised RGB blob, one session run."""
        return np.asarray(
            self._session.run(None, {self._input_name: self._blob(img, face)})[0]
        )

    def _blob(self, img: np.ndarray, face: dict) -> np.ndarray:
        """One face's (1, ...) input tensor: aligned, RGB, normalised, graph dtype."""
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
        return np.ascontiguousarray(blob)

    def _features(self, img: np.ndarray, faces: list[dict]) -> list[np.ndarray]:
        """All faces through as few runs as BATCH_MAX allows, order preserved.

        Every blob is built (and every landmark validated) BEFORE the first
        run, so a malformed face is the same ValueError -> 400 it always was.
        """
        if not self.batch_active:
            return super()._features(img, faces)
        blobs = [self._blob(img, face) for face in faces]
        out: list[np.ndarray] = []
        for start in range(0, len(blobs), BATCH_MAX):
            batch = np.concatenate(blobs[start:start + BATCH_MAX], axis=0)
            rows = np.asarray(self._session.run(None, {self._input_name: batch})[0])
            out.extend(rows[i:i + 1] for i in range(rows.shape[0]))
        return out
