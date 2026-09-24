"""Face attributes — gender and age from an InsightFace-style genderage graph.

WHY. Run f0bfc5 (a Punjab wedding-hall overview camera, 74 guests) flooded
its review queue with 500 pairs, and pair #3 put a man in a blue shirt
beside an elderly woman in glasses at face cosine 0.345. Nothing in the
pipeline knew gender or age, so nothing could set that pair aside. This
module is that knowledge: one extra session run per face, inside the
embedder's lock, returned beside the embedding so the match service can
store it per template and the review queue can print
"man / woman · ages 41 / 63" next to the cosine.

THE MODEL is InsightFace buffalo_l's genderage.onnx: 1x3x96x96 RGB float
in, 1x3 out = [female logit, male logit, age/100]. Its graph carries its
own `_minusscalar0` / `_mulscalar0` normalisation at the top (checked on
the file itself), so the blob is fed as plain 0..255 — exactly what
insightface's Attribute does for a graph with Sub/Mul in its first nodes
(input_mean 0, input_std 1). The CROP is InsightFace's too: centred on the
face BOX, scale 96 / (1.5 * max(w, h)), no rotation — deliberately NOT the
landmark-aligned 112 crop the embedder uses, because the attribute net was
trained on this looser box crop and reads hairline, jaw and neck that the
identity alignment trims away.

OPTIONAL BY DESIGN. EMBED_ATTR_MODEL names the file; unset, the default
<embed>/models/genderage.onnx is used when it exists and the pass is off
otherwise, and the literal `off` disables it even when the default file is
present. Off, /embed answers "attributes": null and the review line says
"not measured" — absent is not zero, and a missing attribute must never
read as a confident answer.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from heco_common.ort import announce_device, providers_for

#: The default weight location — beside the embedder's, gitignored, never
#: committed (InsightFace weights are restricted-tier: see the README).
DEFAULT_ATTR_MODEL = Path(__file__).resolve().parent.parent / "models" / "genderage.onnx"

#: The genderage input frame, and the InsightFace crop margin: the box is
#: scaled so that 1.5x its longer side fills the 96 px window.
INPUT_SIZE = 96
BOX_MARGIN = 1.5

#: The graph's output order. InsightFace: `gender = np.argmax(pred[:2])`,
#: 0 female / 1 male; `age = pred[2] * 100`.
GENDER_LABELS = ("F", "M")
OUTPUT_DIM = 3


def attr_model_from_env(value: str | None) -> tuple[Path | None, bool]:
    """Resolve EMBED_ATTR_MODEL to `(path, explicit)`.

    `explicit` says whether an operator NAMED the file: a named file that
    fails to load is an error worth surfacing in /health, whereas the
    default file simply being absent is the pass being off. `or ""`, not a
    default argument, for the same reason as EMBED_MODEL — compose renders
    an unset ${EMBED_ATTR_MODEL-} as an empty string, and Path("") is the
    working directory, not "unset".
    """
    text = (value or "").strip()
    if text.lower() == "off":
        return None, True
    if text:
        return Path(text), True
    return DEFAULT_ATTR_MODEL, False


ATTR_MODEL_PATH, ATTR_MODEL_EXPLICIT = attr_model_from_env(os.environ.get("EMBED_ATTR_MODEL"))


def box_geometry(box) -> tuple[float, float, float]:
    """`(cx, cy, side)` of a contract face box, or ValueError.

    Validated here and not left to numpy because a box with a missing key
    or a zero side would otherwise become a NaN scale, a black crop and a
    confident-looking gender for nothing — the same 200-OK-garbage shape
    the landmark guard exists to ban. A malformed box is a contract
    violation from the faces service, and it 400s like malformed landmarks.
    """
    if not isinstance(box, dict) or not all(k in box for k in ("x", "y", "w", "h")):
        raise ValueError("face.box must have x, y, w, h")
    try:
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    except (TypeError, ValueError) as exc:
        raise ValueError("face.box x, y, w, h must be numbers") from exc
    side = max(w, h)
    if not np.isfinite([x, y, w, h]).all() or side <= 0.0:
        raise ValueError("face.box w, h must be finite and one of them positive")
    return x + w / 2.0, y + h / 2.0, side


def attribute_transform(box, size: int = INPUT_SIZE) -> np.ndarray:
    """Build the 2x3 affine of InsightFace's `face_align.transform(..., rotation 0)`.

    Uniform scale about the box centre, then translate the centre to the
    window's middle. Pure, so the crop is testable at value level against
    the documented formula.
    """
    cx, cy, side = box_geometry(box)
    scale = size / (side * BOX_MARGIN)
    return np.array(
        [[scale, 0.0, size / 2.0 - cx * scale], [0.0, scale, size / 2.0 - cy * scale]],
        dtype=np.float32,
    )


def crop_face(img: np.ndarray, box, size: int = INPUT_SIZE) -> np.ndarray:
    """Warp out the `size` x `size` BGR crop the attribute graph consumes."""
    return cv2.warpAffine(img, attribute_transform(box, size), (size, size), borderValue=0.0)


def postprocess(pred) -> dict:
    """[F logit, M logit, age/100] -> {gender, genderP, age}.

    genderP is the softmax probability OF THE REPORTED GENDER (so it is
    always >= 0.5), computed max-shifted: a graph is free to answer with
    logits in the hundreds and exp() must not overflow into NaN. Age is a
    float in years, unrounded — the match service takes medians, and
    rounding before a median throws away real signal.
    """
    out = np.asarray(pred, dtype=np.float64).ravel()
    if out.shape[0] != OUTPUT_DIM:
        raise ValueError(f"genderage output has {out.shape[0]} values, expected {OUTPUT_DIM}")
    logits = out[:2] - out[:2].max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = int(np.argmax(out[:2]))
    return {
        "gender": GENDER_LABELS[idx],
        "genderP": float(probs[idx]),
        "age": float(out[2] * 100.0),
    }


class AttributeModel:
    """One genderage ORT session; `predict()` answers for one face box.

    Built like ArcFaceEmbedder: providers from HECO_DEVICE, the device
    truth announced and served, and every graph shape this class could
    never feed refused at init — ORT checks dtype and dimensions at run()
    time, so a wrong export accepted here would be a green /health and a
    500 on every /embed, the exact healthy-but-dead shape the deploy
    integrity guards ban. Refusing at init rides the lazy loader out as
    /health attrError with the reason.
    """

    def __init__(self, model_path: Path, device: str | None = None):
        """Build the session and read dtype/layout/dims from the graph."""
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — place InsightFace genderage.onnx there or set "
                "EMBED_ATTR_MODEL (see the embed README)"
            )
        import onnxruntime as ort  # deferred, like every family loader

        self.model_name = model_path.name
        providers, provider_options = providers_for(device)
        # ONE intra-op thread, deliberately. ORT's default pool (one thread
        # per core) spin-waits after every run, and inside the embedder's
        # loop those spinning threads steal the cores cv2's SFace is about
        # to use: measured 2026-09-24 on this 8-thread box, 4 faces on a
        # 640x480 frame, p50 wall per /embed — pass off 38 ms, pass on with
        # the default pool 69 ms (alignMs itself 29 -> 54), pass on with
        # intra_op_num_threads=1 40.5 ms. A 96x96 net does not need a pool;
        # single-threaded it also answers faster (attrMs 4.4 -> 3.5 ms).
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), options, providers=providers, provider_options=provider_options
        )
        self.device_requested = (device or "CPU").upper()
        self.providers_active = list(self._session.get_providers())
        announce_device("embed-attributes", self.device_requested, self.providers_active)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        itype = inp.type
        if itype == "tensor(float16)":
            self._np_dtype = np.float16
        elif itype == "tensor(float)":
            self._np_dtype = np.float32
        else:
            raise ValueError(
                f"{model_path.name} declares input dtype {itype} — the attribute pass feeds "
                "tensor(float) or tensor(float16)"
            )
        shape = list(inp.shape)
        self._nchw = len(shape) == 4 and shape[1] in (1, 3)
        channels = shape[1] if self._nchw else (shape[3] if len(shape) == 4 else None)
        if isinstance(channels, int) and channels != 3:
            raise ValueError(
                f"{model_path.name} expects {channels}-channel input; the attribute pass "
                "feeds 3-channel RGB"
            )
        spatial = shape[2:4] if self._nchw else shape[1:3]
        static = [d for d in spatial if isinstance(d, int)]
        if static and any(d != INPUT_SIZE for d in static):
            raise ValueError(
                f"{model_path.name} declares a {'x'.join(str(d) for d in spatial)} input; "
                f"the attribute crop is {INPUT_SIZE}x{INPUT_SIZE}"
            )
        out_shape = list(self._session.get_outputs()[0].shape)
        if out_shape and isinstance(out_shape[-1], int) and out_shape[-1] != OUTPUT_DIM:
            raise ValueError(
                f"{model_path.name} answers {out_shape[-1]} values per face; a genderage "
                f"graph answers {OUTPUT_DIM} ([F logit, M logit, age/100])"
            )

    def blob(self, img: np.ndarray, box) -> np.ndarray:
        """Build the exact tensor fed to the graph: RGB, 0..255, graph dtype and layout."""
        rgb = cv2.cvtColor(crop_face(img, box), cv2.COLOR_BGR2RGB)
        arr = rgb.astype(self._np_dtype)  # no mean/std: the graph normalises itself
        arr = arr.transpose(2, 0, 1)[None] if self._nchw else arr[None]
        return np.ascontiguousarray(arr)

    def predict(self, img: np.ndarray, box) -> dict:
        """{gender, genderP, age} for the face in `box` of the BGR frame `img`."""
        pred = self._session.run(None, {self._input_name: self.blob(img, box)})[0]
        return postprocess(pred)


def build_attributes(model_path: Path | None = None, device: str | None = None) -> AttributeModel:
    """Build the attribute model at `model_path` (default: the env-resolved one)."""
    path = model_path or ATTR_MODEL_PATH
    if path is None:
        raise ValueError("attribute model is switched off (EMBED_ATTR_MODEL=off)")
    return AttributeModel(path, device)
