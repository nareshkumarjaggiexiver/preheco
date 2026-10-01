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

A SECOND FAMILY, faceage (2026-10-01). genderage's age spread over one
guest's clean views on the Sharon clip was 10.4 years (43 guests, 211
crops), too wide for any age rule beyond child-against-adult. faceage-dino
(DINOv3 ViT-L/16 backbone, CORAL ordinal age head, gender head; MAE 3.56
years on LAGENDA) read the same crops with a 3.6-year spread, and a
15-year gap between two guests' medians separated 49 % of the 703 pairs
with 0 misfires in 36 guests. It is the same pass with a different crop,
normalisation and decoding — `Family` holds the differences and the graph's
own outputs say which family a file is, so EMBED_ATTR_MODEL simply names
the other file. It costs 14.6 ms a face on TensorRT (fp32 compute, see
`Family.trt_fp16`) against 1.5 for genderage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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

#: ImageNet's channel mean and standard deviation (RGB, on 0..1 pixels):
#: what faceage's preprocessor normalises with.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Family:
    """One attribute graph family: how it is fed and how its answer is read.

    `size` is the square crop; `margin` its side as a multiple of the box's
    longer side; `output_widths` the per-face width of each graph output in
    graph order — which is also how a file is told apart at load (see
    `family_of`): one output 3 wide is genderage, two outputs 100 and 2
    wide are faceage, anything else is refused by name.
    """

    name: str
    size: int
    margin: float
    output_widths: tuple[int, ...]
    #: 0..1 then ImageNet mean/std (faceage), or RGB 0..255 as-is (genderage
    #: normalises inside its graph).
    imagenet: bool
    #: Bicubic resampling (the faceage card's); genderage's warp is bilinear.
    cubic: bool
    #: fp16 compute on TensorRT. faceage's DINOv3 backbone carries one
    #: residual channel near 1.57e5 — past fp16's 65 504 — and a fp16
    #: engine answers NaN for every face (measured 2026-10-01 on the box's
    #: 4060, ORT 1.30 / TensorRT 10.16); its engine is built with fp32
    #: compute, and the weights stay the file's fp16.
    trt_fp16: bool


#: InsightFace genderage: a 96 px crop at 1.5x the box, one output
#: [F logit, M logit, age/100].
GENDERAGE = Family("genderage", INPUT_SIZE, BOX_MARGIN, (OUTPUT_DIM,), False, False, True)
#: faceage-dino: a 224 px crop at 1.2x the box's longer side — the card asks
#: for 10 % padding on each side of the box (without it, its error grows
#: from 3.56 to 3.76 years), and the square at 1.2x the longer side is the
#: crop the 2026-10-01 calibration measured with — ImageNet-normalised; two
#: outputs, `age_logits` (100 CORAL thresholds, age = the sum of their
#: sigmoids) and `gender_logits` (0 female, 1 male).
FACEAGE = Family("faceage", 224, 1.2, (100, 2), True, True, False)
FAMILIES = (GENDERAGE, FACEAGE)


def family_of(name: str, output_widths: tuple) -> Family:
    """The family a graph's output widths name, or ValueError.

    A width the graph leaves dynamic (None) matches anything in its place,
    as the genderage check always allowed; the refusal names what each
    family answers, so an embedder dropped into EMBED_ATTR_MODEL is told
    apart from a mis-exported attribute graph by the message alone.
    """
    for family in FAMILIES:
        if len(output_widths) == len(family.output_widths) and all(
            w is None or w == want for w, want in zip(output_widths, family.output_widths)
        ):
            return family
    if len(output_widths) == 1:
        raise ValueError(
            f"{name} answers {output_widths[0]} values per face; a genderage graph answers "
            f"{OUTPUT_DIM} ([F logit, M logit, age/100]) and a faceage graph answers "
            "age_logits[100] and gender_logits[2]"
        )
    raise ValueError(
        f"{name} answers {len(output_widths)} outputs {list(output_widths)} wide; a "
        f"genderage graph answers one output {OUTPUT_DIM} wide and a faceage graph two, "
        "100 (age_logits) and 2 (gender_logits)"
    )


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


def attribute_transform(box, size: int = INPUT_SIZE, margin: float = BOX_MARGIN) -> np.ndarray:
    """Build the 2x3 affine of InsightFace's `face_align.transform(..., rotation 0)`.

    Uniform scale about the box centre, then translate the centre to the
    window's middle: `margin` times the box's longer side fills the `size`
    window. Pure, so the crop is testable at value level against the
    documented formula.
    """
    cx, cy, side = box_geometry(box)
    scale = size / (side * margin)
    return np.array(
        [[scale, 0.0, size / 2.0 - cx * scale], [0.0, scale, size / 2.0 - cy * scale]],
        dtype=np.float32,
    )


def crop_face(img: np.ndarray, box, size: int = INPUT_SIZE, margin: float = BOX_MARGIN,
              cubic: bool = False) -> np.ndarray:
    """Warp out the `size` x `size` BGR crop the attribute graph consumes."""
    return cv2.warpAffine(
        img, attribute_transform(box, size, margin), (size, size),
        flags=cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR, borderValue=0.0,
    )


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


def postprocess_faceage(age_logits, gender_logits) -> dict:
    """faceage's `age_logits` (100 CORAL thresholds) + `gender_logits` -> {gender, genderP, age}.

    CORAL: threshold k answers "older than k years?", and the age is the
    sum of the thresholds' sigmoids — a continuous 0..100, the card's own
    decoding. The sigmoid is written as tanh so a threshold in the
    thousands cannot overflow exp(). Gender reads as genderage's: argmax,
    genderP the max-shifted softmax mass of the reported gender.
    """
    ages = np.asarray(age_logits, dtype=np.float64).ravel()
    sexes = np.asarray(gender_logits, dtype=np.float64).ravel()
    want_age, want_sex = FACEAGE.output_widths
    if ages.shape[0] != want_age or sexes.shape[0] != want_sex:
        raise ValueError(
            f"faceage output has {ages.shape[0]} age and {sexes.shape[0]} gender values, "
            f"expected {want_age} and {want_sex}"
        )
    age = float(np.sum(0.5 * (1.0 + np.tanh(0.5 * ages))))
    logits = sexes - sexes.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = int(np.argmax(sexes))
    return {"gender": GENDER_LABELS[idx], "genderP": float(probs[idx]), "age": age}


class AttributeModel:
    """One attribute ORT session (either family); `predict()` answers for one face box.

    Built like ArcFaceEmbedder: providers from HECO_DEVICE, the device
    truth announced and served, and every graph shape this class could
    never feed refused at init — ORT checks dtype and dimensions at run()
    time, so a wrong export accepted here would be a green /health and a
    500 on every /embed, the exact healthy-but-dead shape the deploy
    integrity guards ban. Refusing at init rides the lazy loader out as
    /health attrError with the reason.
    """

    def __init__(self, model_path: Path, device: str | None = None, batch: bool = False,
                 batch_max: int = 16):
        """Build the session and read dtype/layout/dims from the graph.

        `batch` (the embedder's EMBED_BATCH) lets predict_many answer a whole
        request in one run when the graph's batch dimension is dynamic.
        """
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — place InsightFace genderage.onnx there or set "
                "EMBED_ATTR_MODEL (see the embed README)"
            )
        import onnxruntime as ort  # deferred, like every family loader

        self.model_name = model_path.name
        # `model`: a TensorRT engine keyed on the weights (heco_common.ort).
        # The family — and with it the TensorRT options it needs — is read
        # off this session's signature; a family needing other options is
        # rebuilt below, before any run, together with the batch profile.
        providers, provider_options = providers_for(device, model=model_path)
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
        if not is_trt(device):
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
        # The outputs name the family (the input shape alone cannot: both
        # families are an RGB square), and the family says how big the
        # input must be and how the answer is read.
        outputs = self._session.get_outputs()
        self.family = family_of(model_path.name, tuple(
            o.shape[-1] if o.shape and isinstance(o.shape[-1], int) else None for o in outputs
        ))
        self._output_index = {width: i for i, width in enumerate(self.family.output_widths)}
        size = self.family.size
        spatial = shape[2:4] if self._nchw else shape[1:3]
        static = [d for d in spatial if isinstance(d, int)]
        if static and any(d != size for d in static):
            raise ValueError(
                f"{model_path.name} declares a {'x'.join(str(d) for d in spatial)} input; "
                f"the {self.family.name} crop is {size}x{size}"
            )
        self.batch_max = int(batch_max)
        self.batch_active = bool(batch) and not isinstance(shape[0], int)
        sample = (3, size, size) if self._nchw else (size, size, 3)
        trt_extra: dict = {}
        if is_trt(device) and "TensorrtExecutionProvider" in self.providers_active:
            if self.batch_active:
                # Same reason as the embedder: an explicit 1..max profile, or
                # every new batch size rebuilds the engine inside a request.
                trt_extra.update(trt_batch_profile(inp.name, sample, self.batch_max))
            if not self.family.trt_fp16:
                trt_extra["trt_fp16_enable"] = "False"
        if trt_extra:
            # One rebuild with everything the family and the batch need,
            # before the warm-up: the engine is built once, the right way.
            providers, provider_options = providers_for(device, trt_extra, model=model_path)
            self._session = ort.InferenceSession(
                str(model_path), options, providers=providers, provider_options=provider_options
            )
            self.providers_active = list(self._session.get_providers())
        #: TensorRT's own read-back (trt_truth) — None off TRT and whenever
        #: the engine did not come up; embed /health serves it beside the
        #: embedder's under device.attributes.
        self.trt = None
        if is_trt(device):
            # Build (or load) the TensorRT engine at load, not in a request —
            # and read the truth after it (a failed build drops ORT to CUDA).
            self._session.run(None, {self._input_name: np.zeros((1, *sample), self._np_dtype)})
            self.providers_active = list(self._session.get_providers())
            self.trt = trt_truth(self._session)
            announce_device("embed-attributes", self.device_requested, self.providers_active)

    def blob(self, img: np.ndarray, box) -> np.ndarray:
        """Build the exact tensor fed to the graph: the family's crop, RGB, graph dtype and layout."""
        fam = self.family
        rgb = cv2.cvtColor(crop_face(img, box, fam.size, fam.margin, fam.cubic), cv2.COLOR_BGR2RGB)
        if fam.imagenet:
            arr = ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).astype(self._np_dtype)
        else:
            arr = rgb.astype(self._np_dtype)  # no mean/std: the graph normalises itself
        arr = arr.transpose(2, 0, 1)[None] if self._nchw else arr[None]
        return np.ascontiguousarray(arr)

    def _decode(self, outputs: list, row: int) -> dict:
        """One face's answer, read from the graph outputs the family's way."""
        if self.family is FACEAGE:
            age_i, sex_i = (self._output_index[w] for w in FACEAGE.output_widths)
            return postprocess_faceage(outputs[age_i][row], outputs[sex_i][row])
        return postprocess(np.asarray(outputs[0])[row])

    def predict(self, img: np.ndarray, box) -> dict:
        """{gender, genderP, age} for the face in `box` of the BGR frame `img`."""
        outputs = self._session.run(None, {self._input_name: self.blob(img, box)})
        return self._decode(outputs, 0)

    def predict_many(self, img: np.ndarray, boxes: list) -> list[dict]:
        """predict() for every box, in order — one run per batch_max when batching.

        Every crop (and every box validation) happens before the first run,
        so a malformed box is the same ValueError it is in predict().
        """
        if not self.batch_active:
            return [self.predict(img, box) for box in boxes]
        blobs = [self.blob(img, box) for box in boxes]
        out: list[dict] = []
        for start in range(0, len(blobs), self.batch_max):
            batch = np.concatenate(blobs[start:start + self.batch_max], axis=0)
            outputs = self._session.run(None, {self._input_name: batch})
            out.extend(self._decode(outputs, row) for row in range(batch.shape[0]))
        return out


def build_attributes(model_path: Path | None = None, device: str | None = None,
                     batch: bool = False, batch_max: int = 16) -> AttributeModel:
    """Build the attribute model at `model_path` (default: the env-resolved one)."""
    path = model_path or ATTR_MODEL_PATH
    if path is None:
        raise ValueError("attribute model is switched off (EMBED_ATTR_MODEL=off)")
    return AttributeModel(path, device, batch, batch_max)
