"""Head covering — SigLIP B/16 zero-shot reads: turban, bare, dupatta or scarf, cap or hat.

WHY. Run 8b8b87's review queue put a Sikh man in a sky-blue turban beside a
bare-headed man (p00005/p00009), and nothing in the pipeline could say they
were two people: the runner's colour head descriptor cannot tell a dark
turban from black hair, so the match service's head rule compares headwear
against headwear only. SigLIP B/16 reads the head semantically. Offline, on
461 head crops labelled by eye from the same wedding (siglip evaluation,
2026-09-25), no confident call was wrong — 31 turban, 312 bare — and every
turban it did not call was "unsure", never bare.

WHAT THIS MODULE OWNS: everything between a picture and the numbers. The
caller sends an image (the whole frame, or a context crop of it) and each
face's detector box in that image's pixels; this module cuts the TWO views
of each head, preprocesses them, runs the image tower and scores the views
against the prompt set. The geometry, the preprocessing and the scoring are
the reference reader's (siglip ``onnx/headwear_ref.py``) line for line,
because they are part of the contract: swapping ``cv2.INTER_AREA`` for PIL's
bicubic resize changed 16 of 461 calls and made one false turban (a girl
with a hair bow). ``tests/headwear_ref.py`` is that file, verbatim, and the
tests pin this module against it.

  loose  the face box widened 0.30 face widths a side, raised 1.0 face
         heights, down to the face bottom — sees a whole turban;
  tight  0.15 a side, raised 0.8, everything outside the inscribed ellipse
         grey (127) — a neighbour's turban in a corner is masked.

Each view is letterboxed to a square with grey 127, BGR -> RGB, resized to
224 with ``cv2.INTER_AREA``, scaled to [-1, 1]; the tower answers an
L2-normalised 768-d embedding. A prompt's logit is ``scale * cos + bias``
(117.33, -12.93: the checkpoint's own), a class's logit the MAX over its
prompts. The reading is 8 floats — turban, bare, dupatta_or_scarf,
cap_or_hat of the loose view, then of the tight view — LOGITS, not a
verdict: the match service makes the call at review time from config.

THE FILES. ``EMBED_HEADWEAR_MODEL`` names the image graph
(``siglip_b16_224_image_fp32.onnx``, 372 MB, never committed);
``headwear_prompts.json`` and ``headwear_text_embeds.npy`` sit beside it.
The JSON records both other files' sha256 and the loader refuses a set that
does not match — logits from another graph or prompt set are on another
scale. ``stamp`` (graph sha256[:12] + "+" + prompt JSON sha256[:12]) names
the set on every reply; the match service keeps one per gallery.

OFF UNLESS NAMED. Unset, empty or ``off``, the reader does not exist: no
session, no /health key, and POST /headwear answers exactly what an unknown
route does. Unlike the genderage pass there is no default file that turns
it on by being present — dropping the weights into models/ changes nothing
until the variable names them. A named file that will not load is served as
``/health headwear.error``, never as ``ok: false``: embedding and counting do
not stop for an advisory read.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from heco_common.ort import announce_device, is_trt, providers_for, trt_truth

#: The environment variable that names the image graph (the only switch).
HEADWEAR_ENV = "EMBED_HEADWEAR_MODEL"
#: The prompt set, beside the graph.
PROMPTS_FILE = "headwear_prompts.json"
#: The side the views are resized to (SigLIP B/16 at 224).
INPUT_SIZE = 224
#: The letterbox / mask fill, all channels.
GREY = 127
#: The wire order of the classes and the views: the match service decodes
#: the 8 floats by position, so a prompt set in any other order is refused.
CLASSES = ("turban", "bare", "dupatta_or_scarf", "cap_or_hat")
VIEWS = ("loose", "tight")
#: Faces per session run (two images each). Also the TensorRT profile's
#: ceiling, so no request size ever forces an engine rebuild.
MAX_FACES = 8
#: Intra-op threads on the CPU provider, with spinning off: the reader runs
#: beside the embedder in one process, and spinning pool threads are what
#: slowed cv2's SFace when the genderage pass shared its loop (attributes.py).
CPU_THREADS = 4


def headwear_model_from_env(value: str | None) -> Path | None:
    """Resolve EMBED_HEADWEAR_MODEL: None (off) unless it names a file.

    Unset, empty (compose renders an unset ``${EMBED_HEADWEAR_MODEL-}`` as
    ``""``) and the literal ``off`` all mean off.
    """
    text = (value or "").strip()
    if not text or text.lower() == "off":
        return None
    return Path(text)


HEADWEAR_MODEL_PATH = headwear_model_from_env(os.environ.get(HEADWEAR_ENV))


# ------------------------------------------------ geometry + preprocessing
# The reference reader's functions, kept verbatim in behaviour (tests pin
# them against tests/headwear_ref.py byte for byte on the pixels).


def face_box(box) -> dict:
    """Validate a contract face box: floats ``{x, y, w, h}``, or ValueError.

    Both sides must be positive and every value finite: a head region is
    cut from the box's width AND height, and a NaN or zero would become an
    empty crop read as a confident nothing — the same 200-OK garbage the
    landmark guard bans. A malformed box is the caller's bug and 400s.
    """
    if not isinstance(box, dict) or not all(k in box for k in ("x", "y", "w", "h")):
        raise ValueError("face.box must have x, y, w, h")
    try:
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    except (TypeError, ValueError) as exc:
        raise ValueError("face.box x, y, w, h must be numbers") from exc
    if not np.isfinite([x, y, w, h]).all() or w <= 0.0 or h <= 0.0:
        raise ValueError("face.box x, y, w, h must be finite, w and h positive")
    return {"x": x, "y": y, "w": w, "h": h}


def head_box(
    box: dict, img_w: int, img_h: int, side: float, up: float
) -> tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` of one head region in the image.

    The face box widened ``side`` face widths each side and raised ``up``
    face heights above its top, down to its bottom; each edge rounded
    (Python's round, half to even — the reference's), then clamped.
    """
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    x0, x1 = x - side * w, x + w + side * w
    y0, y1 = y - up * h, y + h
    return (int(round(max(0.0, x0))), int(round(max(0.0, y0))),
            int(round(min(img_w, x1))), int(round(min(img_h, y1))))


def head_view(
    img: np.ndarray, box: dict, side: float, up: float, ellipse: bool
) -> np.ndarray | None:
    """Cut one square BGR view of the head, ellipse-masked or not, letterboxed in grey.

    None when the clamped region is empty (a box outside the image):
    there is nothing to read.
    """
    height, width = img.shape[:2]
    x0, y0, x1, y1 = head_box(box, width, height, side, up)
    if x1 <= x0 or y1 <= y0:
        return None
    c = img[y0:y1, x0:x1].copy()
    h, w = c.shape[:2]
    if ellipse:
        yy, xx = np.mgrid[0:h, 0:w]
        c[((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) / 2) / (h / 2)) ** 2 > 1.0] = GREY
    s = max(h, w)
    sq = np.full((s, s, 3), GREY, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return sq


def pixel_values(views: list[np.ndarray], size: int = INPUT_SIZE) -> np.ndarray:
    """Square BGR uint8 views -> float32[N, 3, size, size] in [-1, 1].

    BGR -> RGB, ``cv2.INTER_AREA`` to ``size`` (the resize is part of the
    contract), /255, (x - 0.5) / 0.5, HWC -> CHW.
    """
    out = np.empty((len(views), 3, size, size), np.float32)
    for i, v in enumerate(views):
        rgb = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        out[i] = ((r - 0.5) / 0.5).transpose(2, 0, 1)
    return out


def covers(box: dict, image_w: int, image_h: int, side: float, up: float, crop: dict) -> bool:
    """Say whether an image cut from a larger frame holds every pixel a view needs.

    ``crop`` = ``{x, y, frameW, frameH}``: where the image's top-left sits in
    the frame, and the frame's size. A view that reaches past an image edge
    is fine where that edge IS the frame's edge (the reference clamps there
    too) and wrong anywhere else — the region would be clipped by the cut,
    not by the frame, and read differently from the evaluation. The runner
    cuts 0.5 face widths a side and 1.2 face heights up, over the views'
    0.30 / 1.0; this is what makes a narrower cut loud instead of quietly
    different.
    """
    x, y, w, h = (box[k] for k in ("x", "y", "w", "h"))
    if x - side * w < 0 and crop["x"] > 0:
        return False
    if y - up * h < 0 and crop["y"] > 0:
        return False
    if x + w + side * w > image_w and crop["x"] + image_w < crop["frameW"]:
        return False
    if y + h > image_h and crop["y"] + image_h < crop["frameH"]:
        return False
    return True


def _sha256(path: Path) -> str:
    """Hash a file with sha256 in 1 MiB chunks (the graph is 372 MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_set(model_path: Path) -> tuple[dict, np.ndarray, str]:
    """``(config, text_embeds, prompts_sha256)`` for the graph at ``model_path``.

    The JSON sits beside the graph and must describe THIS graph and THIS
    text bank (their sha256, checked by the caller and here), the four
    classes in wire order, both views, one row per prompt. Anything else is
    a ValueError naming what is wrong: a mismatched set would load, answer,
    and put another scale's logits into the gallery.
    """
    prompts = model_path.parent / PROMPTS_FILE
    if not prompts.is_file():
        raise FileNotFoundError(
            f"{prompts} missing — {PROMPTS_FILE} and its text embeddings sit beside "
            f"{model_path.name} (see the embed README)"
        )
    raw = prompts.read_bytes()
    cfg = json.loads(raw)
    if tuple(cfg.get("classes") or ()) != CLASSES:
        raise ValueError(f"{PROMPTS_FILE} classes must be {list(CLASSES)} in that order")
    views = cfg.get("views") or {}
    if set(views) != set(VIEWS):
        raise ValueError(f"{PROMPTS_FILE} must define exactly the views {list(VIEWS)}")
    for name in VIEWS:
        v = views[name]
        if v.get("mask") not in (None, "ellipse"):
            raise ValueError(f"{PROMPTS_FILE} view {name}: mask must be null or 'ellipse'")
        _number(v, "side_face_widths", f"view {name}")
        _number(v, "up_face_heights", f"view {name}")
    _number(cfg, "logit_scale", "the prompt set")
    _number(cfg, "logit_bias", "the prompt set")
    text_path = model_path.parent / str(cfg.get("text_embeds") or "")
    if not text_path.is_file():
        raise FileNotFoundError(f"{text_path} missing — named by {PROMPTS_FILE}")
    if _sha256(text_path) != cfg.get("text_embeds_sha256"):
        raise ValueError(
            f"{text_path.name} does not match the sha256 {PROMPTS_FILE} records — "
            "a text bank from another prompt set; refusing the mismatched set"
        )
    text = np.load(text_path).astype(np.float32)
    rows = cfg.get("prompts") or []
    if text.ndim != 2 or text.shape[0] != len(rows) or not rows:
        raise ValueError(
            f"{text_path.name} holds {text.shape} embeddings for {len(rows)} prompts"
        )
    if any(p.get("class") not in CLASSES for p in rows):
        raise ValueError(f"{PROMPTS_FILE}: every prompt must name one of {list(CLASSES)}")
    missing = [c for c in CLASSES if all(p["class"] != c for p in rows)]
    if missing:
        raise ValueError(f"{PROMPTS_FILE}: no prompt for class {missing}")
    return cfg, text, hashlib.sha256(raw).hexdigest()


def _number(obj: dict, key: str, where: str) -> float:
    """``obj[key]`` as a finite float, or a ValueError naming the prompt-set field."""
    try:
        value = float(obj[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{PROMPTS_FILE}: {where} needs a numeric {key}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{PROMPTS_FILE}: {where} {key} must be finite")
    return value


class HeadwearReader:
    """One SigLIP image-tower session and its prompt set; ``read()`` answers per face.

    Built like AttributeModel: providers from HECO_DEVICE (CPU, CUDA, TRT
    fp16 with the engine keyed on the weights' sha256), the device truth read
    AFTER a warm-up run under TensorRT (a failed engine build drops ORT to
    CUDA inside the first run, without raising), and every graph this reader
    could never feed refused at init — the lazy loader then serves the reason
    as ``/health headwear.error``.
    """

    def __init__(self, model_path: Path, device: str | None = None):
        """Verify the set, build the session, warm it (under TensorRT) and read the truth."""
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — place siglip_b16_224_image_fp32.onnx, "
                f"{PROMPTS_FILE} and headwear_text_embeds.npy there or fix {HEADWEAR_ENV}"
            )
        cfg, text, prompts_sha = load_prompt_set(model_path)
        onnx_sha = _sha256(model_path)
        if onnx_sha != cfg.get("onnx_sha256"):
            raise ValueError(
                f"{model_path.name} does not match the sha256 {PROMPTS_FILE} records — "
                "an image graph from another export; refusing the mismatched set"
            )
        import onnxruntime as ort  # deferred, like every family loader

        self.model_name = model_path.name
        self.stamp = f"{onnx_sha[:12]}+{prompts_sha[:12]}"
        self.text = text
        self.row_class = np.array([CLASSES.index(p["class"]) for p in cfg["prompts"]])
        self.scale, self.bias = float(cfg["logit_scale"]), float(cfg["logit_bias"])
        self.views = [
            (name, float(cfg["views"][name]["side_face_widths"]),
             float(cfg["views"][name]["up_face_heights"]), cfg["views"][name]["mask"] == "ellipse")
            for name in VIEWS
        ]
        self._input_name = str((cfg.get("preprocessing") or {}).get("input") or "pixel_values")
        tail = f"3x{INPUT_SIZE}x{INPUT_SIZE}"
        profile = {
            "trt_profile_min_shapes": f"{self._input_name}:1x{tail}",
            "trt_profile_opt_shapes": f"{self._input_name}:{len(VIEWS)}x{tail}",
            "trt_profile_max_shapes": f"{self._input_name}:{len(VIEWS) * MAX_FACES}x{tail}",
        }
        providers, provider_options = providers_for(device, profile, model=model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = CPU_THREADS
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(model_path), options, providers=providers, provider_options=provider_options
        )
        self.device_requested = (device or "CPU").upper()
        self.providers_active = list(self._session.get_providers())
        if not is_trt(device):
            announce_device("embed-headwear", self.device_requested, self.providers_active)
        inp = self._session.get_inputs()[0]
        if inp.name != self._input_name:
            raise ValueError(
                f"{model_path.name} takes input {inp.name!r}; {PROMPTS_FILE} says "
                f"{self._input_name!r}"
            )
        if inp.type == "tensor(float16)":
            self._np_dtype = np.float16
        elif inp.type == "tensor(float)":
            self._np_dtype = np.float32
        else:
            raise ValueError(
                f"{model_path.name} declares input dtype {inp.type} — the reader feeds "
                "tensor(float) or tensor(float16)"
            )
        shape = list(inp.shape)
        if (
            len(shape) != 4
            or (isinstance(shape[1], int) and shape[1] != 3)
            or any(isinstance(d, int) and d != INPUT_SIZE for d in shape[2:])
        ):
            raise ValueError(
                f"{model_path.name} declares a {shape} input; the reader feeds "
                f"[N, 3, {INPUT_SIZE}, {INPUT_SIZE}]"
            )
        out_shape = list(self._session.get_outputs()[0].shape)
        if out_shape and isinstance(out_shape[-1], int) and out_shape[-1] != text.shape[1]:
            raise ValueError(
                f"{model_path.name} answers {out_shape[-1]}-d embeddings; the prompt set's "
                f"are {text.shape[1]}-d"
            )
        self._lock = threading.Lock()
        #: TensorRT's own read-back (trt_truth) — None off TRT and whenever
        #: the engine did not come up.
        self.trt = None
        if is_trt(device):
            # Build (or load) the engine at load, not in the runner's first read.
            zeros = np.zeros((len(VIEWS), 3, INPUT_SIZE, INPUT_SIZE), self._np_dtype)
            self._session.run(None, {self._input_name: zeros})
            self.providers_active = list(self._session.get_providers())
            self.trt = trt_truth(self._session)
            announce_device("embed-headwear", self.device_requested, self.providers_active)

    def class_logits(self, emb: np.ndarray) -> np.ndarray:
        """[N, D] image embeddings -> [N, 4] class logits (max over each class's prompts)."""
        per_prompt = self.scale * (emb @ self.text.T) + self.bias
        out = np.full((len(emb), len(CLASSES)), -np.inf, np.float32)
        for k in range(len(CLASSES)):
            out[:, k] = per_prompt[:, self.row_class == k].max(1)
        return out

    def embed_views(self, views: list[np.ndarray]) -> np.ndarray:
        """Square BGR views -> [N, D] float32 image embeddings, MAX_FACES faces a run."""
        out = []
        step = len(VIEWS) * MAX_FACES
        for start in range(0, len(views), step):
            blob = pixel_values(views[start:start + step]).astype(self._np_dtype, copy=False)
            with self._lock:
                emb = self._session.run(None, {self._input_name: blob})[0]
            out.append(np.asarray(emb, dtype=np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.text.shape[1]), np.float32)

    def read(
        self, img: np.ndarray, boxes: list, crop: dict | None = None
    ) -> list[list[float] | None]:
        """Answer 8 logits per face (the loose view's 4 classes, then the tight view's).

        ``boxes`` are face boxes in ``img``'s pixels; every one is validated
        (and, with ``crop``, checked to be covered) before the first run, so a
        malformed request is one ValueError. A face whose region lies wholly
        outside the image reads None — absent is not zero.
        """
        faces = [face_box(b) for b in boxes]
        height, width = img.shape[:2]
        if crop is not None:
            for i, box in enumerate(faces):
                for name, side, up, _ in self.views:
                    if not covers(box, width, height, side, up, crop):
                        raise ValueError(
                            f"faces[{i}]: the image does not cover the {name} head region — "
                            "cut the context crop with a wider margin (0.5 face widths a "
                            "side, 1.2 face heights up)"
                        )
        views: list[np.ndarray] = []
        owners: list[int] = []
        for i, box in enumerate(faces):
            pair = [head_view(img, box, side, up, ell) for _, side, up, ell in self.views]
            if any(v is None for v in pair):
                continue
            views.extend(pair)
            owners.append(i)
        readings: list[list[float] | None] = [None] * len(faces)
        if not views:
            return readings
        logits = self.class_logits(self.embed_views(views))
        logits = logits.reshape(len(owners), len(VIEWS) * len(CLASSES))
        for i, row in zip(owners, logits, strict=True):
            readings[i] = [float(v) for v in row]
        return readings


def build_headwear(model_path: Path | None = None, device: str | None = None) -> HeadwearReader:
    """Build the reader at ``model_path`` (default: the env-resolved one)."""
    path = model_path or HEADWEAR_MODEL_PATH
    if path is None:
        raise ValueError(f"the head-covering reader is off ({HEADWEAR_ENV} unset or 'off')")
    return HeadwearReader(path, device)


def parity(model_path: Path, crops_dir: Path, meta_path: Path, device: str) -> dict:
    """Compare ``device`` with the CPU fp32 reference on real context crops.

    For the rollout's FP16 check on the box: ``meta_path`` maps crop id ->
    ``{"face": [x, y, w, h]}`` in the crop's pixels (the evaluation's
    ``context_meta.json`` shape), ``crops_dir`` holds ``<id>.png``. Reports
    the smallest image-embedding cosine and the largest class-probability
    difference over both views, and how many reads changed call at the
    shipped bars (turban 0.80, bare 0.50). Target: cosine >= 0.999.
    """
    meta = json.loads(Path(meta_path).read_text())
    ids = sorted(k for k in meta if (Path(crops_dir) / f"{k}.png").is_file())
    ref, dev = HeadwearReader(model_path, "CPU"), HeadwearReader(model_path, device)
    cos_min, dp_max, changed = 1.0, 0.0, []
    for cid in ids:
        img = cv2.imread(str(Path(crops_dir) / f"{cid}.png"))
        x, y, w, h = meta[cid]["face"]
        box = {"x": x, "y": y, "w": w, "h": h}
        views = [head_view(img, box, s, u, e) for _, s, u, e in ref.views]
        e_ref, e_dev = ref.embed_views(views), dev.embed_views(views)
        cos = (e_ref * e_dev).sum(1) / (
            np.linalg.norm(e_ref, axis=1) * np.linalg.norm(e_dev, axis=1))
        cos_min = min(cos_min, float(cos.min()))
        calls = []
        for e in (e_ref, e_dev):
            z = ref.class_logits(e)
            z = z - z.max(1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            calls.append((p, _call(p)))
        dp_max = max(dp_max, float(np.abs(calls[0][0] - calls[1][0]).max()))
        if calls[0][1] != calls[1][1]:
            changed.append({"crop": cid, "cpu": calls[0][1], device: calls[1][1]})
    return {
        "crops": len(ids), "device": device, "active": dev.providers_active, "trt": dev.trt,
        "cosineMin": cos_min, "probMaxAbsDiff": dp_max, "callsChanged": changed,
        "verdict": "ok" if cos_min >= 0.999 and not changed else "CHECK",
    }


def _call(p: np.ndarray, tau: dict | None = None) -> str:
    """Call one read from its [2, 4] view probabilities, as the reference does."""
    tau = tau or {"turban": 0.8, "bare": 0.5}
    arg = p.argmax(1)
    if not (arg == arg[0]).all() or CLASSES[arg[0]] not in tau:
        return "unsure"
    cls = CLASSES[arg[0]]
    return cls if float(p[:, arg[0]].min()) >= tau[cls] else "unsure"


if __name__ == "__main__":  # python -m app.headwear parity <crops_dir> <meta.json> [DEVICE]
    if len(sys.argv) < 4 or sys.argv[1] != "parity" or HEADWEAR_MODEL_PATH is None:
        sys.exit(f"usage: {HEADWEAR_ENV}=<graph> python -m app.headwear parity "
                 "<crops_dir> <meta.json> [DEVICE, default HECO_DEVICE]")
    t0 = time.perf_counter()
    report = parity(HEADWEAR_MODEL_PATH, Path(sys.argv[2]), Path(sys.argv[3]),
                    sys.argv[4] if len(sys.argv) > 4 else (os.environ.get("HECO_DEVICE") or "CPU"))
    report["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(report, indent=1))
