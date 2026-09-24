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

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from heco_common.ort import (
    announce_device,
    distinct_input_dims,
    is_trt,
    providers_for,
    trt_truth,
)

from . import scrfd

log = logging.getLogger("faces")

#: Default model location — populated by `make models`, never committed.
DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
)

#: The models this service knows how to run. Unknown names infer scrfd when
#: they say so, else yunet — an experiment is never blocked by a table.
#: `score_min` is the family's OPERATING POINT: YuNet's and SCRFD's scores
#: are not commensurable (every InsightFace reference runs det_thresh=0.5;
#: real SCRFD true positives on small/angled gate faces commonly score
#: 0.5–0.8), so sharing YuNet's 0.8 would run scrfd 0.3 above calibration
#: and silently gut its recall in exactly the A/B the swap exists for.
MODEL_SPECS = {
    "face_detection_yunet_2023mar.onnx": {"family": "yunet"},
    "scrfd_2.5g_kps.onnx": {"family": "scrfd", "input": 640, "score_min": 0.5},
    "scrfd_10g_kps.onnx": {"family": "scrfd", "input": 640, "score_min": 0.5},
}


def spec_for(model_path: Path) -> dict:
    """Look up the model's spec row, inferring it from the filename when unlisted."""
    known = MODEL_SPECS.get(model_path.name)
    if known:
        return dict(known)
    family = "scrfd" if "scrfd" in model_path.name.lower() else "yunet"
    return {
        "family": family,
        **({"input": 640, "score_min": 0.5} if family == "scrfd" else {}),
    }


#: Planner-applied selection, persisted beside the weights (bind-mounted, so
#: it survives restarts). Precedence: .selected > env > default — the file
#: is the operator's LATEST intent through the planner; same rule as persons.
#: State lives BESIDE the weights, not among them: the models mount is
#: read-only by design (weights are immutable-by-mount; verify-models
#: guards their content) — bitten live on the .94 first apply, where the
#: swap succeeded and the durability write then 500'd the response. The
#: state dir is its own small writable mount.
STATE_DIR = DEFAULT_MODEL.parent.parent / "state"
SELECTED_FILE = STATE_DIR / "selected"


def selected_model() -> str | None:
    try:
        name = SELECTED_FILE.read_text().strip()
    except OSError:
        return None
    if not name or "/" in name or not (DEFAULT_MODEL.parent / name).is_file():
        return None
    return name


def persist_selection(name: str) -> None:
    """Record an applied selection atomically; refuse when not durable —
    persist runs BEFORE the swap, so a selection that cannot stick is
    refused rather than applied-until-a-random-restart (same contract as
    persons; the .94 first apply is the incident)."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SELECTED_FILE.with_suffix(".tmp")
        tmp.write_text(name + "\n")
        tmp.replace(SELECTED_FILE)
    except OSError as exc:
        raise RuntimeError(
            f"cannot persist the selection ({exc}) — the state dir is missing, "
            "read-only, or not writable by this service's uid (a missing dir is "
            "created root-owned by the docker daemon — pre-create "
            "services/faces/state with the right owner) or apply the change as "
            "deployment env"
        ) from exc


def _model_path(value: str | None) -> Path:
    if not value:
        return DEFAULT_MODEL
    return Path(value) if "/" in value else DEFAULT_MODEL.parent / value


MODEL_PATH = _model_path(selected_model() or (os.environ.get("FACES_MODEL") or None))
#: FACES_SCORE_MIN is the YUNET knob and stays so — the scrfd family has
#: its own operating point below, because the two score spaces do not line
#: up (see MODEL_SPECS).
SCORE_MIN = float(os.environ.get("FACES_SCORE_MIN") or "0.8")
NMS_IOU = float(os.environ.get("FACES_NMS_IOU") or "0.3")
TOP_K = int(os.environ.get("FACES_TOP_K") or "5000")
DEVICE = os.environ.get("HECO_DEVICE") or "CPU"
#: The scrfd operating point, operator-reachable per family (empty-string-
#: safe, the repo convention). Set, it overrides the MODEL_SPECS row so the
#: knob works even for listed models; unset, the row (or this 0.5 default,
#: InsightFace's det_thresh) rules.
SCRFD_SCORE_MIN = float(os.environ.get("FACES_SCRFD_SCORE_MIN") or "0.5")
_SCRFD_SCORE_MIN_OVERRIDDEN = bool(os.environ.get("FACES_SCRFD_SCORE_MIN"))

#: Crops smaller than this per side are skipped. The rationale is YuNet's —
#: it runs at NATIVE resolution and cannot resolve inputs this small — and
#: it does NOT transfer to scrfd, which letterboxes everything into a fixed
#: square: there the operative constraint is the downscale ratio, guarded
#: per frame in ScrfdDetector.detect (min_resolvable_face_px).
MIN_CROP_SIDE = 16


def _parse_input_size(raw: str | None, fallback: int) -> int | tuple[int, int]:
    """FACES_SCRFD_INPUT as a square ("640") or a frame-shaped pair ("1472x832").

    WHY A RECTANGLE IS WORTH THE PARSER. The family letterboxes into its
    input, so a 16:9 frame in a SQUARE canvas spends 44% of the pixels on
    padding — and the padding is where the resolution for small faces went.
    Measured on the 4060 with scrfd_10g: 1472x832 costs 20.5 ms and leaves a
    67 px source face 25.7 net px; the square 640 that same frame would use
    leaves it 11 and finds nothing, which is why the caller was cropping.

    Both forms are validated to stride-32 multiples: the decode builds its
    anchor grid as input // stride for strides 8/16/32, so a size that is not
    a multiple silently drops the remainder row and shifts every box.
    """
    if not raw:
        return fallback
    text = raw.strip().lower().replace("*", "x")
    try:
        parts = [int(p) for p in text.split("x")]
    except ValueError as exc:
        raise ValueError(f"FACES_SCRFD_INPUT={raw!r} — want 640 or 1472x832") from exc
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2 or any(p <= 0 or p % 32 for p in parts):
        raise ValueError(
            f"FACES_SCRFD_INPUT={raw!r} — both sides must be positive multiples of 32 "
            "(the decode grids at strides 8/16/32)"
        )
    w, h = parts  # WxH as an operator writes it; the detector wants (h, w)
    return (h, w)


#: Operator override for the scrfd network input. Unset, MODEL_SPECS rules
#: (640) and nothing changes.
SCRFD_INPUT = os.environ.get("FACES_SCRFD_INPUT") or None


#: SCRFD's input normalisation, (x - 127.5) / 128, as a 256-entry table.
#: Every entry is a multiple of 1/256 and so exact in float32: a lookup and
#: the arithmetic agree bit for bit (pinned in test_detector.py).
_SCRFD_NORM_LUT = ((np.arange(256, dtype=np.float32) - 127.5) / 128.0).astype(np.float32)


def scrfd_blob(canvas: np.ndarray) -> np.ndarray:
    """Turn a BGR uint8 canvas into SCRFD's (1, 3, H, W) float32 RGB blob.

    The same numbers as ``((rgb.astype(float32) - 127.5) / 128).transpose``
    — which made a uint8 RGB copy and then four full float32 passes over the
    canvas (convert, subtract, divide, the contiguous copy of the transpose):
    at 1472x832 that is ~15 MB per pass. Here each channel is split once and
    looked up straight into its plane of the output. Measured on the laptop,
    1472x832: 24 ms -> 4 ms, byte-identical.
    """
    h, w = canvas.shape[:2]
    blob = np.empty((1, 3, h, w), dtype=np.float32)
    # split is B, G, R; the graph wants R, G, B — hence reversed.
    for ch, plane in enumerate(reversed(cv2.split(canvas))):
        out = cv2.LUT(plane, _SCRFD_NORM_LUT, dst=blob[0, ch])
        if not np.may_share_memory(out, blob):
            blob[0, ch] = out  # cv2 declined the dst: copy, never leave it empty
    return blob


def build_detector(model_path: Path | None = None, device: str | None = None):
    """The family factory: one detect() contract, family chosen by the file."""
    path = model_path or MODEL_PATH
    spec = spec_for(path)
    if spec["family"] == "scrfd":
        score_min = (
            SCRFD_SCORE_MIN
            if _SCRFD_SCORE_MIN_OVERRIDDEN
            else spec.get("score_min", SCRFD_SCORE_MIN)
        )
        return ScrfdDetector(
            path,
            _parse_input_size(SCRFD_INPUT, spec.get("input", 640)),
            device or DEVICE,
            score_min,
        )
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
        # The operating point rides on the instance so /health and /model
        # can serve it uniformly across families — a golden-replay ledger
        # must record which threshold produced it.
        self.score_min = SCORE_MIN
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

    #: Faces smaller than this in NETWORK pixels sit below what the stride-8
    #: anchor grid resolves — the floor the whole-frame downscale guard in
    #: detect() measures against.
    MIN_NET_FACE_PX = 16
    #: The POC face floor in SOURCE pixels (CONTRACTS.md / quality.py):
    #: the size detect() is expected to resolve; used only to decide when
    #: the letterbox has made that promise unkeepable.
    POC_FLOOR_PX = 56

    def __init__(
        self,
        model_path: Path,
        input_size: int = 640,
        device: str | None = None,
        score_min: float | None = None,
    ):
        """Build and PROVE the session; any refusal here is the feature.

        ONNX Runtime checks input dimensions at run() time, not at session
        creation, so a loadable-but-unrunnable weight (a fixed-320 export
        configured 640, a wrong-contract graph) would otherwise pass
        validate-before-swap and brick every subsequent /detect behind
        ok:true — the exact healthy-but-dead failure the deploy-integrity
        guards exist for. Hence: reconcile the graph's static input shape
        against the configured size, then dry-run one zero frame through
        the session AND the decode, exactly as detect() will feed them.
        """
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing — fetch it (restricted tier: make models-restricted) first"
            )
        import onnxruntime as ort  # deferred, like every family loader

        self.model_name = model_path.name
        # (h, w). Square unless an operator asked for a frame-shaped input —
        # see _parse_input_size: letterboxing 16:9 into a square throws away
        # 44% of the canvas on padding, and with it the resolution the small
        # faces need.
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.score_min = SCRFD_SCORE_MIN if score_min is None else float(score_min)
        model: str | bytes = str(model_path)
        if is_trt(device):
            # The InsightFace exports name BOTH spatial dims "?", which
            # TensorRT reads as one dimension: a 1472x832 input then fails
            # the engine build and ORT drops to CUDA. Renamed apart in
            # memory; the file on the read-only mount is untouched.
            model = distinct_input_dims(model_path.read_bytes()) or model
        # Under TRT the engine directory is keyed on exactly the graph handed
        # to ORT (heco_common.ort.model_key). Loaded from BYTES the EP's own
        # engine name has no file name in it at all, so any same-architecture
        # weight (POST /model, FACES_MODEL) ran the old engine: measured, a
        # shifted bias "built" in 0.08 s and answered the original's scores.
        providers, provider_options = providers_for(device, model=model)
        self._session = ort.InferenceSession(
            model, providers=providers, provider_options=provider_options
        )
        self.device_requested = (device or "CPU").upper()
        self.providers_active = list(self._session.get_providers())
        #: TensorRT's own view of its options; None when it is not in the
        #: session (set after the dry run below, see there).
        self.trt = None
        if not is_trt(device):
            announce_device("faces", self.device_requested, self.providers_active)
        if len(self._session.get_outputs()) != 9:
            raise ValueError(
                f"{model_path.name} has {len(self._session.get_outputs())} outputs — "
                "the scrfd family speaks the 9-tensor *_kps export (3 strides x "
                "score/box/kps); a different export needs its own family"
            )
        # Static-shape reconcile: a fixed-shape export declares integer H/W
        # on the graph; a mismatch with the configured size is a config
        # error the operator can act on NOW, not an ORT stack trace later.
        shape = self._session.get_inputs()[0].shape
        static_hw = [d for d in shape[-2:] if isinstance(d, int)]
        configured_hw = [d for d in self.input_size if isinstance(d, int)]
        if static_hw and list(static_hw) != configured_hw[: len(static_hw)]:
            raise ValueError(
                f"{model_path.name} expects a {'x'.join(map(str, static_hw))} input, "
                f"configured {'x'.join(map(str, self.input_size))} — check MODEL_SPECS/spec_for"
            )
        self._input_name = self._session.get_inputs()[0].name
        self._lock = threading.Lock()
        self._warned_sizes: set[tuple[int, int]] = set()
        # Dry run: one zero frame through the session and select_faces. A
        # dynamic graph that still refuses our shape, or an export whose
        # tensors do not reshape into the *_kps contract, raises HERE —
        # apply_model turns it into a 400 and the old model keeps serving;
        # at boot the same raise makes /health honestly unhealthy.
        outputs = self._session.run(
            None, {self._input_name: np.zeros((1, 3, *self.input_size), np.float32)}
        )
        scrfd.select_faces(outputs, self.input_size, (1.0, 1.0), self.score_min, NMS_IOU)
        if is_trt(device):
            # The dry run is also where TensorRT builds (or loads) its
            # engine — and where a failed build makes ORT silently rebuild
            # the session on CUDA. The truth is read AFTER it, or /health
            # would claim a TensorRT that is not running (measured: it did).
            self.providers_active = list(self._session.get_providers())
            self.trt = trt_truth(self._session)
            announce_device("faces", self.device_requested, self.providers_active)

    def detect(self, img: np.ndarray) -> list[dict]:
        """Detect faces in a BGR image; returns contract face dicts (no quality)."""
        h, w = img.shape[:2]
        if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE:
            return []
        # Letterbox keep-ratio into the square input, zeros pad — the decode
        # divides by the ACHIEVED per-axis scale on the way out.
        ih, iw = self.input_size
        ratio = min(ih / h, iw / w)
        # Whole-frame guard: letterboxing an 8MP frame into 640 makes the
        # 56 px POC floor ~9 net px — below what stride-8 anchors resolve —
        # so an empty answer would read as "no faces present" when it means
        # "faces this small were invisible". Warn once per frame size; the
        # machine-readable floor rides the /detect reply (main.py).
        if self.POC_FLOOR_PX * ratio < self.MIN_NET_FACE_PX and (w, h) not in self._warned_sizes:
            log.warning(
                "scrfd: %dx%d downscales %.3fx — faces under ~%d source px are "
                "unresolvable; pass `within` person crops or raise input_size",
                w, h, ratio, self.min_resolvable_face_px(w, h),
            )
            # A SET, not a last-size slot: a labeller alternating between two
            # cameras' frame sizes must not turn the once-per-size promise
            # into per-frame spam. Distinct sizes are few; unbounded is fine.
            self._warned_sizes.add((w, h))
        rw, rh = int(w * ratio), int(h * ratio)
        canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
        canvas[:rh, :rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        blob = scrfd_blob(canvas)
        with self._lock:
            outputs = self._session.run(None, {self._input_name: blob})
        # int() truncation means the canvas actually holds rw/w x rh/h of
        # the frame, not `ratio` on both axes — divide by the achieved pair
        # or every box drifts toward the origin (~4 px at 4MP frame edges).
        return scrfd.select_faces(
            outputs, self.input_size, (rw / w, rh / h), self.score_min, NMS_IOU
        )

    def min_resolvable_face_px(self, w: int, h: int) -> int:
        """Smallest source-pixel face the letterbox leaves resolvable.

        Pure arithmetic on the frame size — detect() keeps no per-call
        state, so concurrent callers cannot race it. Served additively on
        whole-frame /detect replies so `faces: []` on a big frame cannot be
        mistaken for "no faces present".
        """
        ih, iw = self.input_size
        ratio = min(ih / h, iw / w)
        return round(self.MIN_NET_FACE_PX / ratio)
