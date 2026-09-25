# ruff: noqa — VERBATIM copy of the siglip evaluation's onnx/headwear_ref.py (sha256 of the lines below: 5c96f2f974baffdc897db642343af57baa32c082efe3de231b275b4a97b918e6). Never edit: tests/test_headwear.py pins app/headwear.py against it.
"""Reference head-covering reader — what a service would run. numpy + opencv + onnxruntime ONLY
(no torch, no transformers). Everything it needs is in this folder:

  siglip_b16_224_image_fp32.onnx   image tower: pixel_values[N,3,224,224] -> image_embeds[N,768] (L2-normalised)
  headwear_text_embeds.npy         float32[14,768]: one L2-normalised text embedding per PROMPT
  headwear_prompts.json            classes, prompt->class rows, logit scale/bias, the two views, preprocessing, thresholds

Per face, TWO views of the head are read in one batch:
  loose  face box widened 0.30 face widths a side, raised 1.0 face heights    (sees a whole turban)
  tight  face box widened 0.15 face widths a side, raised 0.8 face heights,
         outside the inscribed ellipse greyed                                  (a neighbour's head is masked)
each letterboxed to a square with grey 127, resized to 224 (INTER_AREA), scaled to [-1, 1].
Class logit = max over the class's prompts of scale * cos + bias; probabilities = softmax over the 4 classes.
CALL: "turban" / "bare" only when BOTH views argmax that class and min(p_view1, p_view2) >= its threshold;
anything else is "unsure". The 2 x 4 raw logits are returned too — they are what should be stored.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
GREY = 127


def head_box(face_box: dict, img_w: int, img_h: int, side: float, up: float) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the face box widened `side` face widths each side and raised `up` face heights
    above its top, down to the face box bottom; edges rounded, clamped to the frame."""
    x, y, w, h = (float(face_box[k]) for k in ("x", "y", "w", "h"))
    x0, x1 = x - side * w, x + w + side * w
    y0, y1 = y - up * h, y + h
    return (int(round(max(0.0, x0))), int(round(max(0.0, y0))),
            int(round(min(img_w, x1))), int(round(min(img_h, y1))))


def head_view(frame_bgr: np.ndarray, face_box: dict, side: float, up: float, ellipse: bool) -> np.ndarray:
    """One square BGR view of the head: the region, optionally ellipse-masked, letterboxed with grey."""
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = head_box(face_box, W, H, side, up)
    c = frame_bgr[y0:y1, x0:x1].copy()
    h, w = c.shape[:2]
    if ellipse:
        yy, xx = np.mgrid[0:h, 0:w]
        c[((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) / 2) / (h / 2)) ** 2 > 1.0] = GREY
    s = max(h, w)
    sq = np.full((s, s, 3), GREY, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return sq


def preprocess(views_bgr: list[np.ndarray], size: int = 224) -> np.ndarray:
    """Square BGR uint8 views -> float32[N,3,size,size] in [-1, 1]."""
    out = np.empty((len(views_bgr), 3, size, size), np.float32)
    for i, v in enumerate(views_bgr):
        rgb = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        out[i] = ((r - 0.5) / 0.5).transpose(2, 0, 1)
    return out


class HeadwearReader:
    def __init__(self, folder: Path = HERE, threads: int = 4, providers: list[str] | None = None):
        cfg = json.load(open(folder / "headwear_prompts.json"))
        self.cfg = cfg
        self.classes = cfg["classes"]
        self.text = np.load(folder / cfg["text_embeds"]).astype(np.float32)
        self.row_class = np.array([self.classes.index(p["class"]) for p in cfg["prompts"]])
        self.scale, self.bias = float(cfg["logit_scale"]), float(cfg["logit_bias"])
        self.tau = cfg["thresholds"]
        self.views = [(name, float(v["side_face_widths"]), float(v["up_face_heights"]), v["mask"] == "ellipse")
                      for name, v in cfg["views"].items()]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(folder / cfg["onnx"]), so,
                                         providers=providers or ["CPUExecutionProvider"])

    def class_logits(self, emb: np.ndarray) -> np.ndarray:
        """[N,768] image embeddings -> [N,4] class logits (max over each class's prompts)."""
        per_prompt = self.scale * (emb @ self.text.T) + self.bias
        out = np.full((len(emb), len(self.classes)), -np.inf, np.float32)
        for k in range(len(self.classes)):
            out[:, k] = per_prompt[:, self.row_class == k].max(1)
        return out

    def read(self, frame_bgr: np.ndarray, face_boxes: list[dict]) -> list[dict]:
        if not face_boxes:
            return []
        views = [head_view(frame_bgr, b, side, up, ell) for b in face_boxes for _, side, up, ell in self.views]
        emb = self.sess.run(None, {"pixel_values": preprocess(views)})[0]
        logits = self.class_logits(emb).reshape(len(face_boxes), len(self.views), len(self.classes))
        z = logits - logits.max(2, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(2, keepdims=True)
        out = []
        for lg, pr in zip(logits, p):
            arg = pr.argmax(1)
            call, score = "unsure", None
            if (arg == arg[0]).all() and self.classes[arg[0]] in self.tau:
                cls = self.classes[arg[0]]
                score = float(pr[:, arg[0]].min())
                if score >= self.tau[cls]:
                    call = cls
            out.append({"call": call, "score": score,
                        "views": {name: {"argmax": self.classes[a], "probs": [float(v) for v in pv],
                                         "logits": [float(v) for v in lv]}
                                  for (name, *_), a, pv, lv in zip(self.views, arg, pr, lg)}})
        return out
