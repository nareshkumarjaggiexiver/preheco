"""SCRFD decode — the second face-detector family, as pure functions.

SCRFD (InsightFace) is the realistic strong alternative to YuNet, and it is
NON-COMMERCIAL — under the open-catalog policy that makes it a
restricted-tier entry (grant `contact`), admissible for evaluation and
blocked from billable events until procured, like any other restricted
model. The adapter ships ahead of any weights for the same reason the
rtdetr family did: the family contract is code we can pin and test; the
weights are a lock row away the day a vendor conversation starts.

The `*_kps` exports emit nine tensors — per stride (8, 16, 32): scores
(N,1), boxes (N,4) as l/t/r/b DISTANCES in stride units from each anchor
centre, and five landmarks (N,10) as offsets — two anchors per grid cell.
Everything here is that arithmetic, plus NMS; the ORT session lives in
detector.py with the other families.

Every function is pure numpy in, contract dicts out — tested against
synthetic tensors with known truth, no weights required.
"""

from __future__ import annotations

import functools

import numpy as np

#: The three detection strides of every published SCRFD variant.
STRIDES = (8, 16, 32)
#: Anchors per grid cell in the *_kps exports.
NUM_ANCHORS = 2


def anchor_centers(height: int, width: int, stride: int) -> np.ndarray:
    """(h*w*NUM_ANCHORS, 2) grid centres in INPUT pixels for one stride."""
    ys, xs = np.mgrid[:height, :width]
    centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
    return np.repeat(centers, NUM_ANCHORS, axis=0)


@functools.lru_cache(maxsize=16)
def _cached_centers(height: int, width: int, stride: int) -> np.ndarray:
    """Return anchor_centers built once per grid, read-only.

    The input size never changes between frames, so the grid is the same
    every call; frozen so no caller can scribble on the cached array.
    """
    centers = anchor_centers(height, width, stride)
    centers.flags.writeable = False
    return centers


def decode_stride(scores, boxes, kps, stride: int, input_hw: tuple[int, int]):
    """One stride's tensors into (score, xyxy box, landmarks) rows.

    Boxes arrive as l/t/r/b distances FROM the anchor centre, in stride
    units — the decode is centre ± distance*stride. Landmarks are (dx, dy)
    pairs from the same centre.
    """
    h = input_hw[0] // stride
    w = input_hw[1] // stride
    centers = anchor_centers(h, w, stride)
    scores = np.asarray(scores).reshape(-1)
    boxes = np.asarray(boxes).reshape(-1, 4) * stride
    kps = np.asarray(kps).reshape(-1, 5, 2) * stride
    n = min(len(centers), len(scores))
    centers, scores, boxes, kps = centers[:n], scores[:n], boxes[:n], kps[:n]
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = centers[:, 0] - boxes[:, 0]
    xyxy[:, 1] = centers[:, 1] - boxes[:, 1]
    xyxy[:, 2] = centers[:, 0] + boxes[:, 2]
    xyxy[:, 3] = centers[:, 1] + boxes[:, 3]
    landmarks = kps + centers[:, None, :]
    return scores, xyxy, landmarks


def _decode_kept(scores, boxes, kps, stride: int, input_hw: tuple[int, int], score_min: float):
    """decode_stride for the anchors at or above `score_min` ONLY.

    At 1472x832 a frame has 50,232 anchors and a few dozen clear the score;
    decoding all of them (boxes, five landmarks, their centres) and THEN
    masking was the bulk of select_faces. Same arithmetic, element for
    element, on the kept rows in the same order — so the same bits out
    (pinned against decode_stride + mask in test_scrfd.py).
    """
    h = input_hw[0] // stride
    w = input_hw[1] // stride
    centers = _cached_centers(h, w, stride)
    scores = np.asarray(scores).reshape(-1)
    n = min(len(centers), len(scores))
    keep = np.flatnonzero(scores[:n] >= score_min)
    centers = centers[keep]
    boxes = np.asarray(boxes).reshape(-1, 4)[keep] * stride
    kps = np.asarray(kps).reshape(-1, 5, 2)[keep] * stride
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = centers[:, 0] - boxes[:, 0]
    xyxy[:, 1] = centers[:, 1] - boxes[:, 1]
    xyxy[:, 2] = centers[:, 0] + boxes[:, 2]
    xyxy[:, 3] = centers[:, 1] + boxes[:, 3]
    return scores[keep], xyxy, kps + centers[:, None, :]


def nms(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy IoU NMS, best first — the standard detector epilogue."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        y1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        x2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        y2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        area_r = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / np.clip(area_i + area_r - inter, 1e-9, None)
        order = rest[iou <= iou_thr]
    return keep


def select_faces(outputs: list, input_hw, scale: tuple[float, float], score_min: float,
                 nms_iou: float) -> list[dict]:
    """The nine SCRFD tensors into contract face dicts, in source pixels.

    `scale` is the ACHIEVED per-axis letterbox scale (sx, sy) = (rw/w,
    rh/h) — what the canvas actually holds after the integer-truncated
    resize, not the nominal ratio, because whenever the frame does not
    divide the input cleanly the truncation shifts every coordinate toward
    the origin (up to ~4 px at 4MP frame edges — letterbox arithmetic that
    would read as model disagreement in a golden diff).

    Coordinates are UNCLAMPED source pixels, exactly like the YuNet rows: a
    face straddling a crop edge keeps its extrapolated box, which is the
    truer width the 56/80 px floors are calibrated against (the crop edge
    is an artifact of the person box, not of the face). Output shape is
    IDENTICAL to the YuNet path — box, five landmarks in the same order,
    conf — so the quality gate and the embed alignment downstream cannot
    tell the families apart, which is the whole contract.
    """
    all_scores, all_xyxy, all_lm = [], [], []
    for si, stride in enumerate(STRIDES):
        scores, xyxy, lm = _decode_kept(
            outputs[si], outputs[si + 3], outputs[si + 6], stride, input_hw, score_min)
        if scores.size:
            all_scores.append(scores)
            all_xyxy.append(xyxy)
            all_lm.append(lm)
    if not all_scores:
        return []
    sx, sy = scale
    scores = np.concatenate(all_scores)
    xyxy = np.concatenate(all_xyxy)
    xyxy[:, 0::2] /= sx
    xyxy[:, 1::2] /= sy
    lm = np.concatenate(all_lm)
    lm[..., 0] /= sx
    lm[..., 1] /= sy
    keep = nms(xyxy, scores, nms_iou)
    return [
        {
            "box": {
                "x": round(float(xyxy[i, 0]), 1),
                "y": round(float(xyxy[i, 1]), 1),
                "w": round(float(xyxy[i, 2] - xyxy[i, 0]), 1),
                "h": round(float(xyxy[i, 3] - xyxy[i, 1]), 1),
            },
            "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in lm[i]],
            "conf": round(float(scores[i]), 4),
        }
        for i in keep
    ]
