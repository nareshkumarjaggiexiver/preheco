"""YOLOX raw-output decoding, person filtering and NMS as pure functions.

The released YOLOX ONNX graphs emit raw per-anchor predictions
`(N, 5 + num_classes)` — `[dx, dy, log_w, log_h, obj, cls...]` — that must be
decoded against the stride grids (mirroring `demo_postprocess` in Megvii's
ONNXRuntime demo) before thresholding and NMS. Everything here is numpy-only
and unit-tested against golden fixtures; no model is needed.
"""

import numpy as np

#: COCO class index for "person" — the only class this service reports.
PERSON_CLASS = 0

#: FPN strides of the standard (non-P6) YOLOX heads.
STRIDES = (8, 16, 32)


def decode_predictions(
    outputs: np.ndarray,
    img_size: tuple[int, int],
    strides: tuple[int, ...] = STRIDES,
) -> np.ndarray:
    """Decode raw YOLOX head outputs into network-space `[cx, cy, w, h, ...]`.

    `outputs` is `(N, 5 + C)` or `(1, N, 5 + C)`; `img_size` is the (h, w) the
    network ran at. Box centres become `(pred_xy + grid) * stride` and sizes
    `exp(pred_wh) * stride`; objectness/class columns pass through untouched.
    Returns a new `(N, 5 + C)` float array (input is not mutated).
    """
    preds = np.array(outputs, dtype=np.float64, copy=True)
    if preds.ndim == 3:
        preds = preds[0]
    grids = []
    expanded_strides = []
    for stride in strides:
        hsize, wsize = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride))
    grid = np.concatenate(grids, 0)
    stride_col = np.concatenate(expanded_strides, 0)
    if preds.shape[0] != grid.shape[0]:
        raise ValueError(
            f"prediction count {preds.shape[0]} does not match grid {grid.shape[0]} "
            f"for img_size={img_size} strides={strides}"
        )
    preds[:, :2] = (preds[:, :2] + grid) * stride_col
    preds[:, 2:4] = np.exp(preds[:, 2:4]) * stride_col
    return preds


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Single-class numpy NMS; returns kept indices in descending-score order."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = (boxes_xyxy[:, i] for i in range(4))
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou <= iou_thr]
    return keep


def select_persons(
    decoded: np.ndarray,
    ratio: float,
    conf_min: float,
    nms_iou: float,
    img_w: int,
    img_h: int,
) -> list[dict]:
    """Turn decoded predictions into contract person boxes.

    Steps: score = objectness x person-class probability; threshold at
    `conf_min`; convert `cxcywh` to corners; undo the letterbox `ratio`; clamp
    to the source image; NMS at `nms_iou`. Returns CONTRACTS.md-shaped dicts
    `{x, y, w, h, conf}` in source-image pixels, best first.
    """
    scores = decoded[:, 4] * decoded[:, 5 + PERSON_CLASS]
    mask = scores >= conf_min
    if not mask.any():
        return []
    boxes = decoded[mask, :4]
    scores = scores[mask]
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy /= ratio
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, img_w)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, img_h)
    keep = nms(xyxy, scores, nms_iou)
    return [
        {
            "x": round(float(xyxy[i, 0]), 1),
            "y": round(float(xyxy[i, 1]), 1),
            "w": round(float(xyxy[i, 2] - xyxy[i, 0]), 1),
            "h": round(float(xyxy[i, 3] - xyxy[i, 1]), 1),
            "conf": round(float(scores[i]), 4),
        }
        for i in keep
    ]
