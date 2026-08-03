"""Golden tests for pure YOLOX decoding, NMS and person selection.

The decode fixture (tests/fixtures/decode_golden.json) was generated once by
an independent loop-based reference implementation (see fixtures/README) so
the vectorised production code is checked against separately-derived numbers.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.postprocess import decode_predictions, nms, select_persons

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_decode_zero_preds_yields_grid_centres():
    # img 32x32, strides 8/16/32 -> 16 + 4 + 1 = 21 anchors
    preds = np.zeros((21, 85))
    out = decode_predictions(preds, (32, 32))
    assert out.shape == (21, 85)
    np.testing.assert_allclose(out[0, :4], [0, 0, 8, 8])
    np.testing.assert_allclose(out[5, :4], [8, 8, 8, 8])  # stride-8 grid (1,1)
    np.testing.assert_allclose(out[16, :4], [0, 0, 16, 16])  # first stride-16
    np.testing.assert_allclose(out[20, :4], [0, 0, 32, 32])  # the stride-32 cell


def test_decode_offsets_and_log_sizes():
    preds = np.zeros((21, 85))
    preds[0, :4] = [0.5, 0.25, np.log(2.0), 0.0]
    out = decode_predictions(preds, (32, 32))
    np.testing.assert_allclose(out[0, :4], [4.0, 2.0, 16.0, 8.0])


def test_decode_accepts_batch_dim_and_does_not_mutate():
    preds = np.zeros((1, 21, 85), dtype=np.float32)
    out = decode_predictions(preds, (32, 32))
    assert out.shape == (21, 85)
    assert (preds == 0).all()


def test_decode_rejects_wrong_anchor_count():
    with pytest.raises(ValueError):
        decode_predictions(np.zeros((20, 85)), (32, 32))


def test_decode_golden_fixture():
    golden = json.loads((FIXTURES / "decode_golden.json").read_text())
    out = decode_predictions(np.array(golden["preds"]), tuple(golden["img_size"]))
    np.testing.assert_allclose(out, np.array(golden["expected"]), rtol=1e-6, atol=1e-9)


def test_nms_suppresses_overlap_keeps_disjoint():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, iou_thr=0.45) == [0, 2]


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


def _pred_row(cx, cy, w, h, obj, cls_idx, cls_prob):
    row = np.zeros(85)
    row[:4] = [cx, cy, w, h]
    row[4] = obj
    row[5 + cls_idx] = cls_prob
    return row


def test_select_persons_filters_maps_and_nms():
    decoded = np.stack(
        [
            _pred_row(100, 60, 40, 80, 0.9, 0, 0.9),  # person A, score .81
            _pred_row(102, 60, 40, 80, 0.8, 0, 0.9),  # near-duplicate of A -> NMS'd
            _pred_row(300, 200, 40, 40, 1.0, 0, 0.5),  # person C, score .50
            _pred_row(150, 150, 40, 40, 0.99, 1, 0.99),  # bicycle -> class-filtered
            _pred_row(50, 50, 20, 20, 0.5, 0, 0.2),  # score .10 < confMin
        ]
    )
    boxes = select_persons(decoded, ratio=0.5, conf_min=0.3, nms_iou=0.45, img_w=640, img_h=480)
    assert [b["conf"] for b in boxes] == [0.81, 0.5]
    assert boxes[0] == {"x": 160.0, "y": 40.0, "w": 80.0, "h": 160.0, "conf": 0.81}
    assert boxes[1] == {"x": 560.0, "y": 360.0, "w": 80.0, "h": 80.0, "conf": 0.5}


def test_select_persons_clamps_to_image():
    decoded = _pred_row(10, 10, 60, 60, 1.0, 0, 1.0)[None, :]
    boxes = select_persons(decoded, ratio=1.0, conf_min=0.3, nms_iou=0.45, img_w=100, img_h=100)
    assert boxes[0]["x"] == 0.0 and boxes[0]["y"] == 0.0
    assert boxes[0]["w"] == 40.0  # right edge at 40, left clamped at 0


def test_select_persons_none_above_threshold():
    decoded = _pred_row(10, 10, 5, 5, 0.1, 0, 0.1)[None, :]
    assert select_persons(decoded, 1.0, 0.3, 0.45, 100, 100) == []
