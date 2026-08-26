"""SCRFD decode against synthetic tensors with known truth — no weights."""

import numpy as np

from app.scrfd import NUM_ANCHORS, STRIDES, anchor_centers, decode_stride, select_faces


def _tensors(input_hw=(640, 640)):
    """Nine zero tensors of the correct shapes, ready to be seeded."""
    outs = []
    for group in range(3):  # scores, boxes, kps
        for stride in STRIDES:
            n = (input_hw[0] // stride) * (input_hw[1] // stride) * NUM_ANCHORS
            width = {0: 1, 1: 4, 2: 10}[group]
            outs.append(np.zeros((n, width), dtype=np.float32))
    return outs


def test_anchor_centres_are_grid_times_stride_with_two_anchors():
    c = anchor_centers(2, 2, 8)
    assert c.shape == (8, 2)
    assert (c[0] == [0, 0]).all() and (c[1] == [0, 0]).all(), "two anchors per cell"
    assert (c[2] == [8, 0]).all(), "next cell is one stride across"


def test_decode_puts_a_face_where_the_arithmetic_says():
    """centre (160,160) at stride 8, distances 2 units each way = a 32px box
    centred there; landmarks offset by (1,1) units = centre + 8px."""
    outs = _tensors()
    idx = ((160 // 8) * (640 // 8) + (160 // 8)) * NUM_ANCHORS
    outs[0][idx] = 0.9
    outs[3][idx] = [2, 2, 2, 2]
    outs[6][idx] = [1, 1] * 5
    faces = select_faces(outs, (640, 640), 1.0, 0.5, 0.4, 640, 640)
    assert len(faces) == 1
    f = faces[0]
    assert (f["box"]["x"], f["box"]["y"], f["box"]["w"], f["box"]["h"]) == (144.0, 144.0, 32.0, 32.0)
    assert f["landmarks"][0] == [168.0, 168.0]
    assert f["conf"] == 0.9


def test_ratio_undoes_the_letterbox_and_clamps_to_the_source():
    """A face decoded at input pixel 160 under ratio 0.5 lands at source 320
    — and the contract shape is identical to the yunet family's."""
    outs = _tensors()
    idx = ((160 // 8) * (640 // 8) + (160 // 8)) * NUM_ANCHORS
    outs[0][idx] = 0.8
    outs[3][idx] = [2, 2, 2, 2]
    outs[6][idx] = [0, 0] * 5
    faces = select_faces(outs, (640, 640), 0.5, 0.5, 0.4, 1280, 960)
    assert faces[0]["box"]["x"] == 288.0  # 144 / 0.5
    assert len(faces[0]["landmarks"]) == 5


def test_nms_keeps_the_best_of_overlapping_detections():
    outs = _tensors()
    base = ((160 // 8) * (640 // 8) + (160 // 8)) * NUM_ANCHORS
    for i, score in ((base, 0.9), (base + 1, 0.7)):  # same cell, both anchors
        outs[0][i] = score
        outs[3][i] = [2, 2, 2, 2]
    faces = select_faces(outs, (640, 640), 1.0, 0.5, 0.4, 640, 640)
    assert len(faces) == 1 and faces[0]["conf"] == 0.9


def test_below_threshold_is_empty_not_an_error():
    assert select_faces(_tensors(), (640, 640), 1.0, 0.5, 0.4, 640, 640) == []
