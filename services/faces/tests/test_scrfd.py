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
    faces = select_faces(outs, (640, 640), (1.0, 1.0), 0.5, 0.4)
    assert len(faces) == 1
    f = faces[0]
    assert (f["box"]["x"], f["box"]["y"], f["box"]["w"], f["box"]["h"]) == (144.0, 144.0, 32.0, 32.0)
    assert f["landmarks"][0] == [168.0, 168.0]
    assert f["conf"] == 0.9


def test_scale_undoes_the_letterbox_and_geometry_passes_through_unclamped():
    """A face decoded at input pixel 160 under scale 0.5 lands at source 320
    — and, like the yunet family, coordinates are emitted VERBATIM: a box
    reaching past the crop edge keeps its extrapolated geometry (negative x
    included), because the crop edge is a person-box artifact and the
    extrapolated width is what the 56/80 px floors are calibrated against.
    Clamping only here made the families distinguishable downstream."""
    outs = _tensors()
    idx = ((160 // 8) * (640 // 8) + (160 // 8)) * NUM_ANCHORS
    outs[0][idx] = 0.8
    outs[3][idx] = [2, 2, 2, 2]
    outs[6][idx] = [0, 0] * 5
    # a second face straddling the left edge: centre (0, 320), l distance 5
    # units = decoded x1 of -40 in canvas pixels
    edge = ((320 // 8) * (640 // 8) + 0) * NUM_ANCHORS
    outs[0][edge] = 0.9
    outs[3][edge] = [5, 2, 2, 2]
    outs[6][edge] = [0, 0] * 5
    faces = select_faces(outs, (640, 640), (0.5, 0.5), 0.5, 0.4)
    by_conf = {f["conf"]: f for f in faces}
    assert by_conf[0.8]["box"]["x"] == 288.0  # 144 / 0.5
    assert len(by_conf[0.8]["landmarks"]) == 5
    assert by_conf[0.9]["box"]["x"] == -80.0  # -40 / 0.5, NOT clipped to 0
    assert by_conf[0.9]["box"]["w"] == 112.0  # (16 - -40) / 0.5, full width


def test_achieved_scale_maps_the_canvas_content_edge_to_the_frame_edge():
    """2688x1520 (4MP) does not divide 640: rh = int(1520*ratio) = 361, so
    the canvas holds 361/1520 of the frame — NOT the nominal ratio. Decode
    must divide by the achieved per-axis scale, or a detection touching the
    content edge lands ~4 px inside the frame and every box drifts toward
    the origin (letterbox arithmetic masquerading as model disagreement)."""
    w, h = 2688, 1520
    ratio = min(640 / h, 640 / w)
    rw, rh = int(w * ratio), int(h * ratio)
    assert rh / h != ratio, "premise: truncation changed the vertical scale"
    outs = _tensors()
    # stride 8, cell (44, 40) -> anchor centre (320, 352); distances reach
    # the canvas content corners (0, 0) and (rw, rh) exactly
    idx = ((352 // 8) * (640 // 8) + (320 // 8)) * NUM_ANCHORS
    outs[0][idx] = 0.9
    outs[3][idx] = [320 / 8, 352 / 8, (rw - 320) / 8, (rh - 352) / 8]
    outs[6][idx] = [(rw - 320) / 8, (rh - 352) / 8] * 5  # all at (rw, rh)
    faces = select_faces(outs, (640, 640), (rw / w, rh / h), 0.5, 0.4)
    box = faces[0]["box"]
    assert (box["x"], box["y"]) == (0.0, 0.0)
    assert box["x"] + box["w"] == float(w), "canvas x=rw is frame x=w exactly"
    assert box["y"] + box["h"] == float(h), "canvas y=rh is frame y=h exactly"
    assert faces[0]["landmarks"][0] == [float(w), float(h)]


def test_nms_keeps_the_best_of_overlapping_detections():
    outs = _tensors()
    base = ((160 // 8) * (640 // 8) + (160 // 8)) * NUM_ANCHORS
    for i, score in ((base, 0.9), (base + 1, 0.7)):  # same cell, both anchors
        outs[0][i] = score
        outs[3][i] = [2, 2, 2, 2]
    faces = select_faces(outs, (640, 640), (1.0, 1.0), 0.5, 0.4)
    assert len(faces) == 1 and faces[0]["conf"] == 0.9


def test_below_threshold_is_empty_not_an_error():
    assert select_faces(_tensors(), (640, 640), (1.0, 1.0), 0.5, 0.4) == []
