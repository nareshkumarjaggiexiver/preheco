"""Unit tests for the cv2 frame-annotation helpers (synthetic frames only)."""

import numpy as np
from app import annotate


def _blank(h: int = 240, w: int = 320) -> np.ndarray:
    """A black BGR frame to draw onto."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_to_jpeg_is_valid_jpeg():
    """Encoding a frame yields JPEG bytes (SOI marker 0xFFD8)."""
    data = annotate.to_jpeg(_blank())
    assert isinstance(data, bytes) and data[:2] == b"\xff\xd8"


def test_draw_persons_marks_the_frame_without_mutating_source():
    """Drawing returns a changed copy and leaves the original untouched."""
    img = _blank()
    out = annotate.draw_persons(img, [{"x": 40, "y": 30, "w": 60, "h": 120, "conf": 0.91}])
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.any()  # something was drawn
    assert not img.any()  # source frame is unchanged (drawn on a copy)


def test_draw_faces_colours_by_quality_band():
    """Canon / sub-canon / gated faces draw in distinct colours."""
    img = _blank()
    faces = [
        {"box": {"x": 10, "y": 10, "w": 90, "h": 110}},  # canon -> green
        {"box": {"x": 120, "y": 10, "w": 64, "h": 80}},  # sub-canon -> amber
        {"box": {"x": 230, "y": 10, "w": 40, "h": 50}},  # gated -> red
    ]
    out = annotate.draw_faces(img, faces, min_px=56.0, canon_px=80.0)
    colours = {tuple(int(c) for c in px) for px in out.reshape(-1, 3) if px.any()}
    # At least three distinct non-black colours appear (the three bands + text).
    assert len(colours) >= 3


def test_draw_faces_marks_a_wide_face_the_gate_rejected():
    """A face can be comfortably wide and still not be counted.

    Colouring on width alone would draw a reassuring green box around a guest
    the gate discarded for pose or focus — the overlay exists precisely so an
    operator can see who was not counted and why.
    """
    img = _blank()
    box = {"x": 10, "y": 10, "w": 120, "h": 150}  # far above the 80 px canon

    def drawn(faces):
        """Non-black pixels of the overlay (antialiased text included)."""
        out = annotate.draw_faces(img, faces, min_px=56.0, canon_px=80.0)
        return out.reshape(-1, 3)[out.reshape(-1, 3).any(axis=1)]

    rejected = drawn([{"box": box, "gateReason": "frontality"}])
    # BGR: pure red means the blue and green channels are untouched.
    assert (rejected[:, 0] == 0).all() and (rejected[:, 1] == 0).all(), (
        "a gated face is red whatever its width"
    )

    # ...and the same box, kept, is green — not a warning at all.
    kept = drawn([{"box": box, "gateReason": None}])
    assert (kept[:, 1] > 0).any() and not (kept[:, 2] > 0).any()


def test_draw_tracks_and_matches_run():
    """Track and match overlays draw ids/labels without raising."""
    img = _blank()
    tracks = [{"id": 7, "box": {"x": 20, "y": 20, "w": 50, "h": 90}, "ageFrames": 5, "hits": 4}]
    assert annotate.draw_tracks(img, tracks).any()
    box_a = {"x": 20, "y": 20, "w": 50, "h": 90}
    box_b = {"x": 90, "y": 20, "w": 50, "h": 90}
    verdicts = [
        {"personKey": "p00001", "cosine": 0.42, "isStaff": False, "box": box_a},
        {"personKey": "st-1", "cosine": 0.6, "isStaff": True, "box": box_b},
    ]
    assert annotate.draw_matches(img, verdicts).any()


def test_render_dispatches_each_stage_to_jpeg():
    """render() returns JPEG bytes for every visual stage, ingest included."""
    img = _blank()
    pbox = {"x": 40, "y": 30, "w": 60, "h": 120}
    fbox = {"x": 40, "y": 30, "w": 85, "h": 110}
    last = {
        "boxes": [{**pbox, "conf": 0.9}],
        "tracks": [{"id": 1, "box": pbox, "ageFrames": 3, "hits": 2}],
        "faces": [{"box": fbox}],
        "verdicts": [{"personKey": "p00001", "cosine": 0.4, "isStaff": False, "box": fbox}],
    }
    for stage in annotate.STAGES:
        data = annotate.render(stage, img, last, 56.0, 80.0)
        assert data[:2] == b"\xff\xd8", stage


def test_render_unknown_stage_falls_back_to_raw_frame():
    """An unexpected stage name never raises inside the best-effort loop."""
    data = annotate.render("count", _blank(), {}, 56.0, 80.0)
    assert data[:2] == b"\xff\xd8"


def test_draw_zones_scales_normalized_points_and_labels():
    """Zones draw from normalized 0..1 vertices scaled by the frame size.

    The fill is translucent (the operator must see the partition BEHIND the
    zone to judge their own drawing), so pixels inside the polygon are dim
    magenta and pixels far outside stay black.
    """
    img = _blank(h=240, w=320)
    zones = [{"label": "mirror", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]}]
    out = annotate.draw_zones(img, zones)
    assert not img.any(), "drawn on a copy"
    inside = out[60, 80]  # (0.25, 0.25) of the frame — inside the zone
    assert inside[0] > 0 and inside[2] > 0, "translucent magenta fill (B and R)"
    assert inside[1] == 0
    assert not out[230, 310].any(), "far corner untouched"
    assert not annotate.draw_zones(img, []).any(), "no zones: an untouched copy"


def test_draw_faces_marks_a_zone_eaten_face_in_magenta():
    """A zone-excluded face is neither red (gated) nor green (kept).

    It may be a perfectly good face in the wrong PLACE — behind the frosted
    partition the zone was drawn for — so it borrows the zone's own colour.
    """
    img = _blank()
    box = {"x": 10, "y": 10, "w": 90, "h": 110}  # canon-wide: green if kept
    out = annotate.draw_faces(
        img, [{"box": box, "excludedByZone": True}], min_px=56.0, canon_px=80.0
    )
    px = out.reshape(-1, 3)[out.reshape(-1, 3).any(axis=1)]
    assert (px[:, 0] > 0).any() and (px[:, 2] > 0).any(), "magenta: blue + red"
    assert (px[:, 1] == 0).all(), "no green anywhere — this face was not kept"


def test_render_face_detect_draws_zones_from_the_snapshot():
    """The face-detect overlay carries the zone polygons the loop recorded."""
    img = _blank()
    last = {
        "faces": [],
        "zones": [{"label": "tv", "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]}],
    }
    data = annotate.render("face-detect", img, last, 56.0, 80.0)
    assert data[:2] == b"\xff\xd8"
