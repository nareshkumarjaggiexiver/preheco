"""Unit tests for the cv2 frame-annotation helpers (synthetic frames only)."""

import cv2
import numpy as np
import pytest
from app import annotate


def _decode(data: bytes) -> np.ndarray:
    """JPEG bytes back to a BGR image, for size assertions."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


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


# ------------------------------- tap-frame downscale (the 4K death spiral, P1)


def test_render_downscales_a_4k_frame_to_console_width():
    """Every stage's tap frame leaves render() at most 1280 px wide.

    A tap round used to annotate and JPEG-encode FIVE full 3840x2160 frames —
    ~1-1.5 s per round on the T440, the entire gap between healthy stage
    timings and the observed 0.4 fps lock-in — for a console that displays
    them small.  Aspect must be preserved: 3840x2160 -> 1280x720.
    """
    img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    last = {
        "boxes": [{"x": 400, "y": 300, "w": 600, "h": 1200, "conf": 0.9}],
        "tracks": [],
        "faces": [{"box": {"x": 420, "y": 320, "w": 85, "h": 110}}],
        "verdicts": [],
    }
    for stage in annotate.STAGES:
        out = _decode(annotate.render(stage, img, last, 56.0, 80.0))
        assert out.shape[1] == 1280, stage
        assert out.shape[0] == 720, f"{stage}: aspect must be preserved"


def test_render_never_upscales_a_small_frame():
    """A frame already under the cap passes through at its own size.

    Upscaling would blur the source and spend encode time buying nothing.
    """
    out = _decode(annotate.render("ingest", _blank(h=240, w=320), {}, 56.0, 80.0))
    assert out.shape[:2] == (240, 320)


def test_downscale_is_exact_at_the_cap_and_identity_below_it():
    """The helper itself: 1280 stays 1280 (no work), 1281 shrinks to 1280."""
    at_cap = np.zeros((100, 1280, 3), dtype=np.uint8)
    assert annotate.downscale(at_cap) is at_cap  # untouched, not even copied
    over = annotate.downscale(np.zeros((100, 1281, 3), dtype=np.uint8))
    assert over.shape[1] == 1280


# ------------------------------- labels that survive the downscale
#
# REPORTED FROM THE FIELD, 2026-08-07: "in the frame when you do the facebox
# the number is not very clear."  It was not a rendering bug — the label was
# drawn correctly and then destroyed.  Annotations go on at NATIVE resolution
# and the finished frame is shrunk to TAP_MAX_WIDTH, so a fixed 0.5-scale,
# 1 px label on a 2560 px camera reaches the screen at an effective 0.25 with
# a half-pixel stroke.  The BOXES survived (2 px of solid colour); six
# characters of hairline text did not, so an operator could see which faces
# were matched and not who to — on the one overlay a disputed count turns on.


def test_label_metrics_cancel_the_downscale_the_frame_is_about_to_get():
    """Sized in the coordinates it will be READ in, not the ones it is drawn in."""
    at_cap = np.zeros((720, annotate.TAP_MAX_WIDTH, 3), dtype=np.uint8)
    assert annotate.label_metrics(at_cap) == (0.5, 1), "no downscale, no compensation"

    # 2x the cap: the downscale halves everything, so the label starts double.
    twice = np.zeros((1440, annotate.TAP_MAX_WIDTH * 2, 3), dtype=np.uint8)
    scale, thick = annotate.label_metrics(twice)
    assert scale == 1.0 and thick == 2

    # 4K: the measured camera. Effective on-screen size lands back at ~0.5.
    uhd = np.zeros((2160, 3840, 3), dtype=np.uint8)
    scale, thick = annotate.label_metrics(uhd)
    assert scale == pytest.approx(0.5 * 3840 / annotate.TAP_MAX_WIDTH)
    assert scale * (annotate.TAP_MAX_WIDTH / 3840) == pytest.approx(0.5)

    # A frame SMALLER than the cap is never upscaled, so it is never compensated.
    small = np.zeros((240, 320, 3), dtype=np.uint8)
    assert annotate.label_metrics(small) == (0.5, 1)


def test_a_match_label_is_still_legible_after_the_frame_is_shrunk():
    """THE REGRESSION, end to end: the glyphs survive 4K -> screen.

    Measured as GLYPH MASS — the pixels that form the characters, counted the
    same way for both designs so the comparison is fair rather than flattering.
    In each label strip the most common colour is the ground (the footage for
    the old thin-text label, the chip for the new one) and everything else is
    ink, so this counts what a reader actually has to read, not how the ink
    happens to be coloured.
    """
    uhd = np.full((2160, 3840, 3), 210, dtype=np.uint8)  # a bright doorway
    box = {"x": 1500, "y": 900, "w": 300, "h": 380}
    label = "p00003 0.69"

    now = annotate.downscale(annotate.draw_matches(uhd, [
        {"box": box, "personKey": "p00003", "cosine": 0.6874, "isNew": False},
    ]))

    # The old design, rendered identically: fixed 0.5 scale, 1 px, no chip.
    before = uhd.copy()
    x, y, w, h = annotate._xywh(box)
    cv2.rectangle(before, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(before, label, (x, y - 6), annotate._FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    before = annotate.downscale(before)

    def glyph_mass(img):
        """Pixels forming the characters in the label strip, whatever colour."""
        k = annotate.TAP_MAX_WIDTH / 3840
        sx, sy = int(x * k), int(y * k)
        strip = img[max(0, sy - 26):sy, sx:sx + 160].reshape(-1, 3)
        if not len(strip):
            return 0
        colours, counts = np.unique(strip, axis=0, return_counts=True)
        ground = colours[counts.argmax()]
        return int((np.abs(strip.astype(int) - ground.astype(int)).sum(axis=1) > 90).sum())

    assert glyph_mass(before) > 0, "the fixture must really draw the old label"
    assert glyph_mass(now) > 3 * glyph_mass(before), (
        f"glyph mass after downscale: {glyph_mass(now)} px now "
        f"vs {glyph_mass(before)} px before"
    )


def test_a_label_on_a_face_at_the_very_top_is_not_clipped_away():
    """A face detected against the top edge still says who it is.

    The label prefers to sit ABOVE its box and drops inside when there is no
    room — because a guest walking close to the camera is exactly when the
    face box reaches the frame edge, and exactly when knowing who they are
    matters most.
    """
    img = _blank()
    out = annotate.draw_matches(img, [
        {"box": {"x": 20, "y": 0, "w": 90, "h": 110}, "personKey": "p00001", "isNew": True},
    ])
    # Ink inside the top strip of the box means the chip was placed there.
    assert (out[0:40, 20:150].astype(int).sum(axis=2) > 0).any(), "the label is on the frame"


# ------------------------------- face cards (the guest register's own picture)
#
# REPORTED 2026-08-07: "sometimes all images of that guest come with other
# people ... it is hard to find who it is." The register showed whole annotated
# frames, and in a busy doorway every frame holds three people. A card is one
# guest's face, cut out of the frame, so the register can show WHO rather than
# WHERE.


def test_a_face_card_is_the_face_plus_a_little_head_and_shoulder():
    """Padded past the detector box, which clips the hairline and the chin."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (10, 10, 10)
    img[200:340, 300:400] = (0, 0, 255)  # the "face"
    card = annotate.face_crop(img, {"x": 300, "y": 200, "w": 100, "h": 140})
    assert card is not None
    assert card.shape[1] == annotate.FACE_CARD_WIDTH
    # 35% each side of a 100 px box => 170 px of source, so the face fills
    # roughly 100/170 of the width — most of the card, but not all of it.
    red = (card[:, :, 2] > 100).sum(axis=0) > 0
    assert 0.5 < red.mean() < 0.75, f"face fills {red.mean():.2f} of the card"


def test_a_face_at_the_frame_edge_still_gets_a_card():
    """Clamped, not refused — the guests who walk closest to the lens are
    exactly the ones whose boxes hang over the edge."""
    img = np.full((600, 800, 3), 40, dtype=np.uint8)
    assert annotate.face_crop(img, {"x": 0, "y": 0, "w": 90, "h": 110}) is not None
    assert annotate.face_crop(img, {"x": 740, "y": 520, "w": 90, "h": 110}) is not None


def test_an_impossible_box_yields_no_card_rather_than_a_broken_one():
    """No card is a state the console renders; a 0-px image is not."""
    img = np.full((600, 800, 3), 40, dtype=np.uint8)
    assert annotate.face_crop(img, {"x": 10, "y": 10, "w": 0, "h": 110}) is None
    assert annotate.face_crop(img, {"x": 10, "y": 10, "w": 90, "h": -5}) is None
    assert annotate.face_crop(img, {"x": 5000, "y": 10, "w": 90, "h": 110}) is None
    assert annotate.face_crop(img, {}) is None


def test_a_card_is_cut_from_the_NATIVE_frame_not_the_downscaled_one():
    """A distant guest's card must not also pay the tap round's 3x reduction.

    The register is where an operator decides whether two identities are one
    person. That decision should be made on the pixels the camera gave.
    """
    uhd = np.zeros((2160, 3840, 3), dtype=np.uint8)
    uhd[900:1040, 1500:1600] = (0, 0, 255)
    card = annotate.face_crop(uhd, {"x": 1500, "y": 900, "w": 100, "h": 140})
    assert card.shape[1] == annotate.FACE_CARD_WIDTH
    # Cut from native, a 100 px face becomes 192/170 of its size: UPscaled.
    # Cut from the 1280-wide tap frame it would have been 33 px first.
    assert (card[:, :, 2] > 100).sum() > 8000, "the card carries real detail"
