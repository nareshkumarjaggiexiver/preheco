"""The v3 torso descriptor: shape, the three None conditions, colour vs pattern.

Every image here is SYNTHETIC and known: a plain colour, a stripe, a skin
tone.  The assertions pin the wire contract (64 floats summing to 1.0, None
for the three unmeasurable cases, None for a v2/v3 length mismatch) and the
two behaviours v3 was built for on run f0bfc5 — a colour clash the old
chin-down crop could not see, and a pattern difference a colour histogram
cannot see at all.
"""

import cv2
import numpy as np
import pytest
from heco_counting import appearance as ap

#: A torso-sized synthetic frame: the face sits at the top, the person box
#: extends well below it, so the v3 band (0.5..3.0 face heights under the
#: face) is fully inside both the box and the image.
FACE = {"x": 100.0, "y": 20.0, "w": 80.0, "h": 100.0}
PERSON = {"x": 50.0, "y": 0.0, "w": 200.0, "h": 460.0}
SHAPE = (480, 300, 3)


def plain(bgr) -> np.ndarray:
    """A frame of one flat colour (BGR)."""
    img = np.zeros(SHAPE, dtype=np.uint8)
    img[:] = bgr
    return img


def striped(bgr_a, bgr_b, period: int = 16) -> np.ndarray:
    """Horizontal stripes alternating two colours, ``period`` px per pair."""
    img = plain(bgr_a)
    for y in range(0, SHAPE[0], period):
        img[y : y + period // 2] = bgr_b
    return img


RED = (0, 0, 200)
BLUE = (200, 0, 0)
DARK_RED = (0, 0, 120)
#: A skin tone squarely inside the YCrCb window (Cr ~150, Cb ~110).
SKIN = (140, 170, 220)


def parts(d):
    """Split a descriptor into its colour, texture, edge and reserved parts."""
    v = np.asarray(d)
    return v[: ap.COLOUR_BINS], v[ap.TEXTURE_OFFSET : ap.TEXTURE_OFFSET + ap.TEXTURE_BINS], \
        v[ap.EDGE_OFFSET : ap.EDGE_OFFSET + ap.EDGE_BINS], v[ap.EDGE_OFFSET + ap.EDGE_BINS :]


# ------------------------------------------------------------- the wire shape


def test_descriptor_is_64_floats_summing_to_one():
    """The wire shape: 64 plain floats, non-negative, summing to 1.0."""
    d = ap.torso_descriptor(plain(RED), FACE, PERSON)
    assert d is not None
    assert len(d) == ap.APPEARANCE_DIM == 64
    assert all(isinstance(x, float) for x in d)
    assert sum(d) == pytest.approx(1.0, abs=1e-9)
    assert min(d) >= 0.0


def test_parts_carry_their_contract_weights_and_reserved_bins_are_zero():
    """Colour 0.9, texture 0.07, edge 0.03; the last 12 bins stay zero.

    The first cut shipped 0.7/0.2/0.1 and measured on the run's own crops
    that 0.3 of pattern weight lifted every impostor pair by ~0.2 (plain
    cloth agrees with plain cloth on weave) and separated nothing; the
    pattern parts are a 0.1 tie-break now.
    """
    assert (ap.W_COLOUR, ap.W_TEXTURE, ap.W_EDGE) == (0.9, 0.07, 0.03)
    colour, tex, edge, reserved = parts(ap.torso_descriptor(plain(RED), FACE, PERSON))
    assert colour.sum() == pytest.approx(ap.W_COLOUR, abs=1e-9)
    assert tex.sum() == pytest.approx(ap.W_TEXTURE, abs=1e-9)
    assert edge.sum() == pytest.approx(ap.W_EDGE, abs=1e-9)
    assert reserved.shape == (12,) and not reserved.any()


def test_band_starts_below_the_neck_and_stops_at_the_person_box():
    """The crop rule, in pixels: y0 = face bottom + 0.5 fh, y1 = min(face
    bottom + 3.0 fh, person bottom), x = box inset 15% a side."""
    assert ap._band(FACE, PERSON, 300, 480) == (80, 220, 170, 420)
    short = dict(PERSON, h=300.0)  # the box ends before 3 face heights
    assert ap._band(FACE, short, 300, 480) == (80, 220, 170, 300)
    # The face-column clip binds only on a box wider than the column (an
    # arm out: 700 px against an 80 px face): x = face centre +- 1.5 widths.
    assert ap._band(FACE, dict(PERSON, x=-200.0, w=700.0), 300, 480) == (20, 260, 170, 420)


# ------------------------------------------------------------- None conditions


def test_no_person_box_is_none():
    """None condition 1: a face with no containing person box has no torso."""
    assert ap.torso_descriptor(plain(RED), FACE, None) is None


def test_crop_under_24px_is_none():
    """None condition 2: a band under 24 px in either dimension."""
    # A person box only 26 px wide insets to 18 px: too thin to histogram.
    thin = {"x": 127.0, "y": 0.0, "w": 26.0, "h": 460.0}
    assert ap.torso_descriptor(plain(RED), FACE, thin) is None
    # A person box ending 20 px under the band's top: too short.
    short = {"x": 50.0, "y": 0.0, "w": 200.0, "h": 190.0}
    assert ap.torso_descriptor(plain(RED), FACE, short) is None


def test_fewer_than_100_unmasked_pixels_is_none():
    """None condition 3: shadow (V < 30) everywhere leaves nothing to read."""
    assert ap.torso_descriptor(plain((10, 10, 10)), FACE, PERSON) is None


def test_a_face_at_the_edge_of_its_box_is_none():
    """None condition 4 (post-contract): p00047, half behind a pillar.

    Her face centre sat 0.17 face widths from the person box's right edge
    and the band beside it was the pillar, which the review then presented
    as a measured clothes reading.  A face within 0.4 widths of either side
    is at the edge of a cut box; a centred face is >= 1.25 from both.
    """
    at_left = dict(PERSON, x=140.0 - 0.3 * 80.0)     # face centre 0.3 widths in
    at_right = dict(PERSON, x=140.0 + 0.3 * 80.0 - 200.0)
    assert ap.torso_descriptor(plain(RED), FACE, at_left) is None
    assert ap.torso_descriptor(plain(RED), FACE, at_right) is None
    inside = dict(PERSON, x=140.0 - 0.5 * 80.0)      # 0.5 widths: measured
    assert ap.torso_descriptor(plain(RED), FACE, inside) is not None
    assert ap.FACE_EDGE_WIDTHS == 0.4


def test_an_unmeasurable_pattern_makes_the_sighting_none(monkeypatch):
    """None condition 5 (post-contract): no interior cloth pixel after the resize.

    Handing the pattern weight to colour capped every comparison of that
    sighting at 0.9 — the absent reading behaved as a 0.1 clash.  Absent is
    not zero: the sighting is not measured.
    """
    monkeypatch.setattr(ap, "_texture_parts", lambda g, m: (np.zeros(10), np.zeros(3)))
    assert ap.torso_descriptor(plain(RED), FACE, PERSON) is None


def test_saturated_orange_is_cloth_not_skin_at_any_exposure():
    """The exposure cliff: an orange kurta (H 15, S 255) at V 150 vs V 155.

    The Cr/Cb window holds fully saturated orange for every V up to 152, so
    a 3% exposure move flipped the whole garment between a quarter weight
    and full weight and two sightings of it intersected at 0.79.  No skin
    is that saturated; at S >= 150 a pixel is cloth whatever Cr/Cb say.
    """
    def kurta_with_dupatta(v):
        bgr = cv2.cvtColor(np.array([[[15, 255, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        img = plain(tuple(int(c) for c in bgr))
        img[:, :150] = (245, 245, 245)  # a white dupatta over half the band
        return img
    dim, lit = kurta_with_dupatta(150), kurta_with_dupatta(155)
    assert ap.intersection(
        ap.torso_descriptor(dim, FACE, PERSON), ap.torso_descriptor(lit, FACE, PERSON)
    ) > 0.95
    w = ap.skin_weights(kurta_with_dupatta(150))
    assert w[:, 200:].min() == 1.0, "saturated orange weighs as cloth"
    assert ap.skin_weights(plain(SKIN)).max() == ap.SKIN_WEIGHT, "skin still a quarter"


def test_skin_only_band_is_none_not_a_descriptor():
    """A band that is all skin is no cloth reading: None, never a histogram
    of skin that would then 'match' every other bare chest."""
    assert ap.torso_descriptor(plain(SKIN), FACE, PERSON) is None


def test_skin_pixels_count_a_quarter_in_the_colour_part():
    """Half skin, half red cloth: the skin half carries SKIN_WEIGHT of a
    pixel each, so red holds 1 / (1 + 0.25) = 80% of the colour mass (v2
    histogrammed the skin at full weight as hue bin 1)."""
    half = plain(RED)
    half[:, :150] = SKIN
    c_mixed = parts(ap.torso_descriptor(half, FACE, PERSON))[0]
    c_pure = parts(ap.torso_descriptor(plain(RED), FACE, PERSON))[0]
    red_share = np.minimum(c_mixed, c_pure).sum() / c_pure.sum()
    assert red_share == pytest.approx(1.0 / (1.0 + ap.SKIN_WEIGHT), abs=0.02)


def test_skin_toned_garment_still_reads_as_its_own_colour():
    """The p00065 lesson: a pink checked shirt was 95-100% 'skin' to the
    YCrCb window.  A band that is 95% skin-toned cloth and 5% dark thread
    must read as the skin-toned colour, not as the 5% the window missed."""
    shirt = plain(SKIN)
    shirt[::20] = (40, 40, 40)  # a dark thread every 20 rows, ~5% of pixels
    d = ap.torso_descriptor(shirt, FACE, PERSON)
    assert d is not None  # the 5% clears the 100-pixel None rule
    colour = parts(d)[0]
    # The chromatic bin the skin tone itself lands in, from its HSV reading.
    h, s, _ = cv2.cvtColor(np.array([[SKIN]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    assert s >= ap.S_ACHROMATIC
    skin_bin = (int(h) * ap.H_BINS // 180) * ap.S_BINS + (
        (int(s) - ap.S_ACHROMATIC) * ap.S_BINS // (256 - ap.S_ACHROMATIC)
    )
    # 95% at a quarter weight (0.2375) against 5% at full weight (0.05):
    # the garment's own hue holds ~83% of the colour mass.
    assert colour[skin_bin] / colour.sum() == pytest.approx(0.2375 / 0.2875, abs=0.03)


# ------------------------------------------------------------- what it sees


def test_plain_red_vs_plain_blue_clash_on_colour():
    """Different plain colours share NO colour mass; the pattern parts (0.1 of
    the descriptor) agree because both are plain — so the intersection is the
    pattern share and nothing more."""
    d_red = ap.torso_descriptor(plain(RED), FACE, PERSON)
    d_blue = ap.torso_descriptor(plain(BLUE), FACE, PERSON)
    sim = ap.intersection(d_red, d_blue)
    assert np.minimum(parts(d_red)[0], parts(d_blue)[0]).sum() == pytest.approx(0.0, abs=1e-9)
    assert sim == pytest.approx(ap.W_TEXTURE + ap.W_EDGE, abs=1e-6)
    assert sim < 0.35  # under the runner's heal clash floor


def test_same_colour_plain_vs_striped_differ_in_texture_not_colour():
    """A stripe is the failure v2 could not see (#4, a white shirt vs a white
    striped shirt at 0.71): the hue is the same, the pattern parts are not."""
    d_plain = ap.torso_descriptor(plain(RED), FACE, PERSON)
    d_stripe = ap.torso_descriptor(striped(RED, DARK_RED), FACE, PERSON)
    c_plain, t_plain, e_plain, _ = parts(d_plain)
    c_stripe, t_stripe, e_stripe, _ = parts(d_stripe)
    # Same hue bin: the chromatic mass overlaps almost entirely.
    assert np.minimum(c_plain, c_stripe).sum() > 0.6
    # Texture: plain cloth is entirely the "flat" code (bin 0, no neighbour
    # clears the noise threshold); the stripe moves at least a third of the
    # texture mass into edge codes.
    assert t_plain[0] == pytest.approx(ap.W_TEXTURE, abs=1e-9)
    assert t_stripe[1:].sum() > ap.W_TEXTURE / 3.0
    assert np.minimum(t_plain, t_stripe).sum() < 0.7 * ap.W_TEXTURE
    # Edges: plain cloth has no "high" gradient at all; the stripe does.
    assert np.minimum(e_plain, e_stripe).sum() < 0.6 * ap.W_EDGE
    assert e_stripe[2] > 0.0 and e_plain[2] == pytest.approx(0.0, abs=1e-9)
    assert ap.intersection(d_plain, d_stripe) < ap.intersection(d_plain, d_plain)


def test_identical_images_intersect_at_one():
    """The same band twice: full overlap in every part."""
    d = ap.torso_descriptor(striped(RED, BLUE), FACE, PERSON)
    assert ap.intersection(d, d) == pytest.approx(1.0, abs=1e-9)


def test_pattern_is_body_relative_not_pixel_relative():
    """The same stripe seen twice as far away (half the pixels) reads the same
    texture, because the band is resized to a fixed width before LBP."""
    near = striped(RED, DARK_RED, period=32)
    far = np.zeros((240, 150, 3), np.uint8)
    far[:] = near[::2, ::2]
    d_near = ap.torso_descriptor(near, FACE, PERSON)
    d_far = ap.torso_descriptor(
        far, {k: v / 2 for k, v in FACE.items()}, {k: v / 2 for k, v in PERSON.items()}
    )
    assert ap.intersection(d_near, d_far) > 0.9


# ------------------------------------------------------------- intersection


def test_intersection_mixed_versions_is_none_not_zero_not_raise():
    """A 48-float v2 row beside a 64-float v3 one is not comparable: None."""
    d3 = ap.torso_descriptor(plain(RED), FACE, PERSON)
    v2 = [1.0 / 48] * 48
    assert ap.intersection(d3, v2) is None
    assert ap.intersection(v2, d3) is None
    assert ap.intersection([], []) is None


def test_intersection_is_bounded_and_symmetric():
    """Intersection lives in 0..1 and does not care which side is which."""
    a = ap.torso_descriptor(plain(RED), FACE, PERSON)
    b = ap.torso_descriptor(striped(BLUE, RED), FACE, PERSON)
    assert 0.0 <= ap.intersection(a, b) <= 1.0
    assert ap.intersection(a, b) == pytest.approx(ap.intersection(b, a), abs=1e-12)
