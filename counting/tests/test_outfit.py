"""The lower-body outfit descriptor: wire shape, band, None rules, lighting.

Every image here is SYNTHETIC and known: a standing person box on a plain
frame whose upper rows are one colour (the shirt) and lower rows another (the
trousers).  The assertions pin the wire contract the match service relies on
(64 floats summing to 1.0 in the torso descriptor's exact layout, so one
intersection compares both) and the None rules that keep a seated, cut,
hidden or bare lower body from ever reading as a clash.
"""

import cv2
import numpy as np
import pytest
from heco_counting import appearance as ap
from heco_counting import outfit as of

#: A standing guest on a 1000 x 400 frame: the box is 4x as tall as wide and
#: 13 face widths tall, its bottom 150 px above the frame's bottom edge.
#: Band: hip line 150 + 3 x 80 = 390, knee 50 + 0.72 x 800 = 626, sides
#: 100 + 60 .. 300 - 60.
SHAPE = (1000, 400, 3)
PERSON = {"x": 100.0, "y": 50.0, "w": 200.0, "h": 800.0}
FACE = {"x": 170.0, "y": 70.0, "w": 60.0, "h": 80.0}
#: Where the shirt stops and the trousers start in the synthetic frames —
#: above the band's top edge, so the band reads trousers only.
WAIST_Y = 300

RED = (0, 0, 200)
BLUE = (200, 0, 0)
WHITE = (235, 235, 235)
CHARCOAL = (60, 60, 60)
#: Black trousers as this hall's camera sees them: under the torso's shadow
#: floor (V 30), which is why the lower band counts dark as cloth.
BLACK = (15, 15, 15)
#: A skin tone squarely inside the YCrCb window (Cr ~150, Cb ~110).
SKIN = (140, 170, 220)
ACHROMATIC = slice(ap.H_BINS * ap.S_BINS, ap.COLOUR_BINS)


def dressed(shirt, trousers) -> np.ndarray:
    """A frame whose rows above WAIST_Y are ``shirt`` and below are ``trousers``."""
    img = np.zeros(SHAPE, dtype=np.uint8)
    img[:WAIST_Y] = shirt
    img[WAIST_Y:] = trousers
    return img


def lower(img, person=PERSON, face=FACE, **kw):
    """The descriptor with the module's defaults for everything not given."""
    return of.lower_descriptor(img, person, face, **kw)


# ------------------------------------------------------------- the wire shape


def test_descriptor_is_64_floats_summing_to_one_in_the_torso_layout():
    """64 plain floats, non-negative, summing to 1.0; colour 0.9 / texture
    0.07 / edges 0.03 / 12 reserved zeros — the torso's layout exactly, so
    the match service stores and intersects it the same way."""
    d = lower(dressed(WHITE, RED))
    assert d is not None
    assert len(d) == ap.APPEARANCE_DIM == 64
    assert all(isinstance(x, float) for x in d)
    assert sum(d) == pytest.approx(1.0, abs=1e-9)
    assert min(d) >= 0.0
    v = np.asarray(d)
    assert v[: ap.COLOUR_BINS].sum() == pytest.approx(ap.W_COLOUR, abs=1e-9)
    assert v[ap.TEXTURE_OFFSET : ap.EDGE_OFFSET].sum() == pytest.approx(ap.W_TEXTURE, abs=1e-9)
    edge_end = ap.EDGE_OFFSET + ap.EDGE_BINS
    assert v[ap.EDGE_OFFSET : edge_end].sum() == pytest.approx(ap.W_EDGE, abs=1e-9)
    assert not v[edge_end:].any()


def test_colour_and_pattern_bin_exactly_like_the_torso():
    """The partitions are copied here (appearance keeps them private); this
    pins them to the torso's on random pixels, so the two descriptors can
    never drift into comparing different histograms under one layout."""
    rng = np.random.default_rng(7)
    crop = rng.integers(0, 256, size=(60, 90, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    weights = rng.choice([0.0, ap.SKIN_WEIGHT, 1.0], size=crop.shape[:2])
    np.testing.assert_allclose(of._colour_part(hsv, weights), ap._colour_part(hsv, weights))
    gray = rng.integers(0, 256, size=(40, 64), dtype=np.uint8)
    mask = rng.random((40, 64)) > 0.3
    mine, theirs = of._texture_parts(gray, mask), ap._texture_parts(gray, mask)
    for a, b in zip(mine, theirs, strict=True):
        np.testing.assert_allclose(a, b)


# ------------------------------------------------------------------ the band


def test_band_is_the_thighs_from_the_hip_line_to_the_knee():
    """Face known: the hip line is 3 face heights under the face's bottom
    edge; unknown: 55% down the box.  The band stops at the knee, 72% down
    the box — not at the feet, where a stride shows the floor — and is
    inset 30% a side."""
    assert (of.HIP_FACE_HEIGHTS, of.HIP_BOX_FRAC, of.KNEE_BOX_FRAC, of.SIDE_INSET_FRAC) == (
        3.0, 0.55, 0.72, 0.30,
    )
    assert of.lower_band(PERSON, FACE, 400, 1000) == (160, 240, 390, 626)
    assert of.lower_band(PERSON, None, 400, 1000) == (160, 240, 490, 626)
    # Clamped to the image: a box overhanging the left edge starts at 0.
    assert of.lower_band(dict(PERSON, x=-60.0), FACE, 400, 1000)[:2] == (0, 80)


def test_the_band_reads_the_trousers_not_the_shirt():
    """Same trousers under different shirts agree; the shirt is the torso's job."""
    a = lower(dressed(WHITE, CHARCOAL))
    b = lower(dressed(RED, CHARCOAL))
    assert of.intersection(a, b) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------- clash / agree


def test_red_and_blue_trousers_clash_same_colour_agrees():
    """Plain red vs plain blue share only the pattern parts (0.1, the floor
    of any two plain garments); the same red under a different shirt is 1.0."""
    red = lower(dressed(WHITE, RED))
    blue = lower(dressed(WHITE, BLUE))
    assert of.intersection(red, blue) == pytest.approx(ap.W_TEXTURE + ap.W_EDGE, abs=1e-6)
    assert of.intersection(red, lower(dressed(BLUE, RED))) == pytest.approx(1.0, abs=1e-9)


def test_black_trousers_are_cloth_not_shadow():
    """Under V 30 the torso masks a pixel as shadow; on legs it is black
    cloth.  It reads as the dark achromatic bin — never a hue, never skin —
    and clashes with a white pyjama instead of vanishing."""
    black = np.asarray(lower(dressed(WHITE, BLACK)))
    assert black[ACHROMATIC][0] == pytest.approx(ap.W_COLOUR, abs=1e-9)
    white = lower(dressed(WHITE, WHITE))
    assert of.intersection(black.tolist(), white) == pytest.approx(
        ap.W_TEXTURE + ap.W_EDGE, abs=1e-6
    )
    # Charcoal (V 60) is in the same dark bin: exposure drift between the
    # two does not read as a change of trousers.
    assert of.intersection(black.tolist(), lower(dressed(WHITE, CHARCOAL))) == pytest.approx(
        1.0, abs=1e-9
    )


# ------------------------------------------------------------- None rules


def test_no_person_box_is_none():
    """Nothing to hang a band from."""
    assert of.lower_descriptor(dressed(WHITE, RED), None, FACE) is None


def test_a_seated_wide_box_is_none():
    """h/w under 2.0: seated or bending — the band would be lap and chair."""
    seated = {"x": 50.0, "y": 400.0, "w": 300.0, "h": 400.0}
    assert lower(dressed(WHITE, RED), person=seated) is None


def test_a_box_under_six_face_widths_is_none_only_when_the_face_is_known():
    """A head-to-waist box with a standing aspect (an occlusion's box) is
    not a full body; without a face the aspect is all there is to go on."""
    short = {"x": 150.0, "y": 50.0, "w": 100.0, "h": 300.0}  # 5 face widths
    assert lower(dressed(WHITE, RED), person=short) is None
    assert lower(dressed(WHITE, RED), person=short, face=None) is not None


def test_a_box_cut_by_the_frame_bottom_is_none():
    """Box bottom within 10 px of the frame's bottom edge: the legs
    continue out of shot and the knee estimate is a guess."""
    img = dressed(WHITE, RED)
    cut = dict(PERSON, h=1000.0 - PERSON["y"] - 5.0)
    assert lower(img, person=cut) is None
    # The caller's frame height wins over the image's: the same box in a
    # crop of a taller frame is not cut.
    assert lower(img, person=cut, frame_h=1400) is not None


def test_a_band_shorter_than_half_a_face_is_none(monkeypatch):
    """A box that is short for its face (cut by the frame but ending just
    above the 10 px margin, or crouching) raises the knee to the hip: the
    46 px band here is only 0.35 face heights and would read the shirt hem."""
    big_face = {"x": 155.0, "y": 60.0, "w": 90.0, "h": 130.0}
    band_h = PERSON["y"] + of.KNEE_BOX_FRAC * PERSON["h"] - (60.0 + 130.0 * 4.0)
    assert of.MIN_CROP_PX <= band_h < of.MIN_BAND_FACE_HEIGHTS * 130.0
    assert lower(dressed(WHITE, RED), face=big_face) is None
    monkeypatch.setattr(of, "MIN_BAND_FACE_HEIGHTS", 0.0)
    assert lower(dressed(WHITE, RED), face=big_face) is not None


def test_someone_in_front_covering_the_band_is_none():
    """Another box whose feet are level with or below this guest's (nearer
    the camera) covering more than 30% of the band hides the legs."""
    img = dressed(WHITE, RED)
    front = {"x": 120.0, "y": 350.0, "w": 170.0, "h": 600.0}  # feet at 950
    assert lower(img, others=[front]) is None
    # A box clipping a corner of the band does not hide the legs.
    corner = {"x": 220.0, "y": 560.0, "w": 150.0, "h": 400.0}
    band = of.lower_band(PERSON, FACE, 400, 1000)
    assert of.covered_fraction(band, of.in_front(PERSON, [corner])) < of.OCCLUDED_FRAC
    assert lower(img, others=[corner]) is not None
    # The guest's own box among 'others' is not someone else.
    assert lower(img, others=[dict(PERSON)]) is not None


def test_someone_behind_does_not_hide_the_legs():
    """A box over the band whose feet are HIGHER in the frame stands
    further from the camera: this guest hides them, not the reverse."""
    behind = {"x": 120.0, "y": 150.0, "w": 170.0, "h": 600.0}  # feet at 750
    assert of.in_front(PERSON, [behind]) == []
    assert lower(dressed(WHITE, RED), others=[behind]) is not None


def test_a_skin_only_band_is_none():
    """Bare legs (or a garment inside the skin window) leave no cloth to read."""
    assert lower(dressed(WHITE, SKIN)) is None


def test_a_blown_out_band_is_none():
    """Fewer than 100 readable pixels: a clipped highlight is not a colour."""
    assert lower(dressed(WHITE, (255, 255, 255))) is None


def test_a_band_under_24_px_is_none():
    """A far, thin box: the inset band is under MIN_CROP_PX wide."""
    thin = {"x": 180.0, "y": 50.0, "w": 50.0, "h": 800.0}  # band 20 px wide
    assert lower(dressed(WHITE, RED), person=thin, face=None) is None


# ------------------------------------------------------------- lighting


def test_gains_move_a_colour_cast_into_the_achromatic_bins():
    """White cloth under a violet stage wash reads chromatic (hue bins);
    the frame's gains undo the cast and the same cloth reads light grey."""
    cast = (230, 170, 190)  # BGR: white under a blue-magenta light
    img = dressed(WHITE, cast)
    raw = np.asarray(lower(img))
    inverse = [1.0 / c for c in cast]
    gains = tuple(g / (sum(inverse) / 3.0) for g in inverse)
    fixed = np.asarray(lower(img, gains=gains))
    chroma = ap.H_BINS * ap.S_BINS
    assert raw[:chroma].sum() == pytest.approx(ap.W_COLOUR, abs=1e-9)
    assert fixed[:chroma].sum() == pytest.approx(0.0, abs=1e-9)
    assert fixed[ACHROMATIC].sum() == pytest.approx(ap.W_COLOUR, abs=1e-9)
    # The pattern parts read the pixels as the camera delivered them.
    np.testing.assert_allclose(raw[ap.COLOUR_BINS :], fixed[ap.COLOUR_BINS :])


def test_unit_gains_are_no_gains():
    """(1, 1, 1) is exactly the descriptor without lighting correction."""
    img = dressed(WHITE, RED)
    assert lower(img, gains=(1.0, 1.0, 1.0)) == lower(img)


def test_intersection_is_the_torso_intersection():
    """One comparison for both descriptors: a v2-length vector is not comparable."""
    d = lower(dressed(WHITE, RED))
    assert of.intersection(d, d) == pytest.approx(1.0, abs=1e-9)
    assert of.intersection(d, [1.0 / 48] * 48) is None
