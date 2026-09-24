"""White balance for the colour descriptors: frame_gains and ``gains=``.

The proof the brief asked for: the SAME garment under a strong red, blue or
amber cast, read with the frame's gains, agrees with its neutral-light read
at >= 0.8 on the colour part, and clearly more than without.  The scene is
synthetic and known: a background of colour patches whose three channels
are permutations of each other (so the scene itself is grey on average, the
shades-of-grey assumption), a textured garment in the torso band, and a
per-channel multiplier as the light.

And the off switch: ``gains=None`` is the descriptor exactly as before.
"""

import itertools

import numpy as np
import pytest
from heco_counting import appearance as ap

FACE = {"x": 100.0, "y": 20.0, "w": 80.0, "h": 100.0}
PERSON = {"x": 50.0, "y": 0.0, "w": 200.0, "h": 460.0}
#: The v3 band of FACE/PERSON: x 80..220, y 170..420.
BAND = (slice(170, 420), slice(80, 220))

#: Casts as BGR multipliers: a red stage wash, a blue one, warm amber lamps.
CASTS = {
    "red": (0.7, 0.9, 1.4),
    "blue": (1.4, 1.0, 0.7),
    "amber": (0.55, 0.95, 1.35),
}
GARMENTS = {
    "green": (60, 140, 60),
    "red": (40, 60, 170),
    "orange": (40, 90, 150),
    "grey-blue": (140, 130, 110),
}


def scene(garment, seed: int = 7) -> np.ndarray:
    """A 960x720 frame: channel-balanced patches, a textured garment in the band."""
    rng = np.random.default_rng(seed)
    img = np.zeros((720, 960, 3), dtype=np.uint8)
    perms = list(itertools.permutations(range(3)))
    base = None
    for n, (y, x) in enumerate(itertools.product(range(0, 720, 24), range(0, 960, 24))):
        if n % 6 == 0:
            base = rng.integers(25, 165, 3)
        img[y:y + 24, x:x + 24] = base[list(perms[n % 6])]
    band = img[BAND]
    band[:] = np.clip(np.asarray(garment)[None, None, :] + rng.integers(-6, 7, band.shape), 0, 255)
    return img


def lit_by(img: np.ndarray, cast) -> np.ndarray:
    """The frame under a coloured light: each channel scaled, 8-bit."""
    out = img.astype(np.float32) * np.asarray(cast, dtype=np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def colour_part(d) -> np.ndarray:
    """The descriptor's colour bins, renormalised to sum 1."""
    c = np.asarray(d, dtype=np.float64)[: ap.COLOUR_BINS]
    return c / c.sum()


def agree(a, b) -> float | None:
    """Colour-part intersection; None when either side was not measurable."""
    if a is None or b is None:
        return None
    return float(np.minimum(colour_part(a), colour_part(b)).sum())


@pytest.mark.parametrize("cast", sorted(CASTS))
def test_frame_gains_invert_the_cast(cast):
    """The estimate is the light's inverse, normalised to average 1 (within 4%)."""
    m = np.asarray(CASTS[cast])
    g = np.asarray(ap.frame_gains(lit_by(scene(GARMENTS["green"]), m)))
    expected = (1.0 / m) / np.mean(1.0 / m)
    assert np.allclose(g, expected, rtol=0.04), (g, expected)
    assert np.mean(g) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("garment", sorted(GARMENTS))
@pytest.mark.parametrize("cast", sorted(CASTS))
def test_the_same_garment_under_a_cast_reads_as_itself_with_gains(garment, cast):
    """WB brings the colour part back to >= 0.8 of the neutral read, and ahead of raw."""
    neutral = ap.torso_descriptor(scene(GARMENTS[garment]), FACE, PERSON)
    under = lit_by(scene(GARMENTS[garment]), CASTS[cast])
    raw = ap.torso_descriptor(under, FACE, PERSON)
    balanced = ap.torso_descriptor(under, FACE, PERSON, ap.frame_gains(under))
    with_wb, without = agree(neutral, balanced), agree(neutral, raw)
    assert with_wb is not None and with_wb >= 0.8, (garment, cast, with_wb)
    if without is not None:
        assert with_wb >= without - 0.15, "WB may cost a hair on a garment the cast spared"


def test_wb_is_clearly_ahead_across_the_board():
    """Averaged over the 12 garment/cast pairs, WB wins by a wide margin.

    Measured: with gains 0.83-0.99 on every pair; without, anywhere from 0.00
    (grey-blue under blue) to 0.995 (red under red) — and three of the twelve
    stop being measurable at all.
    """
    with_wb, without = [], []
    for garment, cast in itertools.product(GARMENTS.values(), CASTS.values()):
        neutral = ap.torso_descriptor(scene(garment), FACE, PERSON)
        under = lit_by(scene(garment), cast)
        balanced = ap.torso_descriptor(under, FACE, PERSON, ap.frame_gains(under))
        with_wb.append(agree(neutral, balanced))
        without.append(agree(neutral, ap.torso_descriptor(under, FACE, PERSON)) or 0.0)
    assert min(with_wb) >= 0.8
    assert np.mean(with_wb) - np.mean(without) >= 0.3, (np.mean(with_wb), np.mean(without))


def test_a_warm_cast_turns_a_cool_garment_into_skin_and_gains_bring_it_back():
    """Grey-blue cloth under red or amber light sits inside the skin window.

    The band is then all skin-toned and the descriptor is None (not
    measurable) — the light, not the garment, decided that.  Balanced, it
    is cloth again.
    """
    for cast in ("red", "amber"):
        under = lit_by(scene(GARMENTS["grey-blue"]), CASTS[cast])
        assert ap.torso_descriptor(under, FACE, PERSON) is None
        assert ap.torso_descriptor(under, FACE, PERSON, ap.frame_gains(under)) is not None


def test_gains_none_is_the_descriptor_as_it_was():
    """The off switch is byte-for-byte: no gains, same floats; unit gains, same floats."""
    img = lit_by(scene(GARMENTS["orange"]), CASTS["amber"])
    before = ap.torso_descriptor(img, FACE, PERSON)
    assert ap.torso_descriptor(img, FACE, PERSON, None) == before
    assert ap.torso_descriptor(img, FACE, PERSON, (1.0, 1.0, 1.0)) == before


def test_a_blown_highlight_stays_excluded_after_a_gain_under_one():
    """Clipping is a sensor fact: 255 x 0.8 = 204 is still not cloth."""
    img = scene(GARMENTS["green"])
    img[180:260, 90:210] = 255
    balanced = ap.apply_gains(img[BAND], (0.8, 0.8, 0.8))
    assert int(balanced.max()) == 204
    lit, _skin, _w = ap._analyse(balanced, sensor_bgr=img[BAND])
    assert not lit[10:90, 10:130].any(), "the blown block stays masked"


def test_frame_gains_are_none_without_enough_usable_samples_and_clamped():
    """Black or all-blown frames have no illuminant; one saturated colour is clamped."""
    assert ap.frame_gains(np.zeros((720, 960, 3), np.uint8)) is None
    assert ap.frame_gains(np.full((720, 960, 3), 255, np.uint8)) is None
    assert ap.frame_gains(np.zeros((10, 10), np.uint8)) is None
    one = np.zeros((720, 960, 3), np.uint8)
    one[:] = (5, 200, 20)
    g = ap.frame_gains(one)
    assert g is not None and all(ap.WB_GAIN_MIN <= x <= ap.WB_GAIN_MAX for x in g)


def test_frame_gains_read_a_strided_sample_not_every_pixel():
    """gains_from_samples on the 1/8 grid is exactly frame_gains on the frame."""
    img = lit_by(scene(GARMENTS["green"]), CASTS["blue"])
    assert ap.frame_gains(img) == ap.gains_from_samples(img[:: ap.WB_STRIDE, :: ap.WB_STRIDE])
