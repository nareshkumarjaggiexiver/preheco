"""Head and beard descriptors: the review queue's turban, hair and beard.

Synthetic faces, known pixels.  Pinned: the wire shapes (40 and 4 floats,
each summing to 1, reserved head bins zero); red and orange turbans apart
(< 0.3) and hue wrapping at red; chroma — not saturation — deciding what is
a colour (black hair's saturation is noise); a skin-toned turban still read
as its colour; the beard judged against the same face's cheeks, so a shaded
chin is skin and a white beard is white; and None for everything that
cannot be measured.
"""

import cv2
import numpy as np
import pytest
from heco_counting import appearance as ap

FACE = {"x": 50.0, "y": 60.0, "w": 80.0, "h": 100.0}
#: Right eye, left eye, nose tip, right and left mouth corners (YuNet order).
LANDMARKS = [[70, 95], [110, 95], [90, 118], [76, 138], [104, 138]]
SKIN = (110, 140, 190)          # BGR, inside the YCrCb skin window
TURBAN_RED = (0, 0, 200)
TURBAN_ORANGE = (20, 110, 235)


def face(turban=None, chin=None, skin=SKIN, h=220, w=200) -> np.ndarray:
    """A skin-coloured frame; ``turban`` paints rows 0..84, ``chin`` y 120..160."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = skin
    if turban is not None:
        img[0:85, :] = turban
    if chin is not None:
        img[120:160, 60:120] = chin
    return img


def inter(a, b) -> float:
    """Histogram intersection of two descriptors — over the histogram (0..26),
    as the match service compares heads; 27/28 are the headwear share."""
    return float(np.minimum(np.asarray(a)[:27], np.asarray(b)[:27]).sum())


def hue_of(bgr) -> int:
    """OpenCV's 8-bit hue of one BGR colour."""
    return int(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])


# ------------------------------------------------------------------ head


def test_head_is_40_floats_a_histogram_the_headwear_share_and_reserved_zeros():
    """The wire shape the match service validates: bins 0..26 sum to 1, 27 is
    the headwear share, 28 its flag, 29..39 reserved."""
    d = ap.head_descriptor(face(TURBAN_RED), FACE, LANDMARKS)
    assert d is not None and len(d) == ap.HEAD_DIM == 40
    assert sum(d[:27]) == pytest.approx(1.0) and min(d) >= 0.0
    assert 0.0 <= d[ap.HEAD_WEAR_SLOT] <= 1.0 and d[ap.HEAD_WEAR_FLAG] == 1.0
    assert not any(d[29:]), "bins 29..39 are reserved"


def test_the_headwear_share_leaves_skin_out():
    """A turban is cloth outside the skin window: most of the window. A bald
    scalp is chromatic too — its skin — and read 0.91 "headwear" on f0bfc5's
    p00062; outside the skin window it is nearly nothing. So is a skin-toned
    turban (the peach one): it reads bare, which only keeps a pair asked."""
    turban = ap.head_descriptor(face(TURBAN_RED), FACE, LANDMARKS)
    assert sum(turban[:24]) > 0.9 and turban[ap.HEAD_WEAR_SLOT] > 0.3
    scalp = ap.head_descriptor(face(), FACE, LANDMARKS)  # skin to the top
    assert sum(scalp[:24]) > 0.9, "the histogram still reads the scalp's colour"
    assert scalp[ap.HEAD_WEAR_SLOT] < 0.05
    peach = ap.head_descriptor(face((120, 160, 225)), FACE, LANDMARKS)
    assert peach[ap.HEAD_WEAR_SLOT] < 0.05


def test_a_red_turban_and_an_orange_one_are_different_heads():
    """Pair #1 of run f0bfc5 in two colours: the head must tell them apart."""
    red = ap.head_descriptor(face(TURBAN_RED), FACE, LANDMARKS)
    orange = ap.head_descriptor(face(TURBAN_ORANGE), FACE, LANDMARKS)
    assert inter(red, orange) < 0.3
    assert inter(red, red) == pytest.approx(1.0)


def test_hue_wraps_at_red():
    """H 179 and H 0 are both red: soft bins share their mass across the wrap."""
    almost_red = (7, 0, 200)
    assert hue_of(almost_red) == 179 and hue_of(TURBAN_RED) == 0
    a = ap.head_descriptor(face(almost_red), FACE, LANDMARKS)
    b = ap.head_descriptor(face(TURBAN_RED), FACE, LANDMARKS)
    assert inter(a, b) >= 0.8


def test_black_grey_and_white_hair_are_three_heads():
    """Achromatic heads bin by brightness: black, grey and white apart."""
    black, grey, white = (
        ap.head_descriptor(face(c), FACE, LANDMARKS)
        for c in ((18, 18, 20), (120, 120, 122), (225, 225, 228))
    )
    assert inter(black, grey) < 0.5 and inter(black, white) < 0.2 and inter(grey, white) < 0.6
    assert sum(black[:24]) < 0.1, "black hair is not a colour"


def test_chroma_not_saturation_decides_what_is_a_colour():
    """Black hair under a warm lamp: saturation ~116 (noise), chroma 10.

    Saturation is (max - min) / max and explodes in the dark; run f0bfc5's
    black hair read S 40-93 at V 13-22.  A dark MAROON turban (chroma 50)
    is still a colour.
    """
    warm_black = (12, 14, 22)
    hsv = cv2.cvtColor(np.uint8([[warm_black]]), cv2.COLOR_BGR2HSV)[0, 0]
    assert hsv[1] > 100, "OpenCV calls it saturated"
    hair = ap.head_descriptor(face(warm_black), FACE, LANDMARKS)
    assert sum(hair[:24]) < 0.1 and hair[24] > 0.8, "…but it is black"
    maroon = ap.head_descriptor(face((30, 20, 70)), FACE, LANDMARKS)
    assert sum(maroon[:24]) > 0.8, "a dark but coloured turban keeps its hue"


def test_a_skin_toned_turban_is_still_read_as_its_colour():
    """Pair #1's peach turban sat wholly inside the skin window.

    Masked (the first draft) it vanished; at a quarter weight it outvotes
    nothing but itself, and reads as a colour, not as an empty head.
    """
    peach = (120, 160, 225)
    sat = cv2.cvtColor(np.uint8([[peach]]), cv2.COLOR_BGR2HSV)[0, 0, 1]
    ycrcb = cv2.cvtColor(np.uint8([[peach]]), cv2.COLOR_BGR2YCrCb)[0, 0]
    assert 133 <= ycrcb[1] <= 173 and 77 <= ycrcb[2] <= 127 and sat < 150, "in the window"
    d = ap.head_descriptor(face(peach), FACE, LANDMARKS)
    assert d is not None and sum(d[:24]) > 0.9
    assert inter(d, ap.head_descriptor(face((200, 60, 20)), FACE, LANDMARKS)) < 0.1


def test_the_region_stops_at_the_person_box_top():
    """Above the person box is wall, not headwear."""
    img = face(TURBAN_RED)
    img[0:40, :] = (200, 0, 0)          # a blue wall above the head
    person = {"x": 20.0, "y": 43.0, "w": 160.0, "h": 170.0}
    clipped = ap.head_descriptor(img, FACE, LANDMARKS, person_box=person)
    unclipped = ap.head_descriptor(img, FACE, LANDMARKS)
    red = ap.head_descriptor(face(TURBAN_RED), FACE, LANDMARKS)
    assert inter(clipped, red) > 0.9 > inter(unclipped, red)
    x0, x1, y0, y1 = ap.head_region(FACE, LANDMARKS, 200, 220, person)
    assert y0 == 38, "person top 43 less 0.05 face heights"


def test_head_is_none_when_it_cannot_be_measured():
    """No eyes, no eye distance, a sliver of a window, a blown head: None."""
    img = face(TURBAN_RED)
    assert ap.head_descriptor(img, FACE, None) is None
    assert ap.head_descriptor(img, FACE, [[1, 1]] * 5) is None, "no eye distance"
    assert ap.head_descriptor(img, FACE, LANDMARKS[:1]) is None
    tiny = {"x": 50.0, "y": 60.0, "w": 4.0, "h": 5.0}
    assert ap.head_descriptor(img, tiny, [[51, 62], [53, 62], [52, 63], [51, 64], [53, 64]]) is None
    blown = np.full((220, 200, 3), 255, np.uint8)
    assert ap.head_descriptor(blown, FACE, LANDMARKS) is None


def test_head_gains_none_is_the_plain_read_and_gains_change_the_colour():
    """gains=None and unit gains are the plain read; real gains move colour."""
    img = face(TURBAN_ORANGE)
    plain = ap.head_descriptor(img, FACE, LANDMARKS)
    assert ap.head_descriptor(img, FACE, LANDMARKS, None) == plain
    assert ap.head_descriptor(img, FACE, LANDMARKS, (1.0, 1.0, 1.0)) == plain
    assert ap.head_descriptor(img, FACE, LANDMARKS, (2.0, 1.0, 0.5)) != plain


# ------------------------------------------------------------------ beard


def test_beard_is_four_fractions_summing_to_one():
    """The wire shape: skin, dark, grey, white."""
    b = ap.beard_descriptor(face(chin=(20, 20, 25)), FACE, LANDMARKS)
    assert b is not None and len(b) == 4
    assert sum(b) == pytest.approx(1.0) and min(b) >= 0.0


def test_a_black_beard_is_dark_and_a_white_one_white():
    """The three hair classes, each on a chin under an ordinary cheek."""
    black = ap.beard_descriptor(face(chin=(20, 20, 25)), FACE, LANDMARKS)
    white = ap.beard_descriptor(face(chin=(200, 205, 210)), FACE, LANDMARKS)
    grey = ap.beard_descriptor(face(chin=(115, 118, 120)), FACE, LANDMARKS)
    assert black[1] > 0.8, black
    assert white[3] > 0.8, white
    assert grey[2] > 0.8, grey


def test_a_shaven_chin_is_skin_even_in_shadow():
    """Judged against the cheek: skin at 0.6 of its brightness is still skin.

    Absolute cuts failed on run f0bfc5's dim hall — every shaven man read
    50-90% "dark" at V < 70.  A shaded chin keeps the cheek's saturation.
    """
    shaven = ap.beard_descriptor(face(), FACE, LANDMARKS)
    assert shaven == pytest.approx([1.0, 0.0, 0.0, 0.0])
    shade = tuple(int(round(c * 0.6)) for c in SKIN)
    shaded = ap.beard_descriptor(face(chin=shade), FACE, LANDMARKS)
    assert shaded[0] > 0.95, shaded
    dim = ap.beard_descriptor(face(skin=tuple(int(c * 0.45) for c in SKIN)), FACE, LANDMARKS)
    assert dim[0] > 0.95, "a dim face is judged against its own dim cheeks"


def test_beard_is_none_when_the_chin_cannot_be_seen():
    """No landmarks, a fully bowed head, or cheeks too dark to judge by."""
    img = face(chin=(20, 20, 25))
    assert ap.beard_descriptor(img, FACE, None) is None
    head_down = [[70, 95], [110, 95], [90, 140], [76, 138], [104, 138]]
    assert ap.nose_drop(head_down) >= 1.0
    assert ap.beard_descriptor(img, FACE, head_down) is None, "nose below the mouth"
    dark = np.full((220, 200, 3), 8, np.uint8)
    assert ap.beard_descriptor(dark, FACE, LANDMARKS) is None, "cheeks too dark to judge"


def test_the_beard_is_not_read_off_a_steeply_bowed_head():
    """Nose 0.8 of the way to the mouth line or further: no reading.

    Run f0bfc5: reads with a nose drop of 0.85-1.0 named a conflicting beard
    class 6.9% of the time (a white collar in the window read "white" on a
    shaven man) against 0-1.2% below 0.85.
    """
    img = face(chin=(20, 20, 25))
    bowed = [[70, 95], [110, 95], [90, 130], [76, 138], [104, 138]]
    assert 0.8 <= ap.nose_drop(bowed) < 1.0
    assert ap.beard_descriptor(img, FACE, bowed) is None
    level = [[70, 95], [110, 95], [90, 125], [76, 138], [104, 138]]
    assert ap.nose_drop(level) < 0.8
    assert ap.beard_descriptor(img, FACE, level) is not None


# ------------------------------------------------------------------ skin


def test_skin_tone_is_the_cheeks_log_chromaticity():
    """Two floats, log(R/G) and log(B/G) of the cheek window."""
    s = ap.skin_tone(face(), LANDMARKS)
    b, g, r = SKIN
    assert s == pytest.approx([np.log(r / g), np.log(b / g)], abs=1e-6)
    assert len(s) == ap.SKIN_DIM


def test_a_light_on_the_face_moves_the_skin_reading_by_its_own_log():
    """A per-channel light is an additive shift of the log ratios — the same
    whatever the skin — which is what makes one tolerance mean one light."""
    for skin in (SKIN, (60, 90, 150)):
        plain = ap.skin_tone(face(skin=skin), LANDMARKS)
        gains = (0.9, 1.0, 1.1)
        warm = face(skin=tuple(int(round(c * m)) for c, m in zip(skin, gains, strict=True)))
        lit = ap.skin_tone(warm, LANDMARKS)
        assert lit[0] - plain[0] == pytest.approx(np.log(1.1), abs=0.02)
        assert lit[1] - plain[1] == pytest.approx(np.log(0.9), abs=0.02)


def test_skin_tone_reads_under_the_gains_and_is_none_when_unmeasurable():
    """Gains apply first; no landmarks or a black face read nothing."""
    img = face()
    assert ap.skin_tone(img, LANDMARKS, (1.0, 1.0, 1.0)) == ap.skin_tone(img, LANDMARKS)
    assert ap.skin_tone(img, LANDMARKS, (1.2, 1.0, 0.8)) != ap.skin_tone(img, LANDMARKS)
    assert ap.skin_tone(img, None) is None
    assert ap.skin_tone(np.full((220, 200, 3), 20, np.uint8), LANDMARKS) is None
