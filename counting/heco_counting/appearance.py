"""Torso-appearance descriptor: an ADVISORY clothing signal, never a verdict.

WHY THIS EXISTS.  Face evidence on this camera is thin exactly where it
matters: same-person sightings have missed at cosine 0.294/0.308/0.361
(threshold 0.363), while the closest measured impostor pair sits at 0.377 —
the genuine and impostor distributions OVERLAP, so no threshold move can
separate them.  Clothing is constant within one event, so a cheap torso
histogram adds an independent axis — but only as a VETO on actions the face
evidence already justified (the track heal, template enrolment), never as a
reason to mint, merge, match or count.  The measured impostor ceiling itself
says why: the 0.377 pair was two DIFFERENT men BOTH IN LIGHT SHIRTS.  Their
torsos would have AGREED, so clothing agreement proves nothing about identity;
only a CLASH carries information, and only against a decision face evidence
was about to make anyway.

v2 (48 floats, 2026-08-06) split the histogram into CHROMATIC-BY-HUE and
ACHROMATIC-BY-BRIGHTNESS partitions, because the hue of a desaturated pixel
is noise and a light shirt scattered across hue bins differently every frame
(the same man read 0.88 / 0.48 / 0.77 / 0.88 on unchanged clothing).  That
partition is kept unchanged in v3, bins 0..38.

WHAT v2 GOT WRONG, measured on run f0bfc5 (2026-09-24, wedding-hall overview
camera, 74 guests, the review queue's top five pairs):

* The crop started at the CHIN.  From the face's bottom edge down 2.5 face
  heights is neck, chest skin, hair on the shoulders, and only then cloth.
  Review pair #5 — a girl in a pale YELLOW top vs a woman in a DARK GREEN
  dress — scored clothes 0.72 with both crops dominated by skin (hue bin 1 at
  low saturation) and dark hair.
* It had no notion of PATTERN.  Review pair #4 — a plain white kurta vs a
  white-and-grey STRIPED shirt — scored 0.71: to a colour histogram a fine
  stripe is "mostly light with a little dark", i.e. a light shirt.

THE v3 DESCRIPTOR: 64 floats, three L1-normalised parts, weighted so the
concatenation still sums to 1.0 and histogram intersection stays 0..1.

* bins 0..38  — COLOUR, weight 0.9: the v2 partition (12 hue x 3 saturation
  chromatic bins, then 3 SOFT brightness bins for achromatic pixels) over a
  band that starts BELOW the neck — face-bottom + 0.5 face heights down to
  face-bottom + 3.0 face heights or the person box's bottom, whichever comes
  first; horizontally the person box inset 15% a side AND clipped to the
  face's own column (face centre +- 1.5 face widths).  Skin-toned pixels
  (the standard YCrCb window, Cr 133..173 / Cb 77..127) are DOWN-WEIGHTED
  4x rather than dropped — see "why skin is weighted, not masked" below.
* bins 39..48 — TEXTURE, weight 0.07: the 10-bin rotation-invariant uniform
  LBP(8,1) histogram of the grey band, on the non-skin pixels, with a noise
  threshold of 8 grey levels on each neighbour comparison.  Without the
  threshold the histogram is sensor noise: on this camera a plain kurta and
  a striped shirt gave IDENTICAL distributions (0.04/0.07/0.06/0.09/0.17…
  for both); with it, plain cloth concentrates in the "flat" bin (0.53) and
  stripes (0.31) and heavy print (0.23) do not.
* bins 49..51 — EDGE DENSITY, weight 0.03: the fraction of those pixels whose
  Sobel gradient magnitude is low / mid / high (< 6, 6..24, >= 24 grey levels
  per pixel).  Plain cloth is mostly "low"; a print moves mass to "high"
  (the dark-green printed dress reads 0.06/0.29/0.65, the plain kurta
  0.30/0.52/0.19).
* bins 52..63 — reserved, always zero (headroom for a partition change
  without another contract bump, as v2's reserved bins were).

Texture and edges are read on the band resized to a FIXED WIDTH (64 px), so
"stripes across the torso" reads the same at 4 m and at 12 m: a garment's
pattern is a body-relative quantity, and at native 4K a close sighting would
otherwise histogram sensor noise and codec blocks that a far one cannot see.

WHY SKIN IS WEIGHTED, NOT MASKED (the contract asked for non-skin pixels;
the run said no).  The standard YCrCb window is a SKIN-TONE window, and
skin-toned cloth is common at a wedding: the pink checked shirt of p00065
was 95-100% "skin" in every sighting, the pale yellow top of p00048 31-91%.
Masking them left the descriptor reading the 2-5% of leftover pixels —
noise and the camera's own overlay text — which read "light" and agreed with
a plain white kurta at 0.83 (v2 had them at 0.24, a clear clash), and the
pink shirt's own sightings agreed with each other at only 0.63.  Weighting
skin-toned pixels at 0.25 keeps a skin-toned GARMENT reading as its colour
(95% of the band at a quarter weight still outvotes 5% of noise 5:1) while
a bare arm across a red shirt contributes a fifth of what it did.  Measured
on the same run: the pink-vs-white pair back to 0.38, the worst within-
identity mean up from 0.63 to 0.73.  The None rule still counts NON-skin
pixels, so a band that is nothing but skin remains "not measurable".

The window stops at saturation 150.  Swept over every (H, S, V): OpenCV
hue bins 0 and 1 (red through yellow-orange) are 64-71% "skin" under the
Cr/Cb window, and a fully saturated orange (H 15, S 255) is inside it for
every V up to 152 — so the 1.0/0.25 step at the window edge turned a 3%
exposure move (V 150 to 155) on an orange kurta into a 0.30 swing in its
own colour share, and two sightings of one garment intersected at 0.79:
the cliff v2's soft V bins were added to remove, back at the skin
boundary.  No skin is as saturated as 150 (S); saturated red and orange
cloth is, and is now cloth whatever Cr/Cb say.  A RAMP on the weight
(0.25 rising to 1.0 over 2, 4, 8 or 16 Cr/Cb units outside the window) was
measured on the same crops and rejected: every width lowered the pale
yellow top's within-identity mean (0.69 -> 0.67 / 0.65 / 0.65 / 0.67) for
0.00-0.07 on the pink-vs-white pair — a garment straddling the window edge
gets lighting-dependent intermediate weights under a ramp, where a step
only moves when the drift crosses the edge.  The residual is a garment
sitting exactly at the Cr/Cb edge at S < 150 under exposure drift; not
seen on this run.

WHY THE BAND IS CLIPPED TO THE FACE'S COLUMN.  A person box grows with the
pose: with an arm stretched out, p00048's box was 3.3x to 7.5x her face
width, and the band's inset 15% of THAT was half wall.  The face's column,
+-1.5 face widths, covers a torso (shoulders are ~2.5 face widths) and cuts
the background: the white-vs-striped pair went from 0.80 to 0.75 with
within-identity agreement unchanged (0.93).  What the clip does NOT do: it
binds only when the inset box is wider than the column, i.e. the person box
exceeds ~4.3 face widths, and nothing here knows whether the band is on
the person at all.  A face at the edge of a narrow box (p00047, half
behind a pillar: her band was the pillar) and an arm-out box only 3.2 face
widths wide (p00048 at seq 9844: 23% wall) still read background, and the
review shows that number as measured.  The evidence for "the band is cloth"
is the None rules, which are about pixel counts, not placement.

RED AND ORANGE ARE ONE COLOUR HERE.  Twelve hard hue bins, inherited from
v2: pure red (0,0,200) and a turban orange (20,110,235) both land in bin 0
(OpenCV H 0..14) at the same saturation bin and intersect at 1.000, and hue
does not wrap (H 179 and H 0 are both red, in bins 11 and 0).  An operator
reading `clothes` on a red-vs-orange garment pair should know the number
cannot tell them apart; soft hue assignment (as the V bins already have) is
the contract-neutral fix if it ever matters.  Also: the colour part is the
v2 partition computed exactly — cv2.calcHist's float LUT put S=112 and
S=184 one bin low (2 of 216 saturation levels); the integer bincount here
does not, so v2-vs-v3 comparisons carry that hair of a shift.

WHAT v3 DOES AND DOES NOT ACHIEVE, on the same run, 15 identities, <= 8
sightings each (scratchpad appearance-eval/, 2026-09-24).  Within-identity
mean intersection rose from 0.85 (v2) to 0.87, the worst identity from 0.67
to 0.73.  The measured impostor pairs that should clash still do (#3 man in
blue vs woman in cream 0.27, #6 0.33, #1 pink-check vs white 0.38).  The two
pairs that motivated the change are NOT separated: #4 white kurta vs striped
shirt reads 0.74 (v2 0.71) against within-identity means of 0.93; #5 yellow
top vs green printed dress 0.72 (v2 0.68) against 0.75 / 0.80.  The reason
is structural to the colour part, which no band, threshold or weight within
this contract moves (twenty variants swept, all within 0.03 of each other):
with only three soft brightness bins a white kurta and a light-grey striped
shirt share 0.37 of their mass in the middle bin, and because brightness is
deliberately ignored for chromatic pixels, gold embroidery and a pale yellow
top share the same low-saturation yellow bins (0.17).  The pattern parts can
move a pair by at most 0.05 at these weights.  Separating those two pairs
needs either brightness in the chromatic partition or a finer achromatic
one — both a contract change — or, better, the co-presence, age and stature
evidence the review carries alongside this signal.

ABSENT IS NOT ZERO (codebase-wide convention, like ``gatedUnmeasured`` /
``zoneUnmeasured``): a sighting with no descriptor — no containing person box,
a crop under 24 px, fewer than 100 lit non-skin pixels, an undecodable frame,
an old run recorded before this existed — returns/means None, and None never
vetoes anything.  A signal that could not be measured must not be treated as
a signal that measured "different".  The same applies ACROSS VERSIONS: a
48-float v2 descriptor from an older gallery row compared with a 64-float v3
one is "not comparable", and :func:`intersection` answers None for it rather
than an exception or a 0.0 that would read as a maximal clash.

A NOTE FOR THE CLASH FLOORS.  Two plain garments of DIFFERENT colours share
the pattern parts: the floor of their intersection is the pattern weight,
0.10, not 0.0 as it was for v2 (plain red vs plain blue measures exactly
0.10).  The first cut weighted pattern 0.3 and measured why that is too
much: on the run's own crops the texture and edge parts overlap at a
median 0.77 ACROSS identities (0.93 within), so they lifted every impostor
pair towards agreement — pink check vs white kurta 0.19 -> 0.38, blue
shirt vs cream suit 0.02 -> 0.27 — and the first of those crossed the
runner's 0.35 heal-clash floor, while on the two pairs the parts were
built for (#4 plain vs striped, #5 yellow vs green) their overlap was
HIGHER than colour's (0.86 / 0.88: a plain kurta at 4K has buttons,
pocket flaps and folds).  At 0.1 the pattern parts are a tie-break that
can never clash on weave alone and lift an impostor pair by at most 0.1;
the runner's 0.35 floor reads a clean colour clash with 0.25 to spare.
"""

import cv2
import numpy as np

#: The wire length of a v3 descriptor.  v2 was 48; the match service accepts
#: both and never compares one with the other.
APPEARANCE_DIM = 64

#: Chromatic partition: 12 hue bins (OpenCV HSV: H 0..179) x 3 saturation
#: bins over the SATURATED range (S 40..255) = bins 0..35.
H_BINS = 12
S_BINS = 3
#: Below this saturation a pixel is ACHROMATIC — its hue is noise (the light-
#: shirt failure v1 measured) — and it bins by brightness instead.
S_ACHROMATIC = 40
#: Achromatic partition: 3 coarse brightness bins (dark / mid / light cloth)
#: over V 30..252 = bins 36..38, filled by SOFT (triangular) assignment — each
#: pixel splits its mass between the two nearest bin centres.  Hard binning
#: has a cliff: V=205 and V=218 (one shirt, auto-exposure drift) landed in
#: adjacent bins with ZERO overlap.  Soft assignment makes the histogram a
#: continuous function of brightness, so a small drift moves a little mass
#: instead of all of it.
V_BINS = 3
#: Sizes of the three parts and where each starts in the 64-float wire.
COLOUR_BINS = H_BINS * S_BINS + V_BINS  # 39
TEXTURE_BINS = 10                        # LBP(8,1) riu2: 9 uniform + 1 other
EDGE_BINS = 3                            # low / mid / high gradient
TEXTURE_OFFSET = COLOUR_BINS             # 39
EDGE_OFFSET = COLOUR_BINS + TEXTURE_BINS  # 49
#: Part weights: each part is L1-normalised, then scaled by these, so the
#: whole descriptor sums to 1.0 and intersection stays 0..1.  Colour keeps
#: nearly all of it because it is the part that separates a red shirt from
#: a blue one at any distance; the pattern parts are a 0.1 tie-break.
#: Swept on run f0bfc5 (113 crops, 15 identities): 0.7/0.2/0.1 lifted the
#: clear impostor pairs by ~0.2 of constant agreement (plain cloth agrees
#: with plain cloth on weave) and moved the two target pairs by nothing;
#: 0.8/0.15/0.05 was better on every pair; colour alone better still.  0.1
#: keeps the pattern reading on the wire for a later, separate signal
#: without letting it vote against a colour clash.
W_COLOUR = 0.9
W_TEXTURE = 0.07
W_EDGE = 0.03

#: V-channel mask bounds for BOTH partitions: below = shadow (nothing reads
#: reliably in the dark), above = specular blowout (a highlight, not cloth).
#: v1 masked at 240 and threw away the brightest pixels of every light shirt;
#: 252 keeps them.  Lowering the floor to 20 or 12 was tried for the dark
#: green dress of pair #5 (40-60% of its band is under 30): the hue of those
#: pixels is noise and every within-identity mean fell — kept at 30.
V_MIN = 30
V_MAX = 252

#: The standard YCrCb skin window (Chai & Ngan 1999; the one every OpenCV
#: skin-segmentation recipe uses).  It is a skin-TONE window: it also
#: matches pink, salmon, peach, beige and pale yellow cloth, which is why
#: pixels inside it are down-weighted (SKIN_WEIGHT) rather than dropped —
#: see the module docstring for the measurements.
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)
#: No skin is this saturated (OpenCV S, 0..255); saturated red and orange
#: cloth is, and the Cr/Cb window alone calls 60% of it skin even at
#: S >= 120.  At or above this the pixel is cloth whatever Cr/Cb say.
SKIN_S_MAX = 150
#: The weight of a lit skin-toned pixel in the colour histogram, against 1.0
#: for every other lit pixel.  0.25: a bare arm across a shirt counts a
#: quarter, a skin-toned shirt still outvotes the few pixels the window
#: misses.  Swept 0.1 / 0.25 / 0.5: 0.1 let the pink-vs-white pair climb
#: back to 0.52, 0.5 cost 0.01 on the within-identity minimum; 0.25 it is.
SKIN_WEIGHT = 0.25

#: The None conditions (wire contract): a torso crop under this many
#: pixels in either dimension carries too little cloth to histogram honestly…
MIN_CROP_PX = 24
#: …and so does a crop where shadow, blowout and the skin window leave fewer
#: than this many full-weight pixels.  A fourth, added after the contract:
#: a band whose resized mask keeps no interior cloth pixel has no pattern
#: reading, and the descriptor is then None rather than colour-only — a
#: colour-only descriptor renormalised to 1.0 caps every comparison at 0.9
#: and reads the missing pattern as 0.1 of disagreement.
MIN_UNMASKED_PX = 100

#: Where the band STARTS below the face, in face heights.  v2 started at 0
#: (the chin) and read neck, chest skin and shoulder hair before any cloth;
#: half a face height down is below the collar on every standing sighting
#: measured, including head-down poses where the face box sits high.
BAND_TOP_FACE_HEIGHTS = 0.5
#: How far below the face the band extends, in face heights.  The crop stops
#: at the person box's bottom edge when that comes first (a guest close to the
#: camera, seated, or partially occluded).
TORSO_DEPTH_FACE_HEIGHTS = 3.0

#: Horizontal inset of the person box on each side, keeping the histogram on
#: the wearer's cloth rather than the background the box edges usually clip.
SIDE_INSET_FRAC = 0.15
#: The band is further clipped to the face's column: face centre +- this
#: many face widths.  Shoulders are ~2.5 face widths, so 3 covers a torso in
#: any standing pose; a person box with an arm out (3.3x-7.5x the face width
#: on this run) would otherwise fill the band with wall.  1.25 and 2.0 were
#: swept: 1.25 clips a turned torso, 2.0 lets the wall back in.
FACE_CLIP_WIDTHS = 1.5
#: A face whose centre sits within this many face widths of the person box's
#: LEFT or RIGHT edge is a face at the edge of a cut box — an occlusion or
#: the frame edge — and the band beside it is whatever is beside the person,
#: not their torso: p00047 (half behind a pillar, margin 0.17) read the
#: pillar and the review presented it as a measured 0.78 against a dress.
#: Shoulders put a centred face >= 1.25 widths from either edge; on run
#: f0bfc5's 113 crops only the three occluded or cut sightings were under
#: 0.4 (0.06, 0.17, 0.28), the next at 0.53.  Such a sighting is None.
FACE_EDGE_WIDTHS = 0.4

#: The band is resized to this width (aspect kept) before the texture and
#: edge parts are read, making "pattern across the torso" a body-relative
#: measure independent of the sighting's distance.  The colour part is read
#: at native resolution; a histogram of colours does not care about scale.
#: 48 and 96 were swept: within 0.02 of 64 everywhere.
TEXTURE_WIDTH_PX = 64
#: Grey levels a neighbour must exceed the centre pixel by to count as "1" in
#: the LBP code.  After the area downscale to 64 px the residual noise is
#: ~2 levels, a fold or stripe edge 10+; 8 is where plain cloth first reads as
#: mostly flat (bin 0 at 0.5) while stripes (0.3) and print (0.2) do not.
#: 12 / 16 / 24 read the same to within 0.02 on every measured pair.
LBP_NOISE_T = 8
#: Sobel-magnitude band edges for the edge-density part, in per-pixel 8-bit
#: contrast units (the 3x3 Sobel's gain of 4 is divided out).  Below LOW is
#: plain cloth and sensor noise; above HIGH is a stripe, a print or a hard
#: fold.  4/16, 8/32 and 10/30 were swept: no measured pair moved by 0.01.
EDGE_LOW = 6.0
EDGE_HIGH = 24.0


def _band(face_box: dict, person_box: dict, img_w: int, img_h: int):
    """The integer crop window (x0, x1, y0, y1) of the torso band, or None.

    Vertically the band hangs BELOW the face (0.5 to 3.0 face heights under
    its bottom edge, or to the person box's bottom).  Horizontally it is the
    person box inset 15% a side, clipped to the face's column (+- 1.5 face
    widths).  The window is clamped to the image, because detector boxes
    legitimately overhang frames.
    """
    fx, fy = float(face_box.get("x", 0.0)), float(face_box.get("y", 0.0))
    fw, fh = float(face_box.get("w", 0.0)), float(face_box.get("h", 0.0))
    px, py = float(person_box.get("x", 0.0)), float(person_box.get("y", 0.0))
    pw, ph = float(person_box.get("w", 0.0)), float(person_box.get("h", 0.0))
    face_bottom = fy + fh
    y0 = face_bottom + BAND_TOP_FACE_HEIGHTS * fh
    y1 = min(face_bottom + TORSO_DEPTH_FACE_HEIGHTS * fh, py + ph)
    face_cx = fx + fw / 2.0
    x0 = max(px + SIDE_INSET_FRAC * pw, face_cx - FACE_CLIP_WIDTHS * fw)
    x1 = min(px + pw - SIDE_INSET_FRAC * pw, face_cx + FACE_CLIP_WIDTHS * fw)
    ix0, ix1 = max(int(round(x0)), 0), min(int(round(x1)), img_w)
    iy0, iy1 = max(int(round(y0)), 0), min(int(round(y1)), img_h)
    if (ix1 - ix0) < MIN_CROP_PX or (iy1 - iy0) < MIN_CROP_PX:
        return None
    return ix0, ix1, iy0, iy1


def _face_at_box_edge(face_box: dict, person_box: dict) -> bool:
    """Whether the face centre is within FACE_EDGE_WIDTHS of the box's sides."""
    fw = float(face_box.get("w", 0.0))
    if fw <= 0:
        return False
    cx = float(face_box.get("x", 0.0)) + fw / 2.0
    px, pw = float(person_box.get("x", 0.0)), float(person_box.get("w", 0.0))
    return min(cx - px, px + pw - cx) < FACE_EDGE_WIDTHS * fw


def masks(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(lit, skin)`` boolean masks of a band crop.

    Lit: V in [V_MIN, V_MAX] (shadow and specular blowout excluded, as in
    v2).  Skin: the YCrCb skin-tone window, at saturation under SKIN_S_MAX.
    Exposed so the evaluation tooling can draw exactly what the descriptor
    weighted; the colour histogram itself uses :func:`skin_weights`.
    """
    lit, skin, _ = _analyse(crop_bgr)
    return lit, skin


def skin_weights(crop_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel colour weight: SKIN_WEIGHT inside the skin window, 1.0 outside.

    A pixel at or above SKIN_S_MAX saturation is cloth (1.0) whatever its
    Cr/Cb.  Lighting is not applied here; the caller zeroes unlit pixels.
    """
    return _analyse(crop_bgr)[2]


def _analyse(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lit, skin, weight)`` for one crop — one colour conversion each way."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    lit = (v >= V_MIN) & (v <= V_MAX)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin = (
        (cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1])
        & (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1])
        & (s < SKIN_S_MAX)
    )
    weight = np.where(skin, SKIN_WEIGHT, 1.0)
    return lit, skin, weight


def cloth_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """The full-weight pixels: lit and NOT skin-toned (the None-rule count,
    and the pixels the pattern parts read)."""
    lit, skin = masks(crop_bgr)
    return lit & ~skin


def _colour_part(hsv: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The v2 partition as a WEIGHTED histogram, un-normalised (39 floats).

    ``weights`` is per pixel: 0 for masked pixels, SKIN_WEIGHT for lit
    skin-toned ones, 1 for the rest.  bincount with weights replaces
    calcHist, which cannot weight.
    """
    hue, sat, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    used = weights > 0.0
    chroma = used & (sat >= S_ACHROMATIC)
    achroma = used & (sat < S_ACHROMATIC)
    out = np.zeros(COLOUR_BINS, dtype=np.float64)
    if chroma.any():
        hbin = (hue[chroma].astype(np.int64) * H_BINS) // 180
        sbin = ((sat[chroma].astype(np.int64) - S_ACHROMATIC) * S_BINS) // (256 - S_ACHROMATIC)
        idx = np.clip(hbin, 0, H_BINS - 1) * S_BINS + np.clip(sbin, 0, S_BINS - 1)
        out[: H_BINS * S_BINS] = np.bincount(
            idx, weights=weights[chroma], minlength=H_BINS * S_BINS
        )
    if achroma.any():
        width = (V_MAX + 1 - V_MIN) / V_BINS
        pos = (v[achroma].astype(np.float64) - V_MIN) / width - 0.5
        lo = np.floor(pos).astype(np.int64)
        frac = pos - lo
        w = weights[achroma]
        base = H_BINS * S_BINS
        for offset, share in ((0, 1.0 - frac), (1, frac)):
            idx = np.clip(lo + offset, 0, V_BINS - 1)
            out[base : base + V_BINS] += np.bincount(idx, weights=share * w, minlength=V_BINS)
    return out


def _texture_parts(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Un-normalised LBP(8,1) riu2 (10) and Sobel edge-density (3) histograms.

    ``gray`` and ``mask`` are the band already resized to TEXTURE_WIDTH_PX.
    LBP uses the 3x3 neighbourhood (radius 1, no interpolation) in circular
    order, a neighbour counting as "1" only when it exceeds the centre by
    LBP_NOISE_T; a code with at most two 0/1 transitions is "uniform" and
    bins by its number of set bits (0..8), anything else lands in bin 9.
    Both parts read only the interior pixels the mask keeps, so skin and
    shadow shape neither.
    """
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1] + LBP_NOISE_T
    ring = (
        g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:], g[2:, 2:],
        g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2], g[:-2, :-2],
    )
    bits = [(n >= c) for n in ring]
    ones = np.zeros(c.shape, dtype=np.int16)
    trans = np.zeros(c.shape, dtype=np.int16)
    for i, b in enumerate(bits):
        ones += b
        trans += b != bits[(i + 1) % 8]
    code = np.where(trans <= 2, ones, TEXTURE_BINS - 1)
    inner = mask[1:-1, 1:-1]
    lbp = np.bincount(code[inner].ravel(), minlength=TEXTURE_BINS)[:TEXTURE_BINS].astype(np.float64)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = (np.hypot(gx, gy) / 4.0)[1:-1, 1:-1][inner]
    edge = np.array([
        float((mag < EDGE_LOW).sum()),
        float(((mag >= EDGE_LOW) & (mag < EDGE_HIGH)).sum()),
        float((mag >= EDGE_HIGH).sum()),
    ])
    return lbp, edge


def _l1(part: np.ndarray) -> np.ndarray | None:
    total = float(part.sum())
    return None if total <= 0.0 else part / total


def torso_descriptor(
    image_bgr: np.ndarray, face_box: dict, person_box: dict | None
) -> list[float] | None:
    """The 64-float torso descriptor for one face, or None when unmeasurable.

    Crop rule (wire contract): within ``person_box``, vertically from ``face
    bottom + 0.5 x face height`` down to ``min(face bottom + 3.0 x face
    height, person box bottom)``, horizontally the person box inset 15% each
    side clipped to the face's column (face centre +- 1.5 face widths) — the
    chest and torso BELOW the neck of the person the face belongs to.  The
    band starts half a face height under the chin because the v2 crop that
    started at the chin read skin and hair first (a yellow top vs a dark
    green dress intersected at 0.72 on run f0bfc5).

    Returns the L1-normalised 64-float v3 descriptor: colour (v2's 12x3 H x S
    chromatic bins 0..35 + 3 soft brightness bins 36..38, skin-toned pixels
    at a quarter weight) weighted 0.9, LBP(8,1) texture bins
    39..48 weighted 0.07, Sobel edge-density bins 49..51 weighted 0.03,
    reserved zeros 52..63 — or **None** when any of the contract's three
    conditions holds: no containing person box, a crop under 24 px in
    either dimension, or fewer than 100 full-weight pixels (V < 30 shadow /
    V > 252 blowout / YCrCb skin tone) — or, added after the contract, when
    the face sits at the box's left or right edge (FACE_EDGE_WIDTHS: the
    band would be the occluder, not the torso) or the band's resized mask
    keeps no interior cloth pixel for the pattern parts.  None means "could
    not measure", and per the module convention it must never be treated
    as a clash.
    """
    if person_box is None:
        return None
    if _face_at_box_edge(face_box, person_box):
        return None
    img_h, img_w = image_bgr.shape[:2]
    band = _band(face_box, person_box, img_w, img_h)
    if band is None:
        return None
    ix0, ix1, iy0, iy1 = band
    crop = image_bgr[iy0:iy1, ix0:ix1]
    lit, skin, skin_w = _analyse(crop)
    cloth = lit & ~skin
    if int(cloth.sum()) < MIN_UNMASKED_PX:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    weights = np.where(lit, skin_w, 0.0)
    colour = _l1(_colour_part(hsv, weights))
    if colour is None:  # defensive: cloth.sum() >= 100 makes this unreachable
        return None

    # Texture and edges on the band at a fixed, body-relative scale.
    ch, cw = crop.shape[:2]
    tw = TEXTURE_WIDTH_PX
    th = max(int(round(ch * tw / cw)), 3)
    interp = cv2.INTER_AREA if cw > tw else cv2.INTER_LINEAR
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (tw, th), interpolation=interp)
    small_mask = cv2.resize(
        cloth.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    lbp, edge = _texture_parts(gray, small_mask)
    lbp = _l1(lbp)
    edge = _l1(edge)

    # The resized mask can, in principle, keep no interior pixel (a sliver of
    # cloth one pixel wide); then the pattern parts are unmeasured and so is
    # the sighting.  Handing their weight to colour instead capped every
    # comparison at W_COLOUR — the absent reading behaved as a 0.1 clash,
    # against the module's rule that absent is not zero.
    if lbp is None or edge is None:
        return None
    out = np.zeros(APPEARANCE_DIM, dtype=np.float64)
    out[:COLOUR_BINS] = W_COLOUR * colour
    out[TEXTURE_OFFSET:TEXTURE_OFFSET + TEXTURE_BINS] = W_TEXTURE * lbp
    out[EDGE_OFFSET:EDGE_OFFSET + EDGE_BINS] = W_EDGE * edge
    return out.tolist()


def intersection(a: list[float], b: list[float]) -> float | None:
    """Histogram intersection: sum of element-wise minimums, range 0..1.

    Both sides are L1-normalized by construction, so identical descriptors
    score 1.0 and disjoint colour distributions score 0.0.  Chosen over
    cosine/chi-square because it is the bounded, monotone "how much of the
    same cloth do these two sightings share" reading — trivially explainable
    when a veto shows up in a run's log next to its similarity.

    None — not 0.0, not an exception — when the two descriptors differ in
    length: a v2 (48) row beside a v3 (64) sighting is "not comparable", and
    absent is not zero.
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape or va.ndim != 1 or va.size == 0:
        return None
    return float(np.minimum(va, vb).sum())
