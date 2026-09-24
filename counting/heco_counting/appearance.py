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


def _analyse(
    crop_bgr: np.ndarray, sensor_bgr: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lit, skin, weight)`` for one crop — one colour conversion each way.

    ``sensor_bgr`` is the crop as the CAMERA delivered it when ``crop_bgr``
    has been white-balanced (:func:`apply_gains`): shadow and blowout are
    facts about the sensor — a highlight clipped at 255 and scaled by a 0.8
    gain reads 204, which is not cloth — so ``lit`` is read there, while the
    skin window reads the balanced colours.  None (no balancing) reads both
    off ``crop_bgr``, exactly as before gains existed.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    if sensor_bgr is not None:
        v = _max3(sensor_bgr)  # OpenCV's 8-bit V is max(B, G, R)
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
    image_bgr: np.ndarray,
    face_box: dict,
    person_box: dict | None,
    gains: tuple[float, float, float] | None = None,
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

    ``gains`` — the frame's white-balance gains from :func:`frame_gains` —
    are applied to the band before the skin window and the colour bins read
    it; the lit mask and the pattern parts still read the pixels as the
    camera delivered them (see :func:`_analyse`).  None is exactly the
    descriptor as it was before gains existed.
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
    if gains is None:
        colour_crop = crop
        lit, skin, skin_w = _analyse(crop)
    else:
        colour_crop = apply_gains(crop, gains)
        lit, skin, skin_w = _analyse(colour_crop, sensor_bgr=crop)
    cloth = lit & ~skin
    if int(cloth.sum()) < MIN_UNMASKED_PX:
        return None

    hsv = cv2.cvtColor(colour_crop, cv2.COLOR_BGR2HSV)
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


# ------------------------------------------------------------------ lighting
#
# A garment's colour on the wire is the garment times the LIGHT.  One event
# runs under several: warm hall lamps, the stage's coloured washes, a DJ's
# flashes, daylight at the door.  frame_gains() estimates the illuminant of
# a whole frame so every colour descriptor can be read as if under neutral
# light; it is advisory like the descriptors themselves, and off unless the
# runner's HECO_APPEARANCE_WB asks for it.

#: Shades-of-grey Minkowski norm (Finlayson & Trezzi 2004): p = 1 is
#: grey-world, p = infinity is max-RGB; p = 6 is where their evaluation put
#: the best single value, and it keeps one large coloured object (a red
#: stage backdrop) from dragging the estimate the way grey-world's mean does.
WB_P = 6
#: The estimate reads every 8th pixel each way — a 4K frame becomes 480 x
#: 270 = 129,600 samples, far more than an illuminant needs.  Striding, not
#: resizing: area-averaging 8 MP costs more than the estimate itself.
WB_STRIDE = 8
#: A sample whose brightest channel is under this is sensor noise, not a
#: colour: its ratios are what the codec left, not what the light did.
WB_DARK = 16
#: A sample with ANY channel at or above this is clipped: the channel stopped
#: at 255 while the light kept going, so its ratio is not the light's.
WB_BLOWN = 250
#: Fewer usable samples than this and the frame is "not measured" (None):
#: a black frame or a frame that is all blown sky has no illuminant to read.
WB_MIN_SAMPLES = 1000
#: Gains are normalised to average 1 (brightness is not the job) and then
#: clamped: a correction beyond 2x on one channel is the estimate failing —
#: a frame filled by one saturated garment — not a light that coloured.
WB_GAIN_MIN = 0.5
WB_GAIN_MAX = 2.0


def frame_gains(image_bgr: np.ndarray) -> tuple[float, float, float] | None:
    """Per-channel white-balance gains ``(gb, gg, gr)`` for one frame, or None.

    Shades-of-grey (p = WB_P) on a 1/WB_STRIDE sampling of the frame,
    ignoring near-black (max channel < WB_DARK) and clipped (any channel
    >= WB_BLOWN) samples; the gains are the inverse of the estimated
    illuminant, normalised so the three average 1, then clamped to
    [WB_GAIN_MIN, WB_GAIN_MAX].  None when fewer than WB_MIN_SAMPLES
    samples are usable — absent is not zero, and a caller handed None reads
    the colours as the camera delivered them.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None
    return gains_from_samples(image_bgr[::WB_STRIDE, ::WB_STRIDE])


def gains_from_samples(small_bgr: np.ndarray) -> tuple[float, float, float] | None:
    """:func:`frame_gains` on an image that is ALREADY the sample grid.

    Split out so an evaluation holding 1/8-scale frames (not the 4K
    originals) computes exactly the estimate the runner would.
    """
    b, g, r = small_bgr[:, :, 0], small_bgr[:, :, 1], small_bgr[:, :, 2]
    top = np.maximum(np.maximum(b, g), r)
    usable = (top >= WB_DARK) & (top < WB_BLOWN)
    n = int(usable.sum())
    if n < WB_MIN_SAMPLES:
        return None
    est = np.empty(3, dtype=np.float64)
    for i, ch in enumerate((b, g, r)):
        x = ch[usable].astype(np.float32) * np.float32(1.0 / 255.0)
        x2 = x * x
        est[i] = float(np.mean(x2 * x2 * x2, dtype=np.float64)) ** (1.0 / WB_P)
    if not np.all(est > 0.0):
        return None
    g = 1.0 / est
    g = np.clip(g / g.mean(), WB_GAIN_MIN, WB_GAIN_MAX)
    return float(g[0]), float(g[1]), float(g[2])


def apply_gains(crop_bgr: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """The crop under neutral light: each channel scaled by its gain, 8-bit."""
    g = np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(np.rint(crop_bgr.astype(np.float32) * g), 0, 255).astype(np.uint8)




# ---------------------------------------------------------------- head + beard
#
# Two more readings for the review queue, and like the torso ADVISORY: they
# may set a question aside, never answer one.  Run f0bfc5's review pair #1
# was a maroon turban and a black beard against a peach turban and a white
# beard, and the torso could not say so (a white shirt against a pink check
# agreed at 0.42 in some pair of reads).  Measured on the 45 identities of
# that queue's first 32 pairs (<= 24 reads each, SCRFD landmarks re-detected
# offline, scratchpad head-eval/):
#
# HEAD.  Headwear and hair colour above the eyes.  An identity's own reads
# agree at a median 0.87 (min 0.63); one person split across a time gap at
# a best cross of 0.83-0.99 (9 splits); pair #1 at 0.36.  Four choices, each
# against the first draft, each measured:
#
# * CHROMA, not saturation, decides "chromatic": OpenCV's S is (max-min)/max
#   and explodes in the dark — black hair (V 13-22) read S 40-93 and a random
#   hue.  Chroma C = max - min >= 20 keeps 97-100% of the maroon, peach and
#   sky-blue turbans' pixels and 0-7% of black or grey hair's.  A pale pink
#   turban (S ~40, C ~27) is pink rather than "white".
# * Skin-window pixels count a QUARTER, as on the torso: the peach turban of
#   pair #1 sits wholly inside the window (Cr ~154, Cb ~112, S ~95); masked,
#   it vanished and the descriptor read its creases and the wall.
# * No shadow floor: black hair is signal, not shadow (V 13-22 on this
#   camera, under the torso's floor of 30).
# * The region's top is clipped to the PERSON box's top when one is given:
#   0.6 face heights above a bare head is wall, and without the clip one
#   identity's own reads fell to 0.48 and a same-person split to 0.76.
#
# BEARD.  [skin, dark, grey, white] fractions of the chin, each pixel judged
# RELATIVE TO THE SAME FACE'S CHEEKS: absolute cuts failed on this dim hall —
# the white beard of pair #1 read 78% "dark" at V < 70 and most shaven men
# 50-92%.  Against the cheek: DARK under 0.4 of its brightness (the seven
# full beards' median reads 47-75% dark, 21 shaven or moustached men 3-48%),
# PALE when the saturation is under half the cheek's (a white beard is
# desaturated hair — pair #1's elder reads 59% pale and 0% dark — while a
# shaded chin is skin that kept its saturation), WHITE when a pale pixel is
# at least 0.9 of the cheek's brightness.

#: The head descriptor's wire length: 24 hue + 3 brightness + 13 reserved.
HEAD_DIM = 40
#: 24 hue bins of 15 degrees, SOFT-assigned and wrapping: a turban red
#: (H 0) and orange (H 12.6) share no bin, and H 179 is red like H 0.
HEAD_H_BINS = 24
#: Achromatic pixels by brightness: black / grey / white hair.
HEAD_V_BINS = 3
HEAD_RESERVED = HEAD_DIM - HEAD_H_BINS - HEAD_V_BINS  # 13
#: Chromatic from this chroma (max - min channel, 8-bit) up.
HEAD_CHROMA_MIN = 20
#: Weight of a skin-window pixel (1.0 everywhere else).
HEAD_SKIN_WEIGHT = 0.25
#: The region starts this many face heights ABOVE the face box's top edge
#: (a turban rises well above the box a face detector draws)...
HEAD_ABOVE_FACE_HEIGHTS = 0.6
#: ...but not above the person box's top, less this many face heights.
HEAD_PERSON_TOP_MARGIN = 0.05
#: ...and ends this many inter-eye distances above the eye line: the brow
#: and the eyes themselves are not headwear.
HEAD_EYE_MARGIN_IED = 0.15
#: Horizontally the face box widened by this fraction of its width a side.
HEAD_SIDE_FRAC = 0.15
#: A head window under this many pixels either way, or with fewer lit
#: pixels than HEAD_MIN_PX, is not measured.
HEAD_MIN_SIDE_PX = 8
HEAD_MIN_PX = 60

#: The beard window: nose tip down to the face box's bottom, between the
#: mouth corners widened by this many inter-eye distances a side.
BEARD_SIDE_IED = 0.25
BEARD_MIN_SIDE_PX = 6
BEARD_MIN_PX = 40
#: The reference skin: the band from this many IED under the eye line down
#: to the nose tip, across the eyes widened BEARD_CHEEK_SIDE_IED a side —
#: under-eye and upper cheek, skin on every face, bearded or not.
BEARD_CHEEK_TOP_IED = 0.25
BEARD_CHEEK_SIDE_IED = 0.2
BEARD_CHEEK_MIN_PX = 20
#: A cheek darker than this (median max channel) is too dark to judge by.
BEARD_REF_V_MIN = 20
#: Pixel classes relative to the cheek: DARK under this fraction of its
#: brightness...
BEARD_DARK_REL_V = 0.4
#: ...PALE (grey or white hair) under this fraction of its saturation...
BEARD_PALE_REL_S = 0.5
#: ...and a pale pixel is WHITE from this fraction of its brightness up.
BEARD_WHITE_REL_V = 0.9
#: The nose tip this far or further from the eye line toward the mouth line
#: (0 = at the eyes, 1 = at the mouth): the head is too far down for the
#: chin window to hold the chin. Was 1.0 (only a fully bowed head); against
#: eye labels on run f0bfc5 (38 identities, 803 reads) reads naming a
#: conflicting class were 0-1.2% below a drop of 0.85 and 6.9% from 0.85 to
#: 1.0 — a shaven man read "white" on both his steepest close-ups (the white
#: collar in the window), a girl "grey" — and a duplicate is usually minted
#: on exactly that head-down view. 0.8 drops that bin (~4% of reads).
BEARD_MAX_NOSE_DROP = 0.8


def _eyes(landmarks) -> tuple[float, float] | None:
    """``(eye_line_y, ied)`` from the first two landmarks (the eyes), or None."""
    try:
        (rx, ry), (lx, ly) = landmarks[0][:2], landmarks[1][:2]
        rx, ry, lx, ly = float(rx), float(ry), float(lx), float(ly)
    except (TypeError, ValueError, IndexError):
        return None
    ied = float(np.hypot(lx - rx, ly - ry))
    if not np.isfinite(ied) or ied <= 0.0:
        return None
    return (ry + ly) / 2.0, ied


def _window(x0, x1, y0, y1, img_w: int, img_h: int, min_side: int):
    """An integer window clamped to the image, or None under ``min_side``."""
    ix0, ix1 = max(int(round(x0)), 0), min(int(round(x1)), img_w)
    iy0, iy1 = max(int(round(y0)), 0), min(int(round(y1)), img_h)
    if (ix1 - ix0) < min_side or (iy1 - iy0) < min_side:
        return None
    return ix0, ix1, iy0, iy1


def head_region(
    face_box: dict, landmarks, img_w: int, img_h: int, person_box: dict | None = None
):
    """The head window ``(x0, x1, y0, y1)``, or None.

    From HEAD_ABOVE_FACE_HEIGHTS above the face box's top (clipped to the
    person box's top when that is lower) down to HEAD_EYE_MARGIN_IED above
    the eye line; the face box widened HEAD_SIDE_FRAC a side.
    """
    eyes = _eyes(landmarks)
    if eyes is None:
        return None
    eye_y, ied = eyes
    fx, fy = float(face_box.get("x", 0.0)), float(face_box.get("y", 0.0))
    fw, fh = float(face_box.get("w", 0.0)), float(face_box.get("h", 0.0))
    if fw <= 0 or fh <= 0:
        return None
    top = fy - HEAD_ABOVE_FACE_HEIGHTS * fh
    if person_box is not None:
        person_top = float(person_box.get("y", 0.0)) - HEAD_PERSON_TOP_MARGIN * fh
        if top < person_top < fy:
            top = person_top
    return _window(
        fx - HEAD_SIDE_FRAC * fw, fx + fw + HEAD_SIDE_FRAC * fw,
        top, eye_y - HEAD_EYE_MARGIN_IED * ied, img_w, img_h, HEAD_MIN_SIDE_PX,
    )


def beard_region(face_box: dict, landmarks, img_w: int, img_h: int):
    """The beard window ``(x0, x1, y0, y1)``: nose tip to box bottom, or None."""
    eyes = _eyes(landmarks)
    if eyes is None:
        return None
    try:
        nose_y = float(landmarks[2][1])
        mouth_x = (float(landmarks[3][0]), float(landmarks[4][0]))
    except (TypeError, ValueError, IndexError):
        return None
    fy, fh = float(face_box.get("y", 0.0)), float(face_box.get("h", 0.0))
    if fh <= 0:
        return None
    ied = eyes[1]
    return _window(
        min(mouth_x) - BEARD_SIDE_IED * ied, max(mouth_x) + BEARD_SIDE_IED * ied,
        nose_y, fy + fh, img_w, img_h, BEARD_MIN_SIDE_PX,
    )


def cheek_region(landmarks, img_w: int, img_h: int):
    """The reference-skin window under the eyes, or None."""
    eyes = _eyes(landmarks)
    if eyes is None:
        return None
    eye_y, ied = eyes
    try:
        xs = (float(landmarks[0][0]), float(landmarks[1][0]))
        nose_y = float(landmarks[2][1])
    except (TypeError, ValueError, IndexError):
        return None
    return _window(
        min(xs) - BEARD_CHEEK_SIDE_IED * ied, max(xs) + BEARD_CHEEK_SIDE_IED * ied,
        eye_y + BEARD_CHEEK_TOP_IED * ied, nose_y, img_w, img_h, 2,
    )


def nose_drop(landmarks) -> float | None:
    """Where the nose tip sits between the eye line (0) and the mouth line (1)."""
    eyes = _eyes(landmarks)
    if eyes is None:
        return None
    try:
        nose_y = float(landmarks[2][1])
        mouth_y = (float(landmarks[3][1]) + float(landmarks[4][1])) / 2.0
    except (TypeError, ValueError, IndexError):
        return None
    span = mouth_y - eyes[0]
    return None if span <= 0 else (nose_y - eyes[0]) / span


def _max3(img: np.ndarray) -> np.ndarray:
    """Per-pixel max over the 3 channels — numpy's reduce over an axis of
    length 3 cost 70% of the head and beard time; this is 6x faster."""
    return np.maximum(np.maximum(img[:, :, 0], img[:, :, 1]), img[:, :, 2])


def _min3(img: np.ndarray) -> np.ndarray:
    """Per-pixel min over the 3 channels (see :func:`_max3`)."""
    return np.minimum(np.minimum(img[:, :, 0], img[:, :, 1]), img[:, :, 2])


def _skin_window(crop_bgr: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """The torso's skin window: YCrCb, capped at SKIN_S_MAX saturation."""
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return (
        (cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1])
        & (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1])
        & (sat < SKIN_S_MAX)
    )


def _soft_hue(hue: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """HEAD_H_BINS hue histogram, triangular assignment, WRAPPING at red."""
    pos = hue.astype(np.float64) * HEAD_H_BINS / 180.0 - 0.5
    lo = np.floor(pos).astype(np.int64)
    frac = pos - lo
    out = np.bincount(lo % HEAD_H_BINS, weights=(1.0 - frac) * weights, minlength=HEAD_H_BINS)
    out += np.bincount((lo + 1) % HEAD_H_BINS, weights=frac * weights, minlength=HEAD_H_BINS)
    return out


def _soft_v(v: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """HEAD_V_BINS brightness histogram over 0..V_MAX, triangular, clamped."""
    width = (V_MAX + 1) / HEAD_V_BINS
    pos = v.astype(np.float64) / width - 0.5
    lo = np.floor(pos).astype(np.int64)
    frac = pos - lo
    out = np.zeros(HEAD_V_BINS, dtype=np.float64)
    for offset, share in ((0, 1.0 - frac), (1, frac)):
        out += np.bincount(
            np.clip(lo + offset, 0, HEAD_V_BINS - 1), weights=share * weights,
            minlength=HEAD_V_BINS,
        )
    return out


def head_descriptor(
    image_bgr: np.ndarray,
    face_box: dict,
    landmarks,
    gains: tuple[float, float, float] | None = None,
    person_box: dict | None = None,
) -> list[float] | None:
    """The 40-float head descriptor — headwear or hair colour — or None.

    Region (:func:`head_region`): 0.6 face heights above the face box (or
    the person box's top, when lower) down to the eye line less 0.15 IED;
    the box widened 0.15 of its width a side; clamped to the image.  Every
    lit (not blown) pixel votes, skin-window pixels at HEAD_SKIN_WEIGHT:
    those with chroma >= HEAD_CHROMA_MIN into 24 soft, wrapping hue bins
    (0..23), the rest into 3 soft brightness bins (24..26: black, grey,
    white); 27..39 are reserved zeros; the whole sums to 1.  ``gains``
    (:func:`frame_gains`) are applied first, as for the torso.

    None — never a zero vector — without two eye landmarks, for a window
    under HEAD_MIN_SIDE_PX, or with fewer than HEAD_MIN_PX lit pixels.
    """
    img_h, img_w = image_bgr.shape[:2]
    win = head_region(face_box, landmarks, img_w, img_h, person_box)
    if win is None:
        return None
    x0, x1, y0, y1 = win
    crop = image_bgr[y0:y1, x0:x1]
    colour = crop if gains is None else apply_gains(crop, gains)
    lit = _max3(crop) <= V_MAX
    if int(lit.sum()) < HEAD_MIN_PX:
        return None
    hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
    weight = np.where(lit, np.where(_skin_window(colour, hsv[:, :, 1]), HEAD_SKIN_WEIGHT, 1.0), 0.0)
    chromatic = (_max3(colour).astype(np.int16) - _min3(colour)) >= HEAD_CHROMA_MIN
    out = np.zeros(HEAD_DIM, dtype=np.float64)
    sel = lit & chromatic
    if sel.any():
        out[:HEAD_H_BINS] = _soft_hue(hsv[:, :, 0][sel], weight[sel])
    sel = lit & ~chromatic
    if sel.any():
        out[HEAD_H_BINS:HEAD_H_BINS + HEAD_V_BINS] = _soft_v(hsv[:, :, 2][sel], weight[sel])
    total = float(out.sum())
    if total <= 0.0:
        return None
    return (out / total).tolist()


def beard_descriptor(
    image_bgr: np.ndarray,
    face_box: dict,
    landmarks,
    gains: tuple[float, float, float] | None = None,
) -> list[float] | None:
    """``[skinFrac, darkFrac, greyFrac, whiteFrac]`` of the chin, or None.

    Region (:func:`beard_region`): nose tip down to the face box's bottom,
    between the mouth corners widened 0.25 IED a side.  Each lit pixel is
    judged against the SAME face's cheeks (:func:`cheek_region`, median
    brightness and saturation): DARK under BEARD_DARK_REL_V of the cheek's
    brightness; otherwise PALE under BEARD_PALE_REL_S of its saturation —
    WHITE from BEARD_WHITE_REL_V of its brightness up, GREY below; the rest
    SKIN.  The four fractions sum to 1.

    None without eye/nose/mouth landmarks, with the nose at or below the
    mouth line (the chin is out of sight), for a window under
    BEARD_MIN_SIDE_PX or BEARD_MIN_PX lit pixels, or when the cheeks are
    too few or too dark to judge by.
    """
    drop = nose_drop(landmarks)
    if drop is None or drop >= BEARD_MAX_NOSE_DROP:
        return None
    img_h, img_w = image_bgr.shape[:2]
    win = beard_region(face_box, landmarks, img_w, img_h)
    cheek = cheek_region(landmarks, img_w, img_h)
    if win is None or cheek is None:
        return None

    def read(w):
        crop = image_bgr[w[2]:w[3], w[0]:w[1]]
        lit = _max3(crop) <= V_MAX
        colour = crop if gains is None else apply_gains(crop, gains)
        v = _max3(colour).astype(np.int16)
        return v[lit].astype(np.float64), (v - _min3(colour))[lit].astype(np.float64)

    cv, cc = read(cheek)
    if cv.size < BEARD_CHEEK_MIN_PX:
        return None
    v_ref = float(np.median(cv))
    s_ref = float(np.median(cc / np.maximum(cv, 1.0)))
    if v_ref < BEARD_REF_V_MIN or s_ref <= 0.0:
        return None
    v, c = read(win)
    if v.size < BEARD_MIN_PX:
        return None
    dark = v < BEARD_DARK_REL_V * v_ref
    pale = ~dark & (c / np.maximum(v, 1.0) < BEARD_PALE_REL_S * s_ref)
    white = pale & (v >= BEARD_WHITE_REL_V * v_ref)
    n = float(v.size)
    return [
        float((~dark & ~pale).sum()) / n,
        float(dark.sum()) / n,
        float((pale & ~white).sum()) / n,
        float(white.sum()) / n,
    ]


# ----------------------------------------------------------------- the light
#
# A fourth reading for the review queue, and the only one that is evidence
# about the LIGHT rather than the person: the face's own skin colour.  A
# duplicate is minted exactly when a face fails to match, and a change of
# light is one cause — so the pair the queue most needs to show is the one
# whose clothes, turban and beard were read under two different lights.
# Replayed on run f0bfc5's crops (34 single-person identities split at their
# time median, the late half under a per-channel shift): a +/-8% shift set
# one genuine duplicate aside (a white shirt, cross 0.99 -> 0.28; a white
# beard read "none"), +/-15% two or three, a stage wash seven or eight; the
# frame's white balance (HECO_APPEARANCE_WB) cures a WHOLE-frame cast and
# cannot see a light on one person (a spotlight, a videographer's lamp).
# Skin moves with the light that falls on it: the same splits read the
# cheek's median log(R/G), log(B/G) apart by 0.020 (median; p90 0.048)
# under one light, and by 0.048-0.10 under a +/-8% shift, 0.11-0.23 under
# +/-15%, 0.25-0.95 under the washes.  The match service holds a colour
# set-aside back when two identities' skin readings disagree by more than
# its tolerance (HECO_REVIEW_LIGHT_TOL).

#: The skin reading's wire length: [log(R/G), log(B/G)] of the cheek window.
SKIN_DIM = 2
#: A cheek pixel is read when its brightest channel is at least this (a
#: ratio of two small numbers is noise)...
SKIN_V_MIN = 40
#: ...its dimmest at least this (log of ~0)...
SKIN_CHANNEL_MIN = 8
#: ...and at most V_MAX (clipped).  Fewer usable pixels: not measured.
SKIN_MIN_PX = 20


def skin_tone(
    image_bgr: np.ndarray,
    landmarks,
    gains: tuple[float, float, float] | None = None,
) -> list[float] | None:
    """``[log(R/G), log(B/G)]`` — the medians over the cheek window — or None.

    The window is the beard reading's reference skin (:func:`cheek_region`,
    under the eyes down to the nose tip), read under ``gains`` when given,
    as every other colour reading is.  Log ratios because a light that
    scales the three channels moves them by the same amount whatever the
    skin, so a shift between two readings is the light's, in the units the
    tolerance is set in.  None without eye and nose landmarks or with fewer
    than SKIN_MIN_PX usable pixels — absent is not zero.
    """
    img_h, img_w = image_bgr.shape[:2]
    win = cheek_region(landmarks, img_w, img_h)
    if win is None:
        return None
    crop = image_bgr[win[2]:win[3], win[0]:win[1]]
    if gains is not None:
        crop = apply_gains(crop, gains)
    px = crop.reshape(-1, 3).astype(np.float64)
    top, low = px.max(axis=1), px.min(axis=1)
    usable = (top <= V_MAX) & (top >= SKIN_V_MIN) & (low >= SKIN_CHANNEL_MIN)
    if int(usable.sum()) < SKIN_MIN_PX:
        return None
    b, g, r = px[usable, 0], px[usable, 1], px[usable, 2]
    return [float(np.median(np.log(r / g))), float(np.median(np.log(b / g)))]
