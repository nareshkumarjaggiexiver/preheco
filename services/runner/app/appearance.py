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

THE v2 PARTITION: CHROMATIC BY HUE, ACHROMATIC BY BRIGHTNESS.  v1 was a pure
12x4 Hue x Saturation histogram with V only as a mask, on the argument that
brightness is the axis illumination moves along.  True for saturated cloth —
and measurably WRONG for the clothes this camera actually sees: a light shirt
is nearly desaturated, and the hue of a desaturated pixel is noise, so the
same white shirt scattered across hue bins differently every frame.  Measured
(2026-08-06, run cbdc6b): the same man's sightings read 0.88 / 0.48 / 0.77 /
0.88 — a 0.40 spread on unchanged clothing.  v1's V-mask also DISCARDED
V > 240, i.e. the brightest pixels of the very shirt it was trying to read.

v2 keeps the same 48-float wire and splits it:

* bins 0..35  — CHROMATIC pixels (S >= 40): 12 hue x 3 saturation.  Hue means
  something here, and brightness still deliberately plays no part.
* bins 36..38 — ACHROMATIC pixels (S < 40): 3 coarse brightness bins (dark /
  mid / light cloth) over V 30..252, SOFT-assigned (each pixel splits its
  mass between the two nearest bin centres) so the histogram is a continuous
  function of brightness — auto-exposure drift moves a little mass, never all
  of it.  White, grey and black cloth land here, described by the one
  attribute a desaturated fabric reliably has.
* bins 39..47 — reserved, always zero (the wire stays 48 and a future
  partition change has headroom without another contract bump).

Masks: V < 30 is shadow in both partitions (nothing reads reliably in the
dark); V > 252 is specular blowout (a highlight, not cloth) — deliberately
far above v1's 240 so bright shirts are finally IN the histogram.

ABSENT IS NOT ZERO (codebase-wide convention, like ``gatedUnmeasured`` /
``zoneUnmeasured``): a sighting with no descriptor — no containing person box,
a crop under 24 px, fewer than 100 unmasked pixels, an undecodable frame, an
old run recorded before this existed — returns/means None, and None never
vetoes anything.  A signal that could not be measured must not be treated as a
signal that measured "different".
"""

import cv2
import numpy as np

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

#: V-channel mask bounds for BOTH partitions: below = shadow (nothing reads
#: reliably in the dark), above = specular blowout (a highlight, not cloth).
#: v1 masked at 240 and threw away the brightest pixels of every light shirt;
#: 252 keeps them.
V_MIN = 30
V_MAX = 252

#: The three None conditions (wire contract): a torso crop under this many
#: pixels in either dimension carries too little cloth to histogram honestly…
MIN_CROP_PX = 24
#: …and so does a crop where masking leaves fewer than this many pixels.
MIN_UNMASKED_PX = 100

#: How far below the face the torso extends, in face-heights.  The crop stops
#: at the person box's bottom edge when that comes first (a guest close to the
#: camera, or partially occluded).
TORSO_DEPTH_FACE_HEIGHTS = 2.5

#: Horizontal inset of the person box on each side, keeping the histogram on
#: the wearer's cloth rather than the background the box edges usually clip.
SIDE_INSET_FRAC = 0.15


def torso_descriptor(
    image_bgr: np.ndarray, face_box: dict, person_box: dict | None
) -> list[float] | None:
    """The 48-float torso descriptor for one face, or None when unmeasurable.

    Crop rule (wire contract, fixed): within ``person_box``, vertically from
    the face box's bottom edge down to ``min(face bottom + 2.5 x face height,
    person box bottom)``, horizontally the person box inset 15% each side —
    i.e. the chest/torso of the person the face belongs to, avoiding both the
    face itself and the background at the box edges.  The crop is additionally
    clamped to the image, because detector boxes legitimately overhang frames.

    Returns the L1-normalized 48-float v2 descriptor (chromatic 12x3 H x S in
    bins 0..35, achromatic 6 V bins in 36..41, reserved zeros in 42..47 — see
    the module docstring for why the partition exists), or **None** when any
    of the contract's three conditions holds: no containing person box, a
    crop under 24 px in either dimension, or fewer than 100 pixels surviving
    the V-mask (V < 30 shadow / V > 252 blowout).  None means "could not
    measure", and per the module convention it must never be treated as a
    clash.
    """
    if person_box is None:
        return None
    # Only the face's vertical extent matters: the crop hangs BELOW the face
    # (its x comes from the person box, whose width tracks the shoulders).
    fy, fh = float(face_box.get("y", 0.0)), float(face_box.get("h", 0.0))
    px, py = float(person_box.get("x", 0.0)), float(person_box.get("y", 0.0))
    pw, ph = float(person_box.get("w", 0.0)), float(person_box.get("h", 0.0))

    face_bottom = fy + fh
    y0 = face_bottom
    y1 = min(face_bottom + TORSO_DEPTH_FACE_HEIGHTS * fh, py + ph)
    x0 = px + SIDE_INSET_FRAC * pw
    x1 = px + pw - SIDE_INSET_FRAC * pw

    img_h, img_w = image_bgr.shape[:2]
    ix0, ix1 = max(int(round(x0)), 0), min(int(round(x1)), img_w)
    iy0, iy1 = max(int(round(y0)), 0), min(int(round(y1)), img_h)
    if (ix1 - ix0) < MIN_CROP_PX or (iy1 - iy0) < MIN_CROP_PX:
        return None

    crop = image_bgr[iy0:iy1, ix0:ix1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, sat, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (v >= V_MIN) & (v <= V_MAX)
    if int(lit.sum()) < MIN_UNMASKED_PX:
        return None

    chroma = lit & (sat >= S_ACHROMATIC)
    achroma = lit & (sat < S_ACHROMATIC)

    out = np.zeros(48, dtype=np.float64)
    if chroma.any():
        hist = cv2.calcHist(
            [hsv], [0, 1], chroma.astype(np.uint8), [H_BINS, S_BINS],
            [0, 180, S_ACHROMATIC, 256],
        )
        out[: H_BINS * S_BINS] = hist.flatten()
    if achroma.any():
        width = (V_MAX + 1 - V_MIN) / V_BINS
        pos = (v[achroma].astype(np.float64) - V_MIN) / width - 0.5
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        base = H_BINS * S_BINS
        for offset, weight in ((0, 1.0 - frac), (1, frac)):
            idx = np.clip(lo + offset, 0, V_BINS - 1)
            np.add.at(out, base + idx, weight)
    total = float(out.sum())
    if total <= 0.0:  # defensive: lit.sum() >= 100 makes this unreachable
        return None
    return (out / total).tolist()


def intersection(a: list[float], b: list[float]) -> float:
    """Histogram intersection: sum of element-wise minimums, range 0..1.

    Both sides are L1-normalized by construction, so identical descriptors
    score 1.0 and disjoint colour distributions score 0.0.  Chosen over
    cosine/chi-square because it is the bounded, monotone "how much of the
    same cloth do these two sightings share" reading — trivially explainable
    when a veto shows up in a run's log next to its similarity.
    """
    return float(np.minimum(np.asarray(a, dtype=np.float64),
                            np.asarray(b, dtype=np.float64)).sum())
