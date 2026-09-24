"""Lower-body outfit descriptor: an ADVISORY clothing signal for the thighs.

WHY THIS EXISTS.  The operator of a Punjab wedding hall asked the duplicate
review to "keep track of trouser colour, shirt colour, turban colour".  The
torso descriptor (:mod:`heco_counting.appearance`) reads the shirt; this reads
what the same guest wears below it — trousers, pyjama, salwar, lehenga, a
saree's pleats — in the SAME 64-float layout, so the match service can store
it and compare it with the same histogram intersection.  Like the torso it
is only ever evidence for a CLASH, never for agreement: most men at this
wedding wear dark trousers, so two different identities agree on their legs
far more than on their shirts (median best agreement 0.72 against the
torso's 0.49 on run c84098).

THE BAND is the THIGHS of a STANDING person, not the whole lower body:

* top — the hip line, face bottom + HIP_FACE_HEIGHTS (3.0) face heights.  On
  run c84098's 1,699 standing face sightings that lands at a median 0.53 of
  the box height (p5 0.46, p95 0.69): the crotch.  Without a face, box top +
  HIP_BOX_FRAC (0.55) of its height — the weaker anchor, because a box moves
  with arms, stride and detector jitter (the worst same-person split fell
  from 0.74 with the face line to 0.21 with the box line); a caller that has
  the face should pass it.
* bottom — the knee, KNEE_BOX_FRAC (0.72) of the box height below its top.
  The first cut ran to the box bottom minus 3%, as the brief asked, and the
  visual check said no: a walking guest's stride opens a triangle of FLOOR
  between the shins, and this hall's floor is black marble with white
  chevrons — dark plus light, which is what a white pyjama and a charcoal
  trouser both partly read as.  Stopping at the knee keeps the gap mostly out.
* sides — the box inset SIDE_INSET_FRAC (30%) a side: an arm swing or a
  turned body widens the box, never the legs.  A face-column clip like the
  torso's was measured and rejected: the face is not on the legs' centre
  line (its centre sits 0.30-0.72 across the box, p5-p95), and at +-1 face
  width the worst same-person split fell 0.86 -> 0.80, p5 0.90 -> 0.82.

WHY DARK IS CLOTH HERE (the torso masks it as shadow).  Below V 30, on legs,
a pixel is mostly BLACK TROUSERS.  Under the torso's mask the lit remainder
of a dark-trousered band is the floor and a hand, and a man in charcoal
trousers agreed 0.78 with a man in a cream pathani suit (review pair
p00014/p00019).  Counting V < 30 as the dark achromatic bin — its hue and
saturation are noise, so it is never chromatic and never skin — put that
pair at 0.50 and tightened every same-person reading (early half against
late half of 60 identities: p5 0.71 -> 0.86).  Blown highlights (V > 252)
are still masked.

WHAT IT CANNOT DO.  Navy and dark brown are one colour at V < 30: the three
review pairs of two men in dark trousers checked by eye agree at 0.88-0.94
whatever else differs.  The three soft brightness bins are the torso's too:
a cream kurta reads mostly MID grey, so against charcoal trousers with pale
wall showing through a stride the best of 12-16 reads climbs from 0.50 to
0.60.  A long kurta or kameez hangs over the thighs, and an untucked
shirt's hem reaches the hip line, so for those the band partly reads the
torso's garment.  Skin-toned cloth (pink, peach, beige, pale gold) is
down-weighted with bare skin, exactly as the torso does.

ABSENT IS NOT ZERO (the codebase-wide convention).  None — never a zero, never
a clash — when the box is not a standing full body (h/w under 2.0, or under
6 face widths when the face is known: a seated guest's band is lap and
chair), when its bottom reaches within 10 px of the frame's bottom (the legs
continue out of shot), when the band is shorter than half a face height (a
box the frame cut just above that margin), when people IN FRONT cover more
than 30% of the band, when the band is under 24 px, or when fewer than 100
readable non-skin pixels remain.  "In front" is read from the feet: a box
whose bottom edge is level with or below this one's stands nearer the
camera; a box whose feet are higher stands behind, and this guest's legs
hide it, not the other way round.  On c84098 that reading measured 22 more
sightings than counting every overlapping box — six checked by eye, each
the guest's own trousers — and left the worst same-person split at 0.86.

MEASURED on c84098 (Sharon wedding, overview camera; scratchpad
outfit-eval/, 2026-09-24): measurable on 52.5% of the run's 2,808 face
sightings (seated or head-to-waist 27.7%, cut by the frame 11.8%, band too
small 4.1%, someone in front 3.8%); a person against themselves (early half
vs late half, up to 8 reads) never under 0.74, p5 0.87, median 0.95.
"""

import cv2
import numpy as np

from heco_counting.appearance import (
    APPEARANCE_DIM,
    COLOUR_BINS,
    EDGE_BINS,
    EDGE_HIGH,
    EDGE_LOW,
    EDGE_OFFSET,
    H_BINS,
    LBP_NOISE_T,
    MIN_CROP_PX,
    MIN_UNMASKED_PX,
    S_ACHROMATIC,
    S_BINS,
    SKIN_CB,
    SKIN_CR,
    SKIN_S_MAX,
    SKIN_WEIGHT,
    TEXTURE_BINS,
    TEXTURE_OFFSET,
    TEXTURE_WIDTH_PX,
    V_BINS,
    V_MAX,
    V_MIN,
    W_COLOUR,
    W_EDGE,
    W_TEXTURE,
    intersection,
)

__all__ = [
    "APPEARANCE_DIM",
    "apply_gains",
    "cloth_mask",
    "covered_fraction",
    "in_front",
    "intersection",
    "lower_band",
    "lower_descriptor",
    "standing_box",
]

#: The hip line, in face heights below the face box's bottom edge.  2.5 /
#: 3.0 / 3.5 measured within 0.03 of each other on the same-person split;
#: 3.0 is where the median standing guest's crotch is.
HIP_FACE_HEIGHTS = 3.0
#: The hip line without a face: this fraction of the box height below its
#: top.  0.50 / 0.55 / 0.60 measured; all weaker than the face line.
HIP_BOX_FRAC = 0.55
#: The band's bottom edge — the knee — as a fraction of the box height below
#: its top.  0.97 (the brief's "3% above the feet") put the stride gap's
#: floor into walking guests' bands: worst same-person split 0.76 and a
#: different-identity median of 0.79, against 0.86 and 0.72 at 0.72.  0.68
#: and 0.64 read within noise of 0.72 and measured fewer sightings.
KNEE_BOX_FRAC = 0.72
#: Horizontal inset of the person box on each side.  0.20 and 0.35 measured
#: against 0.30: 0.20 lets the swinging hand and the floor beside the legs
#: in (worst same-person split 0.74 against 0.86); 0.35 was no better.
SIDE_INSET_FRAC = 0.30
#: A standing full body is at least this tall for its width (seated and
#: bending bodies are squatter) — the match service's stature rule.
STANDING_ASPECT = 2.0
#: …and at least this many face widths tall when the face is known: a
#: head-to-waist box behind a table has a standing aspect but no legs.
STANDING_FACE_WIDTHS = 6.0
#: A box whose bottom edge is within this many pixels of the frame's bottom
#: is cut by the frame: the legs continue out of shot.
FRAME_BOTTOM_MARGIN_PX = 10.0
#: The band is hidden when more than this fraction of it lies under the
#: boxes of people standing in front.
OCCLUDED_FRAC = 0.30
#: Another box stands IN FRONT when its bottom edge is at or below this
#: box's bottom edge less this fraction of this box's height — nearer the
#: camera, or level with it (side by side counts as in front: the
#: conservative reading).
FRONT_TOLERANCE_FRAC = 0.02
#: The band must be at least this many face heights tall when the face is
#: known.  A shorter one means the box is shorter than its face says a
#: standing body is — cut by the frame yet ending just above the 10 px
#: margin, or crouching — and the "knee" has risen to the shirt hem.  0.5
#: removed 10 of 1,039 measured sightings on c84098, every one a box ending
#: 14-32 px above the frame's bottom edge at 6.3-6.9 face heights tall (a
#: whole guest is ~8.6), and raised the same-person split p5 from 0.83 to
#: 0.87; 0.75 / 1.0 / 1.25 cost 22 / 41 / 99 sightings for nothing further.
MIN_BAND_FACE_HEIGHTS = 0.5


def _face_known(face_box: dict | None) -> bool:
    return (
        face_box is not None
        and float(face_box.get("w", 0.0)) > 0.0
        and float(face_box.get("h", 0.0)) > 0.0
    )


def standing_box(person_box: dict, face_box: dict | None = None) -> bool:
    """Whether ``person_box`` reads as a STANDING full body.

    At least STANDING_ASPECT times as tall as wide, and — when the face is
    known — at least STANDING_FACE_WIDTHS face widths tall.
    """
    pw, ph = float(person_box.get("w", 0.0)), float(person_box.get("h", 0.0))
    if pw <= 0.0 or ph <= 0.0 or ph / pw < STANDING_ASPECT:
        return False
    return not (_face_known(face_box) and ph < STANDING_FACE_WIDTHS * float(face_box["w"]))


def lower_band(
    person_box: dict, face_box: dict | None, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """The integer crop window ``(x0, x1, y0, y1)`` of the thigh band, or None.

    Vertically from the hip line (face bottom + HIP_FACE_HEIGHTS face heights,
    or box top + HIP_BOX_FRAC of its height without a face) to the knee (box
    top + KNEE_BOX_FRAC of its height); horizontally the box inset
    SIDE_INSET_FRAC a side; clamped to the image.  None when the window is
    under MIN_CROP_PX in either dimension, or empty.
    """
    px, py = float(person_box.get("x", 0.0)), float(person_box.get("y", 0.0))
    pw, ph = float(person_box.get("w", 0.0)), float(person_box.get("h", 0.0))
    if _face_known(face_box):
        y0 = float(face_box.get("y", 0.0)) + float(face_box["h"]) * (1.0 + HIP_FACE_HEIGHTS)
    else:
        y0 = py + HIP_BOX_FRAC * ph
    y1 = py + KNEE_BOX_FRAC * ph
    x0 = px + SIDE_INSET_FRAC * pw
    x1 = px + pw - SIDE_INSET_FRAC * pw
    if _face_known(face_box) and (y1 - y0) < MIN_BAND_FACE_HEIGHTS * float(face_box["h"]):
        return None
    ix0, ix1 = max(int(round(x0)), 0), min(int(round(x1)), img_w)
    iy0, iy1 = max(int(round(y0)), 0), min(int(round(y1)), img_h)
    if (ix1 - ix0) < MIN_CROP_PX or (iy1 - iy0) < MIN_CROP_PX:
        return None
    return ix0, ix1, iy0, iy1


def in_front(person_box: dict, others) -> list[tuple[float, float, float, float]]:
    """The boxes of ``others`` standing in front of (or level with) ``person_box``.

    Returned as ``(x, y, w, h)`` tuples.  The person's own box — a caller may
    pass every box of the frame — is not someone else and is skipped.
    """
    own = tuple(float(person_box.get(q, 0.0)) for q in "xywh")
    bottom = own[1] + own[3]
    out = []
    for o in others or ():
        box = tuple(float(o.get(q, 0.0)) for q in "xywh")
        if box == own or box[2] <= 0.0 or box[3] <= 0.0:
            continue
        if box[1] + box[3] >= bottom - FRONT_TOLERANCE_FRAC * own[3]:
            out.append(box)
    return out


def covered_fraction(band: tuple[int, int, int, int], boxes) -> float:
    """The fraction of ``band`` under the union of ``boxes`` (``(x, y, w, h)`` tuples)."""
    ix0, ix1, iy0, iy1 = band
    mask = np.zeros((iy1 - iy0, ix1 - ix0), dtype=bool)
    for x, y, w, h in boxes or ():
        ox0, oy0 = max(int(round(x)) - ix0, 0), max(int(round(y)) - iy0, 0)
        ox1 = min(int(round(x + w)) - ix0, ix1 - ix0)
        oy1 = min(int(round(y + h)) - iy0, iy1 - iy0)
        if ox1 > ox0 and oy1 > oy0:
            mask[oy0:oy1, ox0:ox1] = True
    return float(mask.mean()) if mask.size else 1.0


def apply_gains(crop_bgr: np.ndarray, gains) -> np.ndarray:
    """The crop under neutral light: each channel scaled by its (b, g, r) gain, 8-bit."""
    g = np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(np.rint(crop_bgr.astype(np.float32) * g), 0, 255).astype(np.uint8)


def _analyse(colour_bgr: np.ndarray, sensor_bgr: np.ndarray):
    """``(hsv, readable, skin)`` for one band.

    Readable: not blown out (V <= V_MAX), read on the SENSOR pixels — a
    clipped highlight is clipped whatever the white balance says.  Pixels
    under V_MIN are DARK CLOTH: their hue and saturation are noise, so they
    are rewritten to the darkest achromatic reading (S 0, V V_MIN) and are
    never skin.  Skin: the torso's YCrCb window under SKIN_S_MAX, read on
    the (possibly white-balanced) colour pixels.
    """
    hsv = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2HSV)
    v = sensor_bgr.max(axis=2)  # OpenCV's 8-bit V is max(B, G, R)
    ycrcb = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    dark = v < V_MIN
    skin = (
        (cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1])
        & (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1])
        & (hsv[:, :, 1] < SKIN_S_MAX)
        & ~dark
    )
    hsv[:, :, 1][dark] = 0
    hsv[:, :, 2][dark] = V_MIN
    return hsv, v <= V_MAX, skin


def cloth_mask(crop_bgr: np.ndarray, gains=None) -> np.ndarray:
    """The full-weight pixels of a band crop: readable and not skin-toned.

    The None rule counts these; exposed so evaluation tooling can draw
    exactly what the descriptor weighted.
    """
    colour = crop_bgr if gains is None else apply_gains(crop_bgr, gains)
    _, readable, skin = _analyse(colour, crop_bgr)
    return readable & ~skin


def _colour_part(hsv: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The torso's colour partition as a weighted histogram, un-normalised (39).

    12 hue x 3 saturation chromatic bins for S >= S_ACHROMATIC, then three
    SOFT brightness bins.  A copy of appearance's private helper, so the two
    descriptors can never bin differently under one layout:
    tests/test_outfit.py pins them equal.
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
    """The torso's LBP(8,1) riu2 (10) and Sobel edge-density (3) parts, un-normalised."""
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
    lbp = np.bincount(code[inner].ravel(), minlength=TEXTURE_BINS)[:TEXTURE_BINS]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = (np.hypot(gx, gy) / 4.0)[1:-1, 1:-1][inner]
    edge = np.array([
        float((mag < EDGE_LOW).sum()),
        float(((mag >= EDGE_LOW) & (mag < EDGE_HIGH)).sum()),
        float((mag >= EDGE_HIGH).sum()),
    ])
    return lbp.astype(np.float64), edge


def _l1(part: np.ndarray) -> np.ndarray | None:
    total = float(part.sum())
    return None if total <= 0.0 else part / total


def lower_descriptor(
    image_bgr: np.ndarray,
    person_box: dict | None,
    face_box: dict | None = None,
    frame_h: int | None = None,
    others=None,
    gains=None,
) -> list[float] | None:
    """The 64-float lower-body (thigh) descriptor of one person, or None.

    ``image_bgr`` is the frame, or a crop of it holding the whole box with
    ``frame_h`` the frame's height in the crop's coordinates (default: the
    image's own height); ``person_box`` and ``face_box`` are ``{x, y, w, h}``
    in image pixels; ``others`` are the frame's other person boxes (the
    person's own box among them is ignored); ``gains`` are optional
    per-frame ``(b, g, r)`` white-balance gains, applied to the colours
    before the skin window and the colour bins read them — readability,
    darkness and the pattern parts read the pixels as the camera delivered
    them, so unit gains are exactly no gains.

    Returns the torso's layout: colour bins 0..38 (skin-toned pixels at
    SKIN_WEIGHT, dark pixels in the dark achromatic bin) weighted W_COLOUR,
    LBP texture 39..48 weighted W_TEXTURE, edge density 49..51 weighted
    W_EDGE, reserved zeros 52..63; the whole sums to 1.0.  None under the
    module's None rules, none of which may ever be read as a clash.
    """
    if person_box is None or image_bgr is None:
        return None
    if not standing_box(person_box, face_box):
        return None
    img_h, img_w = image_bgr.shape[:2]
    limit = img_h if frame_h is None else frame_h
    bottom = float(person_box.get("y", 0.0)) + float(person_box.get("h", 0.0))
    if bottom >= limit - FRAME_BOTTOM_MARGIN_PX:
        return None
    band = lower_band(person_box, face_box, img_w, img_h)
    if band is None:
        return None
    front = in_front(person_box, others)
    if front and covered_fraction(band, front) > OCCLUDED_FRAC:
        return None
    ix0, ix1, iy0, iy1 = band
    crop = image_bgr[iy0:iy1, ix0:ix1]
    colour_crop = crop if gains is None else apply_gains(crop, gains)
    hsv, readable, skin = _analyse(colour_crop, crop)
    cloth = readable & ~skin
    if int(cloth.sum()) < MIN_UNMASKED_PX:
        return None
    weights = np.where(readable, np.where(skin, SKIN_WEIGHT, 1.0), 0.0)
    colour = _l1(_colour_part(hsv, weights))

    # Pattern at a fixed, body-relative scale, on the camera's own pixels.
    ch, cw = crop.shape[:2]
    tw = TEXTURE_WIDTH_PX
    th = max(int(round(ch * tw / cw)), 3)
    interp = cv2.INTER_AREA if cw > tw else cv2.INTER_LINEAR
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (tw, th), interpolation=interp)
    small = cv2.resize(cloth.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
    lbp, edge = _texture_parts(gray, small.astype(bool))
    lbp, edge = _l1(lbp), _l1(edge)
    # An empty part is an unmeasured part, and so is the sighting: handing
    # its weight to colour would read the missing pattern as disagreement.
    if colour is None or lbp is None or edge is None:
        return None
    out = np.zeros(APPEARANCE_DIM, dtype=np.float64)
    out[:COLOUR_BINS] = W_COLOUR * colour
    out[TEXTURE_OFFSET : TEXTURE_OFFSET + TEXTURE_BINS] = W_TEXTURE * lbp
    out[EDGE_OFFSET : EDGE_OFFSET + EDGE_BINS] = W_EDGE * edge
    return out.tolist()
