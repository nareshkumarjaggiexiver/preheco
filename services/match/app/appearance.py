"""Torso-appearance similarity — the arithmetic of the advisory tie-breaker.

The descriptor is computed by the RUNNER, never here (no cv2 in this service
on purpose).  Two generations are on the wire and both are accepted:

* **v2, 48 floats** — an L1-normalised histogram of the torso crop below
  the chin: 12 hue × 3 saturation chromatic bins, 3 soft brightness bins for
  achromatic pixels, 12 reserved zeros (OpenCV HSV; V < 30 or V > 252
  masked out).  Galleries written before 2026-09-24 hold only these.
* **v3, 64 floats** — colour (the same 39 bins, skin-toned pixels
  down-weighted, over a band that starts BELOW the neck) weighted 0.9,
  texture (uniform LBP(8,1), 10 bins) 0.07, edge density (3 bins) 0.03,
  then 12 reserved zeros; each part L1-normalised before weighting so the
  whole still sums to 1.0.  The pattern parts are deliberately light: two
  plain garments of any two colours agree on them, so at 0.3 they lifted
  every impostor pair towards agreement (measured 0.19 -> 0.38 on a pink
  check against a white kurta) and separated nothing.  It exists because run
  f0bfc5 scored a yellow top against a dark green dress at 0.72: the v2 crop
  started right under the face and was dominated by neck/chest skin and hair,
  and it had no texture term at all to tell a plain shirt from a striped one.

By the time either reaches /match it is just numbers on the wire; this module
only compares them, and a v2 against a v3 is NOT COMPARABLE — that answer is
``None``, never an exception and never 0.0 (see :func:`intersection`).

WHY clothing gets a voice at all, and why only a whisper.  Clothing is
constant within one event, so a torso descriptor is genuine evidence about
whether two sightings seconds apart are the same body.  It is WEAK evidence
about identity, and the bench measured exactly how weak: the closest impostor
pair on this camera — two DIFFERENT men at cosine 0.377, above the 0.363 face
threshold — were BOTH IN LIGHT SHIRTS.  Any appearance term that could RESCUE
a borderline face match would have merged those two paying guests into one
invoice line, invisibly.  So a similarity computed here may only ever VETO a
write the face pipeline was about to make (refuse a heal, refuse a template
enrolment) and never cause one: nothing is minted, merged, matched or counted
on clothing alone.
"""

import numpy as np

#: The current (v3) descriptor length: 39 colour + 10 texture + 3 edge + 12
#: reserved.  What a new runner sends and what a fresh gallery stores.
APPEARANCE_DIM = 64

#: The v2 length (12 hue × 4 saturation bins).  Still accepted on the wire
#: and still present in every gallery written before v3, so the comparison
#: below must meet it without complaint.
LEGACY_APPEARANCE_DIM = 48

#: Every descriptor length the wire contract admits.
APPEARANCE_DIMS = (LEGACY_APPEARANCE_DIM, APPEARANCE_DIM)


def intersection(
    a: list[float] | np.ndarray, b: list[float] | np.ndarray
) -> float | None:
    """Histogram intersection of two descriptors: the sum of element-wise minimums.

    Both sides are L1-normalised by the wire contract, so the result lives in
    0..1 — 1.0 for identical histograms, 0.0 for mass in entirely disjoint
    bins.  Intersection is chosen over cosine or chi-squared because it is the
    bluntest instrument that works here: partial occlusion of the torso only
    REMOVES mass, which can only lower the score (fail towards "no clash", the
    safe direction), and the fixed 0..1 range keeps the clash knob legible.

    ``None`` when the descriptors differ in length or either is empty.  This
    used to raise, on the argument that a length mismatch was a wire bug —
    and then v3 (64 floats) arrived while every retained gallery still held
    v2 rows (48 floats), so a mismatch became an ordinary Tuesday: a sighting
    from the new runner compared against a template from before the upgrade.
    That comparison has no meaning, and the two honest answers are "not
    measured" or an exception.  Not an exception, because it would surface
    deep inside /match and take the verdict down with it; not 0.0, because
    zero reads as a MAXIMAL CLASH and would veto every enrolment against an
    older template.  So: None, absent is not zero, and every caller already
    handles None because unmeasured torsos always could be.
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.ndim != 1 or vb.ndim != 1 or va.size != vb.size or va.size == 0:
        return None
    return float(np.minimum(va, vb).sum())


def best_intersection(
    query: list[float] | np.ndarray | None, stored: list[np.ndarray]
) -> float | None:
    """Best intersection of ``query`` against an identity's stored descriptors.

    ``None`` — NOT 0.0 — when the query carries no descriptor, the identity
    has none stored (old galleries from before the column existed, sightings
    with no containing person box, torso crops under 24 px or with fewer than
    100 unmasked pixels), or none of the stored ones is the query's length
    (a v3 sighting against an all-v2 identity).  Absent is not zero, the
    codebase-wide convention (``gatedUnmeasured``, ``zoneUnmeasured``): a
    zero here would read as "maximally clashing" and veto enrolments for
    guests whose torso simply could not be measured, and under-counting is
    this pipeline's dominant failure mode.  Stored descriptors of the wrong
    length are skipped, not scored — a mixed identity is compared on the
    rows that CAN be compared.

    BEST rather than mean or worst, because the veto must stay charitable: an
    identity legitimately accumulates different-looking descriptors (lighting
    shifts across a doorway, a jacket over an arm), and a sighting should be
    called a clash only when it agrees with NONE of them.  The same-person
    face misses that motivated this whole feature (0.294/0.308/0.361 against
    the 0.363 threshold) wore identical clothes in every frame — the common
    case the veto must never punish.
    """
    if query is None or not stored:
        return None
    scores = [s for s in (intersection(query, s) for s in stored) if s is not None]
    return max(scores) if scores else None
