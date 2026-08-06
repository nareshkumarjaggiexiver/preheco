"""Torso-appearance similarity — the arithmetic of the advisory tie-breaker.

The descriptor is computed by the RUNNER, never here (no cv2 in this service
on purpose): 48 floats, an L1-normalised 12×4 Hue×Saturation histogram
(OpenCV HSV: H 0–179 → 12 bins, S 0–255 → 4 bins) of the torso crop, with
pixels at V < 40 or V > 240 masked out (shadow/blowout).  By the time it
reaches /match it is just 48 numbers on the wire; this module only compares
them.

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

#: Descriptor length fixed by the wire contract: 12 hue bins × 4 saturation bins.
APPEARANCE_DIM = 48


def intersection(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Histogram intersection of two descriptors: the sum of element-wise minimums.

    Both sides are L1-normalised by the wire contract, so the result lives in
    0..1 — 1.0 for identical histograms, 0.0 for mass in entirely disjoint
    Hue×Saturation bins.  Intersection is chosen over cosine or chi-squared
    because it is the bluntest instrument that works here: partial occlusion
    of the torso only REMOVES mass, which can only lower the score (fail
    towards "no clash", the safe direction), and the fixed 0..1 range keeps
    the clash knob legible.

    Raises ValueError when the descriptors differ in length or are empty — a
    length mismatch is a wire bug, and quietly answering 0.0 for it would
    turn that bug into a maximal-clash veto.
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.ndim != 1 or vb.ndim != 1 or va.size != vb.size or va.size == 0:
        raise ValueError(
            "appearance descriptors must be equal-length non-empty flat vectors;"
            f" got shapes {va.shape} vs {vb.shape}"
        )
    return float(np.minimum(va, vb).sum())


def best_intersection(
    query: list[float] | np.ndarray | None, stored: list[np.ndarray]
) -> float | None:
    """Best intersection of ``query`` against an identity's stored descriptors.

    ``None`` — NOT 0.0 — when the query carries no descriptor or the identity
    has none stored (old galleries from before the column existed, sightings
    with no containing person box, torso crops under 24 px or with fewer than
    100 unmasked pixels).  Absent is not zero, the codebase-wide convention
    (``gatedUnmeasured``, ``zoneUnmeasured``): a zero here would read as
    "maximally clashing" and veto enrolments for guests whose torso simply
    could not be measured, and under-counting is this pipeline's dominant
    failure mode.

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
    return max(intersection(query, s) for s in stored)
