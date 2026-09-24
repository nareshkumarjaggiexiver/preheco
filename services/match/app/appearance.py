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


# ------------------------------------------------ the review queue's evidence
#
# Everything below reads an identity's appearance ACROSS its sightings for
# POST /review/duplicates — never for a verdict.  An identity's reads are
# the torso / head / beard readings its body-log rows carry
# (:meth:`app.store.VectorStore.sighting_evidence`), capped and spread over
# its time on camera (:func:`spread`).

#: The v3 torso length; v2 rows (48) never count in the review's clothing
#: evidence — a v2 partition is not the colour-below-the-neck v3 reads.
TORSO_DIM = APPEARANCE_DIM
#: The head descriptor: 24 soft hue bins, 3 soft brightness bins, 13 reserved.
HEAD_DIM = 40
HEAD_H_BINS = 24
#: The beard reading: [skinFrac, darkFrac, greyFrac, whiteFrac].
BEARD_DIM = 4

#: At most this many reads per identity feed a comparison, spread evenly
#: over its time order: an identity seen for ten minutes holds hundreds of
#: body-log rows, consecutive frames of one walk are one reading many times
#: over, and a pair's best cross is a (reads x reads) table.  24 keeps the
#: largest review (500 pairs) in tens of milliseconds.
READS_PER_KEY = 24


def spread(reads: list, cap: int = READS_PER_KEY) -> list:
    """``reads`` (already in time order) thinned to at most ``cap``, evenly."""
    if len(reads) <= cap:
        return list(reads)
    idx = np.linspace(0, len(reads) - 1, cap).round().astype(int)
    return [reads[i] for i in sorted(set(idx.tolist()))]


def self_agreement(vectors: list[np.ndarray]) -> float | None:
    """Median pairwise histogram intersection among one identity's reads.

    None under two reads — one reading agrees with nothing, it is not
    "agreement 1.0".  Median, not mean or min: one read through an
    occluder must not make a steady garment look unsteady, and one lucky
    pair must not make an unsteady one look steady.
    """
    if len(vectors) < 2:
        return None
    m = np.stack(vectors).astype(np.float64)
    inter = np.minimum(m[:, None, :], m[None, :, :]).sum(-1)
    iu = np.triu_indices(len(vectors), k=1)
    return float(np.median(inter[iu]))


def best_cross(a: list[np.ndarray], b: list[np.ndarray]) -> float | None:
    """The best intersection of any read of one identity with any of the other.

    BEST, as the torso veto is charitable: a pair is called different only
    when NO reading of one agrees with any reading of the other.  None when
    either side has none.
    """
    if not a or not b:
        return None
    ma = np.stack(a).astype(np.float64)
    mb = np.stack(b).astype(np.float64)
    return float(np.minimum(ma[:, None, :], mb[None, :, :]).sum(-1).max())


#: Colour families over the head descriptor's bins: hue bin i covers
#: OpenCV H [7.5 i, 7.5 (i+1)) — 15 degrees each — and bins 24..26 are
#: black / grey / white.  Display only: the head rule compares histograms,
#: never labels.  Bin 0 (0-15 degrees) is shared half-and-half by red and
#: orange, which puts the boundary near 8 degrees: run f0bfc5's maroon
#: turban (H 160-175) reads red and its peach one (H 0-10) orange, where a
#: whole bin 0 in red called pair #1 "red against red".  There is no
#: "brown": a brown is a dark orange and the descriptor bins a chromatic
#: pixel by hue alone, so a brown turban reads orange and brown hair —
#: dark, barely chromatic — black.  A bald scalp reads as its skin: orange.
HEAD_FAMILIES = (
    ("red", ((22, 1.0), (23, 1.0), (0, 0.5))),
    ("orange", ((0, 0.5), (1, 1.0), (2, 1.0))),
    ("yellow", ((3, 1.0), (4, 1.0))),
    ("green", tuple((b, 1.0) for b in range(5, 11))),
    ("blue", tuple((b, 1.0) for b in range(11, 17))),
    ("purple", ((17, 1.0), (18, 1.0))),
    ("pink", ((19, 1.0), (20, 1.0), (21, 1.0))),
    ("black", ((24, 1.0),)),
    ("grey", ((25, 1.0),)),
    ("white", ((26, 1.0),)),
)


def head_label(vectors: list[np.ndarray]) -> str | None:
    """The dominant colour family of an identity's mean head reading, or None."""
    if not vectors:
        return None
    m = np.mean(np.stack(vectors).astype(np.float64), axis=0)
    mass = {
        name: float(sum(m[b] * w for b, w in bins)) for name, bins in HEAD_FAMILIES
    }
    best = max(mass, key=mass.get)
    return best if mass[best] > 0.0 else None


#: One beard reading's class, from its fractions (run f0bfc5, 45 identities
#: of the queue's first 32 pairs, 1,078 reads): DARK when at least half the
#: chin reads darker than 0.4 of the cheek (the seven full beards' median
#: reads 0.47-0.75 dark, 21 shaven or moustached men 0.03-0.48); WHITE (or
#: GREY) when at least half reads pale — desaturated against the cheek —
#: with under 0.25 dark (the white-bearded elder of pair #1: 0.59 pale, 0.00
#: dark); NONE when at most 0.2 is dark and at least 0.7 is skin.  Anything
#: else is unsure and names no class — a moustache, stubble, a hand over the
#: mouth.
BEARD_DARK_MIN = 0.5
BEARD_PALE_MIN = 0.5
BEARD_PALE_DARK_MAX = 0.25
BEARD_NONE_DARK_MAX = 0.2
BEARD_NONE_SKIN_MIN = 0.7
#: An identity's class is the one at least this share of its reads name —
#: counting the unsure reads against it.
BEARD_AGREE = 2.0 / 3.0


def beard_read_class(b: np.ndarray) -> str | None:
    """``none`` / ``dark`` / ``grey`` / ``white`` for one reading, or None (unsure)."""
    skin, dark, grey, white = (float(x) for x in b[:BEARD_DIM])
    if dark >= BEARD_DARK_MIN:
        return "dark"
    if grey + white >= BEARD_PALE_MIN and dark < BEARD_PALE_DARK_MAX:
        return "white" if white >= grey else "grey"
    if dark <= BEARD_NONE_DARK_MAX and skin >= BEARD_NONE_SKIN_MIN:
        return "none"
    return None


def beard_class(vectors: list[np.ndarray]) -> str | None:
    """An identity's beard: the class BEARD_AGREE of its reads name, or None."""
    if not vectors:
        return None
    named = [beard_read_class(v) for v in vectors]
    counts: dict[str, int] = {}
    for c in named:
        if c is not None:
            counts[c] = counts.get(c, 0) + 1
    if not counts:
        return None
    best = max(counts, key=counts.get)
    return best if counts[best] >= BEARD_AGREE * len(vectors) else None


def beards_differ(a: str | None, b: str | None) -> bool:
    """Two confident beard classes one person cannot show in one event.

    None against any beard, or dark against white.  Grey sits between both
    and is never set against either: salt-and-pepper under warm light reads
    grey one walk and dark the next.
    """
    if a is None or b is None or a == b:
        return False
    return "none" in (a, b) or {a, b} == {"dark", "white"}
