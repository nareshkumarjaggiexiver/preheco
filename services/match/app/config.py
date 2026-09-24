"""Environment-driven configuration for the match service.

Every knob is an environment variable so docker-compose and the POC bench can
tune without code changes.
"""

import os
from pathlib import Path

# Cosine-similarity threshold for "same person".
#
# 0.363 is the SFace paper's published operating point for cosine similarity
# (Zhong et al., "SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face
# Recognition") and what OpenCV documents for FaceRecognizerSF.  It is a
# 1:1-verification number, so treat it as a STARTING point only: the POC runs
# open-set 1:N against a growing gallery, where the right threshold is a
# function of gallery size (docs/planning/04-identity-pipeline.md, stage 6).
# POC-tunable via HECO_MATCH_THRESHOLD.  Sub-canon faces (56-79 px, below the
# production 80-100 px canon) are matched like any other embedding but TAGGED,
# so every report can state what share of the gallery is sub-canon evidence.
DEFAULT_THRESHOLD = 0.363

# Faces narrower than this many pixels are tagged sub-canon (the POC geometry:
# 2.8 mm camera at 2.0 m, subjects at 2-3 m, expected face widths ~64-85 px —
# deliberately below the production canon; see CONTRACTS.md "POC geometry").
DEFAULT_CANON_PX = 80.0

# --------------------------------------------------------- multi-template (M1)
#
# A guest used to be represented forever by the FIRST view of them, whatever
# angle that happened to be, and every later sighting was compared against that
# one vector.  The measured consequence (corridor bench, one man walking):
# THREE gallery identities for one person, their views mutually 0.296-0.347
# against a 0.363 threshold — every pair a near miss.  A guest now accumulates
# up to DEFAULT_TEMPLATES_PER_PERSON views and is matched against all of them.
#
# NOTE none of these knobs is a threshold in disguise: a sighting still has to
# clear HECO_MATCH_THRESHOLD against an already-stored template to be counted as
# a re-sighting at all.  Lowering the threshold is M2 and is deliberately
# blocked on impostor data we do not have.
#
# TWO BARS, NOT ONE — the distinction below is easy to misread and expensive to
# misread.  HECO_MATCH_THRESHOLD (0.363) decides MATCHED.  Threshold +
# TEMPLATE_CONFIDENCE (0.413) decides LEARNED FROM.  Between them is a dead
# band: the sighting counts as the same guest but is never kept, so the identity
# does not grow towards it and the chain of views does not extend.  Measured on
# a steady pose sweep: adjacent cosine 0.420 -> 1 identity, 0.410 -> 3, 0.380 ->
# 3.  If a bench fragments AND its matches cluster in 0.363-0.413, this is why,
# and TEMPLATE_CONFIDENCE is the knob to reach for — not the threshold.

# How many templates one identity may hold.  Five is the number the staff
# enrolment flow already uses for a deliberate walk-through (CONTRACTS.md
# "keeps the best N=5"), and it is roughly the number of distinct views a
# corridor crossing yields: frontal, two three-quarters, two profiles.  Set to
# 1 to restore the old single-template behaviour exactly.
DEFAULT_TEMPLATES_PER_PERSON = 5

# A sighting must beat the threshold by this much to be promoted to a template.
# A bare-minimum match is the least certain evidence we have; letting it become
# a template would let an identity annex territory around a point we are not
# sure of, and errors would compound template by template.  0.05 above 0.363 =
# 0.413 required to enrol, while 0.363 still counts as a re-sighting.
DEFAULT_TEMPLATE_CONFIDENCE = 0.05

# ...and it must beat the nearest RIVAL identity by this much.  A sighting that
# sits almost equally close to two people is precisely the one that must not be
# stored: it would become a bridge, and the next probe near it would collapse
# two guests into one.  Over-counting is expensive; silently merging two paying
# guests is worse, because nobody can see it happen.
DEFAULT_TEMPLATE_MARGIN = 0.05

# ...and it must NOT be a near-duplicate of a view we already hold.  Above this
# cosine the sighting adds no coverage, only churn: it would spend a capped
# slot, evict a genuinely different view, and make the identity narrower.  Most
# frames of a walking guest land here (consecutive frames sit at ~0.99), so this
# is also what keeps the write path quiet.
DEFAULT_TEMPLATE_MAX_COSINE = 0.90


# ------------------------------------------- torso appearance tie-breaker (v1)
#
# Histogram-intersection floor BELOW which a sighting's torso descriptor is a
# CLASH with the matched identity's stored descriptors, refusing the enrolment
# of that sighting as a template (the anti-poison veto — app.gallery.match).
# Appearance never touches the match verdict: the closest measured impostor
# pair on this camera (cosine 0.377, above the 0.363 threshold) was two
# DIFFERENT men BOTH IN LIGHT SHIRTS, so clothing must never rescue a face
# match — it may only refuse a write.
#
# 0.50 is REASONED, NOT CALIBRATED — exactly like the M1 template margins.
# The reasoning: two L1-normalised histograms of the SAME torso across
# consecutive frames share most of their mass (intersection well above 0.5),
# while a genuinely different outfit concentrates mass in other Hue×Saturation
# bins (intersection well below).  The midpoint is the least-wrong uncalibrated
# split; the first labelled torso pairs from a real event should move it.
# 0 disables the veto entirely (the off switch, same convention as
# TEMPLATES_PER_PERSON=1 for M1).
DEFAULT_APPEARANCE_CLASH = 0.50


# ------------------------------------------------ near-miss flag on a mint (v2)
#
# Floor of the band [floor .. threshold) in which a MINT's best cosine against
# the existing gallery earns a `nearMiss` flag in the /match response — a
# suggestion to the operator that the fresh key may be a split of that
# existing guest (see app.gallery.match).  The verdict is still a mint.
#
# 0.29 sits just below the measured same-person misses on this camera (0.294 /
# 0.308 / 0.346 / 0.361 against the 0.363 threshold): every split we have
# actually watched happen would have been flagged, while mints further out are
# genuinely new faces and flagging them would train the operator to ignore the
# cue.  0 disables the flag entirely (the codebase's off-switch convention).
DEFAULT_NEARMISS_FLOOR = 0.29


# ------------------------------------- the WEAK (clothing) near-miss band (v2)
#
# The 0.29 floor above turned out to be BLIND to the splits that actually cost
# us a count.  Bench run 6e1a5d (2026-08-06), ground truth ONE person walking
# out of frame, back in, then sitting: that one person produced SIX tracker
# ids, three of which the runner's heal folded back.  The two that SURVIVED as
# extra guests read
#
#     p00005: face 0.212 vs p00001, clothing 0.797
#     p00006: face 0.228 vs p00001, clothing 0.875
#
# Both sit BELOW 0.29, so no banner fired and the operator was never asked —
# while the torso descriptor was shouting 0.80-0.88 at both of them.  Seated
# and turned-away re-entries at this camera's angles land there: the face
# signal has almost nothing to work with, the clothing signal has plenty.
#
# So a SECOND, weaker band exists: cosine in [WEAK_FLOOR .. FLOOR) earns a
# near-miss ONLY when the clothing agrees at or above NEARMISS_CLOTHES.
#
# 0.15 is the weak floor: below it the face evidence is indistinguishable from
# two strangers and the flag would rest on clothing alone.  0 disables the weak
# band and leaves the face band exactly as it was.
DEFAULT_NEARMISS_WEAK_FLOOR = 0.15

# ...and the clothing bar that band requires.  BE HONEST ABOUT THIS NUMBER:
# the closest measured IMPOSTOR pair on this camera — two genuinely different
# men — reached clothing intersection 0.747, and another impostor pair sat at
# 0.503 with face 0.377.  0.78 clears the worst measured impostor by 0.033.
# That is a hair, not a margin, and it is the entire reason the weak band
# produces a SUGGESTION a human confirms and never a merge: at a venue with
# uniformed staff, a dress code, or similar traditional dress, this band is
# EXPECTED to point at the wrong person, and it is the first knob to turn off
# (set HECO_MATCH_NEARMISS_WEAK_FLOOR=0).  Clothing agreement alone never
# proves identity; it only earns the operator a look.
DEFAULT_NEARMISS_CLOTHES = 0.78


def _env_f(name: str, default: float) -> float:
    """Read a float knob where an EMPTY string means unset.

    docker-compose renders ``${VAR-}`` as an empty string, and a bare
    ``float("")`` once took the ingest service down at startup — so every
    knob added since parses absent and empty the same way (the local pattern
    ``main._env_s`` established; this service stays free of heco_common on
    purpose).
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def threshold() -> float:
    """Return the active cosine threshold (env HECO_MATCH_THRESHOLD)."""
    return float(os.environ.get("HECO_MATCH_THRESHOLD", DEFAULT_THRESHOLD))


def canon_px() -> float:
    """Return the face-width floor (px) below which a match is tagged sub-canon."""
    return float(os.environ.get("HECO_MATCH_CANON_PX", DEFAULT_CANON_PX))


def templates_per_person() -> int:
    """Max templates one identity may hold (env HECO_MATCH_TEMPLATES_PER_PERSON).

    Values below 1 are clamped to 1: zero templates would mean a person with no
    stored face, which the gallery cannot represent (it would be re-counted as
    new on every single frame).
    """
    raw = os.environ.get("HECO_MATCH_TEMPLATES_PER_PERSON", DEFAULT_TEMPLATES_PER_PERSON)
    return max(1, int(raw))


def template_confidence() -> float:
    """Margin above the match threshold required to enrol (env …_TEMPLATE_CONFIDENCE)."""
    return float(os.environ.get("HECO_MATCH_TEMPLATE_CONFIDENCE", DEFAULT_TEMPLATE_CONFIDENCE))


def template_margin() -> float:
    """Margin over the nearest rival identity required to enrol (env …_TEMPLATE_MARGIN)."""
    return float(os.environ.get("HECO_MATCH_TEMPLATE_MARGIN", DEFAULT_TEMPLATE_MARGIN))


def template_max_cosine() -> float:
    """Near-duplicate ceiling: above this a sighting is not worth a slot.

    Env HECO_MATCH_TEMPLATE_MAX_COSINE.
    """
    return float(os.environ.get("HECO_MATCH_TEMPLATE_MAX_COSINE", DEFAULT_TEMPLATE_MAX_COSINE))


def nearmiss_floor() -> float:
    """Cosine floor of the near-miss band on a mint (env HECO_MATCH_NEARMISS_FLOOR).

    A mint whose best cosine vs the existing gallery lands in
    ``[floor .. threshold)`` carries a ``nearMiss`` flag with basis ``"face"``
    — operator information, never behaviour (see :func:`app.gallery.match`).
    **0 disables the flag**, both bands: the weak clothing band below is a
    band *under this floor*, so with no floor there is nothing to sit under.
    Empty string means unset (the compose ``${VAR-}`` rendering), like every
    knob in this service.
    """
    return _env_f("HECO_MATCH_NEARMISS_FLOOR", DEFAULT_NEARMISS_FLOOR)


def nearmiss_weak_floor() -> float:
    """Floor of the WEAK near-miss band (env HECO_MATCH_NEARMISS_WEAK_FLOOR).

    A mint whose best cosine lands in ``[weak_floor .. nearmiss_floor)`` earns
    a ``nearMiss`` with basis ``"clothing"`` — but only when the torso
    descriptors agree at or above :func:`nearmiss_clothes`.  It exists because
    bench 6e1a5d measured one person splitting at face 0.212 / clothing 0.797
    and face 0.228 / clothing 0.875: both below the 0.29 face floor, both
    unbannered, both real over-counts.

    **0 disables the weak band** and leaves the face band untouched — the
    first knob to turn off at a venue with uniforms or a dress code, where
    clothing agreement stops carrying information.  Empty string means unset.
    """
    return _env_f("HECO_MATCH_NEARMISS_WEAK_FLOOR", DEFAULT_NEARMISS_WEAK_FLOOR)


def nearmiss_clothes() -> float:
    """Torso-intersection bar the weak band requires (env …_NEARMISS_CLOTHES).

    0.78 sits **0.033 above the worst measured impostor clothing reading
    (0.747, two genuinely different men)**.  That thin margin is the whole
    reason the weak band is a suggestion for a human and never a merge — see
    DEFAULT_NEARMISS_CLOTHES above and :func:`app.gallery._near_miss`.  Raise
    it to make the band quieter; the honest off switch is
    ``HECO_MATCH_NEARMISS_WEAK_FLOOR=0``.  Empty string means unset.
    """
    return _env_f("HECO_MATCH_NEARMISS_CLOTHES", DEFAULT_NEARMISS_CLOTHES)


def review_floor() -> float:
    """Floor of the duplicate-REVIEW band (env HECO_REVIEW_FLOOR).

    Its own knob, no longer borrowed from :func:`nearmiss_weak_floor`.
    Borrowing meant the documented weak-band off switch
    (``HECO_MATCH_NEARMISS_WEAK_FLOOR=0``) silently turned the review floor to
    0 as well and flooded the queue with every pair in the gallery — the two
    knobs answer different questions and must not share a value. Run 27ca33
    measured what the band carries at 0.15: 34 of the top-50 review pairs sat
    below face 0.30, which is noise wearing a rank.
    """
    return _env_f("HECO_REVIEW_FLOOR", 0.15)


# --------------------------------- the review queue's exclusion signals (v4)
#
# Run f0bfc5 (2026-09-23, a Punjab wedding hall seen from an overview camera,
# 74 guests counted) flooded the review queue with 500 pairs.  The top of that
# queue, read by eye:
#
#     #1 a red-turbaned man with a black beard vs an orange-turbaned man with a
#        WHITE beard                                       face 0.357
#     #3 a man in a blue shirt vs an ELDERLY WOMAN in glasses      face 0.345
#     #5 a girl in a yellow top vs a woman in a dark green dress   face 0.338
#
# Every one of those is a question no operator should have been asked: the
# pipeline had, or could have had, evidence that settles it — sex, age, and
# how tall the body is.  None of that evidence may MERGE anyone (the standing
# rule: nothing is minted, merged or counted on anything but the face and a
# human click).  But a pair the evidence says CANNOT be one person may be set
# aside from the queue, and the count of pairs set aside is reported so a
# flood that was silenced is never mistaken for a flood that never happened.
#
# Each signal is one knob, and each knob's zero is its off switch.

# Gender: an identity's templates each carry the attribute model's reported
# sex and its probability; the identity's belief is the template-mean of
# P(male).  A pair is set aside only when BOTH identities are confident at or
# above this bar AND they disagree.  0.8 is reasoned, not calibrated: the
# buffalo_l genderage head is strong on frontal adult faces and unsure on
# children, head-down and turned-away views, and unsure is exactly the state
# in which it must not vote.  One side confident and the other not is NOT a
# disagreement; it is one opinion.  0 turns the signal off.
DEFAULT_REVIEW_GENDER_MIN_P = 0.8

# Age: a CHILD (median template age at or below CHILD_MAX) against an ADULT
# (at or above ADULT_MIN) is set aside.  The 12/20 gap is the point: the
# attribute model's age error on a 40 px face is several years each way, so
# the two bands are kept far enough apart that a 15-year-old read as 11 and
# again as 19 lands in the dead zone between them and is still asked about.
# Either knob at 0 turns the signal off.
DEFAULT_REVIEW_AGE_CHILD_MAX = 12.0
DEFAULT_REVIEW_AGE_ADULT_MIN = 20.0

# Stature: each standing person box in the run is fitted, box height against
# box bottom-y (the camera's perspective: further from the lens = higher in
# frame = smaller), and an identity's stature is the median of its own boxes'
# heights over what that fit predicts — 1.0 is "an average standing adult at
# that spot".  Run f0bfc5's fit: h = 0.602 * y_bottom + 300 px over 1268
# standing sightings, with p00009 (a child) at 0.73 against adults at 1.03-
# 1.04.  Two identities whose ratios differ by this much or more are set
# aside.  0.2 is 35 cm on a 1.75 m anchor — wider than any adult-to-adult
# spread the fit produced (0.98-1.08), narrower than adult-to-child.  0 turns
# the signal off.
DEFAULT_REVIEW_STATURE_GAP = 0.2

# ...and how many standing sightings an identity needs before its ratio is
# trusted at all.  A single box mid-stride, half behind a pillar or caught by
# the tracker's bounding-box lag can be 30% off; the median over eight is
# not.  Under this an identity's stature is null — not measured, never 0.
DEFAULT_REVIEW_STATURE_MIN_N = 8

# Clothing: the torso descriptor RANKS the queue, and since 2026-09-24
# (night) it may also set a pair aside — on both identities' own testimony.
# Each identity needs at least CLOTHES_MIN_N v3 torso reads in its body log,
# spanning two seconds, whose median pairwise intersection is at least
# CLOTHES_SELF_MIN (its clothing reads as ONE garment); then the pair is set
# aside when the best intersection any read of one reaches against any read
# of the other is under CLOTHES_CLASH.  Run f0bfc5, 45 identities of the
# queue's first 32 pairs: own reads agree at a median 0.90 (p10 0.70; the
# two identities under 0.6 were each two people merged), one person split
# across a gap of seconds to minutes still agreed at a best cross of 0.77 to
# 0.97 (nine splits).  The queue's different-people pairs spread 0.10-0.97:
# two white shirts agree, so clothing can only speak for pairs dressed
# differently — 11 of the 32 sat under 0.43, and of those the four whose
# identities both had the reads are set aside at 0.35: #3 (blue shirt vs
# cream suit, 0.10), #6 (0.16), #7 (0.13), #23 (0.26); no same-person split
# came within 0.42 of it.  CLOTHES_CLASH 0 turns the signal off.
DEFAULT_REVIEW_CLOTHES_CLASH = 0.35
DEFAULT_REVIEW_CLOTHES_MIN_N = 3
DEFAULT_REVIEW_CLOTHES_SELF_MIN = 0.6

# Head: turban and hair colour above the eyes, per sighting in the body log,
# under the clothing rule's own-testimony bar (three reads over two seconds
# agreeing at 0.6 each side) and only HEADWEAR against HEADWEAR (both heads
# at least half chromatic — gallery._HEAD_WEAR_MIN says why a covered head
# is never set against a bare one); set aside when the best cross reading is
# under HEAD_CLASH.  Run f0bfc5, 45 identities of the queue's first 32 pairs:
# own head reads agree at a median 0.87 (min 0.63), one person split across
# a time gap at 0.83-0.99 (9 splits), pair #1 — maroon turban against peach
# — at 0.36; blue turban against black hair (#18) 0.37 and pink turban
# against black hair (#15) 0.41 are headwear against hair and stay asked.
# 0.45 sits 0.09 over #1 and 0.38 under the closest same-person split.
# HEAD_CLASH 0 turns the signal off.
DEFAULT_REVIEW_HEAD_CLASH = 0.45

# Beard: none / dark / grey / white per identity, from its body-log reads
# (the class two thirds of its reads name, the unsure reads counting
# against it).  A pair is set aside when both identities have at least
# BEARD_MIN_N reads over two seconds and their classes cannot be one face:
# none against any beard, dark against white — grey is never set against
# either.  Run f0bfc5: pair #1 reads dark against white and is set aside;
# of nine same-person splits none was; the seven full beards' median reads
# are 0.47-0.75 dark against 0.03-0.48 for 21 shaven or moustached men.
# BEARD_MIN_N 0 turns the signal off.
DEFAULT_REVIEW_BEARD_MIN_N = 3

# The height a stature ratio of 1.0 means, in metres.  The user's instruction
# for this deployment: the North Indian adult average is 5'9" = 1.75 m, and
# it is the anchor for every stature estimate and the planner's default
# person height.  It converts ratios to metres for a human to read; the
# exclusion test above is on the RATIO, so this number never moves it.
DEFAULT_ADULT_HEIGHT_M = 1.75


def appearance_clash() -> float:
    """Torso-intersection floor for the enrolment veto (below = clash).

    Env HECO_MATCH_APPEARANCE_CLASH; **0 disables the veto**.  The default
    0.50 is reasoned, not calibrated — see DEFAULT_APPEARANCE_CLASH above —
    and is deliberately env-tunable because the first labelled impostor/torso
    pairs from a venue should be what moves it.  Empty string means unset
    (the compose ``${VAR-}`` rendering), like every knob in this service.
    """
    return _env_f("HECO_MATCH_APPEARANCE_CLASH", DEFAULT_APPEARANCE_CLASH)


def review_gender_min_p() -> float:
    """Gender confidence BOTH sides need before a disagreement sets a pair
    aside (env HECO_REVIEW_GENDER_MIN_P; 0 = off).  Empty means unset."""
    return _env_f("HECO_REVIEW_GENDER_MIN_P", DEFAULT_REVIEW_GENDER_MIN_P)


def review_age_child_max() -> float:
    """Median age at or below which an identity reads as a child
    (env HECO_REVIEW_AGE_CHILD_MAX; 0 = the age signal is off)."""
    return _env_f("HECO_REVIEW_AGE_CHILD_MAX", DEFAULT_REVIEW_AGE_CHILD_MAX)


def review_age_adult_min() -> float:
    """Median age at or above which an identity reads as an adult
    (env HECO_REVIEW_AGE_ADULT_MIN; 0 = the age signal is off)."""
    return _env_f("HECO_REVIEW_AGE_ADULT_MIN", DEFAULT_REVIEW_AGE_ADULT_MIN)


def review_stature_gap() -> float:
    """Stature-ratio gap at or beyond which a pair is set aside
    (env HECO_REVIEW_STATURE_GAP; 0 = off).  A ratio, not metres."""
    return _env_f("HECO_REVIEW_STATURE_GAP", DEFAULT_REVIEW_STATURE_GAP)


def review_stature_min_n() -> int:
    """Standing sightings an identity needs before its stature is trusted
    (env HECO_REVIEW_STATURE_MIN_N).  Clamped to at least 1."""
    return max(1, int(_env_f("HECO_REVIEW_STATURE_MIN_N", DEFAULT_REVIEW_STATURE_MIN_N)))


def review_clothes_clash() -> float:
    """Best cross-torso intersection under which a pair whose identities each
    wear ONE garment is set aside (env HECO_REVIEW_CLOTHES_CLASH; 0 = off)."""
    return _env_f("HECO_REVIEW_CLOTHES_CLASH", DEFAULT_REVIEW_CLOTHES_CLASH)


def review_clothes_min_n() -> int:
    """v3 torso reads each identity needs before its clothing counts
    (env HECO_REVIEW_CLOTHES_MIN_N).  Clamped to at least 2: one read has no
    self-agreement to trust."""
    return max(2, int(_env_f("HECO_REVIEW_CLOTHES_MIN_N", DEFAULT_REVIEW_CLOTHES_MIN_N)))


def review_clothes_self_min() -> float:
    """Median pairwise intersection an identity's own torso reads must reach
    (env HECO_REVIEW_CLOTHES_SELF_MIN)."""
    return _env_f("HECO_REVIEW_CLOTHES_SELF_MIN", DEFAULT_REVIEW_CLOTHES_SELF_MIN)


def review_head_clash() -> float:
    """Best cross head intersection under which a pair whose identities each
    read ONE head is set aside (env HECO_REVIEW_HEAD_CLASH; 0 = off)."""
    return _env_f("HECO_REVIEW_HEAD_CLASH", DEFAULT_REVIEW_HEAD_CLASH)


def review_beard_min_n() -> int:
    """Beard reads each identity needs before its class may set a pair aside
    (env HECO_REVIEW_BEARD_MIN_N; 0 = off)."""
    return max(0, int(_env_f("HECO_REVIEW_BEARD_MIN_N", DEFAULT_REVIEW_BEARD_MIN_N)))


def adult_height_m() -> float:
    """Metres a stature ratio of 1.0 stands for (env HECO_STATURE_ADULT_M).

    Default 1.75 — the North Indian adult average this deployment is anchored
    on.  Display only: the stature exclusion compares ratios.  The name is the
    one docker-compose.yml passes through and demo-up.sh reads back; the first
    cut read HECO_REVIEW_ADULT_M here, so the operator-facing knob printed as
    set and changed nothing.
    """
    return _env_f("HECO_STATURE_ADULT_M", DEFAULT_ADULT_HEIGHT_M)


def data_dir() -> Path:
    """Return the directory holding per-run gallery databases (created lazily)."""
    d = Path(os.environ.get("HECO_MATCH_DATA_DIR", "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d
