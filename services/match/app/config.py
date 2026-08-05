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


def data_dir() -> Path:
    """Return the directory holding per-run gallery databases (created lazily)."""
    d = Path(os.environ.get("HECO_MATCH_DATA_DIR", "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d
