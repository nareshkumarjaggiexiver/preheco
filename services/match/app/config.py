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


def threshold() -> float:
    """Return the active cosine threshold (env HECO_MATCH_THRESHOLD)."""
    return float(os.environ.get("HECO_MATCH_THRESHOLD", DEFAULT_THRESHOLD))


def canon_px() -> float:
    """Return the face-width floor (px) below which a match is tagged sub-canon."""
    return float(os.environ.get("HECO_MATCH_CANON_PX", DEFAULT_CANON_PX))


def data_dir() -> Path:
    """Return the directory holding per-run gallery databases (created lazily)."""
    d = Path(os.environ.get("HECO_MATCH_DATA_DIR", "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d
