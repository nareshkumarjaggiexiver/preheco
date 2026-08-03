"""POC face-quality flag from face width in pixels (pure, unit-tested).

Thresholds trace to the site-planner canon (CONTRACTS.md "POC geometry"):
production canon wants faces >= 80 px wide (100 px comfortable); the POC
baseline — UNV 2.8 mm fixed at a 2.0 m mount, subjects passing 2–3 m — yields
only ~64–85 px, accepted deliberately for the POC. Hence:

    width >= 80 px           -> "ok"        (meets the production canon)
    56 px <= width <= 79 px  -> "sub-canon" (POC-accepted, flagged in reports)
    width < 56 px            -> "reject"    (below the POC embedding floor)

Override via env FACES_CANON_PX / FACES_FLOOR_PX (defaults per contract).
"""

import os

CANON_PX = int(os.environ.get("FACES_CANON_PX", "80"))
FLOOR_PX = int(os.environ.get("FACES_FLOOR_PX", "56"))


def classify_width(width_px: float, canon_px: int = CANON_PX, floor_px: int = FLOOR_PX) -> str:
    """Map a face width in source-image pixels to 'ok' | 'sub-canon' | 'reject'."""
    if width_px >= canon_px:
        return "ok"
    if width_px >= floor_px:
        return "sub-canon"
    return "reject"
