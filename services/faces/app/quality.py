"""POC face-quality signals: the width flag, and the focus proxy beside it.

Thresholds trace to the site-planner canon (CONTRACTS.md "POC geometry"):
production canon wants faces >= 80 px wide (100 px comfortable); the POC
baseline — UNV 2.8 mm fixed at a 2.0 m mount, subjects passing 2–3 m — yields
only ~64–85 px, accepted deliberately for the POC. Hence:

    width >= 80 px           -> "ok"        (meets the production canon)
    56 px <= width <= 79 px  -> "sub-canon" (POC-accepted, flagged in reports)
    width < 56 px            -> "reject"    (below the POC embedding floor)

Override via env FACES_CANON_PX / FACES_FLOOR_PX (defaults per contract).

:func:`crop_sharpness` is the second measured signal.  Width (and its honest
cousin, inter-eye distance) says how much of the face the sensor resolved;
sharpness says whether what it resolved is legible.  They fail independently:
a 90 px face smeared by a guest walking past embeds worse than a 60 px face
standing still, and the width flag alone cannot tell those apart.  Both are
pure functions so the boundaries are unit-testable without a detector.
"""

import os

import cv2
import numpy as np

CANON_PX = int(os.environ.get("FACES_CANON_PX", "80"))
FLOOR_PX = int(os.environ.get("FACES_FLOOR_PX", "56"))

#: Side length every face crop is resized to before the Laplacian.  See
#: :func:`crop_sharpness` for why normalising is not optional.
SHARPNESS_NORM_PX = int(os.environ.get("FACES_SHARPNESS_NORM_PX", "64"))


def classify_width(width_px: float, canon_px: int = CANON_PX, floor_px: int = FLOOR_PX) -> str:
    """Map a face width in source-image pixels to 'ok' | 'sub-canon' | 'reject'."""
    if width_px >= canon_px:
        return "ok"
    if width_px >= floor_px:
        return "sub-canon"
    return "reject"


def crop_sharpness(
    img: np.ndarray, box: dict, norm_px: int = SHARPNESS_NORM_PX
) -> float | None:
    """Variance of the Laplacian over one face crop — a RELATIVE focus proxy.

    Returns None when the box does not overlap the image or is too small to
    measure; the caller must treat that as "unknown", never as "bad" (see the
    runner's gate: a missing signal may not reject a face, because an
    under-count is this pipeline's dominant failure mode and the count is an
    invoice figure).

    Two properties of the measure decide how it may be used, and both are the
    reason this ships report-only with its gate threshold defaulting to off:

    * **It is normalised for size on purpose.**  Laplacian variance grows with
      pixel count and with the spatial frequency the crop happens to contain,
      so an unnormalised value is mostly a restatement of how close the guest
      stood — which ``iedPx`` already measures, honestly.  Resizing every crop
      to the same ``norm_px`` square (INTER_AREA, which low-pass filters as it
      shrinks rather than aliasing) makes the number about focus instead.
    * **It is not absolute.**  It still moves with exposure, contrast and the
      camera's own sharpening, so a threshold that is right for one gate at
      one time of day is not right for another.  It is a within-camera
      ordering, and the floor has to be calibrated against that camera's own
      footage before it is armed.

    Cheap by construction: one greyscale conversion and one 3x3 Laplacian over
    a 64x64 crop, which is why it can run on every detected face.
    """
    h_img, w_img = img.shape[:2]
    # Clamp the far edges from the RAW corner, not from the clamped one: a box
    # lying entirely off the frame must collapse to nothing, not slide back
    # inside and measure some innocent corner of the picture instead.
    x0 = int(round(float(box.get("x", 0))))
    y0 = int(round(float(box.get("y", 0))))
    x, y = max(0, x0), max(0, y0)
    x2 = min(w_img, x0 + int(round(float(box.get("w", 0)))))
    y2 = min(h_img, y0 + int(round(float(box.get("h", 0)))))
    if x2 - x < 2 or y2 - y < 2:
        return None
    crop = img[y:y2, x:x2]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    grey = cv2.resize(grey, (norm_px, norm_px), interpolation=cv2.INTER_AREA)
    return round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 2)
