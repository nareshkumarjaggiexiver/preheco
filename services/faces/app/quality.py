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


#: Landmark topology bounds.  A real face has its eyes above the nose above
#: the mouth, eyes spanning a sane fraction of the box, and a nose that stays
#: near the eye span.  The numbers come from the sibling pipeline's field
#: measurements against this camera family (see
#: docs/planning/13-pipeline-review-face-detection.md) and are deliberately
#: LOOSE: this test exists to reject detections whose landmarks are
#: effectively random, not to judge pose.  Pose is `frontality` and
#: `eyeSpanRatio`, which are separate signals with separate floors.
EYE_SPAN_MIN_FRAC = float(os.environ.get("FACES_EYE_SPAN_MIN_FRAC", "0.15"))
EYE_SPAN_MAX_FRAC = float(os.environ.get("FACES_EYE_SPAN_MAX_FRAC", "0.85"))
NOSE_MARGIN_FRAC = float(os.environ.get("FACES_NOSE_MARGIN_FRAC", "0.30"))


def landmarks_plausible(landmarks, box: dict) -> bool | None:
    """Do these 5 landmarks describe a face at all?  None when unmeasurable.

    YuNet reports (right eye, left eye, nose, right mouth corner, left mouth
    corner).  A detection on clothing or a torso satisfies the detector's own
    confidence — the sibling pipeline observed striped shirts verifying at
    70–91 % — but its landmarks land in effectively random positions, because
    there is no face under them for the regressor to lock onto.  Three cheap
    topological facts separate the two, and none of them is a pose judgement:

    1. **Eyes above nose above mouth.**  True of every human face at every
       yaw and every plausible roll for a mounted camera looking down a queue.
    2. **The eye span is a sane fraction of the box width.**  Under 15 % means
       the two "eyes" collapsed onto one point; over 85 % means they landed on
       opposite edges of something that is not a head.
    3. **The nose sits near the eye span.**  The margin is a fraction of the
       BOX WIDTH rather than of the eye distance, deliberately: on a
       three-quarter view the eyes compress together while the nose
       legitimately projects past the far one, and an eye-distance margin
       would reject exactly the yawed-but-usable faces that
       :func:`~app.main._with_quality`'s ``frontality`` is there to score.

    Returns None — meaning UNKNOWN, never BAD — when the landmarks or the box
    cannot support the test.  The runner's gate is required to read absence
    that way, because rejecting a face nobody managed to measure would drop a
    guest from an invoice.
    """
    if not landmarks or len(landmarks) < 5:
        return None
    try:
        (rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy) = (
            (float(p[0]), float(p[1])) for p in landmarks[:5]
        )
        w = float(box.get("w", 0.0))
    except (TypeError, ValueError, IndexError):
        return None
    if w <= 0:
        return None

    eye_y = (rey + ley) / 2.0
    mouth_y = (rmy + lmy) / 2.0
    if not (eye_y < ny < mouth_y):
        return False

    eye_span = abs(lex - rex)
    if not (EYE_SPAN_MIN_FRAC * w <= eye_span <= EYE_SPAN_MAX_FRAC * w):
        return False

    lo, hi = min(rex, lex), max(rex, lex)
    margin = NOSE_MARGIN_FRAC * w
    return lo - margin <= nx <= hi + margin


def eye_span_ratio(landmarks, box: dict) -> float | None:
    """Horizontal eye separation as a fraction of box width; None if unknown.

    The pose signal ``iedPx`` cannot give: inter-eye distance in PIXELS says
    how much of the face the sensor resolved, and grows as a guest walks
    toward the lens.  The same distance as a fraction of the face's own box
    says how much of the face is TURNED AWAY — it collapses toward zero in
    profile at any resolution.  Size and pose fail independently and the two
    numbers must not be conflated; this is the second one.

    Horizontal separation only (``abs(lex - rex)``, not the euclidean
    distance) because that is the component yaw actually compresses; roll
    would shrink the euclidean distance too and roll is not what this gates.
    """
    if not landmarks or len(landmarks) < 2:
        return None
    try:
        rex = float(landmarks[0][0])
        lex = float(landmarks[1][0])
        w = float(box.get("w", 0.0))
    except (TypeError, ValueError, IndexError):
        return None
    if w <= 0:
        return None
    return round(abs(lex - rex) / w, 3)


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
