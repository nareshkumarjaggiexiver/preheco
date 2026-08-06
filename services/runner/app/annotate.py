"""Per-stage frame annotation: draw a stage's output onto the frame with cv2.

The debug console shows the operator *what* each stage produced, not just how
fast it ran: person boxes, track ids and ages, face boxes coloured by quality
band, and the match verdict (personKey + cosine, staff in grey) on each face.
These are pure functions — a BGR image plus the stage data in, a new BGR image
(or encoded JPEG bytes) out — so they draw identically in a unit test on a
synthetic frame and in the live loop.

Colours are BGR (OpenCV order).  Every draw works on a copy, so a caller can
render several stages from the same source frame without them bleeding into
each other.
"""

import cv2
import numpy as np

from .gate import reported_reason

# BGR palette.
_GREEN = (0, 200, 0)  # person boxes / canon-quality faces
_CYAN = (200, 200, 0)  # track boxes
_AMBER = (0, 191, 255)  # sub-canon faces (below the 80 px production canon)
_RED = (0, 0, 255)  # gated faces (below the 56 px embed floor)
_YELLOW = (0, 255, 255)  # guest match labels
_GREY = (150, 150, 150)  # staff (excluded from the guest count, still tracked)
# Exclusion zones and the faces they ate. Deliberately unlike every colour
# above — amber/red/green all mean QUALITY verdicts, and a zone exclusion is
# not a quality verdict: the face may be perfectly good and simply behind a
# frosted partition. The operator checking their polygon must never confuse
# "too small" with "in the wrong place".
_MAGENTA = (255, 0, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Stages that carry a visual overlay (CONTRACTS.md v1); ingest is the raw frame.
STAGES = ("ingest", "person-detect", "track", "face-detect", "match")

#: Widest a tap frame may leave this module.  The debug console renders these
#: as thumbnails-to-half-screen; 1280 px is already more than it displays.
#: Encoding them at the camera's native size was pure cost, and a measured one:
#: on the T440 a tap round annotates + JPEG-encodes FIVE 3840x2160 frames and
#: uploads them, ~1-1.5 s per round — the entire gap between the stage
#: timings (ingest 42 ms, persons 76, faces 58, embed 74, match 4) and the
#: observed 0.4 fps death-spiral state (see loop._maybe_tap).  Downscaling to
#: 1280 cuts the encoded pixels ~9x for a 4K source.  These tap frames are the
#: runner's ONLY frame-upload path — there is no separate full-resolution
#: keyframe path in this repo — so nothing loses evidence by this cap.
TAP_MAX_WIDTH = 1280


def downscale(img: np.ndarray, max_w: int = TAP_MAX_WIDTH) -> np.ndarray:
    """Shrink a frame to at most ``max_w`` wide (aspect preserved); never grow.

    Annotations are drawn at native resolution FIRST (so box coordinates need
    no scaling) and the finished overlay is resized with ``INTER_AREA`` — the
    correct interpolation for reduction, and the one that keeps 2 px box
    strokes legible instead of aliasing them away.  A frame already at or
    under the cap is returned untouched: upscaling a small source would only
    blur it and cost encode time.
    """
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(
        img, (max_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
    )


def to_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR image to raw JPEG bytes (for the multipart frame POST)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed (empty or invalid image?)")
    return buf.tobytes()


def _xywh(box: dict) -> tuple[int, int, int, int]:
    """Box dict -> integer (x, y, w, h) for drawing."""
    return (
        int(round(box.get("x", 0))),
        int(round(box.get("y", 0))),
        int(round(box.get("w", 0))),
        int(round(box.get("h", 0))),
    )


def _rect(img: np.ndarray, box: dict, colour, label: str | None = None) -> None:
    """Draw one rectangle (+ optional label above it) in place."""
    x, y, w, h = _xywh(box)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
    if label:
        ty = max(y - 6, 10)
        cv2.putText(img, label, (x, ty), _FONT, 0.5, colour, 1, cv2.LINE_AA)


def draw_persons(img: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Green person boxes with detector confidence."""
    out = img.copy()
    for b in boxes:
        conf = b.get("conf")
        _rect(out, b, _GREEN, f"{conf:.2f}" if conf is not None else None)
    return out


def draw_tracks(img: np.ndarray, tracks: list[dict]) -> np.ndarray:
    """Cyan track boxes labelled with the stable id and age in frames."""
    out = img.copy()
    for t in tracks:
        _rect(out, t.get("box", {}), _CYAN, f"id{t.get('id')} a{t.get('ageFrames')}")
    return out


def draw_zones(img: np.ndarray, zones: list[dict]) -> np.ndarray:
    """Exclusion-zone polygons: translucent magenta fill plus a labelled outline.

    The zones are the operator's own drawing (normalized 0..1 vertices, scaled
    here by the frame's actual size), and this overlay is how they check it:
    a face flagged as zone-eaten with no visible boundary around it cannot be
    distinguished from a bug.  Translucent rather than opaque because the
    operator specifically needs to see what is BEHIND the zone — the frosted
    partition, the mirror, the TV screen — to judge whether the polygon covers
    it.
    """
    if not zones:
        return img.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    polys = []
    for z in zones:
        pts = np.array(
            [[int(round(p[0] * w)), int(round(p[1] * h))] for p in z.get("points", [])],
            dtype=np.int32,
        )
        if len(pts) < 3:
            continue
        polys.append((pts, z.get("label")))
        cv2.fillPoly(overlay, [pts], _MAGENTA)
    out = cv2.addWeighted(overlay, 0.25, img, 0.75, 0.0)
    for pts, label in polys:
        cv2.polylines(out, [pts], isClosed=True, color=_MAGENTA, thickness=2)
        if label:
            x, y = int(pts[0][0]), int(pts[0][1])
            cv2.putText(
                out, str(label), (x, max(y - 6, 10)), _FONT, 0.5, _MAGENTA, 1,
                cv2.LINE_AA,
            )
    return out


def draw_faces(img: np.ndarray, faces: list[dict], min_px: float, canon_px: float) -> np.ndarray:
    """Face boxes coloured by gate verdict; rejected faces say WHY.

    Red is "this face was not counted", and the label names the floor that
    dropped it (``72px ied``) — a face can now be comfortably wide and still be
    discarded for pose or focus, so colouring on width alone would draw a green
    box around a guest who was never counted.  Green/amber keep their meaning
    for kept faces: at or below the production canon.

    A face the exclusion-zone filter ate is magenta, matching the zone overlay
    that ate it, and labelled ``zone`` — it was never gated at all, so neither
    red nor green would be telling the truth about it.

    The verdict is read from ``gateReason`` where the gate stamped one, falling
    back to the width comparison for callers that never ran the gate.
    """
    out = img.copy()
    for f in faces:
        box = f.get("box", {})
        w = float(box.get("w", 0.0))
        if f.get("excludedByZone"):
            _rect(out, box, _MAGENTA, f"{w:.0f}px zone")
            continue
        reason = reported_reason(f, min_px)
        colour = _RED if reason else (_GREEN if w >= canon_px else _AMBER)
        label = f"{w:.0f}px" if reason is None else f"{w:.0f}px {reason}"
        _rect(out, box, colour, label)
    return out


def draw_matches(img: np.ndarray, verdicts: list[dict]) -> np.ndarray:
    """Match verdicts on each face: personKey + cosine, staff drawn grey."""
    out = img.copy()
    for v in verdicts:
        box = v.get("box")
        if not box:
            continue
        is_staff = bool(v.get("isStaff", False))
        colour = _GREY if is_staff else _YELLOW
        cos = v.get("cosine")
        tag = "STAFF " if is_staff else ""
        label = f"{tag}{v.get('personKey', '?')}"
        if cos is not None:
            label += f" {cos:.2f}"
        _rect(out, box, colour, label)
    return out


def render(stage: str, img: np.ndarray, last: dict, min_px: float, canon_px: float) -> bytes:
    """Annotate ``img`` for ``stage`` from the frame snapshot; return JPEG bytes.

    ``ingest`` is the raw frame (re-encoded).  Unknown stages fall back to the
    raw frame so a new stage name never raises inside the best-effort loop.

    Every stage — the raw ingest frame included — leaves here at most
    :data:`TAP_MAX_WIDTH` px wide (see :func:`downscale`): the console shows
    these small, and 4K tap frames were the measured bulk of the tap round's
    ~1-1.5 s cost.
    """
    if stage == "person-detect":
        img = draw_persons(img, last.get("boxes", []))
    elif stage == "track":
        img = draw_tracks(img, last.get("tracks", []))
    elif stage == "face-detect":
        img = draw_zones(img, last.get("zones", []))
        img = draw_faces(img, last.get("faces", []), min_px, canon_px)
    elif stage == "match":
        img = draw_matches(img, last.get("verdicts", []))
    return to_jpeg(downscale(img))
