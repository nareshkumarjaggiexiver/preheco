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

# BGR palette.
_GREEN = (0, 200, 0)  # person boxes / canon-quality faces
_CYAN = (200, 200, 0)  # track boxes
_AMBER = (0, 191, 255)  # sub-canon faces (below the 80 px production canon)
_RED = (0, 0, 255)  # gated faces (below the 56 px embed floor)
_YELLOW = (0, 255, 255)  # guest match labels
_GREY = (150, 150, 150)  # staff (excluded from the guest count, still tracked)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Stages that carry a visual overlay (CONTRACTS.md v1); ingest is the raw frame.
STAGES = ("ingest", "person-detect", "track", "face-detect", "match")


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


def draw_faces(img: np.ndarray, faces: list[dict], min_px: float, canon_px: float) -> np.ndarray:
    """Face boxes coloured by quality band: canon green, sub-canon amber, gated red."""
    out = img.copy()
    for f in faces:
        box = f.get("box", {})
        w = float(box.get("w", 0.0))
        colour = _GREEN if w >= canon_px else _AMBER if w >= min_px else _RED
        _rect(out, box, colour, f"{w:.0f}px")
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
    """
    if stage == "person-detect":
        img = draw_persons(img, last.get("boxes", []))
    elif stage == "track":
        img = draw_tracks(img, last.get("tracks", []))
    elif stage == "face-detect":
        img = draw_faces(img, last.get("faces", []), min_px, canon_px)
    elif stage == "match":
        img = draw_matches(img, last.get("verdicts", []))
    return to_jpeg(img)
