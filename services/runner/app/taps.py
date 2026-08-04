"""Structured debug-tap payloads: what each stage produced, for the console.

CONTRACTS.md v1 asks the runner to POST, every ~2 s, "the stage's latest
structured output, truncated to what a human debugger needs (<=32 KB): boxes,
track ids+ages, face quality flags, personKey + cosine per match, unique/staff
counters".  These are the pure builders that shape those payloads from the last
frame's raw stage outputs — no I/O, no OpenCV, so they unit-test trivially and
the loop stays the only place that talks to the planner.

The 32 KB ceiling is respected by capping every list at :data:`ROW_CAP` rows
(a debugger scanning boxes never needs more) and rounding coordinates; the
per-frame counts are always exact even when the row list is truncated.
"""

import json

#: Max rows kept in any tap list — keeps payloads well under the 32 KB ceiling.
ROW_CAP = 40

#: Ceiling from CONTRACTS.md v1 for a single tap payload.
MAX_TAP_BYTES = 32_768


def _r(x: float | None, nd: int = 1) -> float | None:
    """Round for readability; pass None through (a coasting box has no conf)."""
    return None if x is None else round(float(x), nd)


def _box(b: dict) -> dict:
    """Compact a box to rounded x/y/w/h (+ conf when the detector gave one)."""
    return {
        "x": _r(b.get("x")),
        "y": _r(b.get("y")),
        "w": _r(b.get("w")),
        "h": _r(b.get("h")),
        "conf": _r(b.get("conf"), 3),
    }


def quality_band(width_px: float, min_px: float, canon_px: float) -> str:
    """Classify a face width: 'gated' (dropped) | 'sub-canon' | 'canon'."""
    if width_px < min_px:
        return "gated"
    if width_px < canon_px:
        return "sub-canon"
    return "canon"


def ingest_payload(t_ms: int, seq: int | None = None) -> dict:
    """The raw-frame stage tap: just where we are in the source."""
    return {"tMs": t_ms, "seq": seq}


def person_payload(boxes: list[dict]) -> dict:
    """Person-detector output: exact count + a capped list of boxes."""
    return {"count": len(boxes), "boxes": [_box(b) for b in boxes[:ROW_CAP]]}


def track_payload(tracks: list[dict]) -> dict:
    """Tracker output: stable ids with their ages (how long each has lived)."""
    rows = [
        {
            "id": t.get("id"),
            "box": _box(t.get("box", {})),
            "ageFrames": t.get("ageFrames"),
            "hits": t.get("hits"),
        }
        for t in tracks[:ROW_CAP]
    ]
    return {"count": len(tracks), "tracks": rows}


def face_payload(faces: list[dict], min_px: float, canon_px: float) -> dict:
    """Face-detector output with the quality band that decides embedding.

    ``kept`` are faces at/above the embed floor; ``gated`` are the rest — the
    number the quality gate dropped before the expensive embed stage.
    """
    rows = []
    kept = 0
    for f in faces:
        w = float(f.get("box", {}).get("w", 0.0))
        if w >= min_px:
            kept += 1
        if len(rows) < ROW_CAP:
            rows.append(
                {
                    "box": _box(f.get("box", {})),
                    "widthPx": _r(w),
                    "quality": quality_band(w, min_px, canon_px),
                    "conf": _r(f.get("conf"), 3),
                }
            )
    return {"count": len(faces), "kept": kept, "gated": len(faces) - kept, "faces": rows}


def match_payload(
    verdicts: list[dict],
    unique: int,
    staff_crossings: int,
    staff_face_frames: int = 0,
    manual_additions: int = 0,
) -> dict:
    """Matcher verdicts (personKey + cosine + staff flag) with live counters.

    Both staff numbers are reported because they answer different questions
    and only one of them is about people: ``staffCrossings`` counts passes (a
    waiter through the gate ten times is ten), ``staffFaceFrames`` counts
    matched face-frames and moves with the frame rate — useful for debugging
    recall, meaningless in a report.  ``manualAdditions`` is how much of
    ``unique`` an operator attested by hand rather than the pipeline detecting
    it; a console that shows the total must be able to show that split.
    """
    rows = [
        {
            "personKey": v.get("personKey"),
            "cosine": _r(v.get("cosine"), 4),
            "isNew": v.get("isNew"),
            "isStaff": bool(v.get("isStaff", False)),
            "subCanon": bool(v.get("subCanon", False)),
        }
        for v in verdicts[:ROW_CAP]
    ]
    return {
        "unique": unique,
        "staffCrossings": staff_crossings,
        "staffFaceFrames": staff_face_frames,
        "manualAdditions": manual_additions,
        "matched": len(verdicts),
        "matches": rows,
    }


def build_payloads(
    last: dict,
    min_px: float,
    canon_px: float,
    unique: int,
    staff_crossings: int,
    staff_face_frames: int = 0,
    manual_additions: int = 0,
) -> dict[str, dict]:
    """Build the per-stage tap payloads from one frame's captured outputs.

    ``last`` is the loop's snapshot of the frame it just processed (t_ms, boxes,
    tracks, faces, verdicts).  Returns ``{stage: payload}`` for the five stages
    that carry a visual/structured output; the loop posts each best-effort.
    """
    return {
        "ingest": ingest_payload(last.get("t_ms", 0), last.get("seq")),
        "person-detect": person_payload(last.get("boxes", [])),
        "track": track_payload(last.get("tracks", [])),
        "face-detect": face_payload(last.get("faces", []), min_px, canon_px),
        "match": match_payload(
            last.get("verdicts", []), unique, staff_crossings,
            staff_face_frames, manual_additions,
        ),
    }


def within_budget(payload: dict, max_bytes: int = MAX_TAP_BYTES) -> bool:
    """True when the JSON-encoded payload fits the CONTRACTS.md 32 KB ceiling."""
    return len(json.dumps(payload).encode()) <= max_bytes
