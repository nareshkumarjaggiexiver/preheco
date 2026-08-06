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

from .gate import reported_reason


def _near_miss(v: dict) -> dict | None:
    """Compact a verdict's nearMiss for the tap row (rounded, shape preserved).

    The match service flags a MINT whose best face cosine landed just under
    the threshold against an existing guest — measured tonight: the same
    person split at face 0.3464 (threshold 0.363) with clothing intersection
    0.562, found only by eye.  The console renders this as a one-click merge
    SUGGESTION; it is information for the operator, never behaviour, so the
    tap carries it verbatim (rounded like every other float here).  None
    passes through untouched — most verdicts carry no flag.
    """
    nm = v.get("nearMiss")
    if not nm:
        return None
    return {
        "key": nm.get("key"),
        "cosine": _r(nm.get("cosine"), 4),
        "appearanceSim": _r(nm.get("appearanceSim"), 4),
    }


def _overlap(v: dict) -> dict | None:
    """Compact a verdict's overlap for the tap row — same shape as nearMiss.

    The match service flags an ENROLMENT whose just-written template landed at
    or above the match threshold against a DIFFERENT identity: two guests the
    gallery itself can no longer tell apart.  Measured emerging three times in
    two days (cross-identity 0.544 / 0.452 / 0.376), each pair one real person
    found by the operator reading cosine matrices by hand.  Unlike nearMiss it
    fires regardless of clothing — at-or-above threshold is the gallery's own
    standard of "same person" — and like nearMiss it is a suggestion, never a
    merge.
    """
    ov = v.get("overlap")
    if not ov:
        return None
    return {
        "key": ov.get("key"),
        "cosine": _r(ov.get("cosine"), 4),
        "appearanceSim": _r(ov.get("appearanceSim"), 4),
    }

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
    """Person-detector output: exact count + a capped list of boxes.

    ``inZone`` counts bodies whose centre sits inside an operator-drawn
    exclusion zone.  They are still detected and still tracked — zones filter
    FACES, not bodies, because a deleted body breaks track continuity for a
    guest walking past a partition — but the count could not say so, and
    "persons 2" with one person behind glass read as a bug to the operator
    (2026-08-06).  The console renders "persons 2 · 1 in zone" instead.
    """
    rows = []
    for b in boxes[:ROW_CAP]:
        r = _box(b)
        if b.get("inZone"):
            r["inZone"] = True
        rows.append(r)
    return {
        "count": len(boxes),
        "inZone": sum(1 for b in boxes if b.get("inZone")),
        "boxes": rows,
    }


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


#: Report-only signals carried through to the console beside each face, so an
#: operator can see the evidence a rejection was made on and not just its name.
_SIGNALS = ("iedPx", "frontality", "sharpness")


def face_payload(faces: list[dict], min_px: float, canon_px: float) -> dict:
    """Face-detector output with the gate verdict that decides embedding.

    ``kept`` are the faces that will be embedded; ``gated`` are the ones the
    quality gate dropped before the expensive embed stage, which is also where
    a guest stops being countable.  ``gatedBy`` breaks that number down by
    reason and each row carries its own ``gate`` label, because "12 faces
    dropped" is not something an operator can act on and "9 for width, 3 for
    frontality" is.

    Faces stamped ``excludedByZone`` (the loop's exclusion-zone filter, which
    runs BEFORE the gate) are neither kept nor gated: they are reported in
    their own ``excludedByZone`` count and their rows carry the flag, because
    the operator who drew the polygon must be able to see every face it is
    eating — an exclusion zone that hides its victims cannot be checked
    against the frosted partition it was drawn for.

    The verdict is READ from the face (``gateReason``, stamped by app/gate.py),
    never recomputed here.  A second implementation of the gate in the
    reporting path is a second gate, and the two would drift the moment either
    grew a signal — which is exactly what happened while the gate was
    width-only in two places.  The width comparison survives only as the
    fallback for a caller that did not run the gate at all.
    """
    rows = []
    kept = 0
    excluded = 0
    gated_by: dict[str, int] = {}
    for f in faces:
        w = float(f.get("box", {}).get("w", 0.0))
        if f.get("excludedByZone"):
            excluded += 1
            if len(rows) < ROW_CAP:
                rows.append(
                    {
                        "box": _box(f.get("box", {})),
                        "widthPx": _r(w),
                        "quality": quality_band(w, min_px, canon_px),
                        "conf": _r(f.get("conf"), 3),
                        # Never gated (the zone filter runs first), so there is
                        # no gate verdict to read — the zone label says WHICH
                        # polygon ate it instead.
                        "gate": "zone",
                        "excludedByZone": True,
                        "zone": f.get("excludedZone"),
                    }
                )
            continue
        reason = reported_reason(f, min_px)
        if reason is None:
            kept += 1
        else:
            gated_by[reason] = gated_by.get(reason, 0) + 1
        if len(rows) < ROW_CAP:
            row = {
                "box": _box(f.get("box", {})),
                "widthPx": _r(w),
                "quality": quality_band(w, min_px, canon_px),
                "conf": _r(f.get("conf"), 3),
                "gate": reason or "kept",
            }
            for key in _SIGNALS:
                if key in f:
                    row[key] = _r(f[key], 3)
            if f.get("gateUnmeasured"):
                row["gateUnmeasured"] = list(f["gateUnmeasured"])
            rows.append(row)
    return {
        "count": len(faces),
        "kept": kept,
        "gated": len(faces) - kept - excluded,
        "excludedByZone": excluded,
        "gatedBy": gated_by,
        "faces": rows,
    }


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
            # Lets the console show a guest's views accumulating live, and
            # shows an operator whether multi-template is actually engaging.
            "templateN": v.get("templateN"),
            "templateAdded": bool(v.get("templateAdded", False)),
            # Advisory torso-appearance similarity vs the matched identity's
            # stored descriptors (4 dp, None for staff/new/descriptor-less —
            # absent is not zero).  Visibility only: it never moves a verdict.
            "appearanceSim": _r(v.get("appearanceSim"), 4),
            # A mint that near-missed an existing guest ({key, cosine,
            # appearanceSim} or None) — the operator's one-click-merge cue.
            "nearMiss": _near_miss(v),
            "overlap": _overlap(v),
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
