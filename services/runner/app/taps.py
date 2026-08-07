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


def jpeg_dims(data: bytes) -> tuple[int, int] | None:
    """(width, height) straight from a JPEG's SOF marker, or None.

    A byte-level scan, no decoder: the forensic path needs the native
    dimensions of EVERY processed frame so the console can scale ledger box
    coordinates onto the picture, and decoding a 4K frame per frame just to
    read two numbers would be the exact per-frame cost the tap duty guard
    exists to forbid.  Scanning the marker table of a real camera JPEG
    touches a few hundred bytes.
    """
    if not data or len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    n = len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            return None  # lost marker sync: not a JPEG we can trust
        marker = data[i + 1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7:  # standalone markers
            i += 2
            continue
        seg_len = (data[i + 2] << 8) | data[i + 3]
        # SOF0..SOF15 carry the frame header, minus DHT/JPG/DAC which share
        # the numeric range but are not frame headers.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 8 >= n:
                return None
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return (width, height) if width > 0 and height > 0 else None
        i += 2 + seg_len
    return None


def ingest_payload(
    t_ms: int,
    seq: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """The raw-frame stage tap: where we are in the source, and how big the
    NATIVE frame is — the reference the console scales overlay boxes against
    (the uploaded picture is downscaled; the box coordinates are not)."""
    return {"tMs": t_ms, "seq": seq, "width": width, "height": height}


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
        if b.get("excludedByZone"):
            # A detections-mode zone dropped this body before the tracker —
            # shown, never hidden, exactly like the excluded faces.
            r["excludedByZone"] = True
            if b.get("excludedZone") is not None:
                r["excludedZone"] = b.get("excludedZone")
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


#: Newest mint-ledger rows carried per tap round.  Mints are rare (one per
#: guest, ever), so a window of 60 keeps each round comfortably inside the
#: 32 KB ceiling while guaranteeing every mint rides MANY consecutive rounds —
#: the planner scans all rounds, so one landing anywhere is enough.
MINT_LEDGER_CAP = 60


def _same_frame_split(v: dict) -> dict | None:
    """Compact a verdict's same-frame split for the tap row.

    Unlike nearMiss and overlap — which SUGGEST — this one records something
    that already happened: the matcher gave this face an identity that another
    body in the same frame was simultaneously wearing, and the frame overruled
    it.  ``{"from": the key it was taken off, "cosine": the confident match
    that was overruled, "clothes": the torso intersection that corroborated
    it}``.  The console shows it beside the guest so an operator can see the
    machine disagreeing with itself and why; the count already reflects it.
    """
    sp = v.get("sameFrameSplit")
    if not sp:
        return None
    return {
        "from": sp.get("from"),
        "cosine": _r(sp.get("cosine"), 4),
        "clothes": _r(sp.get("clothes"), 4),
    }


def verdict_rows(verdicts: list[dict]) -> list[dict]:
    """One frame's match verdicts, as the console reads them.

    Factored out of :func:`match_payload` because the per-frame decision ledger
    (``frame_record``) needs the SAME rows without the run-wide counters and
    ledgers that only make sense once per tap round.  One builder, one shape:
    an engineer reading a frame record and an operator reading a tap are
    looking at the same fields, and cannot drift into two dialects.
    """
    return [
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
            # This face was taken OFF the key the matcher gave it, because a
            # different body in this same frame had it too (None on every
            # ordinary verdict).  Not a suggestion — it already happened.
            "sameFrameSplit": _same_frame_split(v),
        }
        for v in verdicts[:ROW_CAP]
    ]


def match_payload(
    verdicts: list[dict],
    unique: int,
    staff_crossings: int,
    staff_face_frames: int = 0,
    manual_additions: int = 0,
    mints: list[dict] | None = None,
    retired: list[dict] | None = None,
    co_present: list[list[str]] | None = None,
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
    rows = verdict_rows(verdicts)
    return {
        "unique": unique,
        "staffCrossings": staff_crossings,
        "staffFaceFrames": staff_face_frames,
        "manualAdditions": manual_additions,
        "matched": len(verdicts),
        "matches": rows,
        # THE MINT LEDGER.  A mint is a count-changing event: the moment the
        # tap-round sampling missed one, the guest register showed 1 guest
        # against a headline of 5 with three CORRECT merge banners it could
        # not render (2026-08-06, run cbdc6b: 69 rounds, 4 sampled verdicts,
        # every one p00001 — all four mints fell between rounds).  Sightings
        # may be sampled; mints may not.  Every mint verdict (riders and all)
        # is appended to a bounded ledger in the loop and the newest
        # MINT_LEDGER_CAP ride EVERY round, so the register is guaranteed a
        # card — with first-seen time and any banner — for every guest the
        # count contains.
        "mints": [
            {
                "personKey": m.get("personKey"),
                "tMs": m.get("tMs"),
                "cosine": _r(m.get("cosine"), 4),
                "appearanceSim": _r(m.get("appearanceSim"), 4),
                "subCanon": bool(m.get("subCanon", False)),
                "nearMiss": _near_miss(m),
                "overlap": _overlap(m),
            }
            for m in (mints or [])[-MINT_LEDGER_CAP:]
        ],
        # THE RETIREMENT LEDGER — the mint ledger's mirror.  A key that lived
        # even for a second can already have been captured in an earlier tap
        # round's sampled verdicts, and the register builds from every round
        # it has: measured on run a7529b, the lock folded p00002 on the spot,
        # the count correctly read 1, and the register still showed 2 from a
        # single historical sighting.  Naming what was retired (and into what,
        # by which mechanism) lets the console drop those cards, so the count
        # and the register agree in BOTH directions.
        "retired": [
            {
                "personKey": r.get("personKey"),
                "intoKey": r.get("intoKey"),
                "reason": r.get("reason"),
            }
            for r in (retired or [])[-MINT_LEDGER_CAP:]
        ],
        # PAIRS PROVEN DISTINCT by sharing a frame.  A near-miss rider is
        # judged at MINT time, when the pair has often never been seen
        # together yet, so the constraint alone cannot unsay a suggestion it
        # already made — measured on run 05b3b7, where p00002 and p00007 were
        # matched in one frame AFTER p00007's mint had already raised a
        # "likely duplicate" banner against p00002.  The console suppresses
        # suggestions for these pairs retroactively.
        "coPresent": [list(p) for p in (co_present or [])[-MINT_LEDGER_CAP:]],
    }


def build_payloads(
    last: dict,
    min_px: float,
    canon_px: float,
    unique: int,
    staff_crossings: int,
    staff_face_frames: int = 0,
    manual_additions: int = 0,
    mints: list[dict] | None = None,
    retired: list[dict] | None = None,
    co_present: list[list[str]] | None = None,
) -> dict[str, dict]:
    """Build the per-stage tap payloads from one frame's captured outputs.

    ``last`` is the loop's snapshot of the frame it just processed (t_ms, boxes,
    tracks, faces, verdicts).  Returns ``{stage: payload}`` for the five stages
    that carry a visual/structured output; the loop posts each best-effort.
    """
    return {
        "ingest": ingest_payload(
            last.get("t_ms", 0), last.get("seq"), last.get("w"), last.get("h")
        ),
        "person-detect": person_payload(last.get("boxes", [])),
        "track": track_payload(last.get("tracks", [])),
        "face-detect": face_payload(last.get("faces", []), min_px, canon_px),
        "match": match_payload(
            last.get("verdicts", []), unique, staff_crossings,
            staff_face_frames, manual_additions, mints=mints, retired=retired,
            co_present=co_present,
        ),
    }


#: How many frame records the runner batches before a flush drops the oldest.
#: At 4 fps and a 2 s flush that is ~8 per batch, so this is only ever reached
#: when the planner is unreachable — it bounds MEMORY during an outage, and the
#: drop is counted rather than silent.
FRAME_LEDGER_CAP = 4000


def frame_record(
    last: dict,
    min_px: float,
    canon_px: float,
    ms: dict | None = None,
    events: list[str] | None = None,
) -> dict:
    """EVERY frame's decisions, compactly — the pipeline engineer's evidence.

    THE GAP THIS CLOSES.  Tap rounds are sampled at ``tap_interval_s``, and on
    the measured bench that is **12% of frames** (run 0f5c6d: 41 rounds over
    356 frames; run d48e2e: 9%).  The other ~88% are fully processed — detected,
    tracked, gated, embedded, matched — and every decision is then discarded.
    So "why was frame 217 rejected?" has had no answer for 8 frames in 9, which
    makes deep debugging guesswork dressed up as analysis.

    A record is ~1 KB, so a 4-hour event costs about 58 MB of decisions against
    5.6 GB if the PIXELS were kept too — which is exactly why the two are
    separated: the reasoning is cheap enough to keep always, the imagery is not
    (see the armed capture window).

    Deliberately the SAME per-stage shapes the tap round publishes, reusing the
    same builders — an engineer reading a frame and an operator reading a tap
    must be looking at one dialect, not two that drift.  What is left OUT is
    everything run-wide: the unique count, the mint/retirement ledgers and the
    co-presence pairs belong to a round, not to a frame, and repeating them
    57,600 times would be most of the file.

    ``ms`` is this frame's per-stage wall time and ``events`` the loop's own
    count-changing decisions on it (a mint, a fold, a split, a veto) — the two
    things the stage payloads cannot carry because they are about the frame
    rather than about what was in it.
    """
    return {
        "seq": last.get("seq"),
        "tMs": last.get("t_ms", 0),
        "persons": person_payload(last.get("boxes", [])),
        "tracks": track_payload(last.get("tracks", [])),
        "faces": face_payload(last.get("faces", []), min_px, canon_px),
        "verdicts": verdict_rows(last.get("verdicts", [])),
        "ms": {k: _r(v, 1) for k, v in (ms or {}).items()},
        "events": list(events or []),
    }


def within_budget(payload: dict, max_bytes: int = MAX_TAP_BYTES) -> bool:
    """True when the JSON-encoded payload fits the CONTRACTS.md 32 KB ceiling."""
    return len(json.dumps(payload).encode()) <= max_bytes
