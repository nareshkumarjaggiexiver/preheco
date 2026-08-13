"""The operator's drawing, applied to a frame.

Three rules, and the difference between them is the whole module. Two of them
FILTER — a face inside an exclusion polygon never reaches the gate, a person
box inside a detections-mode polygon never reaches the tracker — and one only
LABELS, because deleting a body would cut a track in half and re-mint the
guest on the far side of the partition.

PER-CAMERA, every one of them. A polygon is drawn on one camera's picture in
that picture's normalized coordinates; it means nothing on another lens
looking at the same doorway.

TWO CONVENTIONS THAT LOOK LIKE DETAILS AND ARE NOT.

The CENTRE rule: a box is inside a zone when its centre is, not when it
overlaps. A guest walking past a partition is counted; the face behind the
partition is not. Overlap would eat everyone who passes near the edge.

NO DIMENSIONS MEANS NO EXCLUSION. Zones are normalized 0..1 and need the
frame's pixel size to become polygons. When a frame arrives without one, every
untested item is counted in a ``zoneUnmeasured`` counter and NOTHING is
dropped — an invisible filter silently removing guests from an invoice figure
is the worse failure by a distance.
"""

from heco_common.geometry import point_in_polygon


def apply_face_zones(faces: list, frame: dict, zones: list[dict], obs) -> list:
    """Drop faces whose box CENTRE falls inside an operator-drawn zone.

    Returns the faces that remain countable; excluded ones are stamped
    ``excludedByZone`` in place and stay in the caller's list, because the
    operator must be able to SEE what their polygon is eating (in the tap
    and on the annotated frame) — an invisible filter on an invoice figure
    is not correctable.

    Why zones exist at all: the live bench minted p00004 from an 87 px
    face seen THROUGH A FROSTED GLASS PARTITION — someone inside an
    office, not at the gate, scoring 0.3203 against their true owner.  No
    quality floor can reject a face for being in the wrong PLACE; only the
    operator knows where the partitions, mirrors and TV screens are, so
    they draw them and the loop enforces them here, before the gate, so an
    excluded face is never embedded or matched.

    The centre rule is deliberate: a box straddling a zone edge counts by
    where its middle is, so a real guest walking PAST a partition (box
    clipping the zone) is still counted while a face BEHIND it (centre
    inside) is not.

    Honesty when it cannot run: zone points are normalized 0..1 and need
    the frame's pixel dimensions to scale by.  A frame that carries none
    gets NO exclusion — under-counting is the dominant failure mode, and a
    filter guessing at geometry is worse than a filter reporting it could
    not run — and every face that passed untested is counted in
    ``zoneUnmeasured``, mirroring the ``gatedUnmeasured`` pattern.

    Two ledgers, deliberately: besides the in-place tap stamp, every
    excluded face is observed on the face-detect stats board as
    ``faceZoneExcluded`` (value = box width px).  The tap flag is the
    per-tick film strip — sampled, so a busy run drops rows — while this
    aggregate's COUNT is the audited whole-run zone-excluded total, the
    same film-strip vs audited-account split the planner's funnel draws
    for every other number.  The width stats ride along free and show
    WHAT size of face the polygons eat.  The first live verification
    (2026-08-05) excluded 41 faces (excludedByZone=41) while unique
    correctly stayed 1 — a total that must be readable from the funnel,
    not only by curling the run status.  A run with no zones never
    observes the metric, so its ABSENCE from old runs' stats means "no
    zones were armed", not zero exclusions.
    """
    if not zones or not faces:
        return faces
    w = frame.get("w")
    h = frame.get("h")
    if not w or not h:
        obs.bump("zoneUnmeasured", len(faces))
        return faces
    w, h = float(w), float(h)
    countable: list = []
    excluded = 0
    for f in faces:
        box = f.get("box", {})
        cx = float(box.get("x", 0.0)) + float(box.get("w", 0.0)) / 2.0
        cy = float(box.get("y", 0.0)) + float(box.get("h", 0.0)) / 2.0
        hit = next(
            (
                z
                for z in zones
                if point_in_polygon(
                    cx, cy, [[p[0] * w, p[1] * h] for p in z["points"]]
                )
            ),
            None,
        )
        if hit is None:
            countable.append(f)
        else:
            f["excludedByZone"] = True
            f["excludedZone"] = hit.get("label")
            # The audited ledger (see docstring): one observation per
            # excluded face, width in px, on the face-detect board the
            # planner's whole-run funnel is built from.
            obs.observe(
                "face-detect", "faceZoneExcluded", float(box.get("w", 0.0))
            )
            excluded += 1
    if excluded:
        obs.bump("excludedByZone", excluded)
    return countable


def apply_detection_zones(boxes: list, frame: dict, zones: list[dict], obs) -> list:
    """Drop person boxes whose centre sits inside a DETECTIONS-mode zone.

    The narrow, opt-in sibling of apply_face_zones: a detections zone marks a
    surface that GENERATES phantom people (a wall TV, a poster) — nothing
    real can be inside it, so the right moment to act is BEFORE the
    tracker, where the phantom would otherwise mint a track, churn ids
    and burn face-detect/embed work every frame. Faces-mode zones never
    come through here: deleting a real guest's body breaks track
    continuity, which is why 'faces' stays the default and the editor
    warns before this mode is chosen.

    Same honesty as apply_face_zones: dropped boxes STAY in the caller's list
    stamped excludedByZone/excludedZone (taps + annotated frame show what
    the polygon ate), each is observed on the person-detect board, and a
    frame with no dimensions filters NOTHING — personsZoneUnmeasured
    counts what passed untested, because a filter guessing at geometry is
    worse than a filter reporting it could not run.
    """
    dz = [z for z in zones if z.get("mode") == "detections"]
    if not dz or not boxes:
        return boxes
    w = frame.get("w")
    h = frame.get("h")
    if not w or not h:
        obs.bump("personsZoneUnmeasured", len(boxes))
        return boxes
    w, h = float(w), float(h)
    trackable: list = []
    dropped = 0
    for b in boxes:
        cx = float(b.get("x", 0.0)) + float(b.get("w", 0.0)) / 2.0
        cy = float(b.get("y", 0.0)) + float(b.get("h", 0.0)) / 2.0
        hit = next(
            (
                z
                for z in dz
                if point_in_polygon(
                    cx, cy, [[p[0] * w, p[1] * h] for p in z["points"]]
                )
            ),
            None,
        )
        if hit is None:
            trackable.append(b)
        else:
            b["excludedByZone"] = True
            b["excludedZone"] = hit.get("label")
            obs.observe(
                "person-detect", "personZoneExcluded", float(b.get("h", 0.0))
            )
            dropped += 1
    if dropped:
        obs.bump("personsZoned", dropped)
    return trackable


def mark_person_zones(boxes: list, frame: dict, zones: list[dict], obs) -> None:
    """Stamp person boxes whose centre sits inside an operator zone.

    DISPLAY ONLY — nothing is filtered.  Person boxes inside zones stay
    detected and stay TRACKED on purpose: deleting them would cut track
    continuity for a guest walking past a partition, and a broken track
    re-mints its person on the far side (the split class every heal and
    banner in this system exists to fight).  What the operator asked for
    (2026-08-06 bench) is narrower and right: the console said
    "persons 2" while one of the two stood inside an excluded zone, and
    the number LOOKED wrong because nothing said the detector knew.  So
    the tap now says "persons 2 · 1 in zone" — same facts, no ambiguity.

    Same centre rule and same honesty as apply_face_zones: no dims, no marks.
    """
    if not zones or not boxes:
        return
    w = frame.get("w")
    h = frame.get("h")
    if not w or not h:
        return
    w, h = float(w), float(h)
    for b in boxes:
        if b.get("excludedByZone"):
            continue # already stamped by the detections filter — one fact, one flag
        cx = float(b.get("x", 0.0)) + float(b.get("w", 0.0)) / 2.0
        cy = float(b.get("y", 0.0)) + float(b.get("h", 0.0)) / 2.0
        if any(
            point_in_polygon(cx, cy, [[p[0] * w, p[1] * h] for p in z["points"]])
            for z in zones
        ):
            b["inZone"] = True
