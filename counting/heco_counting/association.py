"""Which body, which track, which person — the frame's geometry questions.

Three pure functions and one token. They hold no state, do no I/O and know
nothing about a gallery: given this frame's boxes, they say which person box
contains a face, which track it belongs to, and whether two identities were
seen on different bodies.

PER-CAMERA, all of it. Every rule here reasons about one camera's own
geometry and one camera's own tracker ids, and none of it survives a change of
viewpoint — a face centre inside a track box means nothing across two lenses
looking at the same doorway from different angles. The cross-camera question
is a different mechanism entirely, and keeping these functions honest about
their scope is what stops it being answered here by accident.
"""


def track_for(face: dict, tracks: list) -> int | None:
    """The track a face belongs to: box centre inside the track box.

    Ties (overlapping tracks both containing the centre) go to the track
    whose box CENTRE is nearest the face centre.  None when no track
    contains it — a face with no track cannot carry heal bookkeeping,
    because "the SAME track matched someone else" is the entire evidence
    the heal acts on.
    """
    box = face.get("box") or {}
    cx = float(box.get("x", 0.0)) + float(box.get("w", 0.0)) / 2.0
    cy = float(box.get("y", 0.0)) + float(box.get("h", 0.0)) / 2.0
    best_id: int | None = None
    best_d = 0.0
    for t in tracks:
        tb = t.get("box") or {}
        tx, ty = float(tb.get("x", 0.0)), float(tb.get("y", 0.0))
        tw, th = float(tb.get("w", 0.0)), float(tb.get("h", 0.0))
        if not (tx <= cx <= tx + tw and ty <= cy <= ty + th):
            continue
        d = (tx + tw / 2.0 - cx) ** 2 + (ty + th / 2.0 - cy) ** 2
        if best_id is None or d < best_d:
            best_id, best_d = t.get("id"), d
    return best_id


def person_box_for(face: dict, boxes: list) -> dict | None:
    """The person box a face belongs to: box centre inside the person box.

    Same association rule as :func:`track_for` (centre containment, ties
    to the nearest box centre) but over the RAW detector boxes, because the
    torso crop needs the person's full extent this frame — a track box can
    be a stale prediction, and a descriptor histogrammed off background
    would manufacture exactly the clashes the veto must only see in real
    tracker swaps.  None when no box contains the face (a face detected
    outside every person box carries no torso to describe), and per the
    absent-is-not-zero convention None disables the veto rather than
    feeding it.
    """
    fb = face.get("box") or {}
    cx = float(fb.get("x", 0.0)) + float(fb.get("w", 0.0)) / 2.0
    cy = float(fb.get("y", 0.0)) + float(fb.get("h", 0.0)) / 2.0
    best: dict | None = None
    best_d = 0.0
    for b in boxes:
        bx, by = float(b.get("x", 0.0)), float(b.get("y", 0.0))
        bw, bh = float(b.get("w", 0.0)), float(b.get("h", 0.0))
        if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
            continue
        d = (bx + bw / 2.0 - cx) ** 2 + (by + bh / 2.0 - cy) ** 2
        if best is None or d < best_d:
            best, best_d = b, d
    return best


def different_bodies(bodies: dict, a: str | None, b: str | None) -> bool:
    """Were these two identities seen on DIFFERENT bodies in one frame?

    The one certain identity signal this system has, stated precisely.  Two
    faces in TWO person boxes are two bodies, so two people — that is what
    licenses a cannot_link and what forbids a fold.  Two faces in ONE box
    are one body: a guest and the phone, mirror or photo they are holding
    (measured on the 2026-08-06 bench, where a man's phone showed his own
    face and the heal correctly folded it away).  Calling that pair "two
    people" would both assert a false constraint and block a correct fold.

    A face with no containing box pairs with everything: we cannot show it
    shares a body, and asserting difference fails toward OVER-count, which
    is the visible direction.  An identity absent from this frame is not
    co-present with anything here.
    """
    ba, bb = bodies.get(a), bodies.get(b)
    if not ba or not bb:
        return False
    if None in ba or None in bb:
        return True
    return ba.isdisjoint(bb)


def body_token(pbox: dict | None) -> int | None:
    """A token identifying THIS FRAME's person box, for co-presence.

    Names what the call site used to write as a bare ``id(pbox)``. It is a
    PROCESS-LOCAL, FRAME-LOCAL identity: the only guarantee is that two faces
    landing in the same box object get the same token within one frame, and
    that is exactly the guarantee co-presence needs ("two faces in one body
    are one person; in two bodies, two people").

    It is NOT stable across frames, processes or cameras, and must never be
    persisted or sent over a wire — which is the whole reason it has a name
    now. An identifier that looks durable and is not is the kind of thing that
    gets stored, compared across a gate, and quietly folds two guests into one.
    """
    return id(pbox) if pbox is not None else None
