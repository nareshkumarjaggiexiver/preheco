"""The frame's geometry questions, pinned where they now live.

These three functions decide which body a face belongs to, which track it
belongs to, and whether two identities were seen on different bodies. Every
fold in the pipeline reads their answers, so their tie-breaks and their
absent-is-not-zero conventions are load-bearing — and neither had a direct
test before the extraction, only coverage through the loop.
"""

from heco_counting.association import (
    body_token,
    different_bodies,
    person_box_for,
    track_for,
)


def box(x, y, w, h):
    """A box dict in the wire shape."""
    return {"x": x, "y": y, "w": w, "h": h}


def face_at(cx, cy, size=20):
    """A face whose box CENTRE is at (cx, cy)."""
    return {"box": box(cx - size / 2, cy - size / 2, size, size)}


# --------------------------------------------------------------- track_for


def test_a_face_belongs_to_the_track_containing_its_centre():
    """The base rule: centre containment, not overlap or nearest."""
    tracks = [{"id": 7, "box": box(0, 0, 100, 200)}]
    assert track_for(face_at(50, 100), tracks) == 7


def test_a_face_outside_every_track_has_none():
    """None is the honest answer, and the caller must not invent a track.

    'The SAME track matched someone else' is the entire evidence a heal acts
    on, so a face with no track carries no heal bookkeeping at all.
    """
    tracks = [{"id": 7, "box": box(0, 0, 100, 200)}]
    assert track_for(face_at(500, 500), tracks) is None


def test_overlapping_tracks_tie_break_on_nearest_box_centre():
    """The tie-break is centre distance, not list order or box area.

    Two tracks both containing the face centre is the crowded-gate case, and
    picking by order would make the answer depend on the tracker's output
    ordering — which nothing guarantees.
    """
    near = {"id": 1, "box": box(0, 0, 100, 100)}      # centre (50, 50)
    far = {"id": 2, "box": box(0, 0, 400, 400)}       # centre (200, 200)
    assert track_for(face_at(60, 60), [far, near]) == 1
    assert track_for(face_at(60, 60), [near, far]) == 1, "order must not matter"


def test_containment_is_inclusive_of_the_boundary():
    """A face centre exactly on the edge is inside. Pinned because the
    comparison is <= on both sides and a later 'tidy' to < would silently
    drop faces at a box edge — the crowded case, where boxes are tight."""
    tracks = [{"id": 3, "box": box(0, 0, 100, 100)}]
    assert track_for(face_at(0, 0), tracks) == 3
    assert track_for(face_at(100, 100), tracks) == 3


# ---------------------------------------------------------- person_box_for


def test_person_box_for_returns_the_box_OBJECT_not_a_copy():
    """Identity matters: the caller takes body_token(pbox) off this result.

    If this ever returned a copy, every face would land on a distinct token,
    two faces on one body would read as two bodies, and co-presence would
    assert 'different people' for a guest holding a photo of themselves.
    """
    b = box(0, 0, 100, 200)
    assert person_box_for(face_at(50, 100), [b]) is b


def test_a_face_with_no_person_box_has_none():
    """None disables the appearance veto rather than feeding it zero."""
    assert person_box_for(face_at(500, 500), [box(0, 0, 100, 200)]) is None


# -------------------------------------------------------- different_bodies


def test_two_faces_in_two_bodies_are_two_people():
    """The one certain identity signal the pipeline has."""
    bodies = {"p0001": {1}, "p0002": {2}}
    assert different_bodies(bodies, "p0001", "p0002") is True


def test_two_faces_in_ONE_body_say_nothing():
    """A guest and the phone in their hand showing their own face."""
    bodies = {"p0001": {1}, "p0002": {1}}
    assert different_bodies(bodies, "p0001", "p0002") is False


def test_a_key_absent_from_the_frame_is_not_co_present():
    """An identity not seen here shares no evidence with anything here."""
    assert different_bodies({"p0001": {1}}, "p0001", "p0002") is False


def test_a_face_with_no_body_pairs_as_DIFFERENT_with_everything():
    """Fails toward the VISIBLE direction.

    We cannot show the two share a body, and asserting difference fails toward
    over-count — which an operator can see and merge — where the opposite
    fails toward a silent under-count.
    """
    bodies = {"p0001": {None}, "p0002": {2}}
    assert different_bodies(bodies, "p0001", "p0002") is True


# ---------------------------------------------------------------- body_token


def test_body_token_is_stable_within_a_frame_and_none_for_no_box():
    """Frame-local identity: same object same token, distinct objects differ."""
    b = box(0, 0, 10, 10)
    assert body_token(b) == body_token(b), "same object, same token"
    assert body_token(None) is None
    assert body_token(box(0, 0, 10, 10)) != body_token(b), "distinct objects differ"
