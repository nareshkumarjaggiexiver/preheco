"""Retroactive track identity (heco_counting.tracklets).

What each test pins is a way the rule could otherwise assert a FALSE
"different people" — which cannot be undone within a run and blocks the fold
that would heal a real duplicate — or miss the case it exists for: p00074
walking in with p00075 right behind her, p00075's face only usable later on
the same track (Sharon re-run c84098, 20:13).
"""

from heco_counting.tracklets import TrackletBook, bindings, head_sits_in


def box(x, y=100.0, w=100.0, h=300.0):
    """A standing person box; x spaces bodies apart."""
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}


def face_on(b):
    """A face where the head of the body in box ``b`` sits."""
    return {"box": {"x": b["x"] + b["w"] * 0.35, "y": b["y"] + 10.0, "w": b["w"] * 0.3, "h": 40.0}}


def run(book, frames):
    """Feed [(tracks, votes)] frames, then flush; return every proven pair."""
    out = []
    for tracks, votes in frames:
        out += book.observe(tracks, votes)
    return out + book.flush()


A, B = box(0), box(400)


def test_a_face_named_later_covers_the_frames_before_it():
    """The case: B's face is only usable at frame 4, after the frames the two
    shared; B's identity still reaches back and proves A != B."""
    both = [{"id": 1, "box": A}, {"id": 2, "box": B}]
    frames = [(both, [(1, "p00074", 0.9)] if i == 0 else []) for i in range(4)]
    assert run(TrackletBook(), frames) == [], "B never named: nothing proven"
    frames.append(([{"id": 2, "box": B}], [(2, "p00075", 0.95)]))  # later, SAME track
    assert run(TrackletBook(), frames) == [("p00074", "p00075")]


def test_a_pair_is_proven_once_and_only_after_both_stretches_end():
    """Identity must be final before it is used: nothing while B is still open."""
    book = TrackletBook()
    both = [{"id": 1, "box": A}, {"id": 2, "box": B}]
    assert book.observe(both, [(1, "k1", 0.9), (2, "k2", 0.9)]) == []
    assert book.observe([{"id": 2, "box": B}], []) == [], "A ended, B still open"
    assert book.observe([], []) == [("k1", "k2")]
    assert book.flush() == []
    # the same two keys proven again later are not re-reported
    assert run(book, [(both, [(1, "k1", 0.9), (2, "k2", 0.9)])]) == []


def test_an_overlap_ends_the_stretch_so_a_swap_carries_no_name():
    """Tracks 1 and 2 overlap (IoU >= 0.4) at frame 1 — where a tracker swaps
    people.  Track 1's name before the overlap must not reach after it."""
    near = box(40)   # IoU with A = 60*300 / (2*30000 - 18000) = 0.43
    frames = [
        ([{"id": 1, "box": A}, {"id": 3, "box": box(800)}], [(1, "k1", 0.9), (3, "k3", 0.9)]),
        ([{"id": 1, "box": A}, {"id": 2, "box": near}], []),          # contest: both stretches end
        ([{"id": 1, "box": A}, {"id": 4, "box": box(1200)}], [(4, "k4", 0.9)]),
    ]
    got = run(TrackletBook(), frames)
    assert ("k1", "k3") in got, "before the overlap: k1 and k3 were two bodies"
    assert ("k1", "k4") not in got, "after the overlap track 1 may be someone else"


def test_a_face_on_a_contested_track_names_nothing():
    """A face on a track that may be mid-swap names no one."""
    book = TrackletBook()
    near = box(40)
    frames = [([{"id": 1, "box": A}, {"id": 2, "box": near}, {"id": 3, "box": box(800)}],
               [(1, "k1", 0.95), (3, "k3", 0.95)])]
    assert run(book, frames) == [], "k1's vote landed on a track that may be swapping"


def test_one_body_boxed_twice_is_not_two_bodies():
    """Full-body and upper-body boxes of one person: the double-box blocker."""
    upper = box(10, y=100.0, w=80.0, h=120.0)   # inside A (nested), IoU with A < 0.4
    frames = [([{"id": 1, "box": A}, {"id": 2, "box": upper}], [(1, "k1", 0.9), (2, "k2", 0.9)])]
    assert run(TrackletBook(), frames) == []


def test_two_names_on_one_stretch_make_it_impure():
    """A stretch whose faces named two keys proves nothing, and is counted."""
    book = TrackletBook()
    frames = [([{"id": 1, "box": A}, {"id": 2, "box": B}], [(1, "k1", 0.9), (2, "k2", 0.9)]),
              ([{"id": 1, "box": A}, {"id": 2, "box": B}], [(1, "k9", 0.9)])]
    assert run(book, frames) == []
    assert book.impure == 1


def test_one_weak_vote_is_not_a_name_but_two_are_and_one_sure_one_is():
    """One vote under 0.6 is a guess; two votes, or one sure one, are a name."""
    both = [{"id": 1, "box": A}, {"id": 2, "box": B}]
    assert run(TrackletBook(), [(both, [(1, "k1", 0.5), (2, "k2", 0.9)])]) == []
    assert run(TrackletBook(), [(both, [(1, "k1", 0.5), (2, "k2", 0.9)]),
                                (both, [(1, "k1", 0.5)])]) == [("k1", "k2")]
    assert run(TrackletBook(), [(both, [(1, "k1", 0.61), (2, "k2", 0.9)])]) == [("k1", "k2")]


def test_same_name_on_both_proves_nothing():
    """Two stretches with the same name are not a disagreement."""
    both = [{"id": 1, "box": A}, {"id": 2, "box": B}]
    assert run(TrackletBook(), [(both, [(1, "k1", 0.9), (2, "k1", 0.9)])]) == []


def test_bindings_take_only_matched_heads_at_the_floor():
    """Only matched, non-staff faces at the floor, where a head sits, vote."""
    tracks = [{"id": 1, "box": A}, {"id": 2, "box": B}]
    feet = {"box": {"x": A["x"] + 30, "y": A["y"] + 250, "w": 30, "h": 30}}
    faces = [face_on(A), face_on(B), feet, face_on(A), face_on(B)]
    verdicts = [
        {"personKey": "k1", "cosine": 0.8},
        {"personKey": "k2", "cosine": 0.3},                  # under the floor
        {"personKey": "k3", "cosine": 0.9},                  # not where a head sits
        {"personKey": "k4", "cosine": 0.9, "isNew": True},   # a mint names nothing
        {"personKey": "k5", "cosine": 0.9, "isStaff": True},
    ]
    assert bindings(faces, verdicts, tracks, 0.45) == [(1, "k1", 0.8)]
    assert head_sits_in(face_on(A)["box"], A) and not head_sits_in(feet["box"], A)


def test_memory_is_bounded_and_the_cap_is_counted():
    """Closed stretches waiting on an open one are capped, and the cap counts."""
    book = TrackletBook(max_closed=5)
    for i in range(40):
        # a named body that stays + a new named neighbour each frame that leaves
        tracks = [{"id": 1, "box": A}, {"id": 100 + i, "box": B}]
        book.observe(tracks, [(1, "stay", 0.9), (100 + i, f"k{i}", 0.9)])
    assert len(book._closed) <= 5
    assert book.dropped > 0, "the cap bit, and says so"
    book.flush()
