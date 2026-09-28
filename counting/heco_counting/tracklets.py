"""Retroactive track identity: a face names the whole stretch of track it sits on.

THE CASE (operator, 2026-09-24, Sharon re-run c84098). At 20:13 p00074 walks
in with a woman in pink right behind her. Only p00074's face was usable in that
frame, so nothing recorded that the woman in pink — p00075, identified from her
face on the SAME track a little later — was standing there too, and the review
queue asked whether the two were one person. The tracker had followed her body
the whole time; the evidence existed and was thrown away because identity only
ever flowed FORWARD from the moment a face matched.

So identity is held per TRACKLET — one track id's run between moments where the
tracker could have swapped people — and it applies to the tracklet's whole
life, backwards included. Two identified tracklets that were ever two bodies in
one frame are two different people. Replayed on c84098's frame ledger: 67 pairs
proven different (39 the gallery already knew from faces in one frame, 28 new,
p00074/p00075 among them), and not one of them scored >= 0.363 on the face model
— no proof contradicted the faces.

THE GUARDS, each one a way this could assert a false "different people" — which
is worse than none, because a cannot_link cannot be undone within a run and it
blocks the fold that would have healed a real duplicate:

* A tracklet ENDS where the track's box overlaps another track at IoU >= 0.4
  (the runner's contest rule): that is where a tracker swaps two people, so no
  identity is carried across it.
* A tracklet's identity comes only from matched faces (never a fresh mint — a
  mint is a key the heal may yet fold away) at >= the lock floor, with the face
  where a head sits in the track box; one vote is enough only at >= 0.6, else
  two; a tracklet whose faces named two keys is impure and proves nothing.
* Two boxes where the smaller sits >= 70% inside the larger may be ONE body
  boxed twice (full body + upper body — the double-box blocker the co-presence
  review found): such a frame is not evidence of two bodies.
* A pair is judged only when BOTH tracklets have ended, so an identity is final
  before it is used; the runner flushes at the end of a run.

Pure: no I/O, no clock. The runner feeds it one frame at a time and asserts the
pairs it returns through the same cannot_link door as face co-presence.
"""

from __future__ import annotations

from itertools import combinations

from heco_common.geometry import iou_xywh

#: A track overlapping another this much may swap people: the tracklet ends.
CONTEST_IOU = 0.4
#: The smaller box this much inside the larger may be one body boxed twice.
NEST_FRAC = 0.7
#: One face vote names a tracklet only at or above this cosine; below it, two.
SURE_COSINE = 0.6
#: Where a head sits in a person box: top 35% of its height, middle 70% of
#: its width (the runner's own binding rule).
HEAD_TOP_FRAC = 0.35
HEAD_CENTRE_FRAC = 0.35


def _nested(a: dict, b: dict) -> bool:
    """Could these two boxes be one body boxed twice?"""
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    small = min(a["w"] * a["h"], b["w"] * b["h"])
    return small > 0 and (ix * iy) / small >= NEST_FRAC


def head_sits_in(face_box: dict, box: dict) -> bool:
    """Is this face where the head of the body in ``box`` would be?"""
    if box["h"] <= 0 or box["w"] <= 0:
        return False
    cx = face_box["x"] + face_box["w"] / 2.0
    cy = face_box["y"] + face_box["h"] / 2.0
    return (
        (cy - box["y"]) / box["h"] <= HEAD_TOP_FRAC
        and abs(cx - (box["x"] + box["w"] / 2.0)) / box["w"] <= HEAD_CENTRE_FRAC
    )


def _track_for(face_box: dict, tracks: list[dict]) -> dict | None:
    """The track whose box contains the face centre, nearest centre first."""
    cx = face_box["x"] + face_box["w"] / 2.0
    cy = face_box["y"] + face_box["h"] / 2.0
    best, best_d = None, 0.0
    for t in tracks:
        b = t["box"]
        if b["x"] <= cx <= b["x"] + b["w"] and b["y"] <= cy <= b["y"] + b["h"]:
            d = (b["x"] + b["w"] / 2.0 - cx) ** 2 + (b["y"] + b["h"] / 2.0 - cy) ** 2
            if best is None or d < best_d:
                best, best_d = t, d
    return best


def bindings(faces: list[dict], verdicts: list[dict], tracks: list[dict],
             min_cosine: float) -> list[tuple[int, str, float]]:
    """This frame's face votes: (track id, key, cosine) for each face that may
    name its track — matched (not minted), not staff, at >= ``min_cosine``,
    with the face where a head sits in the track box.  ``faces`` and
    ``verdicts`` are index-parallel (the kept faces and their /match replies).
    """
    out = []
    for face, v in zip(faces, verdicts, strict=False):
        key, cos = v.get("personKey"), v.get("cosine")
        if not key or v.get("isStaff") or v.get("isNew") or cos is None or cos < min_cosine:
            continue
        t = _track_for(face["box"], tracks)
        if t is None or t.get("id") is None or not head_sits_in(face["box"], t["box"]):
            continue
        out.append((t["id"], key, float(cos)))
    return out


class TrackletBook:
    """Tracklets, their face votes, and which were two bodies in one frame.

    ``observe`` once per processed frame, in order; it returns the key pairs
    newly proven different (each pair once per book).  ``flush`` at the end of
    a run ends every open tracklet and returns what that settles.  Memory is
    bounded: a closed tracklet is kept only while it has a link to an open
    one, and at most ``max_closed`` closed tracklets are held (the oldest go
    first; ``dropped`` counts them so a cap that bit is visible).
    """

    def __init__(self, max_closed: int = 20000) -> None:
        self._open: dict[int, int] = {}          # track id -> tracklet serial
        self._serial = 0
        self._votes: dict[int, dict[str, list[float]]] = {}
        self._links: dict[int, set[int]] = {}
        self._closed: dict[int, str | None] = {}  # serial -> final identity
        self._asserted: set[tuple[str, str]] = set()
        self._max_closed = max_closed
        self.dropped = 0
        self.impure = 0

    # ------------------------------------------------------------- public

    def observe(
        self, tracks: list[dict], votes: list[tuple[int, str, float]]
    ) -> list[tuple[str, str]]:
        """Feed one frame: its tracks ({id, box}) and face votes from
        :func:`bindings`.  Returns key pairs newly proven different."""
        live = {t["id"] for t in tracks if t.get("id") is not None}
        proven: list[tuple[str, str]] = []
        for tid in [t for t in self._open if t not in live]:
            proven += self._close(self._open.pop(tid))
        for t in tracks:
            if t.get("id") is not None and t["id"] not in self._open:
                self._open[t["id"]] = self._new()
        contested: set[int] = set()
        for a, b in combinations([t for t in tracks if t.get("id") is not None], 2):
            if iou_xywh(a["box"], b["box"]) >= CONTEST_IOU:
                contested.update((a["id"], b["id"]))
            if _nested(a["box"], b["box"]):
                continue  # may be one body boxed twice: no evidence this frame
            sa, sb = self._open[a["id"]], self._open[b["id"]]
            self._links[sa].add(sb)
            self._links[sb].add(sa)
        for tid, key, cos in votes:
            if tid in contested or tid not in self._open:
                continue  # a face on a track that may be swapping names nothing
            self._votes[self._open[tid]].setdefault(key, []).append(cos)
        for tid in contested:  # end here; carry nothing across a possible swap
            proven += self._close(self._open.pop(tid))
            self._open[tid] = self._new()
        return proven

    def flush(self) -> list[tuple[str, str]]:
        """End every open tracklet (end of run) and return what that proves."""
        proven: list[tuple[str, str]] = []
        for tid in list(self._open):
            proven += self._close(self._open.pop(tid))
        return proven

    # ------------------------------------------------------------ private

    def _new(self) -> int:
        self._serial += 1
        s = self._serial
        self._votes[s] = {}
        self._links[s] = set()
        return s

    def _identity(self, serial: int) -> str | None:
        votes = self._votes.get(serial) or {}
        if len(votes) != 1:
            if len(votes) > 1:
                self.impure += 1
            return None
        key, cosines = next(iter(votes.items()))
        return key if len(cosines) >= 2 or max(cosines) >= SURE_COSINE else None

    def _close(self, serial: int) -> list[tuple[str, str]]:
        """Finalise one tracklet's identity and judge its links to tracklets
        that have ALSO ended; links to open ones wait for them to end."""
        ident = self._identity(serial)
        self._votes.pop(serial, None)
        if ident is None:
            # Nameless: it can never prove anything, so it holds nothing.
            for other in self._links.pop(serial, set()):
                self._links.get(other, set()).discard(serial)
                self._forget_if_done(other)
            return []
        self._closed[serial] = ident
        proven = []
        for other in list(self._links.get(serial, ())):
            if other not in self._closed:
                continue  # still open: judged when it ends
            o = self._closed[other]
            if o != ident:
                pair = tuple(sorted((ident, o)))
                if pair not in self._asserted:
                    self._asserted.add(pair)
                    proven.append(pair)
            self._links[other].discard(serial)
            self._links[serial].discard(other)
            self._forget_if_done(other)
        self._forget_if_done(serial)
        self._bound()
        return proven

    def _forget_if_done(self, serial: int) -> None:
        """A closed tracklet with no link left to an open one is settled."""
        if serial in self._closed and not self._links.get(serial):
            self._closed.pop(serial, None)
            self._links.pop(serial, None)

    def _bound(self) -> None:
        while len(self._closed) > self._max_closed:
            oldest = next(iter(self._closed))
            for other in self._links.pop(oldest, set()):
                self._links.get(other, set()).discard(oldest)
            self._closed.pop(oldest)
            self.dropped += 1
