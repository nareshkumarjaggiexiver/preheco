"""Whether this frame needs a face search at all — the face-search cadence.

THE COST THIS SAVES.  On the CUDA box the whole-frame face search (SCRFD-10G
at 1472x832 on a 4K frame) is the largest single cost in the chain and
saturates the GPU on its own — two concurrent inferences measured 1.05x.
Most frames at a wedding are the SAME people standing where they stood a
frame ago, every one of them already identified; searching the whole frame
again only re-confirms answers the run already holds.

THE RULE (lever L4, HECO_FACE_CADENCE).  A frame's face search may be
skipped only when ALL of these hold:

* there is at least one person box — an empty frame is searched, because
  the person detector can miss a body whose face is plainly visible;
* EVERY person box is covered, one to one, by a SETTLED track at IoU >= 0.5
  — a track that holds an identity lock whose last comfortable face match
  is younger than faceReverifyIntervalS.  One to one, so two people standing
  inside one settled box are not both "covered" by it;
* less than ``max_gap_s`` of frame time has passed since the last search
  that actually ran, so even a room of settled guests is looked at again at
  least that often.  Frame time is the frame's tMs, ingest's clock since
  /open: footage time on a live camera, PROCESSING time on a lockstep file
  replay — a replay faster than real time skips more footage per gap.

Anything else — a newcomer, a guest whose lock has gone stale, a body the
tracker has not confirmed, no clock to measure the gap by — searches.  The
rule errs toward looking, because a face never searched is a guest never
counted.

PER-CAMERA and pure: boxes and times in, a reason out.  Nothing here holds
state; the host keeps the last-search time and the locks.
"""

from heco_common.geometry import iou_xywh

#: A person box is covered by a settled track when they overlap this much.
SETTLED_IOU = 0.5


def settled_tracks(
    tracks: list[dict],
    lock_at: dict,
    now_s: float | None,
    fresh_s: float,
    window_s: float,
) -> list[dict]:
    """The tracks that are SETTLED at ``now_s``.

    Settled = the track holds an identity lock (``lock_at`` maps track id to
    the frame time its lock was last refreshed by a comfortable face match)
    and that refresh is younger than ``fresh_s`` (faceReverifyIntervalS) —
    and no older than ``window_s``, the heal window the lock itself expires
    on.  ``fresh_s`` <= 0 settles nobody: an interval of zero means "verify
    every frame", which is exactly a cadence that never skips.
    """
    if now_s is None or fresh_s <= 0:
        return []
    out: list[dict] = []
    for t in tracks:
        at = lock_at.get(t.get("id"))
        if at is None or not t.get("box"):
            continue
        age = now_s - at
        if 0.0 <= age < fresh_s and age <= window_s:
            out.append(t)
    return out


def search_due(
    bodies: list[dict],
    settled: list[dict],
    now_s: float | None,
    last_search_s: float | None,
    max_gap_s: float,
    iou_min: float = SETTLED_IOU,
) -> str | None:
    """Why this frame's face search must run — or None when it may be skipped.

    ``bodies`` are this frame's person boxes (the ones the face search could
    find a face in), ``settled`` the output of :func:`settled_tracks`.  The
    reasons are for the ledger and for tests: ``no-clock`` (the frame carries
    no time, so no gap can be measured), ``first`` (nothing searched yet),
    ``gap`` (``max_gap_s`` of frame time since the last search, or the clock
    went backwards), ``no-bodies``, ``unsettled``.
    """
    if now_s is None:
        return "no-clock"
    if last_search_s is None:
        return "first"
    gap = now_s - last_search_s
    if gap < 0.0 or gap >= max_gap_s:
        return "gap"
    if not bodies:
        return "no-bodies"
    if not covered(bodies, [t["box"] for t in settled], iou_min):
        return "unsettled"
    return None


def may_skip(
    n_settled: int, now_s: float | None, last_search_s: float | None, max_gap_s: float
) -> bool:
    """Whether :func:`search_due` could skip, decided before any person box exists.

    False means this frame WILL be searched whatever the person detector
    returns — so the search can be issued at once, beside it, instead of
    waiting to be ruled on (HECO_PARALLEL_DETECT).
    """
    if now_s is None or last_search_s is None or n_settled == 0:
        return False
    gap = now_s - last_search_s
    return 0.0 <= gap < max_gap_s


def covered(bodies: list[dict], boxes: list[dict], iou_min: float = SETTLED_IOU) -> bool:
    """Can every body be matched to its OWN box at IoU >= ``iou_min``?

    A maximum bipartite matching (augmenting paths — a frame holds tens of
    people, so this is microseconds), not a greedy one: greedy can strand a
    body whose only partner was taken by a neighbour with a better overlap
    elsewhere, which would only ever search more, never less — but the
    rule's meaning is "one settled track per body", and that is what this
    decides.
    """
    if len(bodies) > len(boxes):
        return False
    edges = [
        [j for j, s in enumerate(boxes) if iou_xywh(b, s) >= iou_min] for b in bodies
    ]
    owner: dict[int, int] = {}

    def claim(i: int, seen: set) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or claim(owner[j], seen):
                owner[j] = i
                return True
        return False

    return all(claim(i, set()) for i in range(len(bodies)))


# ------------------------------------------------ the face search region (L7)


def region_px(region: dict | None, w, h) -> dict | None:
    """A normalised region ({x, y, w, h} in 0..1) as whole pixels on a w x h frame.

    Clamped to the frame.  None when there is no region, no frame size to
    scale it by, or nothing of it left after clamping — and None means the
    caller cannot place it, never "search nothing".
    """
    if not region or not w or not h:
        return None
    try:
        fw, fh = int(w), int(h)
        x0 = max(0, round(float(region["x"]) * fw))
        y0 = max(0, round(float(region["y"]) * fh))
        x1 = min(fw, round((float(region["x"]) + float(region["w"])) * fw))
        y1 = min(fh, round((float(region["y"]) + float(region["h"])) * fh))
    except (KeyError, TypeError, ValueError):
        return None
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def bodies_in(boxes: list[dict], region: dict | None) -> list[dict]:
    """The person boxes a region-only face search could find a face in.

    Every box when there is no region; otherwise the boxes that overlap the
    region at all.  A body wholly outside it cannot yield a face from a
    search that never looks there, so it must not hold the cadence open —
    or a region drawn over one doorway would be searched every frame for as
    long as anyone stood anywhere else in the hall.
    """
    if region is None:
        return list(boxes)
    rx0, ry0 = float(region["x"]), float(region["y"])
    rx1, ry1 = rx0 + float(region["w"]), ry0 + float(region["h"])
    out = []
    for b in boxes:
        x0, y0 = float(b.get("x", 0.0)), float(b.get("y", 0.0))
        x1, y1 = x0 + float(b.get("w", 0.0)), y0 + float(b.get("h", 0.0))
        if min(x1, rx1) > max(x0, rx0) and min(y1, ry1) > max(y0, ry0):
            out.append(b)
    return out
