"""SortLite — a small, honest SORT-style multi-object tracker.

What it IS: per-frame association of detection boxes to existing tracks by
IoU (intersection-over-union) on constant-velocity-predicted boxes, greedy
best-first matching, with max-age pruning of coasting tracks and min-hits
confirmation of new ones. Velocity is a smoothed per-frame center delta —
an alpha-filtered constant-velocity model, snap-to-detection on position.

What it is NOT (deliberately, per the planning docs' division of labour):

* no re-identification — appearance is never looked at; a person who leaves
  and returns gets a new id, and repairing that is the face gallery's job
  (docs/planning/03, "long-term identity is deliberately not their job");
* no Kalman filter — no covariance tuning, no measurement-noise model; a
  smoothed velocity is enough to carry a track through the 3-6 seconds of a
  gate crossing at POC scale;
* no Hungarian assignment — matching is greedy over IoU pairs sorted
  descending. Trade-off vs scipy's optimal assignment: greedy can pick a
  globally suboptimal pairing when three or more boxes contest overlapping
  regions, but it is dependency-free, O(n*m log nm), and at POC gate
  densities (a handful of people in frame) the IoU matrix is decisive.
  Swap in scipy.optimize.linear_sum_assignment if crowded-gate evidence
  demands it.

Documented crossing behaviour: two boxes whose *predicted* positions remain
distinguishable (different lanes, or velocity separating them) keep their
ids through a crossing — the constant-velocity prediction carries each track
through the overlap. If two detections become exactly coincident, the
assignment is ambiguous and ids may swap; this is accepted at POC because
identity repair lives downstream in the gallery.
"""

from dataclasses import dataclass


@dataclass
class TrackState:
    """Internal mutable state of one track (centers in pixels)."""

    tid: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    age_frames: int = 1
    misses: int = 0  # frames since last matched detection
    conf: float | None = None
    # Last *matched* center, for velocity measurement across miss gaps.
    last_cx: float = 0.0
    last_cy: float = 0.0

    def __post_init__(self) -> None:
        """Initialise the last-matched center to the birth position."""
        self.last_cx, self.last_cy = self.cx, self.cy

    def predict(self) -> None:
        """Advance one frame under the constant-velocity model."""
        self.cx += self.vx
        self.cy += self.vy
        self.age_frames += 1
        self.misses += 1  # provisional; update() resets on a match

    def update(self, x: float, y: float, w: float, h: float,
               conf: float | None, smooth: float) -> None:
        """Absorb a matched detection: snap position, smooth velocity.

        The velocity measurement is the per-frame delta from the last
        matched center, divided by the gap length so a re-association after
        a dropout does not read the whole gap as one frame of motion.
        """
        cx, cy = x + w / 2.0, y + h / 2.0
        gap = max(1, self.misses)
        mvx, mvy = (cx - self.last_cx) / gap, (cy - self.last_cy) / gap
        self.vx = (1.0 - smooth) * self.vx + smooth * mvx
        self.vy = (1.0 - smooth) * self.vy + smooth * mvy
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.last_cx, self.last_cy = cx, cy
        self.conf = conf
        self.hits += 1
        self.misses = 0

    def box(self) -> tuple[float, float, float, float]:
        """Current box as (x, y, w, h) from the center representation."""
        return (self.cx - self.w / 2.0, self.cy - self.h / 2.0, self.w, self.h)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two (x, y, w, h) boxes; 0.0 when either is degenerate."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = min(ax2, bx2) - max(a[0], b[0])
    ih = min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class SortLite:
    """One run's tracker state: feed detections per frame, get live tracks.

    Parameters mirror classic SORT: ``max_age`` frames a track may coast
    unmatched before being dropped (default 30 — see ``__init__`` for the
    measured reason it is no longer 15), ``min_hits`` matches required before
    a track is reported (suppresses one-frame ghosts; waived during the first
    ``min_hits`` frames of a run so counting starts immediately),
    ``iou_min`` the association gate, ``vel_smooth`` the velocity filter
    alpha (1.0 = trust only the newest delta).
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_min: float = 0.2, vel_smooth: float = 0.5) -> None:
        """Configure thresholds; state starts empty.

        WHY ``max_age`` DEFAULTS TO 30 AND NOT THE ORIGINAL 15 (bench 6e1a5d,
        2026-08-06, ground truth ONE person walking out of frame, back in,
        then sitting down).  That one person produced SIX tracker ids — track
        2 (53 frames), 5 (24), 6 (8), 7 (6), 8 (6), 12 (7).  The run ran at
        3.97 fps, so 15 unmatched frames is **3.75 s** of coasting: leaving
        the frame legitimately killed track 2, but while the subject was
        SEATED the person detector lost them intermittently and every gap
        longer than 3.75 s minted a fresh id — tracks 6/7/8/12 are that one
        seated person, four times over.  30 frames is ~7.5 s at the same rate
        and coasts those gaps.

        THE HONEST CAVEAT: this treats a SYMPTOM.  The disease is YOLOX-nano
        dropping seated bodies, and no amount of coasting fixes a detector
        that cannot see the subject.  Worse, a longer coast WIDENS the window
        in which a ghost track can latch onto a different person walking
        through the dead track's predicted box — and a track that has changed
        person is exactly the evidence the runner's heal and identity lock act
        on, so a wrong coast can become a wrong MERGE, which under-counts
        silently.  That is why the runner's clothing guard (the tracker-swap
        detector) exists and why this number is an env knob
        (``HECO_TRACKER_MAX_AGE``) with the old 15 one restart away.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_min = iou_min
        self.vel_smooth = vel_smooth
        self.tracks: list[TrackState] = []
        self.frame_count = 0
        self._next_id = 1

    def step(
        self, detections: list[tuple[float, float, float, float, float | None]]
    ) -> list[TrackState]:
        """Advance one frame with ``detections`` [(x, y, w, h, conf), ...].

        Returns the tracks that are both *updated this frame* and
        *confirmed* (hits >= min_hits, waived during run warm-up). Coasting
        tracks are predicted but not reported — a track that reappears
        within ``max_age`` frames keeps its id.
        """
        self.frame_count += 1
        for t in self.tracks:
            t.predict()

        # Greedy best-first association on the IoU of predicted boxes.
        pairs: list[tuple[float, int, int]] = []
        for ti, t in enumerate(self.tracks):
            tb = t.box()
            for di, d in enumerate(detections):
                score = iou(tb, (d[0], d[1], d[2], d[3]))
                if score >= self.iou_min:
                    pairs.append((score, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        used_t: set[int] = set()
        used_d: set[int] = set()
        for _score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            x, y, w, h, conf = detections[di]
            self.tracks[ti].update(x, y, w, h, conf, self.vel_smooth)

        # Births for unmatched detections.
        for di, (x, y, w, h, conf) in enumerate(detections):
            if di not in used_d:
                self.tracks.append(
                    TrackState(tid=self._next_id, cx=x + w / 2.0, cy=y + h / 2.0,
                               w=w, h=h, conf=conf)
                )
                self._next_id += 1

        # Deaths: coasted past max_age.
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        warmup = self.frame_count <= self.min_hits
        return [
            t for t in self.tracks
            if t.misses == 0 and (t.hits >= self.min_hits or warmup)
        ]
