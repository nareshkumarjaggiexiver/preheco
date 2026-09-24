"""Streaming stat aggregation for the planner's ingest shapes.

The planner charts per-stage aggregates ({count,min,mean,max} per metric) and
sampled raw rows.  Both are built here, allocation-light, so the run loop can
observe thousands of values per second without keeping them all.

THREAD SAFETY, and why it is not optional here.  The counting loop WRITES
(``frame``/``observe``/``add``) while the reporter thread READS
(``snapshot``/``drain``) — see ``app.reporting``.  The crash this prevents is
specific, not theoretical: ``observe`` grows ``StageStats.metrics`` via
``setdefault``, and several metric names first appear only when the scene
produces them (a face metric on the first face, an embed metric on the first
embedding).  A snapshot iterating that dict at the instant a new name arrives
raises ``RuntimeError: dictionary changed size during iteration`` and kills
the reporter.  The locks are uncontended in the common case — one writer, a
reader every couple of seconds — so they cost a few hundred nanoseconds per
observation against a loop spending milliseconds.
"""

import threading
from dataclasses import dataclass, field

# Contract stage names (CONTRACTS.md "Planner ingest").
STAGES = (
    "ingest",
    "person-detect",
    "track",
    "face-detect",
    "quality",
    "embed",
    "match",
    "count",
)


@dataclass
class MetricAgg:
    """Streaming count/min/mean/max of one metric — no value list kept."""

    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def add(self, value: float) -> None:
        """Fold one observation into the aggregate."""
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def snapshot(self) -> dict:
        """Return the planner's {count,min,mean,max} shape."""
        if self.count == 0:
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": self.count,
            "min": self.min,
            "mean": self.total / self.count,
            "max": self.max,
        }


@dataclass
class StageStats:
    """Frame counter plus named metric aggregates for one pipeline stage."""

    frames: int = 0
    metrics: dict[str, MetricAgg] = field(default_factory=dict)

    def observe(self, name: str, value: float) -> None:
        """Fold one metric observation for this stage."""
        self.metrics.setdefault(name, MetricAgg()).add(value)


class StatsBoard:
    """All per-stage aggregates for one run.

    snapshot() emits one planner stats body per stage that has seen frames —
    the planner upserts per (run, stage), so re-posting growing aggregates
    every flush is the intended usage.
    """

    def __init__(self) -> None:
        """Start with every contract stage present but empty."""
        self.stages: dict[str, StageStats] = {s: StageStats() for s in STAGES}
        self._lock = threading.Lock()
        #: stage -> (window start, the stage's frames then); see observe_window_rate.
        self._windows: dict[str, tuple[float, int]] = {}

    def frame(self, stage: str) -> None:
        """Count one processed frame for a stage."""
        with self._lock:
            self.stages[stage].frames += 1

    def observe(self, stage: str, name: str, value: float) -> None:
        """Record one metric observation under a stage."""
        with self._lock:
            self.stages[stage].observe(name, value)

    def observe_window_rate(
        self, stage: str, now: float, window_s: float, name: str = "windowFps"
    ) -> float | None:
        """Every ``window_s``, observe ``stage``'s frames per second over the window just closed.

        A stage's ``fps`` is frames over the time since the run started, a
        MEAN with no history: the 2026-09-25 live test read ~7 fps for a run
        whose loop processed 13-14.6 fps for as long as the camera sent,
        because the camera was off the network for part of it. Observed as a
        metric, the windows reach the planner as {count, min, mean, max} with
        no new wire field, so the console can show the slow patch or the
        outage (a window with no frames observes 0) beside the mean.

        The first call only opens a window; a call before ``window_s`` has
        passed observes nothing. Returns the rate observed, else None.
        """
        with self._lock:
            st = self.stages[stage]
            opened = self._windows.get(stage)
            if opened is None:
                self._windows[stage] = (now, st.frames)
                return None
            t0, f0 = opened
            if now - t0 < window_s:
                return None
            rate = (st.frames - f0) / (now - t0)
            st.observe(name, rate)
            self._windows[stage] = (now, st.frames)
            return rate

    def snapshot(self, elapsed_s: float) -> list[dict]:
        """Return planner stats bodies for every stage with activity.

        The whole walk happens under the lock, including the inner
        ``v.snapshot()`` calls: releasing between stages would let a metric
        first observed mid-walk appear in one body and not the next, and the
        aggregates are cheap enough that holding it costs the loop nothing
        measurable.
        """
        out = []
        with self._lock:
            for stage, st in self.stages.items():
                if st.frames == 0 and not st.metrics:
                    continue
                fps = st.frames / elapsed_s if elapsed_s > 0 else 0.0
                out.append(
                    {
                        "stage": stage,
                        "frames": st.frames,
                        "fps": round(fps, 3),
                        "metrics": {k: v.snapshot() for k, v in st.metrics.items()},
                    }
                )
        return out


class SampleBuffer:
    """Capped buffer of raw sample rows between flushes.

    The planner accepts batches of <= `cap` rows (200 in the contract), so the
    buffer holds at most one batch: rows arriving after the cap within a flush
    window are dropped and counted — that IS the sampling, cheap and honest.
    """

    def __init__(self, cap: int = 200) -> None:
        """Create an empty buffer holding at most `cap` rows per flush window."""
        self.cap = cap
        self.rows: list[dict] = []
        self.dropped = 0
        self._lock = threading.Lock()

    def add(self, stage: str, t_ms: float, metrics: dict) -> None:
        """Buffer one raw row ({stage, tMs, metrics}) if under the cap."""
        with self._lock:
            if len(self.rows) >= self.cap:
                self.dropped += 1
                return
            self.rows.append({"stage": stage, "tMs": t_ms, "metrics": metrics})

    def drain(self) -> list[dict]:
        """Return and clear the buffered rows (drop counter keeps running).

        Locked against ``add`` because the reporter thread drains while the
        loop is still adding: the read-then-rebind is two bytecodes, and a row
        appended between them would be dropped silently — a lost sample that
        no counter would confess to.
        """
        with self._lock:
            rows, self.rows = self.rows, []
        return rows
