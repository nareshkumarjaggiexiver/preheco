"""Streaming stat aggregation for the planner's ingest shapes.

The planner charts per-stage aggregates ({count,min,mean,max} per metric) and
sampled raw rows.  Both are built here, allocation-light, so the run loop can
observe thousands of values per second without keeping them all.
"""

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

    def frame(self, stage: str) -> None:
        """Count one processed frame for a stage."""
        self.stages[stage].frames += 1

    def observe(self, stage: str, name: str, value: float) -> None:
        """Record one metric observation under a stage."""
        self.stages[stage].observe(name, value)

    def snapshot(self, elapsed_s: float) -> list[dict]:
        """Return planner stats bodies for every stage with activity."""
        out = []
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

    def add(self, stage: str, t_ms: float, metrics: dict) -> None:
        """Buffer one raw row ({stage, tMs, metrics}) if under the cap."""
        if len(self.rows) >= self.cap:
            self.dropped += 1
            return
        self.rows.append({"stage": stage, "tMs": t_ms, "metrics": metrics})

    def drain(self) -> list[dict]:
        """Return and clear the buffered rows (drop counter keeps running)."""
        rows, self.rows = self.rows, []
        return rows
