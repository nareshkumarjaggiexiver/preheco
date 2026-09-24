"""count.windowFps: the processing rate per window, so a mean cannot hide an outage.

THE CASE (2026-09-25, live test on the Sharon CP Plus camera): the run's fps
read ~7 while the loop processed 13-14.6 fps for as long as the camera sent;
the camera was off the network for part of the run and a frames-over-elapsed
mean averaged the outage in. The operator asked for min and max beside it.
"""

import pytest
from app.stats import StatsBoard


def frames(board: StatsBoard, n: int) -> None:
    """Count ``n`` processed frames on the count stage."""
    for _ in range(n):
        board.frame("count")


def test_the_first_call_opens_a_window_and_observes_nothing():
    """Opening a window is not a measurement."""
    b = StatsBoard()
    frames(b, 10)
    assert b.observe_window_rate("count", 100.0, 5.0) is None
    assert "windowFps" not in b.stages["count"].metrics


def test_each_closed_window_observes_its_own_rate_and_an_outage_reads_zero():
    """15, 14, then a camera outage: min 0, max 15, mean over the windows."""
    b = StatsBoard()
    b.observe_window_rate("count", 0.0, 5.0)
    frames(b, 75)
    assert b.observe_window_rate("count", 5.0, 5.0) == pytest.approx(15.0)
    frames(b, 70)
    assert b.observe_window_rate("count", 10.0, 5.0) == pytest.approx(14.0)
    # the camera drops out: no frames for a whole window
    assert b.observe_window_rate("count", 15.0, 5.0) == pytest.approx(0.0)
    agg = b.stages["count"].metrics["windowFps"].snapshot()
    assert agg["count"] == 3
    assert (agg["min"], agg["max"]) == (pytest.approx(0.0), pytest.approx(15.0))
    assert agg["mean"] == pytest.approx(29.0 / 3)


def test_a_call_inside_the_window_waits_and_the_window_keeps_its_start():
    """A flush two seconds in waits; the next one measures from the start."""
    b = StatsBoard()
    b.observe_window_rate("count", 0.0, 5.0)
    frames(b, 30)
    assert b.observe_window_rate("count", 2.0, 5.0) is None
    frames(b, 45)
    assert b.observe_window_rate("count", 6.0, 5.0) == pytest.approx(75 / 6.0)


def test_the_windows_reach_the_planner_in_the_count_stage_snapshot():
    """No new wire field: the metric rides the count stage's stats body."""
    b = StatsBoard()
    b.observe_window_rate("count", 0.0, 5.0)
    frames(b, 50)
    b.observe_window_rate("count", 5.0, 5.0)
    (count,) = [s for s in b.snapshot(10.0) if s["stage"] == "count"]
    assert count["frames"] == 50
    assert count["metrics"]["windowFps"]["max"] == pytest.approx(10.0)
