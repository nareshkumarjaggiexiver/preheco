"""The ingest stage's fps is the SOURCE's delivery rate, on ingest's own clock.

THE CASE (2026-09-25): the Sharon CP Plus camera sends 15 fps (GOP 15), yet
the console read "source fps 15.7" — frames over the runner's clock, which
started after a 9-12 s connection while the video buffered meanwhile was then
decoded in a burst. Ingest's own counter said 15.0.
"""

import pytest
from app.stats import SourceRate, StatsBoard


def feed(rate: SourceRate, frames, fps, t0_ms=0.0, c0=0):
    """``frames`` handed-over frames at ``fps`` on ingest's clock; the last rate."""
    out = None
    for i in range(frames):
        out = rate.observe(t0_ms + i * 1000.0 / fps, c0 + i + 1)
    return out


def test_the_startup_burst_is_not_measured():
    """120 frames decoded in the first second (a backlog), then 15 fps: reads 15."""
    r = SourceRate()
    for i in range(120):  # a burst: 120 frames in one second
        r.observe(i * 1000.0 / 120, i + 1)
    got = feed(r, 15 * 60, 15.0, t0_ms=1000.0, c0=120)
    assert got == pytest.approx(15.0, rel=0.01)


def test_nothing_is_said_before_the_warmup_and_a_span():
    """Under warm-up plus the minimum span there is no rate, not a guess."""
    r = SourceRate(warmup_ms=5000, min_span_ms=2000)
    assert feed(r, 15 * 6, 15.0) is None
    assert feed(r, 15 * 8, 15.0) == pytest.approx(15.0, rel=0.01)


def test_an_outage_counts_as_frames_not_delivered():
    """30 s silent in 90 s: the source delivered at 10 fps over that time."""
    r = SourceRate(warmup_ms=0, min_span_ms=0)
    r.observe(0.0, 0)
    r.observe(30_000.0, 450)   # 15 fps for 30 s
    r.observe(60_000.0, 450)   # 30 s of nothing
    assert r.observe(90_000.0, 900) == pytest.approx(10.0)


def test_a_restarted_capture_is_measured_afresh():
    """The counter going backwards is a new capture, not negative frames."""
    r = SourceRate(warmup_ms=0, min_span_ms=0)
    feed(r, 100, 15.0)
    assert r.observe(10.0, 1) is None, "a new origin: nothing to say yet"
    assert feed(r, 60, 7.5, t0_ms=100.0, c0=1) == pytest.approx(7.5, rel=0.02)


def test_the_board_reports_the_measured_rate_for_ingest_only():
    """set_rate replaces frames ÷ elapsed for that stage; the others are untouched."""
    b = StatsBoard()
    for _ in range(100):
        b.frame("ingest")
        b.frame("count")
    b.set_rate("ingest", 15.0)
    snap = {s["stage"]: s for s in b.snapshot(10.0)}
    assert snap["ingest"]["fps"] == 15.0
    assert snap["count"]["fps"] == pytest.approx(10.0)
    assert snap["ingest"]["frames"] == 100, "the frame count is still the loop's"
