"""Unit tests for the stat aggregation maths and the sample buffer."""

import pytest
from app.stats import MetricAgg, SampleBuffer, StatsBoard


def test_metric_agg_maths():
    """count/min/mean/max over a known value set."""
    agg = MetricAgg()
    for v in [10.0, 20.0, 60.0]:
        agg.add(v)
    snap = agg.snapshot()
    assert snap == {"count": 3, "min": 10.0, "mean": 30.0, "max": 60.0}


def test_metric_agg_empty():
    """An untouched aggregate reports count 0 and no extremes."""
    assert MetricAgg().snapshot() == {"count": 0, "min": None, "mean": None, "max": None}


def test_board_snapshot_shape_and_fps():
    """Snapshot emits the planner stats body shape, only for active stages."""
    board = StatsBoard()
    board.frame("ingest")
    board.frame("ingest")
    board.observe("ingest", "ingestMs", 5.0)
    snap = board.snapshot(elapsed_s=4.0)
    assert len(snap) == 1  # only stages with activity are reported
    body = snap[0]
    assert body["stage"] == "ingest"
    assert body["frames"] == 2
    assert body["fps"] == pytest.approx(0.5)
    assert body["metrics"]["ingestMs"]["count"] == 1


def test_board_rejects_unknown_stage():
    """Stage names are the closed contract set — typos fail loudly."""
    board = StatsBoard()
    with pytest.raises(KeyError):
        board.frame("not-a-stage")


def test_sample_buffer_caps_and_counts_drops():
    """Rows beyond the cap are dropped (that IS the sampling) and counted."""
    buf = SampleBuffer(cap=3)
    for i in range(5):
        buf.add("match", float(i), {"matchCosine": 0.5})
    assert len(buf.rows) == 3
    assert buf.dropped == 2
    rows = buf.drain()
    assert len(rows) == 3
    assert rows[0] == {"stage": "match", "tMs": 0.0, "metrics": {"matchCosine": 0.5}}
    assert buf.rows == []  # drained
    buf.add("match", 9.0, {"matchCosine": 0.1})  # cap applies per flush window
    assert len(buf.rows) == 1
