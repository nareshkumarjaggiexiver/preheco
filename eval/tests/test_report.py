"""Tests for the text report — mainly that a collapse cannot be missed."""

from factories import clip, ingest_max_width_after, planner_run

from eval.metrics import aggregate, score_run
from eval.report import collapse_banner, render_results


def payload(scores, label="unit") -> dict:
    """Minimal result payload for the renderer."""
    return {
        "label": label,
        "startedAt": "T0",
        "endedAt": "T1",
        "eventId": "evt-1",
        "envelope": 0.05,
        "manifest": {"path": "/tmp/m.json", "clipCount": len(scores)},
        "clips": [s.as_dict() for s in scores],
        "aggregate": aggregate(list(scores)),
    }


def test_a_collapse_is_a_banner_at_the_top_not_a_cell_in_a_table():
    """The failure looked like a win in the one column being read. Never again."""
    doc = payload([score_run(clip("gate-a", 9), ingest_max_width_after())])
    text = render_results(doc)
    banner = text.split("HECO eval")[0]
    assert "CRITICAL" in banner
    assert "gate-pass-collapse" in banner
    assert "No throughput number redeems any of the above." in banner


def test_a_healthy_run_gets_no_banner():
    """The banner must mean something, so it cannot fire on a good run."""
    doc = payload([score_run(clip("calm", 100), planner_run(unique=98))])
    assert collapse_banner(doc) == []
    assert render_results(doc).startswith("=")


def test_the_report_shows_the_funnel_the_sizes_and_the_fps_caveat():
    """Everything §5 asks to be reported per clip appears in the text."""
    text = render_results(payload([score_run(clip("calm", 100), planner_run(unique=98))]))
    assert "person boxes" in text and "UNIQUE" in text
    assert "face IED" in text
    assert "reported, never scored" in text
    assert "endReason=source-ended" in text


def test_an_errored_clip_renders_without_inventing_a_score():
    """An unscorable clip prints its reason and no numbers at all."""
    score = score_run(clip("calm", 100), planner_run(end_reason="source-stalled", unique=3))
    text = render_results(payload([score]))
    assert "NOT SCORED" in text
    assert "source-stalled" in text
    assert "COUNT " not in text


def test_the_aggregate_names_the_worst_clip_and_the_unscorable_ones():
    """A batch summary that hides its failures is worse than none."""
    scores = [
        score_run(clip("calm", 100), planner_run(unique=99)),
        score_run(clip("surge", 100, tags=("surge",)), planner_run(unique=70)),
        score_run(clip("veil", 100), planner_run(end_reason="source-stalled")),
    ]
    text = render_results(payload(scores))
    assert "worst clip" in text and "surge" in text
    assert "NOT SCORED" in text
    assert "tag surge" in text
