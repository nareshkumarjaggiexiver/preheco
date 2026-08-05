"""Tests for the A/B guard rail — including today's failure, from real numbers.

The headline test is :func:`test_todays_ingest_max_width_change_is_called_a_regression`.
It reproduces the measured before/after of setting ``INGEST_MAX_WIDTH=1920`` on
the real server and asserts the harness refuses to call a five-fold throughput
win an improvement when the pipeline stopped counting people.
"""

import pytest
from factories import clip, ingest_max_width_after, ingest_max_width_before, planner_run

from eval.compare import compare_results
from eval.metrics import aggregate, score_run
from eval.report import render_comparison


def results(label: str, scores) -> dict:
    """Wrap clip scores in the result-file shape the comparison reads."""
    return {
        "harness": "heco-eval",
        "kind": "results",
        "label": label,
        "clips": [s.as_dict() for s in scores],
        "aggregate": aggregate(list(scores)),
    }


# ------------------------------------------------------- TODAY'S FAILURE


def test_todays_ingest_max_width_change_is_called_a_regression():
    """INGEST_MAX_WIDTH=1920: 1.59 -> 7.89 fps, and it counted nobody.

    Measured on the real server against the real camera: mean detected face
    width 129 -> 48 px, IED 46 -> 23 px, faces passing the 56 px gate 9 -> 0,
    embed and match never ran, unique 9 -> 0. The only number anyone was
    watching improved five-fold. The harness must call this a regression and
    say why.
    """
    ground_truth = clip("gate-a-arrivals", actual_unique=9)
    before = results("before", [score_run(ground_truth, ingest_max_width_before())])
    after = results("after", [score_run(ground_truth, ingest_max_width_after())])

    comparison = compare_results(before, after)

    assert comparison.verdict == "regression"
    assert comparison.is_regression
    joined = " | ".join(comparison.reasons)
    assert "GATE-PASS COLLAPSE" in joined
    assert "counted 9 people before and NOBODY now" in joined
    assert "unique-count error worsened" in joined

    # The fps improvement is real, recorded, and powerless.
    delta = comparison.clips[0]
    assert delta.fps_before == pytest.approx(1.59)
    assert delta.fps_after == pytest.approx(7.89)
    assert comparison.aggregate["meanFpsAfter"] > comparison.aggregate["meanFpsBefore"]
    assert comparison.aggregate["fpsIsNotScored"] is True
    assert any("does not change the verdict" in n for n in comparison.notes)

    # And it is impossible to miss when printed.
    text = render_comparison(comparison)
    assert "REGRESSION" in text
    assert "DO NOT SHIP" in text


def test_the_same_change_scored_alone_already_fails_before_any_comparison():
    """A single run of the broken config must fail on its own merits too."""
    score = score_run(clip("gate-a-arrivals", 9), ingest_max_width_after())
    assert score.verdict == "fail"
    assert score.funnel.faces_passing_gate == 0
    assert score.funnel.faces_detected == 274


# ------------------------------------------------------------- the rules


def test_faster_but_no_more_accurate_is_never_an_improvement():
    """Throughput alone cannot buy the word 'improvement'."""
    c = clip("calm", 100)
    before = results("b", [score_run(c, planner_run(unique=98, fps_count=2.0, fps_ingest=2.0))])
    after = results("a", [score_run(c, planner_run(unique=98, fps_count=9.0, fps_ingest=9.0))])
    comparison = compare_results(before, after)
    assert comparison.verdict == "neutral"


def test_a_real_accuracy_gain_is_called_an_improvement():
    """The harness is not merely a refusal machine."""
    c = clip("calm", 100)
    before = results("b", [score_run(c, planner_run(unique=80))])
    after = results("a", [score_run(c, planner_run(unique=99))])
    comparison = compare_results(before, after)
    assert comparison.verdict == "improvement"
    assert any("improved" in r for r in comparison.reasons)


def test_a_worse_unique_count_error_is_a_regression():
    """The headline moving the wrong way is enough on its own."""
    c = clip("calm", 100)
    before = results("b", [score_run(c, planner_run(unique=99))])
    after = results("a", [score_run(c, planner_run(unique=88))])
    assert compare_results(before, after).verdict == "regression"


def test_a_halved_gate_pass_rate_is_a_regression_even_with_the_count_intact():
    """Half the evidence silently vanishing is the same failure, earlier.

    The count survives this clip, but it now rests on a quarter of the faces —
    the next clip is where it breaks. The rule fires before the count does.
    """
    c = clip("calm", 100)
    before = results(
        "b", [score_run(c, planner_run(unique=100, faces_detected=400, gate_pass=200,
                                       embeds=200, matches=200))]
    )
    after = results(
        "a", [score_run(c, planner_run(unique=100, faces_detected=400, gate_pass=40,
                                       embeds=40, matches=40))]
    )
    comparison = compare_results(before, after)
    assert comparison.verdict == "regression"
    assert any("gate-pass rate fell" in r for r in comparison.reasons)


def test_turning_a_scored_clip_into_an_unscorable_one_is_a_regression():
    """A change that makes runs stall is a regression, not a neutral outcome."""
    c = clip("calm", 100)
    before = results("b", [score_run(c, planner_run(unique=99))])
    after = results("a", [score_run(c, planner_run(end_reason="source-stalled", unique=99))])
    comparison = compare_results(before, after)
    assert comparison.verdict == "regression"
    assert any("was scorable before" in r for r in comparison.reasons)


def test_one_wrecked_clip_outvotes_an_improved_aggregate():
    """§5: a change that helps the aggregate and wrecks the surge is a conversation.

    Three calm clips get better, the baraat surge gets worse. The mean would
    say 'improvement'; the harness says regression and names the clip.
    """
    calm = [clip(f"calm-{i}", 100) for i in range(3)]
    surge = clip("baraat-surge", 100, tags=("surge",))
    before = results(
        "b",
        [score_run(c, planner_run(unique=80)) for c in calm]
        + [score_run(surge, planner_run(unique=99))],
    )
    after = results(
        "a",
        [score_run(c, planner_run(unique=100)) for c in calm]
        + [score_run(surge, planner_run(unique=60))],
    )
    comparison = compare_results(before, after)
    assert comparison.aggregate["meanAbsUniqueCountErrorAfter"] < (
        comparison.aggregate["meanAbsUniqueCountErrorBefore"]
    )
    assert comparison.verdict == "regression"
    assert any(r.startswith("baraat-surge") for r in comparison.reasons)


def test_a_new_critical_check_is_a_regression_on_its_own():
    """Any critical that was passing and now fails stops the change."""
    c = clip("calm", 100)
    before = results("b", [score_run(c, planner_run(unique=100, matches=120, embeds=120))])
    after = results(
        "a", [score_run(c, planner_run(unique=100, gate_pass=120, embeds=120, matches=0))]
    )
    comparison = compare_results(before, after)
    assert comparison.verdict == "regression"
    assert any("match-ran" in r for r in comparison.reasons)


def test_unpaired_clips_are_noted_and_not_compared():
    """A clip missing from one side cannot be a delta; say so, do not guess."""
    before = results("b", [score_run(clip("a", 100), planner_run(unique=99))])
    after = results("a", [score_run(clip("b", 100), planner_run(unique=99))])
    comparison = compare_results(before, after)
    assert comparison.verdict == "inconclusive"
    assert any("BEFORE set only" in n for n in comparison.notes)
    assert any("AFTER set only" in n for n in comparison.notes)


def test_tiny_movements_inside_the_tolerance_claim_nothing():
    """Below the tolerance a difference is clip noise, not evidence."""
    c = clip("calm", 1000)
    before = results("b", [score_run(c, planner_run(unique=998))])
    after = results("a", [score_run(c, planner_run(unique=999))])
    assert compare_results(before, after).verdict == "neutral"


def test_the_comparison_serialises_for_the_record():
    """A verdict has to be storable next to the results it was drawn from."""
    c = clip("calm", 100)
    comparison = compare_results(
        results("b", [score_run(c, planner_run(unique=99))]),
        results("a", [score_run(c, planner_run(unique=70))]),
    )
    doc = comparison.as_dict()
    assert doc["verdict"] == "regression"
    assert doc["clips"][0]["clip"] == "calm"
    assert doc["aggregate"]["fpsIsNotScored"] is True
