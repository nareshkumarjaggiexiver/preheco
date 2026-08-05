"""Tests for the A/B guard rail — including today's failure, from real numbers.

The headline test is :func:`test_todays_ingest_max_width_change_is_called_a_regression`.
It reproduces the measured before/after of setting ``INGEST_MAX_WIDTH=1920`` on
the real server and asserts the harness refuses to call a five-fold throughput
win an improvement when the pipeline stopped counting people.
"""

import json

import pytest
from factories import clip, ingest_max_width_after, ingest_max_width_before, planner_run

from eval.compare import compare_results
from eval.compare import main as compare_main
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


def test_a_clip_added_only_in_the_after_set_is_a_note_not_a_verdict():
    """New ground truth is welcome; it just cannot be a delta."""
    a = clip("a", 100)
    before = results("b", [score_run(a, planner_run(unique=99))])
    after = results(
        "a",
        [score_run(a, planner_run(unique=99)), score_run(clip("new", 100), planner_run(unique=99))],
    )
    comparison = compare_results(before, after)
    assert comparison.verdict == "neutral"
    assert any("AFTER set only" in n for n in comparison.notes)


# ------------------------------- a deleted clip is not a way to pass (flaw 1)


def test_deleting_the_failing_clip_from_the_after_set_is_a_regression():
    """Otherwise `git rm the-clip-that-fails` is a valid way to pass the A/B.

    Before: two clips, one of them counting badly. After: that clip is simply
    not there. Everything remaining looks fine, and the aggregate improves. If
    the harness calls that anything but a regression it is a guard rail with a
    hole, and a guard rail with a hole is worse than none because it is
    believed.
    """
    calm, bad = clip("calm", 100), clip("baraat-surge", 100, tags=("surge",))
    before = results(
        "b", [score_run(calm, planner_run(unique=99)), score_run(bad, planner_run(unique=60))]
    )
    after = results("a", [score_run(calm, planner_run(unique=100))])

    comparison = compare_results(before, after)

    assert comparison.verdict == "regression"
    assert comparison.is_regression
    joined = " | ".join(comparison.reasons)
    assert "baraat-surge" in joined
    assert "ABSENT from the AFTER set" in joined
    assert comparison.aggregate["clipsMissingFromAfter"] == ["baraat-surge"]


def test_deleting_every_clip_is_a_regression_and_not_an_empty_pass():
    """The extreme case: an AFTER set with nothing in it must never exit clean."""
    before = results("b", [score_run(clip("calm", 100), planner_run(unique=99))])
    comparison = compare_results(before, results("a", []))
    assert comparison.verdict == "regression"


def test_a_missing_clip_can_be_waived_but_the_waiver_is_on_the_record():
    """A clip genuinely retired is allowed — deliberately, in writing, by name."""
    calm, gone = clip("calm", 100), clip("retired", 100)
    before = results(
        "b", [score_run(calm, planner_run(unique=99)), score_run(gone, planner_run(unique=99))]
    )
    after = results("a", [score_run(calm, planner_run(unique=99))])

    comparison = compare_results(before, after, waive_missing=["retired"])

    assert comparison.verdict == "neutral"
    assert comparison.aggregate["clipsMissingFromAfter"] == []
    assert comparison.aggregate["clipsMissingWaived"] == ["retired"]
    assert any("WAIVED by hand" in n for n in comparison.notes)


def test_a_waiver_names_one_clip_and_does_not_cover_the_next_one():
    """A waiver is a statement about a clip, not a switch that turns the rule off."""
    calm, gone, alsogone = clip("calm", 100), clip("retired", 100), clip("baraat-surge", 100)
    before = results(
        "b",
        [
            score_run(calm, planner_run(unique=99)),
            score_run(gone, planner_run(unique=99)),
            score_run(alsogone, planner_run(unique=60)),
        ],
    )
    after = results("a", [score_run(calm, planner_run(unique=99))])
    comparison = compare_results(before, after, waive_missing=["retired"])
    assert comparison.verdict == "regression"
    assert comparison.aggregate["clipsMissingFromAfter"] == ["baraat-surge"]


def test_redefining_the_ground_truth_between_the_two_runs_is_not_comparable():
    """Keeping the clip and moving the goalposts must not be softer than deleting it.

    Both errors are relative to ``actualUnique``. Recount the clip 100 -> 60
    and the same pipeline count reads -20% before and +33% after with nothing
    in the pipeline having changed at all.
    """
    before = results("b", [score_run(clip("calm", 100), planner_run(unique=80))])
    after = results("a", [score_run(clip("calm", 60), planner_run(unique=80))])
    comparison = compare_results(before, after)
    assert comparison.verdict == "regression"
    assert any("count of record changed" in r for r in comparison.reasons)


# --------------------------- a rate cannot see its own denominator (flaw 2)


def test_a_gate_collapse_hidden_by_a_steady_pass_rate_is_still_a_regression():
    """The rate is a RATIO, and a ratio is invariant to its own denominator.

    Detection falls 400 -> 40 per run and the gate passes 200 -> 20: the
    pass rate reads 50% both times, the count survives on this clip, and nine
    tenths of the evidence has silently gone. Neither "gate count is zero" nor
    "the rate halved" fires, which is precisely why they are not enough.
    """
    c = clip("calm", 100)
    before = results(
        "b",
        [score_run(c, planner_run(unique=100, frames=400, faces_detected=400, gate_pass=200,
                                  embeds=200, matches=200))],
    )
    after = results(
        "a",
        [score_run(c, planner_run(unique=100, frames=400, faces_detected=40, gate_pass=20,
                                  embeds=20, matches=20))],
    )

    comparison = compare_results(before, after)

    assert comparison.clips[0].gate_rate_before == comparison.clips[0].gate_rate_after
    assert comparison.verdict == "regression"
    joined = " | ".join(comparison.reasons)
    assert "faces per frame" in joined
    assert "faces DETECTED fell" in joined


def test_the_funnel_rules_are_read_per_frame_so_a_shorter_run_is_not_a_regression():
    """Raw rung counts are not comparable: the change under test moves the frame count.

    This is the 2026-08-05 pair the RIGHT way round — reverting the downscale
    gives a fifth of the frames and a fifth of the faces. Yield per frame is
    unchanged, the gate recovers, and the harness must not shout at the fix.
    """
    c = clip("gate-a-arrivals", 9)
    comparison = compare_results(
        results("b", [score_run(c, ingest_max_width_after())]),
        results("a", [score_run(c, ingest_max_width_before())]),
    )
    assert comparison.verdict == "improvement"
    assert not any("per frame" in r for r in comparison.reasons)


# -------------------------- a warning must be able to reach the verdict (flaw 3)


def test_a_new_warning_sinks_an_improvement_claim():
    """face-size-vs-floor is the LEADING indicator and it is only a warning.

    48 px mean detected width against a 56 px floor fired before the count
    collapsed. Here the unique count genuinely improves while that warning
    starts firing: the honest verdict is inconclusive, not "improvement".
    """
    c = clip("calm", 100)
    before = results("b", [score_run(c, planner_run(unique=80))])
    after = results("a", [score_run(c, planner_run(unique=99, face_w=(19.0, 48.0, 55.0)))])

    comparison = compare_results(before, after)

    assert comparison.verdict == "inconclusive"
    assert not comparison.is_regression
    assert any("face-size-vs-floor" in w for w in comparison.concerns)
    assert any("warning-level check(s) newly failed" in r for r in comparison.reasons)
    assert "CONCERN" in render_comparison(comparison)


def test_a_warning_that_was_already_failing_does_not_sink_anything():
    """Only a NEW warning moves the verdict — otherwise nothing could ever improve."""
    c = clip("calm", 100)
    small = {"face_w": (19.0, 48.0, 55.0)}
    before = results("b", [score_run(c, planner_run(unique=80, **small))])
    after = results("a", [score_run(c, planner_run(unique=99, **small))])
    comparison = compare_results(before, after)
    assert comparison.verdict == "improvement"
    assert comparison.concerns == ()


def test_a_newly_failing_envelope_check_is_a_regression_not_a_warning():
    """Severity 'fail' blocks as hard as 'critical' — metrics.failing treats them alike."""
    c = clip("calm", 100)
    before = results("b", [score_run(c, planner_run(unique=99))])
    after = results("a", [score_run(c, planner_run(unique=80))])
    comparison = compare_results(before, after)
    assert comparison.verdict == "regression"
    assert any("check 'unique-count-error' now fails" in r for r in comparison.reasons)


# ------------------------------------------------------------- exit codes


def test_the_exit_codes_distinguish_a_pass_from_a_verdict_nobody_can_check(tmp_path):
    """Inconclusive must not exit 0: a claim that cannot be checked is not a pass."""
    c = clip("calm", 100)
    b_path, a_path = tmp_path / "b.json", tmp_path / "a.json"
    b_path.write_text(json.dumps(results("b", [score_run(c, planner_run(unique=80))])))
    a_path.write_text(
        json.dumps(
            results("a", [score_run(c, planner_run(unique=99, face_w=(19.0, 48.0, 55.0)))])
        )
    )
    assert compare_main([str(b_path), str(a_path)]) == 3

    a_path.write_text(json.dumps(results("a", [score_run(c, planner_run(unique=99))])))
    assert compare_main([str(b_path), str(a_path)]) == 0

    a_path.write_text(json.dumps(results("a", [score_run(c, planner_run(unique=40))])))
    assert compare_main([str(b_path), str(a_path)]) == 2


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
