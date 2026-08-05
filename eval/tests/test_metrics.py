"""Tests for the pure scoring: the funnel, the signed error, and the filters.

Every case here is a claim the harness makes about a run, driven by a
synthetic planner document. No pipeline, no network, no clips.
"""

import pytest
from factories import clip, ingest_max_width_after, planner_run

from eval.metrics import (
    DEFAULT_ENVELOPE,
    ClipScore,
    aggregate,
    check_gate_pass_collapse,
    criticals,
    failing,
    funnel_from_run,
    gate_pass_rate,
    is_scorable,
    score_run,
    unique_count_error,
    warnings,
    within_envelope,
)


def _check(score: ClipScore, name: str):
    """The named check off a score (tests assert on named checks, not order)."""
    return next(c for c in score.checks if c.name == name)


# ------------------------------------------------------------------ funnel


def test_funnel_reads_every_rung_from_the_planner_record():
    """Each rung comes from the stat the runner already emits."""
    funnel = funnel_from_run(
        planner_run(
            frames=400, person_boxes=1200, faces_detected=300, gate_pass=120,
            embeds=118, matches=118, unique=37,
        )
    )
    assert funnel.frames == 400
    assert funnel.person_boxes == 1200
    assert funnel.faces_detected == 300
    assert funnel.faces_passing_gate == 120
    assert funnel.embeds == 118
    assert funnel.matches == 118
    assert funnel.unique == 37


def test_embeds_are_unknown_not_zero_when_the_build_does_not_emit_them():
    """An older build must not be accused of a collapse it did not have."""
    funnel = funnel_from_run(planner_run(embeds=None, include_embed_stage=True))
    assert funnel.embeds is None


def test_embeds_are_zero_when_the_stage_never_ran():
    """No embed stats row at all means the embedder genuinely never ran."""
    funnel = funnel_from_run(planner_run(include_embed_stage=False))
    assert funnel.embeds == 0


def test_frames_fall_back_to_the_run_results_when_ingest_stats_are_missing():
    """A run whose ingest row never flushed still reports its frame count."""
    run = planner_run(frames=120)
    run["stats"] = [s for s in run["stats"] if s["stage"] != "ingest"]
    assert funnel_from_run(run).frames == 120


# ------------------------------------------------------------- the metric


def test_unique_count_error_is_signed_so_deflation_and_inflation_differ():
    """The sign is the whole point: the two failures have opposite causes."""
    assert unique_count_error(90, 100) == pytest.approx(-0.10)
    assert unique_count_error(110, 100) == pytest.approx(0.10)
    assert unique_count_error(100, 100) == 0.0


def test_unique_count_error_refuses_a_zero_denominator():
    """A clip nobody walked through has no relative error to report."""
    with pytest.raises(ValueError, match="denominator"):
        unique_count_error(0, 0)


def test_envelope_is_five_percent_inclusive():
    """The committed envelope is <= 5%, so exactly 5% passes."""
    assert within_envelope(0.05)
    assert within_envelope(-0.05)
    assert not within_envelope(0.0501)
    assert not within_envelope(None)


def test_gate_pass_rate_is_none_when_no_face_was_ever_detected():
    """A rate over zero detections is not zero, it is undefined."""
    assert gate_pass_rate(funnel_from_run(planner_run(faces_detected=0, gate_pass=0))) is None


# ----------------------------------------------------- scorability filter


def test_only_a_source_ended_run_may_be_scored():
    """§5: score only runs that actually finished."""
    assert is_scorable(planner_run(end_reason="source-ended"))[0]


def test_a_stalled_run_is_refused_and_the_reason_says_why():
    """A stalled camera is not a complete count; scoring it fakes a deflation."""
    ok, why = is_scorable(planner_run(end_reason="source-stalled", status="failed"))
    assert not ok
    assert "source-stalled" in why and "deflation" in why


def test_an_operator_stopped_run_is_refused():
    """Stopped early is a partial count against a whole-clip ground truth."""
    ok, why = is_scorable(planner_run(end_reason="operator-stopped"))
    assert not ok
    assert "partial count" in why


def test_a_run_with_no_end_reason_is_refused():
    """No end reason means it never settled — or never said so."""
    ok, why = is_scorable(planner_run(end_reason=None, status="failed"))
    assert not ok
    assert "never settled" in why


def test_score_run_reports_a_stalled_run_as_errored_not_as_a_bad_score():
    """The headline number must be absent, not zero, on an unscorable run."""
    score = score_run(clip(actual_unique=40), planner_run(end_reason="source-stalled", unique=3))
    assert score.status == "errored"
    assert score.verdict == "errored"
    assert score.unique_count_error is None
    assert score.pipeline_unique is None


def test_score_run_errors_when_the_planner_holds_no_results():
    """A settled run with no recorded count has nothing to score."""
    score = score_run(clip(), planner_run(results={}))
    assert score.status == "errored"
    assert "no results.unique" in score.error


# ------------------------------------------------------------------ checks


def test_gate_pass_collapse_is_a_named_critical_check():
    """Faces detected, none passing the gate — the failure this harness exists for."""
    check = check_gate_pass_collapse(funnel_from_run(planner_run(faces_detected=274, gate_pass=0)))
    assert check.name == "gate-pass-collapse"
    assert not check.passed
    assert check.severity == "critical"
    assert "ZERO" in check.detail


def test_gate_pass_collapse_passes_when_faces_survive():
    """The check must not cry wolf on a healthy run."""
    assert check_gate_pass_collapse(funnel_from_run(planner_run(gate_pass=120))).passed


def test_todays_failure_is_loud_on_its_own_even_before_any_comparison():
    """The INGEST_MAX_WIDTH run, scored alone, must fail critically.

    Five times the frame rate and it counted nobody. The gate is where the
    funnel breaks, so the collapse check owns it and the downstream checks
    stay quiet rather than double-reporting one cause as four failures.
    """
    score = score_run(clip(actual_unique=9), ingest_max_width_after())
    assert score.status == "scored"
    assert score.verdict == "fail"
    names = {c.name for c in criticals(score.checks)}
    assert names == {"gate-pass-collapse", "counted-nobody"}
    assert score.funnel.as_dict() == {
        "frames": 1261,
        "personBoxes": 3903,
        "facesDetected": 274,
        "facesPassingGate": 0,
        "embeds": 0,
        "matches": 0,
        "unique": 0,
    }
    assert score.unique_count_error == pytest.approx(-1.0)
    # The faces got smaller, which is the cause and is measured, not inferred.
    assert score.face_size["detected"]["faceBoxWPx"]["mean"] == 48.0
    assert score.face_size["detected"]["faceIedPx"]["mean"] == 23.0
    # ...and the fps that made it look like a win is recorded, unweighted.
    assert score.fps["count"] == 7.89


def test_embed_never_ran_is_critical_when_faces_passed_the_gate():
    """Gate survivors that never reach the embedder are a broken pipeline."""
    score = score_run(
        clip(), planner_run(gate_pass=40, include_embed_stage=False, matches=0, unique=0)
    )
    assert not _check(score, "embed-ran").passed


def test_embed_check_is_skipped_when_the_metric_is_unknown():
    """Unknown is not zero: an older build must not fail this check."""
    score = score_run(clip(), planner_run(embeds=None, gate_pass=40, matches=40))
    assert _check(score, "embed-ran").passed
    assert "unknown" in _check(score, "embed-ran").detail


def test_match_never_ran_is_critical():
    """Embeddings produced but no gallery decision means the count cannot move."""
    score = score_run(clip(), planner_run(gate_pass=40, embeds=40, matches=0, unique=0))
    assert not _check(score, "match-ran").passed


def test_no_frames_is_critical():
    """A run that pulled nothing has nothing to score."""
    score = score_run(clip(), planner_run(frames=0, faces_detected=0, gate_pass=0, unique=0))
    assert not _check(score, "no-frames").passed


def test_no_faces_detected_is_critical():
    """Frames processed and not one face found anywhere."""
    score = score_run(clip(), planner_run(faces_detected=0, gate_pass=0, matches=0, unique=0))
    assert not _check(score, "no-faces-detected").passed


def test_mean_face_size_below_the_floor_warns_without_failing_the_clip():
    """The leading indicator: faces already too small, a few still squeaking through.

    48 px mean against a 56 px floor is exactly what the downscale produced.
    It warns — a warning never fails a clip on its own — so a run that is
    still inside the count envelope is not rejected, only flagged.
    """
    score = score_run(
        clip(actual_unique=10),
        planner_run(unique=10, face_w=(19.0, 48.0, 61.0), gate_pass=4, embeds=4, matches=4),
    )
    assert score.verdict == "pass"
    assert "face-size-vs-floor" in {c.name for c in warnings(score.checks)}


def test_a_low_gate_pass_rate_warns():
    """Counting off 2% of what it sees is worth saying out loud."""
    score = score_run(
        clip(actual_unique=10),
        planner_run(unique=10, faces_detected=1000, gate_pass=20, embeds=20, matches=20),
    )
    assert "gate-pass-rate" in {c.name for c in warnings(score.checks)}
    assert score.verdict == "pass"


def test_an_out_of_envelope_error_fails_the_clip():
    """8% deflation is outside the committed 5% envelope."""
    score = score_run(clip(actual_unique=100), planner_run(unique=92))
    assert score.verdict == "fail"
    assert [c.name for c in failing(score.checks)] == ["unique-count-error"]


def test_a_healthy_run_passes_everything():
    """The control case: nothing fires when the pipeline behaves."""
    score = score_run(clip(actual_unique=100), planner_run(unique=98))
    assert score.verdict == "pass"
    assert failing(score.checks) == []
    assert score.gate["qualityMinPx"] == 56.0


# --------------------------------------------------------------- aggregate


def test_errored_clips_are_counted_and_named_but_never_averaged():
    """An unscorable run must not be able to move an aggregate."""
    good = score_run(clip("a", 100), planner_run(unique=98))
    bad = score_run(clip("b", 100), planner_run(end_reason="source-stalled", unique=1))
    agg = aggregate([good, bad])
    assert agg["scored"] == 1
    assert agg["errored"] == 1
    assert agg["erroredClips"] == ["b"]
    assert agg["meanAbsUniqueCountError"] == pytest.approx(0.02)


def test_aggregate_breaks_down_by_hard_case_tag():
    """§5's hard cases are the reason tags exist at all."""
    surge = score_run(clip("surge", 100, tags=("surge",)), planner_run(unique=80))
    calm = score_run(clip("calm", 100), planner_run(unique=99))
    agg = aggregate([surge, calm])
    assert agg["byTag"]["surge"]["clips"] == 1
    assert agg["byTag"]["surge"]["meanAbsUniqueCountError"] == pytest.approx(0.20)
    assert agg["worstClip"] == "surge"


def test_a_score_round_trips_through_its_stored_form():
    """--resume rehydrates a carried clip identically, checks and all."""
    score = score_run(clip(actual_unique=9), ingest_max_width_after())
    rebuilt = ClipScore.from_dict(score.as_dict())
    assert rebuilt.as_dict() == score.as_dict()
    assert rebuilt.verdict == "fail"
    assert len(criticals(rebuilt.checks)) == len(criticals(score.checks))


def test_the_default_envelope_is_the_committed_five_percent():
    """Pinned so a quiet edit cannot loosen the acceptance test."""
    assert DEFAULT_ENVELOPE == 0.05
