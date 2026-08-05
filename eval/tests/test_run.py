"""Tests for the orchestration: settle, refuse, never touch someone else's run.

Driven entirely by fake clients — no runner, no planner, no pipeline. What is
under test is the DECISION path: when a run counts as settled, when it is an
error rather than a score, and what the harness is allowed to stop.
"""

import logging

from factories import clip, ingest_max_width_after, planner_run

from eval.clients import EvalHttpError
from eval.manifest import Manifest
from eval.metrics import score_run
from eval.run import ClipRunner, EvalConfig, build_payload, evaluate, exit_code, slug

LOG = logging.getLogger("eval-test")
CFG = EvalConfig(label="unit", event_id="evt-1", clip_timeout_s=30.0, poll_s=0.0,
                 settle_grace_s=0.0)


class FakeRunner:
    """Scripted runner: a list of statuses, and a record of what was asked."""

    def __init__(self, states, start_error=None):
        """``states`` are returned by successive ``status`` calls."""
        self.states = list(states)
        self.start_error = start_error
        self.started: list[dict] = []
        self.stopped: list[str] = []

    def start_run(self, event_id, path, label, site_id=None, placement_id=None):
        """Record the start request and hand back a run id."""
        if self.start_error:
            raise EvalHttpError(self.start_error)
        self.started.append(
            {"eventId": event_id, "path": path, "label": label, "siteId": site_id}
        )
        return "run-abc123"

    def status(self, run_id):
        """Return the next scripted status (repeating the last one forever)."""
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def stop(self, run_id):
        """Record a stop request."""
        self.stopped.append(run_id)
        return True


class FakePlanner:
    """Serves one run document, or raises if asked for anything else."""

    def __init__(self, record):
        """Bind the document this planner will answer with."""
        self.record = record
        self.asked: list[str] = []

    def get_run(self, run_id):
        """Record the id asked for and return the document."""
        self.asked.append(run_id)
        if isinstance(self.record, Exception):
            raise self.record
        return self.record


def make(states, record):
    """A ClipRunner over fake clients with an instant clock."""
    ticks = iter(range(10_000))
    runner = FakeRunner(states)
    planner = FakePlanner(record)
    cr = ClipRunner(
        runner, planner, sleep=lambda _s: None, monotonic=lambda: next(ticks), log=LOG
    )
    return cr, runner, planner


SETTLED = {"state": "ended", "plannerRunId": "pr-1", "endReason": "source-ended"}


def test_a_finished_run_is_scored_and_carries_its_run_ids():
    """The traceability requirement: every number points back at a planner row."""
    cr, runner, planner = make(
        [{"state": "running", "plannerRunId": "pr-1"}, SETTLED], planner_run(unique=98)
    )
    score = cr.run_clip(clip("calm", 100), CFG)
    assert score.status == "scored"
    assert score.run_ids == {"runner": "run-abc123", "planner": "pr-1"}
    assert planner.asked == ["pr-1"]
    assert runner.stopped == [], "a run that settled by itself must never be stopped"


def test_the_run_is_started_against_a_file_source_under_an_eval_label():
    """An eval run must be identifiable in the planner's run list at a glance."""
    cr, runner, _ = make([SETTLED], planner_run())
    cr.run_clip(clip("baraat-surge", 60), CFG)
    assert runner.started[0]["path"] == "/srv/heco/clips/baraat-surge.mp4"
    assert runner.started[0]["label"] == "eval/unit/baraat-surge"
    assert runner.started[0]["eventId"] == "evt-1"


def test_a_stalled_run_is_errored_not_scored():
    """§5 again, this time end to end through the orchestration."""
    cr, _, _ = make([SETTLED], planner_run(end_reason="source-stalled", unique=2))
    score = cr.run_clip(clip("calm", 100), CFG)
    assert score.status == "errored"
    assert "source-stalled" in score.error
    assert score.unique_count_error is None


def test_a_run_that_never_settles_is_stopped_and_errored():
    """A wedged run is bounded, and the partial count is refused."""
    cr, runner, _ = make([{"state": "running", "plannerRunId": "pr-1"}], planner_run())
    score = cr.run_clip(clip("calm", 100), EvalConfig(
        label="unit", event_id="evt-1", clip_timeout_s=3.0, poll_s=0.0
    ))
    assert score.status == "errored"
    assert "did not settle" in score.error
    assert runner.stopped == ["run-abc123"]


def test_the_harness_only_ever_stops_runs_it_started():
    """The whole protection against touching a live production run."""
    cr, runner, _ = make([{"state": "running", "plannerRunId": "pr-1"}], planner_run())
    cr.run_clip(clip("calm", 100), EvalConfig(
        label="unit", event_id="evt-1", clip_timeout_s=3.0, poll_s=0.0
    ))
    assert set(runner.stopped) <= set(cr.started)
    assert cr.started == ["run-abc123"]


def test_a_run_that_never_reached_the_planner_is_errored():
    """No durable record means nothing to score, however the loop felt about it."""
    cr, _, _ = make([{"state": "failed", "error": "ingest refused", "plannerRunId": None}], {})
    score = cr.run_clip(clip("calm", 100), CFG)
    assert score.status == "errored"
    assert "nothing durable to score" in score.error


def test_a_failed_start_is_errored_and_nothing_is_started():
    """A refusal from the runner (a live sibling on the camera) is an error."""
    runner = FakeRunner([], start_error="capture slot is held by live run 'run-9'")
    cr = ClipRunner(runner, FakePlanner({}), sleep=lambda _s: None, log=LOG)
    score = cr.run_clip(clip("calm", 100), CFG)
    assert score.status == "errored"
    assert "never started" in score.error
    assert cr.started == []
    assert runner.stopped == []


def test_an_unreadable_planner_record_is_errored():
    """A planner outage must not be scored as a zero count."""
    cr, _, _ = make([SETTLED], EvalHttpError("HTTP 500"))
    score = cr.run_clip(clip("calm", 100), CFG)
    assert score.status == "errored"
    assert "could not be read" in score.error


def test_the_end_reason_falls_back_to_the_runner_and_says_so():
    """When the planner's final PUT is lost, the score records whose word it rests on."""
    record = planner_run(end_reason=None, unique=98)
    cr, _, _ = make([SETTLED], record)
    score = cr.run_clip(clip("calm", 100), CFG)
    assert score.status == "scored"
    assert score.end_reason_source == "runner-status"
    assert score.end_reason == "source-ended"


# ------------------------------------------------------------ the batch


def manifest_of(*clips) -> Manifest:
    """A Manifest around ready-made clips (the loader is tested elsewhere)."""
    return Manifest(clips=tuple(clips), event_id="evt-1", source_path="/tmp/m.json")


def test_resume_carries_scored_clips_forward_without_re_running_them():
    """Clip twelve of fourteen dying at midnight must not cost the first eleven."""
    a, b = clip("a", 100), clip("b", 100)
    done = score_run(a, planner_run(unique=98))
    previous = {"clips": [done.as_dict()]}
    cr, runner, _ = make([SETTLED], planner_run(unique=97))
    scores = evaluate(manifest_of(a, b), CFG, cr, previous)
    assert [s.clip for s in scores] == ["a", "b"]
    assert [c["path"] for c in runner.started] == ["/srv/heco/clips/b.mp4"]
    assert scores[0].as_dict() == done.as_dict(), "a carried clip is unchanged"


def test_resume_re_runs_a_clip_that_errored_last_time():
    """An error is usually a wedged service — exactly the thing worth retrying."""
    a = clip("a", 100)
    failed = score_run(a, planner_run(end_reason="source-stalled"))
    cr, runner, _ = make([SETTLED], planner_run(unique=99))
    scores = evaluate(manifest_of(a), CFG, cr, {"clips": [failed.as_dict()]})
    assert scores[0].status == "scored"
    assert len(runner.started) == 1


def test_the_payload_records_everything_needed_to_re_read_it_later():
    """The machine-readable file is the A/B's input and the audit trail."""
    a = clip("a", 100)
    cr, _, _ = make([SETTLED], planner_run(unique=98))
    scores = evaluate(manifest_of(a), CFG, cr, None)
    payload = build_payload(manifest_of(a), CFG, scores, "T0", "T1")
    assert payload["label"] == "unit"
    assert payload["envelope"] == 0.05
    assert payload["manifest"]["clipCount"] == 1
    assert payload["clips"][0]["runIds"]["planner"] == "pr-1"
    assert payload["aggregate"]["scored"] == 1
    assert exit_code(payload) == 0


def test_the_exit_code_is_non_zero_when_a_clip_fails_or_errors():
    """So a commit can be gated on it."""
    a = clip("a", 9)
    cr, _, _ = make([SETTLED], ingest_max_width_after())
    scores = evaluate(manifest_of(a), CFG, cr, None)
    payload = build_payload(manifest_of(a), CFG, scores, "T0", "T1")
    assert payload["aggregate"]["failed"] == 1
    assert exit_code(payload) == 2


def test_result_files_are_named_from_the_label_so_a_re_run_replaces_itself():
    """Idempotence: the same label writes the same file, never a second one."""
    assert slug("Baseline 2026-08-06") == "baseline-2026-08-06"
    assert slug("!!!") == "eval"
