"""The refactor guard rail's own tests: it must catch what it claims to catch."""

import json

import pytest

from eval.golden import first_difference, load, main, report


def write(tmp_path, name, records):
    """Write records as a capture file and return its path."""
    p = tmp_path / name
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    return str(p)


def frames(n=3, **overrides):
    """A short run of plausible frame records."""
    out = []
    for i in range(n):
        out.append({
            "seq": i, "boxes": [{"x": 10 * i, "y": 5, "w": 100, "h": 200}],
            "faces": [], "verdicts": [], "events": [], "ms": {"step": 1.0 + i},
            **overrides,
        })
    return out


def test_identical_runs_are_called_identical(tmp_path):
    """The baseline: same reasoning, clean verdict."""
    a = write(tmp_path, "a.jsonl", frames())
    b = write(tmp_path, "b.jsonl", frames())
    ok, verdict = report(load(a), load(b))
    assert ok
    assert "IDENTICAL" in verdict


def test_timings_alone_are_not_a_difference(tmp_path):
    """A refactor is EXPECTED to change wall-clock; that is usually the point.

    If timings counted, every performance change would fail the guard rail and
    the guard rail would be turned off — which is how safety nets die.
    """
    slow = frames()
    fast = [{**r, "ms": {"step": r["ms"]["step"] / 10}} for r in frames()]
    a, b = write(tmp_path, "a.jsonl", slow), write(tmp_path, "b.jsonl", fast)
    assert report(load(a), load(b))[0], "timings must be ignored by default"
    assert not report(load(a, keep_timings=True), load(b, keep_timings=True))[0], (
        "--with-timings must still be able to see them"
    )


def test_a_changed_verdict_is_caught(tmp_path):
    """The case the whole tool exists for: reasoning moved, totals may not have."""
    before = frames()
    after = frames()
    after[1]["verdicts"] = [{"personKey": "p0002", "isNew": True}]
    a, b = write(tmp_path, "a.jsonl", before), write(tmp_path, "b.jsonl", after)
    ok, verdict = report(load(a), load(b))
    assert not ok
    assert "record 1" in verdict and "verdicts" in verdict


def test_compensating_errors_do_not_pass(tmp_path):
    """Two changes that cancel in the TOTAL are still two changes.

    This is exactly what comparing unique counts would miss: a guest lost on
    one frame and a phantom minted on another sum to the same number and are
    not the same run. The ledger diff sees both.
    """
    before = frames(4)
    before[1]["verdicts"] = [{"personKey": "p0001", "isNew": True}]
    after = frames(4)
    after[3]["verdicts"] = [{"personKey": "p0001", "isNew": True}]
    a, b = write(tmp_path, "a.jsonl", before), write(tmp_path, "b.jsonl", after)
    assert not report(load(a), load(b))[0]


def test_a_shorter_run_is_caught_and_explained(tmp_path):
    """Different frame counts usually mean the sampler, and the message says so."""
    a = write(tmp_path, "a.jsonl", frames(5))
    b = write(tmp_path, "b.jsonl", frames(3))
    ok, verdict = report(load(a), load(b))
    assert not ok
    assert "before=5 after=3" in verdict
    assert "lockstep" in verdict, "the likely cause must be named, not left to guess"


def test_only_the_first_divergence_is_reported(tmp_path):
    """A run that forked at frame 1 must not print a thousand consequences."""
    before, after = frames(6), frames(6)
    for i in (1, 2, 3, 4, 5):
        after[i]["faces"] = [{"box": {"w": 90}}]
    d = first_difference(load(write(tmp_path, "a.jsonl", before)),
                         load(write(tmp_path, "b.jsonl", after)))
    assert d["index"] == 1, "the first fork is the one to debug"


def test_a_malformed_capture_is_fatal_not_skipped(tmp_path):
    """A capture that cannot be read fully cannot support 'nothing changed'."""
    p = tmp_path / "bad.jsonl"
    p.write_text('{"seq": 0}\nnot json at all\n')
    with pytest.raises(SystemExit):
        load(str(p))


def test_an_empty_capture_is_refused_not_passed(tmp_path):
    """Comparing nothing to nothing must never read as success."""
    a = write(tmp_path, "a.jsonl", [])
    b = write(tmp_path, "b.jsonl", frames())
    assert main([a, b]) == 2


def test_cli_exit_codes(tmp_path):
    """0 identical, 1 different — so CI and a shell `&&` both work."""
    same_a = write(tmp_path, "a.jsonl", frames())
    same_b = write(tmp_path, "b.jsonl", frames())
    diff = write(tmp_path, "c.jsonl", frames(2))
    assert main([same_a, same_b]) == 0
    assert main([same_a, diff]) == 1


def test_a_wall_clock_stamp_is_not_a_decision(tmp_path):
    """tMs must be ignored, and the reason is a measurement, not a preference.

    The tool's first real use diffed two runs of one 60 s clip on the T440:
    1802 frames each, every box, face, gate outcome and verdict identical —
    and it reported DIFFERENT, because frame 1 was read 5 ms later in one run
    than the other. On a file replay tMs is pure scheduling noise. A guard
    rail that fires on that gets switched off within a day, so it does not.
    """
    a = write(tmp_path, "a.jsonl", [{"seq": 1, "tMs": 15, "verdicts": []}])
    b = write(tmp_path, "b.jsonl", [{"seq": 1, "tMs": 10, "verdicts": []}])
    assert report(load(a), load(b))[0], "tMs alone must not be a difference"


def test_a_missing_or_reordered_frame_is_still_caught(tmp_path):
    """seq is NOT a timing key: losing or reordering frames is a real finding."""
    a = write(tmp_path, "a.jsonl", [{"seq": 1, "tMs": 0}, {"seq": 2, "tMs": 5}])
    b = write(tmp_path, "b.jsonl", [{"seq": 2, "tMs": 0}, {"seq": 1, "tMs": 5}])
    assert not report(load(a), load(b))[0], "frame identity must still be compared"


def test_a_missing_capture_says_so_instead_of_traceback(tmp_path):
    """The most common failure deserves a sentence, not a stack trace.

    This tool runs from a shell in the middle of a deploy, and "the capture is
    not where you think it is" is its most frequent outcome — usually a
    docker cp against a container that was recreated underneath it.
    """
    with pytest.raises(SystemExit) as e:
        load(str(tmp_path / "nope.jsonl"))
    assert "cannot read capture" in str(e.value)
