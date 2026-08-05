"""A/B comparison — the guard rail. PURE verdict logic plus a thin CLI.

WHY THIS IS WRITTEN DEFENSIVELY. It is protecting the person running it from
themselves. Every tuning change in the accuracy report is a hypothesis, and
the temptation when the profiler is open is to accept whatever made the loop
faster. That is exactly how ``INGEST_MAX_WIDTH=1920`` got set: 1.59 -> 7.89
fps, a five-fold win on the only number being watched, while faces fell below
the quality floor, the gate passed nothing, and the pipeline silently counted
zero people.

So this module has one rule it will not bend:

    **fps NEVER contributes to the verdict.** It is printed, and it is
    printed next to the accuracy numbers, and that is all it does.

A change is refused the word "improvement" if, on any clip:

* the gate-pass count collapsed to zero, the pass RATE fell by half, or the
  gate-pass or face-detection YIELD PER FRAME fell by half,
* the signed unique-count error got worse by more than the tolerance,
* the count fell to zero where it previously counted people,
* a check of blocking severity (``critical`` or ``fail``) that used to pass
  now fails,
* a clip that used to be scorable is now unscorable (a stalled or failed run
  is an ERROR, and a change that turns scores into errors is a regression, not
  a neutral outcome),
* or a clip that was measured BEFORE is simply absent AFTER — see below.

A WARNING that was not failing before does not on its own prove harm, but it
is not nothing either: it downgrades the verdict to ``inconclusive``, so a
change can never be called an improvement over the top of a new warning. The
leading indicator of the 2026-08-05 collapse (``face-size-vs-floor``, mean
detected width 48 px against a 56 px floor) is a warning, and it fired BEFORE
the count went to zero. A guard rail that cannot hear it is decorative.

A MISSING CLIP IS A REGRESSION. If a clip is in the BEFORE set and absent from
the AFTER set, the honest reading is not "not compared" — it is "the evidence
that this change hurt was removed". Deleting the failing clip from the manifest
must not be a way to turn a regression into a pass, so it is a regression
unless waived by hand with ``--waive-missing LABEL``, which leaves the waiver
in the printed record and in the JSON.

Usage::

    python -m eval.compare before.json after.json [--tolerance 0.005] [--json out.json]

Exit codes: 0 improvement or neutral, 2 regression, 3 inconclusive (nothing
comparable, or a new warning — a claim that cannot be checked is not a pass),
1 on a usage error. So it can gate a commit.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: How much the absolute unique-count error must move before it is called a
#: change at all. Half a percentage point: below that a difference is clip
#: noise, and calling it either way would be false precision.
DEFAULT_TOLERANCE = 0.005

#: Relative fall in a funnel rung that counts as a collapse even when it is not
#: all the way to zero. Half the faces silently stopping is the same failure as
#: all of them, just earlier. Applied to the gate-pass RATE and to the per-frame
#: YIELD of the detect and gate rungs (see :func:`_funnel_yield`).
DEFAULT_GATE_RATE_DROP = 0.5

#: Check severities that stop a change dead. ``fail`` is in here as well as
#: ``critical`` because :func:`eval.metrics.failing` treats them alike — the
#: envelope check is ``fail``, and a clip falling out of the committed envelope
#: is not advisory.
BLOCKING_SEVERITIES = frozenset({"critical", "fail"})

#: Check severities that cannot condemn a change on their own but can stop it
#: being called an improvement. See the module docstring on why a warning has
#: to be able to reach the verdict at all.
ADVISORY_SEVERITIES = frozenset({"warn"})


@dataclass(frozen=True)
class ClipDelta:
    """Before/after for one clip, plus why it is (or is not) a regression."""

    clip: str
    status_before: str
    status_after: str
    error_before: float | None = None
    error_after: float | None = None
    unique_before: int | None = None
    unique_after: int | None = None
    actual_unique: int | None = None
    gate_pass_before: int | None = None
    gate_pass_after: int | None = None
    gate_rate_before: float | None = None
    gate_rate_after: float | None = None
    fps_before: float | None = None
    fps_after: float | None = None
    regressions: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()
    #: Things that got worse WITHOUT proving harm — warning-severity checks
    #: that newly fail. They never make a regression on their own; they make an
    #: improvement claim inconclusive, which is the honest verdict.
    concerns: tuple[str, ...] = ()

    @property
    def abs_error_delta(self) -> float | None:
        """Change in |unique-count error|; negative is better."""
        if self.error_before is None or self.error_after is None:
            return None
        return abs(self.error_after) - abs(self.error_before)

    def as_dict(self) -> dict:
        """JSON shape used in a comparison file."""
        return {
            "clip": self.clip,
            "statusBefore": self.status_before,
            "statusAfter": self.status_after,
            "actualUnique": self.actual_unique,
            "uniqueBefore": self.unique_before,
            "uniqueAfter": self.unique_after,
            "errorBefore": self.error_before,
            "errorAfter": self.error_after,
            "absErrorDelta": self.abs_error_delta,
            "gatePassBefore": self.gate_pass_before,
            "gatePassAfter": self.gate_pass_after,
            "gateRateBefore": self.gate_rate_before,
            "gateRateAfter": self.gate_rate_after,
            "fpsBefore": self.fps_before,
            "fpsAfter": self.fps_after,
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "concerns": list(self.concerns),
        }


@dataclass(frozen=True)
class Comparison:
    """The whole A/B: per-clip deltas, aggregate movement, and one verdict."""

    verdict: str  # "regression" | "improvement" | "neutral" | "inconclusive"
    reasons: tuple[str, ...]
    clips: tuple[ClipDelta, ...]
    aggregate: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    labels: tuple[str, str] = ("before", "after")
    #: Warning-level things that newly fail, gathered from every clip. Printed,
    #: serialised, and enough on their own to hold the verdict at inconclusive.
    concerns: tuple[str, ...] = ()

    @property
    def is_regression(self) -> bool:
        """True when the change must not be shipped on these numbers."""
        return self.verdict == "regression"

    def as_dict(self) -> dict:
        """JSON shape written by ``--json``."""
        return {
            "harness": "heco-eval",
            "schemaVersion": 1,
            "kind": "comparison",
            "labels": {"before": self.labels[0], "after": self.labels[1]},
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "concerns": list(self.concerns),
            "aggregate": self.aggregate,
            "clips": [c.as_dict() for c in self.clips],
            "notes": list(self.notes),
        }


def _clip_index(payload: dict) -> dict[str, dict]:
    """Index a result payload's clips by label (the join key)."""
    return {c["clip"]: c for c in (payload.get("clips") or []) if isinstance(c, dict)}


def _fps(clip: dict) -> float | None:
    """End-to-end fps for a clip: the ``count`` stage, falling back to ingest.

    The fallback is explicit because they are not the same thing — ingest fps
    is the decode rate — but a record with only one of them should still print
    something rather than a blank.
    """
    fps = clip.get("fps") or {}
    value = fps.get("count")
    return value if value is not None else fps.get("ingest")


def _funnel(clip: dict, key: str) -> int | None:
    """One funnel rung out of a clip record, or None when not recorded."""
    funnel = clip.get("funnel") or {}
    value = funnel.get(key)
    return value if isinstance(value, int) else None


def _funnel_yield(clip: dict, rung: str) -> float | None:
    """A funnel rung expressed PER FRAME, or None when it cannot be computed.

    WHY PER FRAME. Raw rung counts are not comparable between two runs of the
    same clip, because the change under test moves how many frames get decoded:
    on 2026-08-05 the same corridor gave 254 frames before and 1,261 after. A
    rule on raw counts would have screamed at the REVERSE change — the one that
    fixed it. Dividing by frames is the comparable form, and it is exactly what
    a pass RATE is not (a rate divides by the rung above it, so it is blind to
    that rung collapsing).
    """
    frames = _funnel(clip, "frames")
    count = _funnel(clip, rung)
    if frames is None or frames <= 0 or count is None:
        return None
    return count / frames


def _relative_fall(before: float | None, after: float | None) -> float | None:
    """How far a number fell as a fraction of where it started; None if unknown.

    Negative when it rose. None when either side is missing or the starting
    value was zero — you cannot fall by a proportion of nothing.
    """
    if before is None or after is None or before <= 0:
        return None
    return (before - after) / before


def _failed_checks(clip: dict, severities: Iterable[str]) -> dict[str, str]:
    """Failing checks on this clip as ``{name: severity}``, filtered by severity.

    Returns the severity as well as the name so the reason can say which kind
    of check moved: a reader who is told "a check now fails" and not which
    class of check will go and look it up, and at 1 a.m. they will not.
    """
    wanted = set(severities)
    return {
        str(c.get("name")): str(c.get("severity"))
        for c in (clip.get("checks") or [])
        if not c.get("passed") and c.get("severity") in wanted and c.get("name")
    }


def compare_clip(
    label: str,
    before: dict | None,
    after: dict | None,
    tolerance: float = DEFAULT_TOLERANCE,
    gate_rate_drop: float = DEFAULT_GATE_RATE_DROP,
) -> ClipDelta:
    """Diff one clip and decide, for that clip alone, better or worse.

    Every regression rule is accuracy-shaped. None of them can be outvoted by
    throughput, because throughput is not consulted here at all.
    """
    b = before or {}
    a = after or {}
    regressions: list[str] = []
    improvements: list[str] = []
    concerns: list[str] = []

    status_before = b.get("status", "absent")
    status_after = a.get("status", "absent")
    err_b = b.get("uniqueCountError")
    err_a = a.get("uniqueCountError")
    gate_b = _funnel(b, "facesPassingGate")
    gate_a = _funnel(a, "facesPassingGate")
    rate_b = b.get("gatePassRate")
    rate_a = a.get("gatePassRate")
    unique_b = b.get("pipelineUnique")
    unique_a = a.get("pipelineUnique")

    # The twin of the missing-clip rule: a clip can also be kept and REDEFINED.
    # Both errors are relative to actualUnique, so if the count of record moved
    # between the two runs the two percentages are not the same measurement and
    # differencing them is arithmetic about nothing. Silently redefining the
    # ground truth must not be a softer way to pass than deleting the clip.
    actual_b, actual_a = b.get("actualUnique"), a.get("actualUnique")
    if actual_b is not None and actual_a is not None and actual_b != actual_a:
        regressions.append(
            f"the count of record changed between the two runs ({actual_b} -> {actual_a}). "
            "The two error figures are measured against different denominators, so this "
            "clip cannot be compared — re-run the BEFORE set against the corrected count"
        )

    if status_before == "scored" and status_after == "errored":
        regressions.append(
            f"was scorable before and is not now ({a.get('error') or 'no reason recorded'})"
        )
    if status_before == "errored" and status_after == "scored":
        improvements.append("became scorable")

    # THE FUNNEL RULES. The pass RATE alone is not enough, because a rate is a
    # ratio and a ratio is invariant to its own denominator: detect 400 faces
    # and pass 200, then detect 40 and pass 20, and the rate reads 50% both
    # times while nine tenths of the evidence has gone. So the rungs are also
    # checked in per-frame yield, which no ratio can hide (see _funnel_yield).
    rate_fall = _relative_fall(rate_b, rate_a)
    gate_yield_b, gate_yield_a = (
        _funnel_yield(b, "facesPassingGate"),
        _funnel_yield(a, "facesPassingGate"),
    )
    det_yield_b, det_yield_a = (
        _funnel_yield(b, "facesDetected"),
        _funnel_yield(a, "facesDetected"),
    )
    gate_yield_fall = _relative_fall(gate_yield_b, gate_yield_a)
    det_yield_fall = _relative_fall(det_yield_b, det_yield_a)

    # One reason per cause: these are ordered worst-first and chained, so a
    # single collapse is not reported four times in four dialects.
    if gate_b is not None and gate_a is not None and gate_b > 0 and gate_a == 0:
        regressions.append(f"GATE-PASS COLLAPSE: {gate_b} faces passed the gate before, ZERO now")
    elif rate_fall is not None and rate_fall >= gate_rate_drop:
        regressions.append(
            f"gate-pass rate fell from {rate_b:.1%} to {rate_a:.1%} (a {rate_fall:.0%} drop)"
        )
    elif gate_yield_fall is not None and gate_yield_fall >= gate_rate_drop:
        regressions.append(
            f"the gate passed {gate_yield_b:.3f} faces per frame before and "
            f"{gate_yield_a:.3f} now (a {gate_yield_fall:.0%} drop). The pass RATE did NOT "
            "halve, because detection fell with it — a rate cannot see its own denominator "
            "collapsing"
        )
    elif gate_b is not None and gate_a is not None and gate_a > gate_b:
        improvements.append(f"gate-pass count rose {gate_b} -> {gate_a}")

    # Detection is the gate rate's denominator, so it needs its own rule: if
    # the detector stops finding faces, the gate can pass all of the few that
    # are left and the rate will read like a triumph.
    if det_yield_fall is not None and det_yield_fall >= gate_rate_drop:
        regressions.append(
            f"faces DETECTED fell from {det_yield_b:.3f} to {det_yield_a:.3f} per frame "
            f"(a {det_yield_fall:.0%} drop) — the count now rests on a fraction of the "
            "evidence, and the gate-pass rate is blind to this because detection is its "
            "denominator"
        )

    if unique_b is not None and unique_a is not None and unique_b > 0 and unique_a == 0:
        regressions.append(f"counted {unique_b} people before and NOBODY now")

    if err_b is not None and err_a is not None:
        delta = abs(err_a) - abs(err_b)
        if delta > tolerance:
            regressions.append(
                f"unique-count error worsened {err_b:+.2%} -> {err_a:+.2%} "
                f"(|error| +{delta:.2%})"
            )
        elif delta < -tolerance:
            improvements.append(
                f"unique-count error improved {err_b:+.2%} -> {err_a:+.2%} "
                f"(|error| {delta:.2%})"
            )

    # Blocking checks (critical AND fail) condemn; warnings only downgrade.
    # Consulting criticals alone was a hole: face-size-vs-floor is a warning,
    # it fired at 48 px against a 56 px floor, and it is the LEADING indicator
    # of the collapse — it fires while a few faces are still squeaking through.
    blocking_b, blocking_a = (
        _failed_checks(b, BLOCKING_SEVERITIES),
        _failed_checks(a, BLOCKING_SEVERITIES),
    )
    for name in sorted(set(blocking_a) - set(blocking_b)):
        regressions.append(f"{blocking_a[name]} check {name!r} now fails and did not before")

    warn_b, warn_a = (
        _failed_checks(b, ADVISORY_SEVERITIES),
        _failed_checks(a, ADVISORY_SEVERITIES),
    )
    for name in sorted(set(warn_a) - set(warn_b)):
        concerns.append(
            f"warning check {name!r} now fails and did not before — not proof of harm, "
            "but not a thing to claim an improvement over either"
        )

    return ClipDelta(
        clip=label,
        status_before=status_before,
        status_after=status_after,
        error_before=err_b,
        error_after=err_a,
        unique_before=unique_b,
        unique_after=unique_a,
        actual_unique=a.get("actualUnique", b.get("actualUnique")),
        gate_pass_before=gate_b,
        gate_pass_after=gate_a,
        gate_rate_before=rate_b,
        gate_rate_after=rate_a,
        fps_before=_fps(b),
        fps_after=_fps(a),
        regressions=tuple(regressions),
        improvements=tuple(improvements),
        concerns=tuple(concerns),
    )


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or None for an empty list."""
    return sum(values) / len(values) if values else None


def compare_results(
    before: dict,
    after: dict,
    tolerance: float = DEFAULT_TOLERANCE,
    gate_rate_drop: float = DEFAULT_GATE_RATE_DROP,
    waive_missing: Sequence[str] = (),
) -> Comparison:
    """Compare two result payloads and return the verdict.

    The verdict is deliberately conservative. "Improvement" requires that
    NOTHING regressed on ANY clip, that no new warning fired, and that the mean
    absolute unique-count error moved past the tolerance in the right
    direction. A change that helps the aggregate but wrecks one clip — the
    baraat surge, say — is not an improvement here, it is a conversation
    (accuracy-and-tuning.md §5).

    ``waive_missing`` names clips that are allowed to be absent from the AFTER
    set (a clip genuinely retired, a file genuinely lost). Everything else that
    vanished is a regression: silence about a missing measurement is how a
    guard rail gets walked around.
    """
    b_index, a_index = _clip_index(before), _clip_index(after)
    paired = [label for label in b_index if label in a_index]
    only_before = [label for label in b_index if label not in a_index]
    only_after = [label for label in a_index if label not in b_index]
    waived = {label for label in only_before if label in set(waive_missing)}
    missing = [label for label in only_before if label not in waived]

    clips = tuple(
        compare_clip(label, b_index[label], a_index[label], tolerance, gate_rate_drop)
        for label in paired
    )

    notes: list[str] = []
    for label in sorted(waived):
        notes.append(
            f"clip {label!r} is missing from the AFTER set and was WAIVED by hand — the "
            "verdict below is a verdict on the clips that remain, not on that one"
        )
    for label in only_after:
        notes.append(f"clip {label!r} is in the AFTER set only — not compared")
    notes.append(
        "fps is reported for information and takes no part in this verdict — a "
        "faster pipeline that counts fewer people is a regression."
    )

    errs_b = [c.error_before for c in clips if c.error_before is not None]
    errs_a = [c.error_after for c in clips if c.error_after is not None]
    mean_abs_b = _mean([abs(e) for e in errs_b])
    mean_abs_a = _mean([abs(e) for e in errs_a])
    fps_b = _mean([c.fps_before for c in clips if c.fps_before is not None])
    fps_a = _mean([c.fps_after for c in clips if c.fps_after is not None])

    aggregate = {
        "clipsCompared": len(clips),
        "clipsOnlyBefore": only_before,
        "clipsOnlyAfter": only_after,
        "clipsMissingFromAfter": missing,
        "clipsMissingWaived": sorted(waived),
        "scoredBefore": sum(1 for c in clips if c.status_before == "scored"),
        "scoredAfter": sum(1 for c in clips if c.status_after == "scored"),
        "meanAbsUniqueCountErrorBefore": mean_abs_b,
        "meanAbsUniqueCountErrorAfter": mean_abs_a,
        "meanAbsUniqueCountErrorDelta": (
            None if mean_abs_b is None or mean_abs_a is None else mean_abs_a - mean_abs_b
        ),
        "meanSignedUniqueCountErrorBefore": _mean(errs_b),
        "meanSignedUniqueCountErrorAfter": _mean(errs_a),
        "meanFpsBefore": fps_b,
        "meanFpsAfter": fps_a,
        "fpsIsNotScored": True,
    }

    # A clip that was measured before and is not measured now is a MISSING
    # MEASUREMENT, and a missing measurement about an invoice figure is a
    # regression. Anything softer makes `git rm` a way of passing the A/B.
    reasons: list[str] = [
        f"{label}: measured in the BEFORE set and ABSENT from the AFTER set. A clip that "
        "disappears cannot be shown to have improved, and deleting the clip that fails is "
        "not a way to pass. Re-run it, or waive it on the record with "
        f"--waive-missing {label}"
        for label in missing
    ]
    for c in clips:
        for r in c.regressions:
            reasons.append(f"{c.clip}: {r}")
    concerns = tuple(f"{c.clip}: {w}" for c in clips for w in c.concerns)

    if reasons:
        verdict = "regression"
    elif not clips:
        verdict = "inconclusive"
        reasons.append("no clip appears in both result sets, so nothing was compared")
    elif concerns:
        # Nothing was proved to have broken, but something that was healthy is
        # now warning. "Improvement" would be a claim the evidence does not
        # support, and "neutral" would be a claim that nothing moved.
        verdict = "inconclusive"
        reasons.append(
            f"{len(concerns)} warning-level check(s) newly failed. No improvement can be "
            "claimed over a warning that was not there before — warnings on this pipeline "
            "are leading indicators, not decoration"
        )
    else:
        delta = aggregate["meanAbsUniqueCountErrorDelta"]
        if delta is not None and delta < -tolerance:
            verdict = "improvement"
            reasons.append(
                f"mean |unique-count error| improved "
                f"{mean_abs_b:.2%} -> {mean_abs_a:.2%}"
            )
        else:
            verdict = "neutral"
            reasons.append(
                "no clip regressed and no clip improved past the "
                f"{tolerance:.2%} tolerance"
            )

    if fps_b is not None and fps_a is not None and verdict == "regression" and fps_a > fps_b:
        notes.append(
            f"mean fps did improve ({fps_b:.2f} -> {fps_a:.2f}). It does not change the "
            "verdict: this is exactly the trade that made the pipeline count nobody."
        )

    return Comparison(
        verdict=verdict,
        reasons=tuple(reasons),
        clips=clips,
        aggregate=aggregate,
        notes=tuple(notes),
        labels=(str(before.get("label") or "before"), str(after.get("label") or "after")),
        concerns=concerns,
    )


def _load(path: str) -> dict:
    """Read one result file, failing with a sentence a human can act on."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"eval.compare: no such result file: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"eval.compare: {path} is not valid JSON: {exc}") from None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the comparison, return the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.compare",
        description=(
            "Compare two eval result files. Refuses to call a change an improvement "
            "if accuracy went backwards, however much fps improved."
        ),
    )
    parser.add_argument("before", help="result JSON from the baseline run")
    parser.add_argument("after", help="result JSON from the changed run")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="movement in |unique-count error| below which nothing is claimed "
        f"(default {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--gate-rate-drop",
        type=float,
        default=DEFAULT_GATE_RATE_DROP,
        help="relative fall in gate-pass rate treated as a collapse "
        f"(default {DEFAULT_GATE_RATE_DROP})",
    )
    parser.add_argument(
        "--waive-missing",
        action="append",
        default=[],
        metavar="LABEL",
        help="allow this clip to be absent from the AFTER set (repeatable). Without it, a "
        "clip that vanishes is a regression — deleting the failing clip must not be a "
        "way to pass. The waiver is printed and stored in the JSON.",
    )
    parser.add_argument("--json", dest="json_out", help="also write the comparison as JSON here")
    args = parser.parse_args(argv)

    # Imported here so the pure comparison logic has no import-time dependency
    # on the renderer (and so tests of the maths need nothing else loaded).
    from .report import render_comparison

    comparison = compare_results(
        _load(args.before),
        _load(args.after),
        args.tolerance,
        args.gate_rate_drop,
        waive_missing=args.waive_missing,
    )
    print(render_comparison(comparison))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(comparison.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
    if comparison.is_regression:
        return 2
    # Inconclusive is NOT a pass. Nothing was compared, or something newly
    # warned; either way a script that gates on this must not sail through.
    return 3 if comparison.verdict == "inconclusive" else 0


if __name__ == "__main__":  # pragma: no cover — thin CLI shell
    sys.exit(main())
