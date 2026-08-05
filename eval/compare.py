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

* the gate-pass count collapsed to zero (or the pass rate fell by half),
* the signed unique-count error got worse by more than the tolerance,
* the count fell to zero where it previously counted people,
* a critical check that used to pass now fails,
* or a clip that used to be scorable is now unscorable (a stalled or failed
  run is an ERROR, and a change that turns scores into errors is a regression,
  not a neutral outcome).

Usage::

    python -m eval.compare before.json after.json [--tolerance 0.005] [--json out.json]

Exit code 0 when the change is an improvement or neutral, 2 on regression, 1
on a usage error — so it can gate a commit.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: How much the absolute unique-count error must move before it is called a
#: change at all. Half a percentage point: below that a difference is clip
#: noise, and calling it either way would be false precision.
DEFAULT_TOLERANCE = 0.005

#: Relative fall in the gate-pass RATE that counts as a collapse even when it
#: is not all the way to zero. Half the faces silently stopping is the same
#: failure as all of them, just earlier.
DEFAULT_GATE_RATE_DROP = 0.5


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


def _failed_criticals(clip: dict) -> set[str]:
    """Names of critical checks that FAILED on this clip."""
    return {
        c.get("name")
        for c in (clip.get("checks") or [])
        if not c.get("passed") and c.get("severity") == "critical"
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

    if status_before == "scored" and status_after == "errored":
        regressions.append(
            f"was scorable before and is not now ({a.get('error') or 'no reason recorded'})"
        )
    if status_before == "errored" and status_after == "scored":
        improvements.append("became scorable")

    if gate_b is not None and gate_a is not None:
        if gate_b > 0 and gate_a == 0:
            regressions.append(
                f"GATE-PASS COLLAPSE: {gate_b} faces passed the gate before, ZERO now"
            )
        elif (
            rate_b is not None
            and rate_a is not None
            and rate_b > 0
            and (rate_b - rate_a) / rate_b >= gate_rate_drop
        ):
            regressions.append(
                f"gate-pass rate fell from {rate_b:.1%} to {rate_a:.1%} "
                f"(a {(rate_b - rate_a) / rate_b:.0%} drop)"
            )
        elif gate_a > gate_b:
            improvements.append(f"gate-pass count rose {gate_b} -> {gate_a}")

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

    new_criticals = _failed_criticals(a) - _failed_criticals(b)
    for name in sorted(n for n in new_criticals if n):
        regressions.append(f"critical check {name!r} now fails and did not before")

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
    )


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or None for an empty list."""
    return sum(values) / len(values) if values else None


def compare_results(
    before: dict,
    after: dict,
    tolerance: float = DEFAULT_TOLERANCE,
    gate_rate_drop: float = DEFAULT_GATE_RATE_DROP,
) -> Comparison:
    """Compare two result payloads and return the verdict.

    The verdict is deliberately conservative. "Improvement" requires that
    NOTHING regressed on ANY clip and that the mean absolute unique-count
    error moved past the tolerance in the right direction. A change that helps
    the aggregate but wrecks one clip — the baraat surge, say — is not an
    improvement here, it is a conversation (accuracy-and-tuning.md §5).
    """
    b_index, a_index = _clip_index(before), _clip_index(after)
    paired = [label for label in b_index if label in a_index]
    only_before = [label for label in b_index if label not in a_index]
    only_after = [label for label in a_index if label not in b_index]

    clips = tuple(
        compare_clip(label, b_index[label], a_index[label], tolerance, gate_rate_drop)
        for label in paired
    )

    notes: list[str] = []
    for label in only_before:
        notes.append(f"clip {label!r} is in the BEFORE set only — not compared")
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

    reasons: list[str] = []
    for c in clips:
        for r in c.regressions:
            reasons.append(f"{c.clip}: {r}")

    if not clips:
        return Comparison(
            verdict="inconclusive",
            reasons=("no clip appears in both result sets, so nothing was compared",),
            clips=clips,
            aggregate=aggregate,
            notes=tuple(notes),
            labels=(str(before.get("label") or "before"), str(after.get("label") or "after")),
        )

    if reasons:
        verdict = "regression"
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
    parser.add_argument("--json", dest="json_out", help="also write the comparison as JSON here")
    args = parser.parse_args(argv)

    # Imported here so the pure comparison logic has no import-time dependency
    # on the renderer (and so tests of the maths need nothing else loaded).
    from .report import render_comparison

    comparison = compare_results(
        _load(args.before), _load(args.after), args.tolerance, args.gate_rate_drop
    )
    print(render_comparison(comparison))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(comparison.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
    return 2 if comparison.is_regression else 0


if __name__ == "__main__":  # pragma: no cover — thin CLI shell
    sys.exit(main())
