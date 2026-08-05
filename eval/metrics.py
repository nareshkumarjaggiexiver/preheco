"""Scoring — PURE functions over a planner run record. No network, no pipeline.

WHY EVERYTHING HERE IS PURE. This module is where a run becomes a verdict, so
it is the part that must be testable to destruction with synthetic records. It
takes the planner's own ``GET /api/pipeline/runs/:id`` document — the durable
record of a run, including ``endReason``, ``results`` and the per-stage
``stats`` — and turns it into a score. It never talks to anything.

THE THREE JOBS.

1. **Refuse to score an incomplete count.** A run whose ``endReason`` is
   ``source-stalled`` settled as *failed* precisely because the camera stopped
   feeding it (accuracy-and-tuning.md §5). Scoring one reads as a huge false
   deflation and would poison an A/B, so it is reported as ERRORED, never as a
   bad score.

2. **The signed headline.** ``unique_count_error = (pipeline - actual) /
   actual``. Signed, because deflation (guests never captured) and inflation
   (one guest split across two gallery keys) have opposite causes and opposite
   fixes, and an absolute error hides which one happened.

3. **The funnel, and the named collapse checks.** frames -> person boxes ->
   faces detected -> faces passing the gate -> embeds -> matches -> unique.
   The gate-pass rung is the one that caught nothing when ``INGEST_MAX_WIDTH``
   was set to 1920: faces shrank below the 56 px floor, gate survivors went
   9 -> 0, embed and match never ran, the count went to zero, and fps looked
   five times better. :func:`check_gate_pass_collapse` exists so that failure
   is a named, loud check instead of a column somebody has to read.

WHERE EACH NUMBER COMES FROM (the runner already emits all of it — see
services/runner/app/loop.py ``_pipeline_step`` and app/stats.py):

===================  ==========================================================
funnel rung          planner stats source
===================  ==========================================================
frames               ``ingest`` stage ``frames``
personBoxes          ``person-detect`` metric ``personBoxHPx`` -> ``count``
facesDetected        ``face-detect`` metric ``faceBoxWPx`` -> ``count``
facesPassingGate     ``quality`` metric ``faceBoxWPx`` -> ``count`` (survivors)
embeds               ``embed`` metric ``faceBoxWPx`` -> ``count``
matches              ``match`` stage ``frames`` (one per /match call)
unique               run ``results.unique``
===================  ==========================================================

``embeds`` is ``None`` — reported as *unknown*, never as zero — when the embed
stage ran but that build did not emit the metric, so an older run is scored
honestly instead of being accused of a collapse it did not have.
"""

from dataclasses import dataclass, field

#: The committed acceptance envelope: <= 5% unique-count error
#: (docs/planning/08-accuracy-and-testing.md, quoted by accuracy-and-tuning §5).
DEFAULT_ENVELOPE = 0.05

#: The only end reason that means "this is a complete count". A stall settled
#: the run as failed on purpose; an operator stop is a partial count against a
#: ground truth that assumed the whole clip.
SCORABLE_END_REASON = "source-ended"

#: Gate-pass rate below this (but above zero) is a warning: the pipeline is
#: still counting, but it is throwing away nearly everything it sees.
LOW_GATE_PASS_RATE = 0.10

#: Severities, most serious first. ``critical`` and ``fail`` fail a clip;
#: ``warn`` is reported and never fails anything on its own.
SEVERITY_ORDER = ("critical", "fail", "warn", "info")


@dataclass(frozen=True)
class Distribution:
    """A planner metric summary: count plus min/mean/max (None when count=0)."""

    count: int
    min: float | None
    mean: float | None
    max: float | None

    def as_dict(self) -> dict:
        """JSON shape used in result files."""
        return {"count": self.count, "min": self.min, "mean": self.mean, "max": self.max}


@dataclass(frozen=True)
class Funnel:
    """Per-clip attrition from frames to counted people.

    ``embeds`` is None when the metric was not emitted by the build under
    test (see the module docstring) — 'unknown', which is not 'zero'.
    """

    frames: int
    person_boxes: int
    faces_detected: int
    faces_passing_gate: int
    embeds: int | None
    matches: int
    unique: int

    def as_dict(self) -> dict:
        """JSON shape used in result files, in funnel order."""
        return {
            "frames": self.frames,
            "personBoxes": self.person_boxes,
            "facesDetected": self.faces_detected,
            "facesPassingGate": self.faces_passing_gate,
            "embeds": self.embeds,
            "matches": self.matches,
            "unique": self.unique,
        }


@dataclass(frozen=True)
class Check:
    """One named, explicit verdict on a run — pass or fail, with a sentence."""

    name: str
    passed: bool
    severity: str
    detail: str

    def as_dict(self) -> dict:
        """JSON shape used in result files."""
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ClipScore:
    """Everything the harness concluded about one clip."""

    clip: str
    path: str
    tags: tuple[str, ...]
    actual_unique: int
    status: str  # "scored" | "errored"
    error: str | None = None
    run_ids: dict = field(default_factory=dict)
    end_reason: str | None = None
    end_reason_source: str | None = None
    run_status: str | None = None
    pipeline_unique: int | None = None
    unique_count_error: float | None = None
    funnel: Funnel | None = None
    gate_pass_rate: float | None = None
    face_size: dict = field(default_factory=dict)
    fps: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    checks: tuple[Check, ...] = ()

    @property
    def verdict(self) -> str:
        """``pass`` only when nothing critical or failing fired."""
        if self.status != "scored":
            return "errored"
        return "fail" if failing(self.checks) else "pass"

    @classmethod
    def from_dict(cls, record: dict) -> "ClipScore":
        """Rebuild a score from its stored form — the inverse of :meth:`as_dict`.

        Needed by ``--resume``: a clip carried forward from a previous result
        file must roll into the aggregate exactly as a fresh one does, checks
        and all. A lossy rehydration would let a resumed batch quietly report
        fewer critical failures than the batch it resumed from.
        """
        funnel_raw = record.get("funnel")
        funnel = (
            Funnel(
                frames=int(funnel_raw.get("frames") or 0),
                person_boxes=int(funnel_raw.get("personBoxes") or 0),
                faces_detected=int(funnel_raw.get("facesDetected") or 0),
                faces_passing_gate=int(funnel_raw.get("facesPassingGate") or 0),
                embeds=(
                    None if funnel_raw.get("embeds") is None else int(funnel_raw["embeds"])
                ),
                matches=int(funnel_raw.get("matches") or 0),
                unique=int(funnel_raw.get("unique") or 0),
            )
            if isinstance(funnel_raw, dict)
            else None
        )
        return cls(
            clip=str(record.get("clip")),
            path=str(record.get("path")),
            tags=tuple(record.get("tags") or ()),
            actual_unique=int(record.get("actualUnique") or 0),
            status=str(record.get("status") or "errored"),
            error=record.get("error"),
            run_ids=dict(record.get("runIds") or {}),
            end_reason=record.get("endReason"),
            end_reason_source=record.get("endReasonSource"),
            run_status=record.get("runStatus"),
            pipeline_unique=record.get("pipelineUnique"),
            unique_count_error=record.get("uniqueCountError"),
            funnel=funnel,
            gate_pass_rate=record.get("gatePassRate"),
            face_size=record.get("faceSize") or {},
            fps=record.get("fps") or {},
            gate=record.get("gate") or {},
            checks=tuple(
                Check(
                    name=str(c.get("name")),
                    passed=bool(c.get("passed")),
                    severity=str(c.get("severity")),
                    detail=str(c.get("detail")),
                )
                for c in (record.get("checks") or [])
            ),
        )

    def as_dict(self) -> dict:
        """The clip's entry in a result file (machine-readable, for the A/B)."""
        return {
            "clip": self.clip,
            "path": self.path,
            "tags": list(self.tags),
            "actualUnique": self.actual_unique,
            "status": self.status,
            "verdict": self.verdict,
            "error": self.error,
            "runIds": dict(self.run_ids),
            "endReason": self.end_reason,
            "endReasonSource": self.end_reason_source,
            "runStatus": self.run_status,
            "pipelineUnique": self.pipeline_unique,
            "uniqueCountError": self.unique_count_error,
            "funnel": self.funnel.as_dict() if self.funnel else None,
            "gatePassRate": self.gate_pass_rate,
            "faceSize": self.face_size,
            "fps": self.fps,
            "gate": dict(self.gate),
            "checks": [c.as_dict() for c in self.checks],
        }


# --------------------------------------------------------------- readers


def stage_map(run: dict) -> dict[str, dict]:
    """Index a run document's ``stats`` list by stage name."""
    out: dict[str, dict] = {}
    for row in run.get("stats") or []:
        stage = row.get("stage")
        if isinstance(stage, str):
            out[stage] = row
    return out


def distribution(stages: dict[str, dict], stage: str, metric: str) -> Distribution | None:
    """Return one stage metric's {count,min,mean,max}, or None if never emitted.

    None and a zero-count Distribution are different facts: None means this
    build never reported the metric, zero means it reported and saw nothing.
    """
    row = stages.get(stage)
    if row is None:
        return None
    summary = (row.get("metrics") or {}).get(metric)
    if not isinstance(summary, dict):
        return None
    count = int(summary.get("count") or 0)
    return Distribution(
        count=count,
        min=_opt_float(summary.get("min")),
        mean=_opt_float(summary.get("mean")),
        max=_opt_float(summary.get("max")),
    )


def _opt_float(value: object) -> float | None:
    """Float or None — planner summaries carry nulls for an empty aggregate."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _metric_count(stages: dict[str, dict], stage: str, metric: str) -> int:
    """Observation count for a stage metric; 0 when the stage never reported."""
    dist = distribution(stages, stage, metric)
    return dist.count if dist else 0


def funnel_from_run(run: dict) -> Funnel:
    """Build the attrition funnel from a planner run document."""
    stages = stage_map(run)
    results = run.get("results") or {}
    ingest = stages.get("ingest") or {}
    frames = int(ingest.get("frames") or 0) or int(results.get("frames") or 0)

    embed_row = stages.get("embed")
    if embed_row is None:
        # No stats row at all: StatsBoard only emits a stage once it has seen a
        # frame, so the embedder genuinely never ran. That is a real zero.
        embeds: int | None = 0
    else:
        dist = distribution(stages, "embed", "faceBoxWPx")
        # Row present but no per-face metric -> a build that does not emit it.
        embeds = dist.count if dist else None

    return Funnel(
        frames=frames,
        person_boxes=_metric_count(stages, "person-detect", "personBoxHPx"),
        faces_detected=_metric_count(stages, "face-detect", "faceBoxWPx"),
        faces_passing_gate=_metric_count(stages, "quality", "faceBoxWPx"),
        embeds=embeds,
        matches=int((stages.get("match") or {}).get("frames") or 0),
        unique=int(results.get("unique") or 0),
    )


def fps_from_run(run: dict) -> dict:
    """Both fps figures, kept apart because they mean different things.

    ``ingest`` is the DECODE rate (the chip the planner's run list shows);
    ``count`` is the end-to-end throughput of the whole pipeline. Quoting the
    first as "the pipeline got faster" is exactly the mistake this harness
    exists to make impossible, so both are recorded and neither is scored.
    """
    stages = stage_map(run)
    return {
        "ingest": _opt_float((stages.get("ingest") or {}).get("fps")),
        "count": _opt_float((stages.get("count") or {}).get("fps")),
    }


def face_sizes_from_run(run: dict) -> dict:
    """Face-size distributions actually seen: at detection and after the gate.

    Reported in both box width and interocular distance because IED is what
    recognition actually depends on (accuracy-and-tuning.md §3.2) and width is
    what the gate is currently expressed in.
    """
    stages = stage_map(run)
    detected = {
        "faceBoxWPx": distribution(stages, "face-detect", "faceBoxWPx"),
        "faceIedPx": distribution(stages, "face-detect", "faceIedPx"),
        "frontality": distribution(stages, "face-detect", "frontality"),
    }
    passed = {"faceBoxWPx": distribution(stages, "quality", "faceBoxWPx")}
    return {
        "detected": {k: v.as_dict() for k, v in detected.items() if v is not None},
        "passed": {k: v.as_dict() for k, v in passed.items() if v is not None},
    }


def gate_from_run(run: dict) -> dict:
    """The quality floors this run actually enforced, from its stored config.

    Read from the run row rather than from the current environment: the count
    is an invoice figure, so which floors produced it has to be recoverable
    from the record months later.
    """
    config = run.get("config") or {}
    keys = (
        "qualityMinPx",
        "qualityCanonPx",
        "qualityMinIedPx",
        "qualityMinFrontality",
        "qualityMinSharpness",
        "gateArmed",
    )
    return {k: config[k] for k in keys if k in config}


# ------------------------------------------------------------- the metric


def unique_count_error(pipeline_unique: int, actual_unique: int) -> float:
    """Signed relative error: negative = deflation, positive = inflation.

    ``actual_unique`` must be >= 1; the manifest loader enforces that, and it
    is re-asserted here because this function is the headline number and a
    silent ZeroDivisionError inside a report generator is not acceptable.
    """
    if actual_unique < 1:
        raise ValueError("actual_unique must be at least 1 — a relative error needs a denominator")
    return (pipeline_unique - actual_unique) / actual_unique


def within_envelope(error: float | None, envelope: float = DEFAULT_ENVELOPE) -> bool:
    """Is the signed error inside +/- the committed envelope?"""
    return error is not None and abs(error) <= envelope + 1e-12


def gate_pass_rate(funnel: Funnel) -> float | None:
    """Share of detected faces that survived the quality gate (None if none seen)."""
    if funnel.faces_detected <= 0:
        return None
    return funnel.faces_passing_gate / funnel.faces_detected


# ------------------------------------------------------- scorability filter


def is_scorable(run: dict) -> tuple[bool, str]:
    """May this run be scored at all? Returns (ok, reason-when-not).

    §5 is explicit: score only runs that actually finished. A ``source-stalled``
    run stopped because the camera vanished, so its count is not a count of the
    whole clip and scoring it reads as a huge false deflation. An
    ``operator-stopped`` run is equally incomplete against a ground truth that
    assumed the entire clip. A missing end reason means the run never settled
    (or never told the planner it had), which is the same problem wearing a
    different hat.
    """
    reason = run.get("endReason")
    if reason == SCORABLE_END_REASON:
        return True, ""
    status = run.get("status")
    if reason is None:
        return False, (
            f"the run has no endReason (status={status!r}) — it never settled, or never "
            "managed to tell the planner it had. An unsettled run is not a complete "
            "count and will not be scored."
        )
    if reason == "source-stalled":
        return False, (
            "endReason=source-stalled — the source stopped feeding this run, so the "
            "count covers only part of the clip. Scoring it would read as a large "
            "false deflation and poison the A/B."
        )
    return False, (
        f"endReason={reason} — only {SCORABLE_END_REASON!r} runs are complete counts "
        "of a clip; anything else is a partial count."
    )


# ------------------------------------------------------------------ checks


def check_gate_pass_collapse(funnel: Funnel) -> Check:
    """THE check: faces were detected and NONE survived the quality gate.

    This is the ``INGEST_MAX_WIDTH=1920`` failure, stated as a rule. Downscaling
    4K to 1080p halved every face; the mean detected width fell to ~48 px, under
    the 56 px floor; the gate passed nothing; embed and match never ran; the
    pipeline counted nobody — and fps improved five-fold, so the only watched
    number said it was a win. A funnel rung that goes to zero is never a win.
    """
    collapsed = funnel.faces_detected > 0 and funnel.faces_passing_gate == 0
    if collapsed:
        detail = (
            f"GATE-PASS COLLAPSE: {funnel.faces_detected} faces detected, ZERO passed "
            "the quality gate. Nothing was embedded, nothing was matched, nobody was "
            "counted. Faces are arriving smaller than the floor — check any change to "
            "ingest resolution, camera zoom or the gate thresholds."
        )
    else:
        detail = (
            f"{funnel.faces_passing_gate} of {funnel.faces_detected} detected faces "
            "passed the quality gate."
        )
    return Check("gate-pass-collapse", not collapsed, "critical", detail)


def check_faces_detected(funnel: Funnel) -> Check:
    """Frames were processed but the detector found no face anywhere."""
    failed = funnel.frames > 0 and funnel.faces_detected == 0
    detail = (
        f"no face was detected in any of {funnel.frames} frames — the face stage is "
        "not seeing anything at all"
        if failed
        else f"{funnel.faces_detected} faces detected across {funnel.frames} frames"
    )
    return Check("no-faces-detected", not failed, "critical", detail)


def check_frames(funnel: Funnel) -> Check:
    """The run pulled no frames at all — there is nothing to score."""
    failed = funnel.frames <= 0
    detail = (
        "the run processed ZERO frames; the source produced nothing"
        if failed
        else f"{funnel.frames} frames processed"
    )
    return Check("no-frames", not failed, "critical", detail)


def check_embed_ran(funnel: Funnel) -> Check:
    """Gate survivors existed but nothing reached the embedder."""
    if funnel.embeds is None:
        return Check(
            "embed-ran",
            True,
            "critical",
            "embed count not reported by this build — rung skipped (unknown, not zero)",
        )
    if funnel.faces_passing_gate == 0:
        # Nothing was offered, so nothing being embedded is not this check's
        # failure — the gate-pass collapse owns it, and one cause should not be
        # reported as four failures.
        return Check("embed-ran", True, "critical", "not applicable (nothing passed the gate)")
    failed = funnel.embeds == 0
    detail = (
        f"{funnel.faces_passing_gate} faces passed the gate but the embedder ran on "
        "NONE of them — the embed stage is broken or unreachable"
        if failed
        else f"{funnel.embeds} faces embedded"
    )
    return Check("embed-ran", not failed, "critical", detail)


def check_match_ran(funnel: Funnel) -> Check:
    """Something was embedded but the matcher never decided anything."""
    upstream = funnel.embeds if funnel.embeds is not None else funnel.faces_passing_gate
    if upstream == 0:
        return Check("match-ran", True, "critical", "not applicable (nothing was embedded)")
    failed = funnel.matches == 0
    detail = (
        f"{upstream} embeddings were produced but the matcher was never called — "
        "no gallery decision was ever made, so the count cannot move"
        if failed
        else f"{funnel.matches} match decisions made"
    )
    return Check("match-ran", not failed, "critical", detail)


def check_counted_anybody(funnel: Funnel, actual_unique: int) -> Check:
    """The clip has people in it and the pipeline counted none of them."""
    failed = actual_unique > 0 and funnel.unique == 0
    detail = (
        f"the pipeline counted NOBODY on a clip with {actual_unique} people in it"
        if failed
        else f"counted {funnel.unique} unique people (ground truth {actual_unique})"
    )
    return Check("counted-nobody", not failed, "critical", detail)


def check_envelope(error: float | None, envelope: float = DEFAULT_ENVELOPE) -> Check:
    """The headline: signed unique-count error against the committed envelope."""
    ok = within_envelope(error, envelope)
    if error is None:
        return Check("unique-count-error", False, "fail", "no unique count was recorded")
    direction = "inflation" if error > 0 else "deflation" if error < 0 else "exact"
    detail = (
        f"unique-count error {error:+.2%} ({direction}) against an envelope of "
        f"+/-{envelope:.0%}"
    )
    return Check("unique-count-error", ok, "fail", detail)


def check_gate_pass_rate(funnel: Funnel) -> Check:
    """Warn when the gate is throwing away nearly everything it is shown."""
    rate = gate_pass_rate(funnel)
    if rate is None or rate == 0.0:
        # Zero is the collapse check's business; do not double-report it.
        return Check("gate-pass-rate", True, "warn", "not applicable (no faces detected)")
    low = rate < LOW_GATE_PASS_RATE
    detail = (
        f"only {rate:.1%} of detected faces passed the gate — the count is resting on "
        "very little evidence"
        if low
        else f"gate-pass rate {rate:.1%}"
    )
    return Check("gate-pass-rate", not low, "warn", detail)


def check_face_size_against_floor(face_size: dict, gate: dict) -> Check:
    """Warn when the AVERAGE detected face is already below the embed floor.

    The leading indicator of the ``INGEST_MAX_WIDTH`` failure: mean detected
    width fell to 48 px against a 56 px floor. This fires while a few large
    faces are still squeaking through, i.e. before the collapse, which is the
    whole point of having it as well as the collapse check.
    """
    floor = gate.get("qualityMinPx")
    mean = ((face_size.get("detected") or {}).get("faceBoxWPx") or {}).get("mean")
    if floor is None or mean is None:
        return Check("face-size-vs-floor", True, "warn", "not applicable (size or floor unknown)")
    low = float(mean) < float(floor)
    detail = (
        f"mean detected face width {float(mean):.0f} px is BELOW the {float(floor):.0f} px "
        "embed floor — most faces are being discarded before they are ever embedded"
        if low
        else f"mean detected face width {float(mean):.0f} px vs a {float(floor):.0f} px floor"
    )
    return Check("face-size-vs-floor", not low, "warn", detail)


def run_checks(
    funnel: Funnel,
    face_size: dict,
    gate: dict,
    actual_unique: int,
    error: float | None,
    envelope: float = DEFAULT_ENVELOPE,
) -> tuple[Check, ...]:
    """Every named check, in reporting order (criticals first)."""
    return (
        check_frames(funnel),
        check_faces_detected(funnel),
        check_gate_pass_collapse(funnel),
        check_embed_ran(funnel),
        check_match_ran(funnel),
        check_counted_anybody(funnel, actual_unique),
        check_envelope(error, envelope),
        check_gate_pass_rate(funnel),
        check_face_size_against_floor(face_size, gate),
    )


def failing(checks: tuple[Check, ...] | list[Check]) -> list[Check]:
    """Checks that failed and are severe enough to fail the clip."""
    return [c for c in checks if not c.passed and c.severity in ("critical", "fail")]


def warnings(checks: tuple[Check, ...] | list[Check]) -> list[Check]:
    """Checks that failed at warning severity (reported, never fatal)."""
    return [c for c in checks if not c.passed and c.severity == "warn"]


def criticals(checks: tuple[Check, ...] | list[Check]) -> list[Check]:
    """Checks that failed at critical severity — the loud ones."""
    return [c for c in checks if not c.passed and c.severity == "critical"]


# ------------------------------------------------------------------ scoring


def score_run(
    clip,
    run: dict,
    run_ids: dict | None = None,
    envelope: float = DEFAULT_ENVELOPE,
    end_reason_source: str = "planner",
) -> ClipScore:
    """Score one clip's planner run document into a :class:`ClipScore`.

    Refuses (status ``errored``) when the run is not a complete count, or when
    the planner holds no ``results`` for it — an unscorable run is an ERROR,
    never a bad score, so it can never be silently averaged into an A/B.
    """
    base = {
        "clip": clip.label,
        "path": clip.path,
        "tags": tuple(clip.tags),
        "actual_unique": clip.actual_unique,
        "run_ids": dict(run_ids or {}),
        "end_reason": run.get("endReason"),
        "end_reason_source": end_reason_source,
        "run_status": run.get("status"),
    }
    ok, why = is_scorable(run)
    if not ok:
        return ClipScore(status="errored", error=why, **base)

    results = run.get("results")
    if not isinstance(results, dict) or results.get("unique") is None:
        return ClipScore(
            status="errored",
            error=(
                "the run settled but the planner holds no results.unique for it — "
                "there is no count to score"
            ),
            **base,
        )

    funnel = funnel_from_run(run)
    face_size = face_sizes_from_run(run)
    gate = gate_from_run(run)
    error = unique_count_error(funnel.unique, clip.actual_unique)
    checks = run_checks(funnel, face_size, gate, clip.actual_unique, error, envelope)
    return ClipScore(
        status="scored",
        pipeline_unique=funnel.unique,
        unique_count_error=error,
        funnel=funnel,
        gate_pass_rate=gate_pass_rate(funnel),
        face_size=face_size,
        fps=fps_from_run(run),
        gate=gate,
        checks=checks,
        **base,
    )


def aggregate(scores: list[ClipScore], envelope: float = DEFAULT_ENVELOPE) -> dict:
    """Roll clip scores up, keeping errored clips visible rather than averaged.

    Errored clips are counted and named but never folded into a mean: an
    unscorable run has no score to average, and quietly dropping it would make
    a broken batch look like a small one.
    """
    scored = [s for s in scores if s.status == "scored"]
    errored = [s for s in scores if s.status != "scored"]
    errors = [s.unique_count_error for s in scored if s.unique_count_error is not None]
    by_tag: dict[str, dict] = {}
    for s in scored:
        for tag in s.tags:
            slot = by_tag.setdefault(tag, {"clips": 0, "meanAbsUniqueCountError": 0.0})
            slot["clips"] += 1
    for tag, slot in by_tag.items():
        vals = [
            abs(s.unique_count_error)
            for s in scored
            if tag in s.tags and s.unique_count_error is not None
        ]
        slot["meanAbsUniqueCountError"] = sum(vals) / len(vals) if vals else None
    worst = max(
        (s for s in scored if s.unique_count_error is not None),
        key=lambda s: abs(s.unique_count_error),
        default=None,
    )
    return {
        "clips": len(scores),
        "scored": len(scored),
        "errored": len(errored),
        "erroredClips": [s.clip for s in errored],
        "passed": sum(1 for s in scored if s.verdict == "pass"),
        "failed": sum(1 for s in scored if s.verdict == "fail"),
        "withinEnvelope": sum(
            1 for s in scored if within_envelope(s.unique_count_error, envelope)
        ),
        "envelope": envelope,
        "meanSignedUniqueCountError": (sum(errors) / len(errors)) if errors else None,
        "meanAbsUniqueCountError": (
            sum(abs(e) for e in errors) / len(errors) if errors else None
        ),
        "worstClip": worst.clip if worst else None,
        "worstUniqueCountError": worst.unique_count_error if worst else None,
        "criticalChecks": [
            {"clip": s.clip, "check": c.name, "detail": c.detail}
            for s in scored
            for c in criticals(s.checks)
        ],
        "byTag": by_tag,
    }
