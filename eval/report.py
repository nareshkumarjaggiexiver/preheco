"""Readable text rendering of results and comparisons — PURE, no I/O.

WHY A SEPARATE RENDERER. The JSON result file is the machine-readable record
that the A/B joins on; a human still has to read the thing at 1 a.m. on a
venue's Wi-Fi. Both come from the same payload dict, so the text can always be
regenerated from a stored result and can never drift from it.

The one design rule: **a collapse must be impossible to miss.** Critical
failures are printed as a banner at the top of the report, before any table,
because the failure this harness was built for looked like a five-fold
improvement in the only column anybody was reading.
"""

from .compare import Comparison

RULE = "=" * 78
THIN = "-" * 78


def _num(value: object, places: int = 0) -> str:
    """Render a number for a table cell; '-' when it was never recorded."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{places}f}"
    return str(value)


def _pct(value: object, places: int = 2, signed: bool = False) -> str:
    """Render a fraction as a percentage; '-' when it was never recorded."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{value:+.{places}%}" if signed else f"{value:.{places}%}"


def _dist(dist: dict | None, unit: str = "px") -> str:
    """Render a {count,min,mean,max} summary on one line."""
    if not dist or not dist.get("count"):
        return "none seen"
    return (
        f"n={dist['count']:,}  min {_num(dist.get('min'), 0)}{unit}  "
        f"mean {_num(dist.get('mean'), 0)}{unit}  max {_num(dist.get('max'), 0)}{unit}"
    )


def _funnel_line(funnel: dict | None) -> str:
    """The funnel on one line, in attrition order."""
    if not funnel:
        return "      not recorded"
    embeds = "unknown" if funnel.get("embeds") is None else _num(funnel.get("embeds"))
    return (
        f"      frames {_num(funnel.get('frames'))}"
        f"  ->  person boxes {_num(funnel.get('personBoxes'))}"
        f"  ->  faces {_num(funnel.get('facesDetected'))}"
        f"  ->  gate {_num(funnel.get('facesPassingGate'))}"
        f"  ->  embeds {embeds}"
        f"  ->  matches {_num(funnel.get('matches'))}"
        f"  ->  UNIQUE {_num(funnel.get('unique'))}"
    )


def collapse_banner(payload: dict) -> list[str]:
    """The loud part: every critical check failure, above everything else."""
    hits = [
        (clip.get("clip"), check)
        for clip in payload.get("clips") or []
        for check in clip.get("checks") or []
        if not check.get("passed") and check.get("severity") == "critical"
    ]
    if not hits:
        return []
    lines = [
        RULE,
        "!!  CRITICAL — THE PIPELINE STOPPED COUNTING SOMEWHERE IN THE FUNNEL  !!",
        RULE,
    ]
    for clip, check in hits:
        lines.append(f"  [{clip}] {check.get('name')}: {check.get('detail')}")
    lines += [
        "",
        "  No throughput number redeems any of the above. A rung of the funnel that",
        "  goes to zero means people were not counted.",
        RULE,
        "",
    ]
    return lines


def render_clip(clip: dict) -> str:
    """One clip's block: identity, headline, funnel, sizes, fps, checks."""
    lines: list[str] = []
    tags = ", ".join(clip.get("tags") or []) or "no tags"
    lines.append(f"{clip.get('clip')}  [{tags}]")
    lines.append(f"    source        {clip.get('path')}")
    run_ids = clip.get("runIds") or {}
    lines.append(
        f"    run ids       runner={run_ids.get('runner') or '-'}  "
        f"planner={run_ids.get('planner') or '-'}"
    )
    lines.append(
        f"    settled       status={clip.get('runStatus') or '-'}  "
        f"endReason={clip.get('endReason') or '-'}"
    )

    if clip.get("status") != "scored":
        lines.append(f"    NOT SCORED    {clip.get('error')}")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"    COUNT         pipeline {_num(clip.get('pipelineUnique'))}  vs  "
        f"actual {_num(clip.get('actualUnique'))}   "
        f"error {_pct(clip.get('uniqueCountError'), 2, signed=True)}"
    )
    lines.append("    funnel")
    lines.append(_funnel_line(clip.get("funnel")))
    face = clip.get("faceSize") or {}
    detected, passed = face.get("detected") or {}, face.get("passed") or {}
    lines.append(f"    face width    detected: {_dist(detected.get('faceBoxWPx'))}")
    lines.append(f"    face IED      detected: {_dist(detected.get('faceIedPx'))}")
    lines.append(f"    face width    after gate: {_dist(passed.get('faceBoxWPx'))}")
    gate = clip.get("gate") or {}
    if gate:
        armed = ", ".join(gate.get("gateArmed") or []) or "width only"
        lines.append(
            f"    gate          floor {_num(gate.get('qualityMinPx'))}px  "
            f"canon {_num(gate.get('qualityCanonPx'))}px  armed: {armed}"
        )
    fps = clip.get("fps") or {}
    lines.append(
        f"    fps           end-to-end {_num(fps.get('count'), 2)}  "
        f"(ingest decode {_num(fps.get('ingest'), 2)}) — reported, never scored"
    )
    lines.append("    checks")
    for check in clip.get("checks") or []:
        mark = "ok  " if check.get("passed") else {"critical": "FAIL", "fail": "FAIL"}.get(
            check.get("severity"), "warn"
        )
        lines.append(f"      [{mark}] {check.get('name')}: {check.get('detail')}")
    lines.append("")
    return "\n".join(lines)


def render_results(payload: dict) -> str:
    """The full human report for one harness run."""
    lines: list[str] = []
    lines += collapse_banner(payload)
    lines.append(RULE)
    lines.append(f"HECO eval — {payload.get('label')}")
    lines.append(RULE)
    manifest = payload.get("manifest") or {}
    lines.append(f"  manifest      {manifest.get('path')}  ({manifest.get('clipCount')} clips)")
    lines.append(f"  started       {payload.get('startedAt')}")
    lines.append(f"  finished      {payload.get('endedAt')}")
    lines.append(f"  event         {payload.get('eventId')}")
    lines.append(f"  envelope      +/-{_pct(payload.get('envelope'), 0)} unique-count error")
    lines.append("")
    lines.append(THIN)
    for clip in payload.get("clips") or []:
        lines.append(render_clip(clip))
    agg = payload.get("aggregate") or {}
    lines.append(THIN)
    lines.append("AGGREGATE")
    lines.append(
        f"  clips {_num(agg.get('clips'))}   scored {_num(agg.get('scored'))}   "
        f"errored {_num(agg.get('errored'))}   passed {_num(agg.get('passed'))}   "
        f"failed {_num(agg.get('failed'))}"
    )
    lines.append(
        f"  mean signed unique-count error   {_pct(agg.get('meanSignedUniqueCountError'), 2, True)}"
    )
    lines.append(
        f"  mean |unique-count error|        {_pct(agg.get('meanAbsUniqueCountError'), 2)}"
    )
    lines.append(
        f"  within envelope                  {_num(agg.get('withinEnvelope'))}"
        f" of {_num(agg.get('scored'))}"
    )
    if agg.get("worstClip"):
        lines.append(
            f"  worst clip                       {agg.get('worstClip')} "
            f"({_pct(agg.get('worstUniqueCountError'), 2, True)})"
        )
    for tag, slot in sorted((agg.get("byTag") or {}).items()):
        lines.append(
            f"  tag {tag:<14} {slot.get('clips')} clip(s), "
            f"mean |error| {_pct(slot.get('meanAbsUniqueCountError'), 2)}"
        )
    if agg.get("erroredClips"):
        lines.append(f"  NOT SCORED                       {', '.join(agg['erroredClips'])}")
    lines.append(THIN)
    return "\n".join(lines)


def render_comparison(comparison: Comparison) -> str:
    """The A/B report: the verdict first, the numbers under it."""
    verdict = comparison.verdict.upper()
    lines = [RULE]
    if comparison.is_regression:
        lines += [
            "!!  REGRESSION — DO NOT SHIP THIS CHANGE ON THESE NUMBERS  !!",
            RULE,
        ]
    else:
        lines += [f"A/B VERDICT: {verdict}", RULE]
    lines.append(f"  before: {comparison.labels[0]}")
    lines.append(f"  after:  {comparison.labels[1]}")
    lines.append("")
    for reason in comparison.reasons:
        lines.append(f"  * {reason}")
    # Concerns are printed even when the verdict is already a regression: the
    # warning is usually the EARLIER symptom, and it is the one that tells the
    # reader which way the failure came in.
    for concern in comparison.concerns:
        lines.append(f"  ? CONCERN  {concern}")
    lines.append("")
    lines.append(THIN)
    header = (
        f"  {'clip':<22}{'error B':>10}{'error A':>10}{'gate B':>9}"
        f"{'gate A':>9}{'fps B':>8}{'fps A':>8}"
    )
    lines.append(header)
    for c in comparison.clips:
        lines.append(
            f"  {c.clip:<22}{_pct(c.error_before, 1, True):>10}"
            f"{_pct(c.error_after, 1, True):>10}"
            f"{_num(c.gate_pass_before):>9}{_num(c.gate_pass_after):>9}"
            f"{_num(c.fps_before, 2):>8}{_num(c.fps_after, 2):>8}"
        )
        for r in c.regressions:
            lines.append(f"      REGRESSION  {r}")
        for w in c.concerns:
            lines.append(f"      concern     {w}")
        for i in c.improvements:
            lines.append(f"      improvement {i}")
    agg = comparison.aggregate
    lines.append(THIN)
    lines.append("AGGREGATE")
    lines.append(
        f"  mean |unique-count error|  "
        f"{_pct(agg.get('meanAbsUniqueCountErrorBefore'), 2)} -> "
        f"{_pct(agg.get('meanAbsUniqueCountErrorAfter'), 2)}  "
        f"(delta {_pct(agg.get('meanAbsUniqueCountErrorDelta'), 2, True)})"
    )
    lines.append(
        f"  mean signed error          "
        f"{_pct(agg.get('meanSignedUniqueCountErrorBefore'), 2, True)} -> "
        f"{_pct(agg.get('meanSignedUniqueCountErrorAfter'), 2, True)}"
    )
    lines.append(
        f"  mean fps                   {_num(agg.get('meanFpsBefore'), 2)} -> "
        f"{_num(agg.get('meanFpsAfter'), 2)}   (NOT part of the verdict)"
    )
    lines.append(f"  clips compared             {_num(agg.get('clipsCompared'))}")
    lines.append("")
    for note in comparison.notes:
        lines.append(f"  note: {note}")
    lines.append(RULE)
    return "\n".join(lines)
