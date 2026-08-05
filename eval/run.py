"""The harness runner: drive ground-truth clips through the REAL pipeline.

    python -m eval.run --manifest eval/manifest.example.json --label baseline

WHY THE REAL PIPELINE. A harness that scores a simulation scores the
simulation. Every clip is replayed through ingest's FILE source (which already
exists — ``POST /open {path}``, ``loop=false``), the runner's own
``POST /runs``, and every stage after it, so what is measured is the thing
that will count the wedding.

THE THREE RULES THIS FILE ENFORCES.

1. **A run that did not finish is an ERROR, not a score.** The waiting logic
   here decides only whether the run SETTLED; :mod:`eval.metrics` then refuses
   anything whose ``endReason`` is not ``source-ended``. A stalled or stopped
   run scored as if complete reads as an enormous false deflation, which is
   worse than no number at all.

2. **Never touch a live run.** The harness starts its own runs and stops only
   ids it started; before it starts anything it asks ingest who holds the
   single capture slot and refuses outright if anyone does. There is no code
   path here that can stop, seize or score a run the harness did not create.

3. **Idempotent and re-runnable.** Results are written per ``--label``, so
   re-running a label replaces that label's file and nothing else. ``--resume``
   carries forward clips already scored in that file, which matters when clip
   twelve of fourteen dies on a wedged service at midnight — but ONLY where the
   manifest entry that produced the stored score is still the manifest entry in
   front of it. Same label, different file or different count of record, and
   the clip is re-run rather than reported from memory.
"""

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from heco_common.logs import safe, setup_logging

from .clients import EvalHttpError, IngestProbe, PlannerReader, RunnerClient
from .manifest import Clip, Manifest, ManifestError, load_manifest, missing_local_paths
from .metrics import DEFAULT_ENVELOPE, ClipScore, aggregate, score_run
from .report import render_results

#: Runner states that mean the run is still using the camera (mirrors
#: services/runner/app/runs.py LIVE_STATES — a run is settled when it is not
#: one of these).
LIVE_STATES = frozenset({"starting", "running", "enrolling"})

#: Default per-clip ceiling. A clip is a file source and ends by itself; this
#: only ever fires when something downstream is wedged, and when it fires the
#: clip is ERRORED, never scored.
DEFAULT_CLIP_TIMEOUT_S = 1800.0

#: How long to let the runner's final planner PUT land after the thread
#: settles. The settle path reports to the planner BEFORE it flips the local
#: state, but the failure path does it the other way round, so a moment's
#: grace stops a genuinely-reported failure being read as an unreported one.
DEFAULT_SETTLE_GRACE_S = 1.5


@dataclass(frozen=True)
class EvalConfig:
    """Everything one harness invocation needs that is not the clip list."""

    label: str
    event_id: str
    site_id: str | None = None
    placement_id: str | None = None
    envelope: float = DEFAULT_ENVELOPE
    clip_timeout_s: float = DEFAULT_CLIP_TIMEOUT_S
    poll_s: float = 2.0
    settle_grace_s: float = DEFAULT_SETTLE_GRACE_S

    def run_label(self, clip: Clip) -> str:
        """The label the planner stores for this clip's run.

        Prefixed with ``eval/`` so an evaluation run is never mistaken for a
        real gate in the planner's run list, and carries both the harness
        label and the clip label so any row can be traced back to the file
        that scored it.
        """
        return f"eval/{self.label}/{clip.label}"


def slug(text: str) -> str:
    """Filesystem-safe form of a label (result files are named from it)."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return cleaned or "eval"


class ClipRunner:
    """Runs one clip through the pipeline and scores it.

    Split out from the CLI so the orchestration — start, wait, settle, refuse
    to score — is testable against fake clients with no pipeline and no
    network.
    """

    def __init__(
        self,
        runner: RunnerClient,
        planner: PlannerReader,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        log=None,
    ) -> None:
        """Bind the two clients and (for tests) the clock and sleeper."""
        self.runner = runner
        self.planner = planner
        self.sleep = sleep
        self.monotonic = monotonic
        self.log = log or setup_logging("eval")
        #: Every runner run id this instance started — the ONLY ids it will
        #: ever stop. See rule 2 in the module docstring.
        self.started: list[str] = []

    def _errored(self, clip: Clip, message: str, **extra) -> ClipScore:
        """Build an errored score; an unscorable clip never gets a number."""
        self.log.error(f"{clip.label}: {message}")
        return ClipScore(
            clip=clip.label,
            path=clip.path,
            tags=tuple(clip.tags),
            actual_unique=clip.actual_unique,
            status="errored",
            error=message,
            **extra,
        )

    def _wait_for_settle(self, run_id: str, cfg: EvalConfig) -> dict | None:
        """Poll the runner until the run leaves the live states; None on timeout."""
        deadline = self.monotonic() + cfg.clip_timeout_s
        last: dict = {}
        while self.monotonic() < deadline:
            last = self.runner.status(run_id)
            if last.get("state") not in LIVE_STATES:
                return last
            self.sleep(cfg.poll_s)
        return None

    def run_clip(self, clip: Clip, cfg: EvalConfig) -> ClipScore:
        """Start, wait, read the planner record, score it — or error out."""
        try:
            run_id = self.runner.start_run(
                event_id=cfg.event_id,
                path=clip.path,
                label=cfg.run_label(clip),
                site_id=cfg.site_id,
                placement_id=cfg.placement_id,
            )
        except EvalHttpError as exc:
            return self._errored(clip, f"the run never started: {safe(exc)}")
        self.started.append(run_id)
        self.log.info(f"{clip.label}: started runner run {run_id}")

        try:
            status = self._wait_for_settle(run_id, cfg)
        except EvalHttpError as exc:
            return self._errored(
                clip, f"lost contact with the runner: {safe(exc)}", run_ids={"runner": run_id}
            )

        if status is None:
            # Only ever a run we started (rule 2). Stopping it makes the count
            # operator-stopped, which the scorability filter then refuses —
            # correctly, because a timed-out run is not a complete count.
            self.runner.stop(run_id)
            return self._errored(
                clip,
                f"the run did not settle within {cfg.clip_timeout_s:.0f}s and was stopped; "
                "a stopped run is not a complete count and will not be scored",
                run_ids={"runner": run_id},
            )

        planner_run_id = status.get("plannerRunId")
        run_ids = {"runner": run_id, "planner": planner_run_id}
        if not planner_run_id:
            return self._errored(
                clip,
                "the run settled without ever creating a planner record "
                f"(state={status.get('state')!r}, error={safe(status.get('error'))}) — "
                "there is nothing durable to score",
                run_ids=run_ids,
            )

        self.sleep(cfg.settle_grace_s)
        try:
            record = self.planner.get_run(str(planner_run_id))
        except EvalHttpError as exc:
            return self._errored(
                clip, f"the planner record could not be read: {safe(exc)}", run_ids=run_ids
            )

        source = "planner"
        if record.get("endReason") is None and status.get("endReason"):
            # The planner never heard how it ended (its final PUT was lost),
            # but the runner knows. Recorded as such, so a reader can see the
            # score rests on the runner's word rather than the durable row.
            record = {**record, "endReason": status.get("endReason")}
            source = "runner-status"

        score = score_run(
            clip, record, run_ids=run_ids, envelope=cfg.envelope, end_reason_source=source
        )
        self.log.info(
            f"{clip.label}: {score.status} "
            + (
                f"unique={score.pipeline_unique} vs actual={clip.actual_unique}"
                if score.status == "scored"
                else f"({score.error})"
            )
        )
        return score


def evaluate(
    manifest: Manifest,
    cfg: EvalConfig,
    clip_runner: ClipRunner,
    previous: dict | None = None,
) -> list[ClipScore]:
    """Run every clip in manifest order, optionally resuming a previous file.

    Sequential on purpose: ingest owns ONE capture slot, so two clips at once
    would fight over it and both counts would be wrong.

    A clip is carried forward from ``previous`` only when the stored score was
    taken from the SAME manifest entry — see :func:`_resume_mismatch`. A label
    match alone is not enough to reuse a number.
    """
    carried = _scored_clips(previous) if previous else {}
    if carried and previous is not None:
        stale = _stale_envelope(previous, cfg)
        if stale:
            clip_runner.log.warning(f"resume: carrying nothing forward — {stale}")
            carried = {}
    scores: list[ClipScore] = []
    for clip in manifest.clips:
        stored = carried.get(clip.label)
        if stored is not None:
            mismatch = _resume_mismatch(clip, stored)
            if mismatch is None:
                clip_runner.log.info(
                    f"{clip.label}: carried forward from the previous result file"
                )
                scores.append(ClipScore.from_dict(stored))
                continue
            clip_runner.log.warning(
                f"{clip.label}: NOT carried forward and will be re-run — {mismatch}"
            )
        scores.append(clip_runner.run_clip(clip, cfg))
    return scores


def _scored_clips(payload: dict) -> dict[str, dict]:
    """Clips in a previous result file that were SCORED (errors are re-run).

    Errored clips are deliberately NOT carried: an error is usually a wedged
    service or a stalled source, which is exactly the thing worth retrying.
    """
    return {
        c["clip"]: c
        for c in (payload.get("clips") or [])
        if isinstance(c, dict) and c.get("status") == "scored" and c.get("clip")
    }


def _resume_mismatch(clip: Clip, stored: dict) -> str | None:
    """Why this stored score may NOT stand in for this clip; None when it may.

    ``--resume`` joins on the clip LABEL, and a label is only a name. Edit the
    path or ``actualUnique`` in the manifest and the label still matches, so
    the old score — taken from a different file, or measured against a
    different count of record — would be reported as the new claim's result,
    silently and with full confidence. The count is an invoice figure; a stored
    score belongs to the manifest entry that produced it, and to no other.

    Tags are bound too, because they are what the per-tag breakdown aggregates
    on: a clip retagged from ``staff`` to ``surge`` and then resumed would file
    its old number under the new heading.
    """
    stored_path = stored.get("path")
    if stored_path != clip.path:
        return (
            f"the manifest now points at {clip.path!r} and the stored score was taken from "
            f"{stored_path!r}"
        )
    stored_actual = stored.get("actualUnique")
    if stored_actual != clip.actual_unique:
        return (
            f"the count of record is now {clip.actual_unique} and the stored score was "
            f"measured against {stored_actual!r}"
        )
    stored_tags = tuple(stored.get("tags") or ())
    if stored_tags != tuple(clip.tags):
        return (
            f"the tags are now {list(clip.tags)} and the stored score carries "
            f"{list(stored_tags)}, so the per-tag breakdown would be filed wrongly"
        )
    return None


def _stale_envelope(previous: dict, cfg: EvalConfig) -> str | None:
    """Reason the whole previous file is unusable for resume; None when it is.

    Same failure as :func:`_resume_mismatch` one level up: every stored check
    was decided against the envelope in force when it ran, so carrying those
    verdicts into a run with a different ``--envelope`` would mix two
    acceptance standards inside one result file.

    A file that records no envelope at all predates this field and is taken at
    its word; re-running the batch is the way to be sure of one.
    """
    stored = previous.get("envelope")
    if stored is None or stored == cfg.envelope:
        return None
    return (
        f"the previous file was scored against an envelope of {stored} and this run uses "
        f"{cfg.envelope}; every clip will be re-scored rather than mix two standards"
    )


def build_payload(
    manifest: Manifest,
    cfg: EvalConfig,
    scores: list[ClipScore],
    started_at: str,
    ended_at: str,
) -> dict:
    """Assemble the machine-readable result file.

    Run ids are in every clip entry on purpose: any number in this file can be
    traced back to the planner row that produced it, months later.
    """
    return {
        "harness": "heco-eval",
        "schemaVersion": 1,
        "kind": "results",
        "label": cfg.label,
        "startedAt": started_at,
        "endedAt": ended_at,
        "eventId": cfg.event_id,
        "siteId": cfg.site_id,
        "envelope": cfg.envelope,
        "manifest": {
            "path": manifest.source_path,
            "clipCount": len(manifest.clips),
            "description": manifest.description,
        },
        "clips": [s.as_dict() for s in scores],
        "aggregate": aggregate(scores, cfg.envelope),
    }


def exit_code(payload: dict) -> int:
    """0 when every clip scored and passed; 2 otherwise (CI can gate on it)."""
    agg = payload.get("aggregate") or {}
    bad = int(agg.get("errored") or 0) + int(agg.get("failed") or 0)
    return 2 if bad else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Define the CLI. Defaults come from the same env the services use."""
    p = argparse.ArgumentParser(
        prog="python -m eval.run",
        description=(
            "Replay ground-truth clips through the real pipeline and score the "
            "unique count against the human count of record."
        ),
    )
    p.add_argument("--manifest", required=True, help="path to the clip manifest (JSON)")
    p.add_argument("--label", required=True, help="name for this evaluation (names the results)")
    p.add_argument("--event-id", default=os.environ.get("HECO_EVAL_EVENT_ID"))
    p.add_argument("--site-id", default=os.environ.get("HECO_EVAL_SITE_ID"))
    p.add_argument("--placement-id", default=os.environ.get("HECO_EVAL_PLACEMENT_ID"))
    p.add_argument(
        "--runner-url", default=os.environ.get("HECO_RUNNER_URL", "http://localhost:7100")
    )
    p.add_argument(
        "--planner-url", default=os.environ.get("PLANNER_URL", "http://localhost:8787")
    )
    p.add_argument("--planner-token", default=os.environ.get("HECO_TOKEN"))
    p.add_argument(
        "--ingest-url", default=os.environ.get("HECO_INGEST_URL", "http://localhost:7101")
    )
    p.add_argument("--out-dir", default="eval/results", help="where result files are written")
    p.add_argument("--envelope", type=float, default=DEFAULT_ENVELOPE)
    p.add_argument("--timeout", type=float, default=DEFAULT_CLIP_TIMEOUT_S, help="per-clip seconds")
    p.add_argument("--poll", type=float, default=2.0, help="run-status poll interval, seconds")
    p.add_argument(
        "--resume",
        action="store_true",
        help="carry forward clips already scored in this label's result file",
    )
    p.add_argument(
        "--check-paths",
        action="store_true",
        help="stat every clip path on THIS machine first (only valid when ingest "
        "shares this filesystem)",
    )
    p.add_argument(
        "--skip-slot-check",
        action="store_true",
        help="do not ask ingest whether its capture slot is free first (not advised)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="validate and print the plan, run nothing"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate, run every clip, write both result files."""
    args = _parse_args(argv)
    log = setup_logging("eval")

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        log.error(str(exc))
        return 1

    event_id = args.event_id or manifest.event_id
    if not event_id:
        log.error(
            "no event id: pass --event-id, set HECO_EVAL_EVENT_ID, or put "
            "\"eventId\" in the manifest. The planner records every run under an event."
        )
        return 1
    if manifest.unknown_keys:
        log.warning(f"manifest has unrecognised top-level keys: {list(manifest.unknown_keys)}")

    if args.check_paths:
        missing = missing_local_paths(manifest)
        if missing:
            log.error(f"clip files not found on this machine: {missing}")
            return 1

    cfg = EvalConfig(
        label=args.label,
        event_id=event_id,
        site_id=args.site_id or manifest.site_id,
        placement_id=args.placement_id,
        envelope=args.envelope,
        clip_timeout_s=args.timeout,
        poll_s=args.poll,
    )

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{slug(cfg.label)}.json"
    text_path = out_dir / f"{slug(cfg.label)}.txt"

    if args.dry_run:
        print(f"would run {len(manifest.clips)} clip(s) as event {cfg.event_id}:")
        for clip in manifest.clips:
            print(f"  {clip.label:<24} {clip.path}  (actual {clip.actual_unique})")
        print(f"would write {json_path} and {text_path}")
        return 0

    if not args.skip_slot_check:
        owner = IngestProbe(args.ingest_url).slot_owner()
        if owner:
            log.error(
                f"ingest's capture slot is held by run {owner!r}. Refusing to start: "
                "an evaluation must never take the camera from a live count. Wait for "
                "that run to finish, or point --ingest-url at the right ingest."
            )
            return 1

    previous = None
    if args.resume and json_path.exists():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        log.info(f"resuming from {json_path}")

    clip_runner = ClipRunner(
        RunnerClient(args.runner_url),
        PlannerReader(args.planner_url, token=args.planner_token),
        log=log,
    )
    started_at = datetime.now(UTC).isoformat()
    scores = evaluate(manifest, cfg, clip_runner, previous)
    payload = build_payload(manifest, cfg, scores, started_at, datetime.now(UTC).isoformat())

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    text = render_results(payload)
    text_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    log.info(f"wrote {json_path} and {text_path}")
    return exit_code(payload)


if __name__ == "__main__":  # pragma: no cover — thin CLI shell
    sys.exit(main())
