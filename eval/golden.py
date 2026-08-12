"""Golden decision replay — the guard rail for a REFACTOR.

``compare.py`` next door answers "is the count right?", which needs labelled
footage and a human count of record.  This module answers a different and much
sharper question: **did anything change at all?**  That one needs no labels,
and it is the question every step of the architecture migration actually has
to answer, because each step claims to move code without moving behaviour.

The artifact is the run's own decision ledger — one JSON record per processed
frame, holding that frame's boxes, faces, gate outcomes, verdicts and events.
The runner writes it to a local file when ``HECO_GOLDEN_PATH`` is set (see
``loop._write_golden``).  So the workflow is:

    1. capture the ledger on the code you have,
    2. make the change,
    3. replay the SAME clip and capture again,
    4. ``python -m eval.golden before.jsonl after.jsonl``.

A clean diff is the strongest statement available about a refactor: not "the
totals matched" but "every frame reasoned identically".  Totals matching is
the weaker claim that lets two compensating errors through — a guest lost here
and a phantom minted there sum to the same number and are not the same run.

WHY A CLIP AND NOT A CAMERA.  This only means anything if the input is
identical between the two captures, which a live camera can never be.  Use a
recording, and use lockstep (``source.lockstep``) so every frame is processed
rather than sampled — otherwise the two runs see different frames and the
diff reports the sampler's timing, not the change's effect.

WHAT IS DELIBERATELY IGNORED.  Wall-clock timings (``ms``) differ between any
two runs of anything and say nothing about reasoning, so they are stripped
before comparison unless ``--with-timings`` is passed.  Everything else —
including the order records appear in — is significant.
"""

import argparse
import json
import sys
from pathlib import Path

#: Keys whose values are wall-clock measurements, not decisions.  A refactor
#: is EXPECTED to change these; that is usually its whole point.
#:
#: ``ms`` is the per-stage cost of the frame.  ``tMs`` is when the frame was
#: read, measured from the run's start — and on a FILE replay it is pure
#: scheduling noise, because the source is consumed as fast as the pipeline
#: can take it and every run therefore stamps slightly different offsets.  It
#: was the first thing this tool ever flagged, on two runs of one clip that
#: agreed on all 1802 frames' reasoning and disagreed by 5 ms on when frame 1
#: arrived; a guard rail that fires on that would be turned off within a day.
#:
#: ``seq`` is deliberately NOT here.  It identifies the frame, so records
#: falling out of order or going missing is a real finding.
TIMING_KEYS = ("ms", "tMs")


def load(path: str | Path, *, keep_timings: bool = False) -> list[dict]:
    """Read a capture file into a list of frame records, oldest first.

    Blank lines are skipped; a malformed line is fatal rather than skipped,
    because a capture that cannot be read completely cannot support a claim
    that nothing changed.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        # A clean sentence, not a traceback. This tool is run from a shell in
        # the middle of a deploy, and "the capture is not where you think it
        # is" is by far its most common failure — usually a docker cp that
        # went to a container that had been recreated underneath it.
        raise SystemExit(f"cannot read capture {path}: {exc.strerror}") from exc
    records = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n}: not valid JSON — {exc}") from exc
        if not keep_timings:
            for key in TIMING_KEYS:
                record.pop(key, None)
        records.append(record)
    return records


def first_difference(before: list[dict], after: list[dict]) -> dict | None:
    """Return the first place the two runs diverge, or None if identical.

    FIRST, not all of them: once reasoning diverges every later frame inherits
    the divergence, so a full diff of a run that forked at frame 12 is a
    thousand lines describing one event.  The first difference is the one to
    debug; everything after it is its consequence.
    """
    # strict=False deliberately: a length mismatch is itself a finding, and
    # it is reported below with the count on each side — raising here would
    # lose the fact that the first N frames DID match.
    for i, (b, a) in enumerate(zip(before, after, strict=False)):
        if b != a:
            return {
                "kind": "record",
                "index": i,
                "seq": b.get("seq", a.get("seq")),
                "fields": _changed_fields(b, a),
                "before": b,
                "after": a,
            }
    if len(before) != len(after):
        longer, which = (
            (before, "before") if len(before) > len(after) else (after, "after")
        )
        return {
            "kind": "length",
            "index": min(len(before), len(after)),
            "counts": {"before": len(before), "after": len(after)},
            "extra_from": which,
            "extra": longer[min(len(before), len(after))],
        }
    return None


def _changed_fields(before: dict, after: dict) -> list[str]:
    """Names of the keys that differ, so the summary line is readable."""
    return sorted(
        k for k in set(before) | set(after) if before.get(k) != after.get(k)
    )


def report(before: list[dict], after: list[dict]) -> tuple[bool, str]:
    """Return (identical, human-readable verdict)."""
    diff = first_difference(before, after)
    if diff is None:
        return True, (
            f"IDENTICAL — {len(before)} frames reasoned exactly the same.\n"
            "Every box, face, gate outcome, verdict and event matched, in order."
        )
    if diff["kind"] == "length":
        counts = diff["counts"]
        return False, (
            f"DIFFERENT — the runs processed different numbers of frames: "
            f"before={counts['before']} after={counts['after']}.\n"
            f"The first {diff['index']} frames matched. The extra records come "
            f"from '{diff['extra_from']}'.\n"
            "If the source is a recording, check both runs used lockstep — a "
            "sampled run sees different frames each time and this diff is then "
            "measuring the sampler, not the change."
        )
    fields = ", ".join(diff["fields"])
    return False, (
        f"DIFFERENT — first divergence at record {diff['index']} "
        f"(frame seq {diff['seq']}), in: {fields}\n\n"
        f"  before: {_only(diff['before'], diff['fields'])}\n"
        f"  after : {_only(diff['after'], diff['fields'])}\n\n"
        "Everything after this frame inherits the divergence; debug this one."
    )


def _only(record: dict, fields: list[str]) -> str:
    """Render just the fields that differ, so the two lines are comparable."""
    return json.dumps({k: record.get(k) for k in fields}, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """CLI: diff two capture files and exit non-zero if they differ."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.golden",
        description="Diff two golden decision captures (HECO_GOLDEN_PATH files).",
    )
    parser.add_argument("before", help="capture from the code you had")
    parser.add_argument("after", help="capture from the code you changed")
    parser.add_argument(
        "--with-timings",
        action="store_true",
        help="also compare per-frame wall-clock timings (normally ignored: "
        "they differ between any two runs and say nothing about reasoning)",
    )
    args = parser.parse_args(argv)

    before = load(args.before, keep_timings=args.with_timings)
    after = load(args.after, keep_timings=args.with_timings)
    if not before or not after:
        print("REFUSED — a capture is empty; there is nothing to compare.")
        return 2
    identical, verdict = report(before, after)
    print(verdict)
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
