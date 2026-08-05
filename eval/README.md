# `eval/` — the accuracy harness

The thing that says whether a change to the pipeline helped. It replays
labelled ground-truth clips through the **real** pipeline, scores the unique
count against the human count of record, and refuses to call a change an
improvement when accuracy went backwards.

This is step 5 of the pilot-critical shortlist in
[`docs/planning/pipeline/accuracy-and-tuning.md`](../../../docs/planning/pipeline/accuracy-and-tuning.md);
§5 of that document is the specification this implements.

---

## Why it exists

On 2026-08-05 `INGEST_MAX_WIDTH=1920` was set on the real server (downscale 4K
to 1080p before the pipeline). Measured against the real camera:

| | before | after |
| --- | --- | --- |
| frames per second | 1.59 | **7.89** |
| mean detected face width | 129 px | 48 px |
| mean interocular distance | 46 px | 23 px |
| faces passing the 56 px gate | 9 | **0** |
| embed / match stages | ran | **never ran** |
| **people counted** | 9 | **0** |

It looked like a five-fold win on the only number anybody was watching, and it
had silently stopped counting people. Nothing in the repository caught it.

So the harness is built on one rule: **fps is reported, never scored.** The
whole funnel is measured, every rung that goes to zero is a named and loud
check, and the A/B refuses the word "improvement" to any change that made the
count worse — however much faster it got.

That exact failure is a test:
`tests/test_compare.py::test_todays_ingest_max_width_change_is_called_a_regression`.

---

## What it measures

**Headline — signed unique-count error**, so deflation (guests never captured)
and inflation (one guest split across two gallery keys) are distinguishable:

```
unique_count_error = (pipeline_unique - actual_unique) / actual_unique
```

Reported against the committed envelope of **± 5 %**.

**The funnel**, per clip, from the stats the runner already emits:

```
frames -> person boxes -> faces detected -> faces PASSING the gate -> embeds -> matches -> unique
```

**The face sizes actually seen** — `faceBoxWPx` and `faceIedPx`, min / mean /
max, at detection and after the gate — and the achieved fps (both the ingest
decode rate and the end-to-end `count`-stage rate, kept apart because they are
not the same number).

**Named checks**, each pass/fail with a sentence:

| check | severity | fires when |
| --- | --- | --- |
| `no-frames` | critical | the run pulled no frames at all |
| `no-faces-detected` | critical | frames processed, not one face found |
| `gate-pass-collapse` | critical | faces detected, **zero** passed the gate |
| `embed-ran` | critical | gate survivors existed, the embedder ran on none |
| `match-ran` | critical | embeddings produced, the matcher never called |
| `counted-nobody` | critical | the clip has people, the count is zero |
| `unique-count-error` | fail | outside the ± 5 % envelope |
| `gate-pass-rate` | warn | under 10 % of detected faces survive the gate |
| `face-size-vs-floor` | warn | the *mean* detected face is already below the embed floor |

`face-size-vs-floor` is the leading indicator — 48 px mean against a 56 px
floor — and it fires while a handful of faces are still squeaking through,
i.e. before the collapse.

---

## What it refuses to do

* **Score a run that did not finish.** Only `endReason == source-ended` is a
  complete count. A `source-stalled` run stopped because the camera vanished;
  scoring it reads as an enormous false deflation and would poison the A/B, so
  it is reported as **ERRORED**, never as a bad score. Same for
  `operator-stopped`, and for a run that never settled at all.
* **Touch a live run.** Before starting anything it asks ingest who holds the
  single capture slot and refuses outright if anyone does (`--skip-slot-check`
  to override). It stops only run ids it started itself; there is no code path
  that can stop, seize or score anybody else's run.
* **Average an unscorable clip away.** Errored clips are counted and named in
  the aggregate, never folded into a mean.

---

## Setup

```bash
cd eval && make venv        # own venv, like every other directory here
make test                   # 88 offline tests: metrics, filters, verdicts
make lint
```

## The manifest

A checked-in JSON file describing each ground-truth clip. The clips themselves
are guests' faces and are **never** committed; the claims about them are.
[`manifest.example.json`](manifest.example.json) is the documented example —
copy it, do not edit it.

```json
{
  "version": 1,
  "eventId": "evt-pilot-2026-09",
  "clips": [
    {
      "label": "baraat-surge",
      "path": "/srv/heco/clips/baraat-surge.mp4",
      "actualUnique": 61,
      "tags": ["surge"],
      "notes": "the crush; heavy occlusion"
    }
  ]
}
```

| field | rule |
| --- | --- |
| `label` | unique — it is the A/B join key |
| `path` | **absolute, as the ingest service sees it** (ingest opens it, usually from inside a container with its own mounts) |
| `actualUnique` | the human count of record; at least 1, because the headline metric is relative |
| `tags` | zero or more of `surge`, `veil`, `children`, `staff`, `evening-light` — a closed set, so a typo cannot quietly split the per-tag breakdown |
| `eventId` / `siteId` | manifest-level defaults; `--event-id` overrides |

## Running it

```bash
# from the repo root
eval/.venv/bin/python -m eval.run --manifest eval/clips.json --label baseline
# or
make eval MANIFEST=eval/clips.json LABEL=baseline
```

Clips run **sequentially** — ingest owns one capture slot — each through
`POST /runs` with a file source (`loop:false`, so the source actually ends).
Results are written per label:

* `eval/results/<label>.json` — machine-readable, the A/B's input
* `eval/results/<label>.txt` — the readable summary

Both carry the runner run id **and** the planner run id for every clip, so any
number can be traced back to the planner record months later. Re-running a
label replaces that label's files and nothing else; `--resume` carries forward
clips already scored (errored clips are always retried).

`--resume` is bound to the **manifest content**, not just the label. A stored
score is reused only when the entry that produced it is still the entry in
front of it — same `path`, same `actualUnique`, same `tags`, same `--envelope`.
Change any of them and that clip is re-run, because a label is only a name:
editing a clip's path and resuming would otherwise report the old file's
number, silently, under the new claim.

Useful flags: `--dry-run`, `--check-paths` (stat clip files locally — only
valid when ingest shares this filesystem), `--timeout` (per clip, default
1800 s), `--envelope`, `--event-id`, `--site-id`, `--runner-url`,
`--planner-url`, `--planner-token`.

Exit code `0` when every clip scored and passed, `2` otherwise, `1` on a
configuration error — so a commit can be gated on it. (`eval.compare` has its
own table below.)

## The A/B

```bash
eval/.venv/bin/python -m eval.compare results/before.json results/after.json
# or
make eval-compare BEFORE=eval/results/before.json AFTER=eval/results/after.json
```

Prints per-clip and aggregate deltas and one verdict:
`regression` / `improvement` / `neutral` / `inconclusive`. It is a
**regression** — printed as a banner, exit code 2 — if on *any* clip:

* the gate-pass count collapsed to zero, or the pass **rate** fell by half;
* the gate-pass or face-detection **yield per frame** fell by half;
* the signed unique-count error worsened past the tolerance;
* the count fell to zero where it previously counted people;
* a check of blocking severity — `critical` **or** `fail` — that used to pass
  now fails;
* a clip that used to be scorable is now unscorable;
* a clip that was measured BEFORE is **absent** AFTER;
* a clip's `actualUnique` differs between the two files — the two errors are
  measured against different denominators, so they are not the same
  measurement and differencing them means nothing.

**Why yield per frame as well as the pass rate.** A rate is a ratio, and a
ratio is invariant to its own denominator. Detect 400 faces and pass 200, then
detect 40 and pass 20: the pass rate reads 50 % both times while nine tenths of
the evidence has gone. The rungs are therefore also compared per frame — per
frame rather than raw, because the change under test moves the frame count
itself (254 frames before the 2026-08-05 downscale, 1,261 after), and a rule on
raw counts would shout at the change that *fixed* it.

**Why a missing clip is a regression.** Otherwise deleting the failing clip
from the manifest is a way to pass. A clip that disappears has not improved; it
has stopped being measured. To retire one, say so on the record with
`--waive-missing LABEL` (repeatable) — the waiver is printed and stored in the
comparison JSON.

**Warnings reach the verdict.** A warning-severity check that newly fails is a
`CONCERN`: never a regression on its own, but enough to hold the verdict at
`inconclusive`. `face-size-vs-floor` is a warning, it fired at 48 px against a
56 px floor, and it fired *before* the count collapsed — an A/B that could not
hear it would have blessed the change on its way down.

"Improvement" therefore requires that **nothing** regressed on **any** clip and
that no new warning fired. A change that improves the aggregate while wrecking
the baraat surge is a conversation, not a ship (§5). The fps delta is printed
underneath, labelled as taking no part in the verdict.

Exit codes: `0` improvement or neutral, `2` regression, `3` inconclusive
(nothing comparable, or a new warning — a claim nobody can check is not a
pass), `1` on a usage error.

---

## One change outside `eval/`

`services/runner/app/loop.py` now observes `faceBoxWPx` on the **embed** stage,
once per face that actually came back from the embedder. That funnel rung was
genuinely not emitted: the embed stage's `frames` counts *frames*, and the gate
survivor count is what was *offered*, not what returned — `zip(..., strict=False)`
silently drops any shortfall. Without it the harness cannot distinguish "the
embedder ran on everything the gate passed" from "the embedder ran on nothing",
which is exactly the rung that went to zero on 2026-08-05. Pinned by
`services/runner/tests/test_loop.py::test_embed_stage_reports_how_many_faces_were_actually_embedded`.

Runs recorded before that change report `embeds: null` — **unknown, which is
not zero** — and the `embed-ran` check stands down rather than accusing an old
run of a collapse it did not have.
