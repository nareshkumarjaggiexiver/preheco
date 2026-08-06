# tracker (port 7103)

**What.** Our own clean-room SORT-style multi-object tracker, stateful per
`runId`. Per frame: IoU association of detections against
constant-velocity-predicted track boxes (greedy best-first matching),
max-age coasting, min-hits confirmation. `/health` reports
`model: "sort-lite-iou-velocity"` — there is no DNN here.

**What it is, honestly** (full write-up in `app/sort.py`):

- *Is*: IoU + smoothed constant-velocity prediction; enough to hold an id
  for the 3–6 seconds of a gate crossing.
- *Is not*: no re-identification (appearance is never used — a person who
  leaves and returns is a new id; the face gallery downstream repairs
  identity, by design per `docs/planning/03`), no Kalman covariance tuning,
  no Hungarian assignment. **Greedy vs Hungarian trade-off:** greedy is
  dependency-free and decisive at POC gate densities; it can mis-pair when
  ≥3 boxes contest one region — swap in
  `scipy.optimize.linear_sum_assignment` if crowded-gate evidence demands.
- **Crossing behaviour (documented):** two boxes on distinguishable paths
  *hold* their ids through a cross (prediction separates them); exactly
  coincident detections are ambiguous and *may* swap — accepted at POC.
- Frame-based, not time-based: `tMs` is accepted on the wire but max-age
  and min-hits count frames.

## API

| method | path | body / response |
| ------ | ---- | --------------- |
| POST | `/reset` | `{runId}` — create/clear that run's state (also re-reads env) |
| POST | `/release` | `{runId}` — forget the run entirely (end of run); idempotent |
| POST | `/track` | `{runId, tMs, boxes:[{x,y,w,h,conf}]}` → `{tracks:[{id, box, ageFrames, hits}]}` |
| GET | `/health` | `{ok, model, version, runs}` — `runs` = resident run trackers |

Per-run state has a LIFECYCLE. Every run gets a fresh id, so the run table only
ever grew: a season of events left one dead tracker per run resident until the
process restarted. The runner calls `/release` when a run ends, and `/track`
evicts runs untouched for `TRACKER_RUN_TTL_S` as the backstop for runs that
died without releasing.

Unknown `runId` on `/track` auto-creates the run. Only tracks *updated this
frame* and past min-hits are returned (min-hits is waived during the first
frames of a run so counting starts immediately); coasting tracks are
predicted silently and re-associate on reappearance within max-age.
`ageFrames` counts frames since birth including gaps; `hits` counts matched
frames. (CONTRACTS.md's table shorthands `age` — the field is `ageFrames`.)

## Run

```sh
make venv
make run          # uvicorn on 7103 (PORT=... to override)
curl -s localhost:7103/track -X POST -H 'content-type: application/json' \
  -d '{"runId":"demo","tMs":0,"boxes":[{"x":10,"y":50,"w":20,"h":40,"conf":0.9}]}'
```

## Test

```sh
make test         # synthetic box sequences + in-process client; no network
make lint
```

Covered: straight pass keeps one id; crossing boxes hold ids (see above);
dropout < max-age re-associates to the same id; dropout > max-age is a new
id; per-run isolation; reset semantics; env tuning.

## Tune

| env | default | meaning |
| --- | ------- | ------- |
| `HECO_TRACKER_MAX_AGE` | `30` | Frames a track may coast unmatched before it is dropped. **Was 15**; bench 6e1a5d (2026-08-06) counted SIX ids for ONE person (tracks 2/5/6/7/8/12) because at 3.97 fps 15 frames is 3.75 s and every seated-detector gap longer than that minted a new id. 30 ≈ 7.5 s. Treats a symptom — the disease is the person detector losing seated bodies — and a longer coast widens the window for a ghost track to latch onto a DIFFERENT person, which the runner's clothing guard exists to catch. |
| `TRACKER_MAX_AGE` | (unset) | Deprecated spelling of the above, still read so an existing deployment's explicit value is not silently ignored. `HECO_TRACKER_MAX_AGE` wins when both are set. |
| `TRACKER_MIN_HITS` | `3` | Matched frames before a track is reported (ghost suppression). |
| `TRACKER_IOU_MIN` | `0.2` | Association gate: pairs below this IoU never match. |
| `TRACKER_RUN_TTL_S` | `3600` | Forget a run's tracker after this many seconds without a `/track`. `0` disables eviction. |
| `TRACKER_VEL_SMOOTH` | `0.5` | Velocity filter alpha (1.0 = trust newest delta only). |

Env is read when a run's tracker is created — POST `/reset` to apply changes
to an existing runId.
