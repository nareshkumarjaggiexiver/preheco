# runner — the pipeline conductor (port 7100)

**What.** Owns the run lifecycle. In **count mode** it creates the planner-side
run record (`heco_common.planner.PlannerClient`), resets the match gallery and
tracker state, opens the source on ingest, then drives the loop —
`frame → persons → tracker → faces (within tracked boxes) → quality gate →
embed → match → unique count` — timing every stage. Aggregates
(count/min/mean/max) and sampled raw rows are POSTed to the site-planner every
2 s; the run ends with a `PUT … {status: ended, notes}` carrying the unique
count, frame count, staff-crossings and sub-canon share.

The quality gate lives here: faces below **56 px** width never reach the
embedder; **56–79 px** pass but are flagged *sub-canon* (POC geometry: 2.8 mm
camera at 2.0 m, faces ~64–85 px — see CONTRACTS.md).

## v1: staff, taps, feedback, enrol

- **Staff whitelist.** A run carrying a `siteId` sends it on every `/match`, so
  the matcher checks the site staff store first. A staff hit is counted as a
  `staffCrossing` and excluded from the guest `unique` count — the track stays
  tracked and visible upstream (suppression would corrupt track association).
- **Debug taps** (every `tap_interval_s`, best-effort). Per stage the runner
  POSTs an annotated JPEG (person boxes / track ids+ages / face boxes coloured
  by quality band / match verdicts, staff grey; ingest posts the raw frame) and
  a structured payload (`app/taps.py`, capped ≤ 32 KB). A planner hiccup or an
  opaque/undecodable frame never blocks the loop — the image upload is simply
  skipped and the structured payload still goes up.
- **Operator feedback** (every `feedback_poll_s`, best-effort). The runner polls
  the planner and applies each open correction to the live gallery via the
  match service — `duplicate → /merge` (unique −1), `false-match → /split`,
  `mark-staff → /mark-staff` (unique −1), `missed`/`note` → acknowledged — then
  PUTs `applied`/`rejected`. Corrections are idempotent, so a dropped status
  update is harmless (the item is retried next poll).
- **Enrol mode** (`mode:'enrol'`, requires `siteId` + `staffId`). A staff
  walk-through: capture faces, keep the best `enrol_best_n` by quality, write
  them to the site staff store (`/staff/enrol`), and `PUT /api/staff/:id` with
  the sample count. No pipeline_run, no counting.

## API

| method | path | body | returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{ok, model, version}` |
| POST | `/runs` | `{eventId, placementId?, source:{url\|path}, plannerUrl?, label?, mode?, siteId?, staffId?}` | `{runId, state}` |
| GET | `/runs/{runId}` | — | live local status (frames, unique, staffCrossings, subCanonShare, sampleCount, state, error) |
| POST | `/runs/{runId}/stop` | — | ends the run after the current frame (RTSP sources never end alone) |

`mode` is `count` (default) or `enrol`. `siteId` opts a count run into the staff
whitelist and is required (with `staffId`) for enrol.

End-of-source: ingest serves the LATEST frame with an increasing `seq` and never
signals EOF explicitly — the runner treats a **stalled seq** (no new frame for
`HECO_SOURCE_STALL_S`) as the end. Stub-style explicit ends (`ended: true`,
missing `imageB64`, HTTP 204/404/410) also end the run.

## Run

```sh
make venv     # python3.12 -m venv .venv && pip install -r requirements.txt
make run      # uvicorn on 0.0.0.0:7100
```

## Test

```sh
make test     # pytest — the loop against fake stage services (httpx.MockTransport)
make lint     # ruff
```

Tests cover orchestration order, aggregation maths, sample-batch capping,
end-of-source/stop/failure settlement, and the v1 additions: staff-crossing
exclusion, tap + annotated-frame posting (and the opaque-frame skip), feedback
merge/split/mark-staff application, and the enrol walk-through — all offline.
The pure helpers (`taps`, `annotate`, `feedback`) are unit-tested directly.

## Tune

| env | default | meaning |
| --- | --- | --- |
| `HECO_INGEST_URL` … `HECO_MATCH_URL` | `http://<service>:<port>` | Stage service URLs (compose DNS names; point at localhost off-compose) |
| `PLANNER_URL` | `http://host.docker.internal:8787` | The site-planner app |
| `HECO_QUALITY_MIN_PX` | `56` | Quality-gate floor: narrower faces never embed |
| `HECO_QUALITY_CANON_PX` | `80` | Below this, matched faces are flagged sub-canon |
| `HECO_FLUSH_INTERVAL_S` | `2.0` | Planner stats/samples cadence |
| `HECO_TAP_INTERVAL_S` | `2.0` | Debug frame + tap cadence (best-effort) |
| `HECO_FEEDBACK_POLL_S` | `3.0` | Operator-feedback poll cadence (best-effort) |
| `HECO_ENROL_BEST_N` | `5` | Enrol: face samples kept per staff walk-through |
| `HECO_REQUEST_TIMEOUT_S` | `30` | Per-stage HTTP timeout |
| `HECO_SOURCE_POLL_S` | `0.02` | Poll interval while ingest's `seq` is unchanged |
| `HECO_SOURCE_STALL_S` | `5.0` | Stalled-seq duration treated as end-of-source |

## Notes

- The planner write side is `heco_common.planner.PlannerClient` (shared
  `common/` package, installed editable); the runner adapts it onto its own
  httpx client — one JSON transport and one multipart `file_transport` — so
  tests fake the planner with `httpx.MockTransport`.
- Annotation uses OpenCV (via `heco_common`, already a dependency); everything
  else is I/O-bound HTTP glue. No compiled rewrite until profiling shows the
  conductor itself is the bottleneck.
- `siteId` reaches the runner from the planner's control proxy (it resolves the
  event's site); the staff store is keyed on it. A count run without `siteId`
  simply skips the staff check and counts every face as a guest.
