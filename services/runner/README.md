# runner — the pipeline conductor (port 7100)

**What.** Owns the run lifecycle. In **count mode** it creates the planner-side
run record (`heco_common.planner.PlannerClient`), resets the match gallery and
tracker state, claims the source on ingest, then drives the loop —
`frame → persons → tracker → faces (within tracked boxes) → quality gate →
embed → match → unique count` — timing every stage. Aggregates
(count/min/mean/max) and sampled raw rows are POSTed to the site-planner every
2 s; the run releases its downstream state (camera, tracker, gallery) and ends
with a `PUT … {status: ended, notes}` carrying the unique count, frame count,
staff crossings, staff face-frames, manual additions and sub-canon share.

**The count is the product; reporting is secondary.** No planner call made from
the frame loop can fail a run or stall it for long: errors are swallowed AND
latency is bounded. A planner restart mid-event used to fail the run, and a
restarted run gets a fresh planner id and therefore a fresh EMPTY gallery — so
a five-second hiccup re-counted every guest already counted.

The quality gate lives here (`app/gate.py`), and it is the pipeline's only
irreversible discard: a face it rejects is never embedded, never matched and
therefore never counted. Faces below **56 px** width never reach the embedder;
**56–79 px** pass but are flagged *sub-canon* (POC geometry: 2.8 mm camera at
2.0 m, faces ~64–85 px — see CONTRACTS.md).

The gate is **composite**: three further floors — inter-eye distance,
frontality and Laplacian sharpness, all measured by the faces service — sit
beside box width, because width is neither the size measure recognition uses
(56 px of box is ~24 px of IED) nor able to see a face turned side-on or
smeared by a walking guest. **All three default to unarmed**, so the shipped
gate is exactly the width-only gate it has always been; no floor has been
measured yet, and guessing one on the only irreversible discard costs guests
off an invoice. A signal that could not be measured never rejects a face — it
is counted as `gatedUnmeasured` instead. Every rejection carries the reason
that caused it (`gateReason`) into the tap payload, the annotated frame and the
run status, so the console can say *which* floor dropped a face.

## v1: staff, taps, feedback, enrol

- **Staff whitelist.** A run carrying a `siteId` sends it on every `/match`, so
  the matcher checks the site staff store first. A staff hit is counted as a
  `staffCrossing` and excluded from the guest `unique` count — the track stays
  tracked and visible upstream (suppression would corrupt track association).
  A **crossing is one pass**: the same member re-seen within
  `HECO_STAFF_COOLDOWN_S` is still the same pass. The raw per-frame figure is
  kept separately as `staffFaceFrames` (useful for recall debugging, useless in
  a report — it moves with the frame rate, not with staff behaviour).
- **Debug taps** (every `tap_interval_s`, best-effort). Per stage the runner
  POSTs an annotated JPEG (person boxes / track ids+ages / face boxes coloured
  by quality band / match verdicts, staff grey; ingest posts the raw frame) and
  a structured payload (`app/taps.py`, capped ≤ 32 KB). A planner hiccup or an
  opaque/undecodable frame never blocks the loop — the image upload is simply
  skipped and the structured payload still goes up.
- **Operator feedback** (every `feedback_poll_s`, best-effort). The runner polls
  the planner and applies each open correction to the live gallery via the
  match service — `duplicate → /merge` (unique −1), `false-match → /split`,
  `mark-staff → /mark-staff` (unique −1, REQUIRES a run with `siteId`),
  `missed → /count/manual` (unique +1, recorded as an operator attestation),
  `note` → acknowledged — then PUTs `applied`/`rejected`. An outcome already
  carried out is remembered, so a dropped status update is re-reported with the
  SAME status and the gallery call is never repeated; an unknown kind or a 4xx
  refusal resolves `rejected` rather than claiming `applied`.
- **Enrol mode** (`mode:'enrol'`, requires `siteId` + `staffId`). A staff
  walk-through: capture faces, keep the best `enrol_best_n` by
  `iedPx × frontality`, write them to the site staff store (`/staff/enrol`),
  and `PUT /api/staff/:id` with the sample count. No counting — but it DOES
  open a `mode:'enrol'` pipeline run and settle it with structured results
  (`sampleCount`, `facesSeen`, `frames`, `multiFaceFramesSkipped`), including
  on the failure path. Erasure already had a permanent ledger while the capture
  that CREATES the biometric had none; the run row is best-effort, so a planner
  outage costs the record and never the enrolment.
  **Single subject only**: a frame contributes a sample only when exactly one
  face passes the gate (others are counted in `multiFaceFramesSkipped`), and an
  enrolment that captured nothing FAILS with an instruction to redo the
  walk-through. The staff store is per-site and persistent, and enrolment
  supersedes prior templates, so a bystander caught in the walk-through would
  be matched as that member at every future event at the venue.
- **Reporting never stalls or kills the count.** The stats flush is best-effort
  (`plannerReportErrors` counts what was lost; stats are upserted aggregates,
  so the next flush repairs the gap), planner traffic uses its own short
  timeouts, and a whole tap round shares `HECO_TAP_BUDGET_S`.
- **End of run the runner hands state back**: ingest `/close`, tracker
  `/release`, match `/reset` (deleting the gallery file) — all best-effort,
  before the closing planner PUT.

## API

| method | path | body | returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{ok, model, version}` |
| POST | `/runs` | `{eventId, placementId?, source:{url\|path}, plannerUrl?, label?, mode?, siteId?, staffId?}` | `{runId, state}` |
| GET | `/runs/{runId}` | — | live local status (frames, unique, manualAdditions, staffCrossings, staffFaceFrames, subCanonShare, feedbackApplied/Rejected, multiFaceFramesSkipped, plannerReportErrors, tapRoundsAbandoned, sampleCount, state, error) |
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
end-of-source/stop/failure settlement, the v1 additions (staff-crossing
exclusion, tap + annotated-frame posting and the opaque-frame skip, feedback
merge/split/mark-staff application, the enrol walk-through), and the v2
regressions: a planner outage never failing a run, run-state release, capture
slot conflicts, staff-crossing debounce, feedback status truthfulness, the
`missed` +1 lever, single-subject enrolment, and the tap budget — all offline.
The pure helpers (`taps`, `annotate`, `feedback`) are unit-tested directly.

## Tune

| env | default | meaning |
| --- | --- | --- |
| `HECO_INGEST_URL` … `HECO_MATCH_URL` | `http://<service>:<port>` | Stage service URLs (compose DNS names; point at localhost off-compose) |
| `PLANNER_URL` | `http://host.docker.internal:8787` | The site-planner app |
| `HECO_QUALITY_MIN_PX` | `56` | Quality-gate floor: narrower faces never embed |
| `HECO_QUALITY_CANON_PX` | `80` | Below this, matched faces are flagged sub-canon |
| `HECO_QUALITY_MIN_IED_PX` | `0.0` | Inter-eye-distance floor; **0 = not armed** (ships unarmed) |
| `HECO_QUALITY_MIN_FRONTALITY` | `0.0` | Pose floor 0..1; **0 = not armed** |
| `HECO_QUALITY_MIN_SHARPNESS` | `0.0` | Laplacian-variance floor; **0 = not armed**, and calibrate per camera |
| `HECO_FLUSH_INTERVAL_S` | `2.0` | Planner stats/samples cadence |
| `HECO_TAP_INTERVAL_S` | `2.0` | Debug frame + tap cadence (best-effort) |
| `HECO_FEEDBACK_POLL_S` | `3.0` | Operator-feedback poll cadence (best-effort) |
| `HECO_ENROL_BEST_N` | `5` | Enrol: face samples kept per staff walk-through |
| `HECO_REQUEST_TIMEOUT_S` | `30` | Per-STAGE HTTP timeout (a stage call is the product) |
| `HECO_PLANNER_TIMEOUT_S` | `5.0` | Retrying planner calls (run record, stats, samples) |
| `HECO_REPORT_TIMEOUT_S` | `2.0` | Best-effort planner calls (taps, frames, feedback) — bounds the stall a wedged planner can cause |
| `HECO_TAP_BUDGET_S` | `3.0` | Ceiling for ONE tap round (5 payloads + 5 JPEGs); the rest is dropped |
| `HECO_STAFF_COOLDOWN_S` | `5.0` | A staff member re-seen within this is the SAME crossing |
| `HECO_SOURCE_POLL_S` | `0.02` | Poll interval while ingest's `seq` is unchanged |
| `HECO_SOURCE_STALL_S` | `45.0` | Stalled-seq duration before a run gives up (a stall settles `failed` and KEEPS the gallery) |
| `HECO_RUN_RETENTION_S` | `600` | How long a settled run stays readable from `GET /runs/:id` before it is reaped |

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
