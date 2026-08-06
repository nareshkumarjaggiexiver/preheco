# runner — the pipeline conductor (port 7100)

**What.** Owns the run lifecycle. In **count mode** it creates the planner-side
run record (`heco_common.planner.PlannerClient`), resets the match gallery and
tracker state, claims the source on ingest, then drives the loop —
`frame → persons → tracker → faces (within tracked boxes) → exclusion zones →
quality gate → embed → match → unique count` — timing every stage. Aggregates
(count/min/mean/max) and sampled raw rows are POSTed to the site-planner every
2 s; the run releases its downstream state (camera, tracker, gallery) and ends
with a `PUT … {status: ended, notes}` carrying the unique count, frame count,
staff crossings, staff face-frames, manual additions, healed splits, the two
appearance-veto tallies, near-miss mints, zone-excluded faces and sub-canon
share.

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

## v2: the track heal and exclusion zones

The live bench (3 real people, counted 5) produced two phantom guests no
threshold can fix — the measured impostor ceiling on this camera is 0.377
while same-person misses measured 0.294/0.308/0.361, so the distributions
overlap and `HECO_MATCH_THRESHOLD` stays at 0.363. Both fixes act on evidence
a threshold cannot see (CONTRACTS.md "the track heal and exclusion zones"):

- **Track heal.** Tonight's p00002 was minted on a re-entry frame at cosine
  0.3084, and the SAME track matched the correct identity p00001 at 0.69 four
  seconds later. When a track that recently minted a new identity later
  matches a DIFFERENT existing identity at ≥ `HECO_HEAL_MIN_COSINE` within
  `HECO_HEAL_WINDOW_S`, the loop folds the mint back via
  `POST /merge {…, onlyIfSingleton: true}` — unique −1, `healedSplits` +1. A
  refusal (the mint grew a second template, or an operator split the pair)
  counts nothing and is the system working. **Residual risk, documented in
  the loop and CONTRACTS.md:** a tracker identity swap inside the window can
  fold a real person into the one they crossed paths with — an under-count
  the guards bound but do not eliminate. Staff hits never heal; the reverse
  ordering (match first, mint later) never heals.
- **Exclusion zones.** Tonight's p00004 was an 87 px face seen through a
  frosted glass partition — a real face in the wrong PLACE. The operator draws
  polygons in the planner (normalized 0..1, forwarded by the control proxy as
  `exclusionZones` on `POST /runs`); the loop drops any face whose box centre
  falls inside one, BEFORE the gate, so it is never embedded or matched.
  Excluded faces stay visible — counted in `excludedByZone`, flagged in the
  face-detect tap, drawn magenta inside the translucent zone overlay — and a
  frame with no dimensions to scale the polygons by excludes nothing and
  counts `zoneUnmeasured` instead (the `gatedUnmeasured` honesty pattern).

## v3: the track identity lock (bench 6e1a5d, 2026-08-06)

Ground truth: **ONE person**, walking out of frame, back in, then sitting
down. The tracker gave that one person **six ids** — track 2 (53 frames), 5
(24), 6 (8), 7 (6), 8 (6), 12 (7) — at 3.97 fps, where the old 15-frame
max-age dies after 3.75 s and every gap while the person sat (the detector
losing a seated body) outlasted it. Three of the resulting mints were healed
back into p00001. **Two survived:** p00005 (face 0.212 vs p00001, clothing
0.797) and p00006 (0.228 / 0.875), both under the 0.29 near-miss floor so not
even a banner fired.

- **Why the heal could not reach them.** The heal is a *cure*: it needs a
  LATER comfortable match on the same track to disown the mint. On a subject
  the detector keeps losing, that match never comes.
- **The lock prevents instead.** Once a verdict on a track matches an existing
  identity at ≥ `HECO_TRACK_LOCK_MIN_COSINE` (0.45 — the heal's floor, above
  the 0.377 measured impostor ceiling), that track is *bound* to that identity
  for `HECO_HEAL_WINDOW_S`. A later `isNew` verdict on the same track is
  folded straight back via `POST /merge {…, onlyIfSingleton: true}` — unique
  −1, `lockedTrackFolds` +1, on the spot. `merged=false` counts nothing.
  Locks are per track id (never across ids), expire with the heal window,
  and staff verdicts neither set nor use one.
- **`distinctTracks`.** How many track ids the run ever saw, in the status,
  the notes and the results. Six ids against one real guest is the signal that
  motivated all of this, and it was previously discoverable only by querying
  the run's SQLite by hand.
- **The asymmetry that governs the switch.** A wrong split over-counts and
  somebody argues about the invoice; a wrong MERGE under-counts *silently* and
  nobody ever sees it. The lock makes track identity more authoritative, which
  amplifies exactly that failure if the tracker swaps people — so
  `HECO_TRACK_LOCK_MIN_COSINE=0` turns it off entirely, its folds are counted
  apart from `healedSplits`, and it carries the same clothing guard as the
  heal. The tracker's longer coast (`HECO_TRACKER_MAX_AGE`, now 30 frames)
  widens the swap window, so the two changes are read together.

## Torso appearance: an advisory veto, never a verdict

Clothing is constant within one event, so the loop computes a cheap
**torso-appearance descriptor** for every gate-surviving face and uses it in
exactly two places — both refusals of actions face evidence had already
justified. It never mints, merges, matches or counts: the measured 0.377
impostor ceiling on this camera was two DIFFERENT men **both in light
shirts**, so torso *agreement* proves nothing about identity; only a *clash*
carries information.

- **The descriptor** (`app/appearance.py`, wire contract fixed): 48 floats —
  an L1-normalized 12×4 Hue×Saturation histogram (OpenCV HSV: H 0–179 → 12
  bins, S 0–255 → 4 bins) of the torso crop, with V < 40 (shadow) and V > 240
  (blowout) pixels masked out. Hue/saturation and not value because brightness
  is the axis illumination moves along as a guest walks the frame. The crop:
  within the containing person box (same centre-containment association as
  the track lookup), from the face box's bottom edge down to
  `min(face bottom + 2.5 × face height, person box bottom)`, horizontally the
  person box inset 15% each side. **No descriptor** (None, never zero) when
  the crop is under 24 px in either dimension, there is no containing person
  box, fewer than 100 unmasked pixels remain, or the frame does not decode —
  and an absent descriptor never vetoes anything, the codebase-wide
  absent-is-not-zero convention (`gatedUnmeasured` / `zoneUnmeasured`).
- **Fold guard — the tracker-swap detector, in three bands not one cliff.**
  The residual risk of BOTH folds (the heal and the identity lock) is a
  tracker identity swap: two people cross paths, the track is handed from
  person A to person B, B matches at ≥ 0.45 inside the window, and A's
  singleton mint is folded into B — an under-count of one. A swapped track
  shows A's clothes on the earlier frame and B's on the folding frame, so
  before each `/merge` the loop compares the remembered descriptor with the
  current frame's and bands the histogram intersection:

  | band | reading | what happens |
  | --- | --- | --- |
  | clash | `< HECO_HEAL_APPEARANCE_CLASH` (**0.35**) | fold refused, `healVetoedByAppearance` +1, info log with both keys and the similarity |
  | uncertain | `[clash, HECO_HEAL_APPEARANCE_UNSURE)` (**0.55**) | fold **proceeds**, `healUncertainAppearance` +1 |
  | corroborated | `≥ unsure` | proceeds silently |

  **Why the cliff went.** The single 0.50 floor VETOED a fold at intersection
  **0.4991** on bench 6e1a5d — nine ten-thousandths, on a fold that was
  probably correct (track 2 had linked p00001 and p00005, and the pipeline
  threw that evidence away). The same run bounds what a mid-range reading can
  mean: the same man's surviving splits read 0.797 and 0.875 while two
  genuinely different men reached 0.747 and another impostor pair 0.503. A
  number in between separates nobody, so only a genuine disagreement blocks
  and everything else is recorded rather than acted on. Both numbers remain
  reasoned, not calibrated. `0` in CLASH disables the refusal; absent
  descriptors on either side veto nothing and count nothing. This SHRINKS the
  residual risk; it does not eliminate it (same-colour swaps are invisible to
  it — that is what track-quality gating remains for). Appearance agreement
  never lowers a cosine floor: the signal only ever fails to object.
- **Enrolment veto (match-side, reported here).** The descriptor rides every
  `/match` body as `appearance`; the matcher may refuse to keep a clashing
  sighting as an *additional template* (anti-poison,
  `HECO_MATCH_APPEARANCE_CLASH`, same default) — the verdict itself is never
  changed. The runner reads each reply's `appearanceVetoed` into
  `enrolVetoedByAppearance` and forwards `appearanceSim` (best intersection
  vs the matched identity's stored descriptors; null for staff hits, new
  mints, or a descriptor-less side) into the match tap, so the console can
  watch the advisory signal live. Both counters land in the run status, the
  end-of-run notes and the structured results, beside `healedSplits`.

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
  **Tap frames are ≤ 1280 px wide** (aspect preserved, drawn at native
  resolution then `INTER_AREA`-resized): a round used to encode five FULL
  3840×2160 frames, ~1–1.5 s on the T440, for a console that shows them
  small — and these are the runner's only frame uploads (no separate
  keyframe path exists).
  **The duty-cycle guard** (`HECO_TAP_DUTY_FACTOR`, default 3) additionally
  requires ≥ K × the previous round's measured cost to elapse since that
  round ENDED before the next fires. The interval alone had two measured
  stable states on the T440 (2.3 fps healthy; one transient stall → EVERY
  frame taps → 0.4 fps locked forever, stage timings identical in both):
  the guard bounds the round at ~25% of loop time and makes the lock-in
  impossible by construction. Postponed rounds count `tapRoundsDeferred`
  (distinct from `tapRoundsAbandoned`, which means a STARTED round ran out
  of `HECO_TAP_BUDGET_S` and lost taps). Each round's cost is observed as
  `tapRoundMs` (count board) and each frame's acquisition wait as
  `frameWaitMs` (ingest board; 0.0 when the source had a fresh frame), so
  the next stall is diagnosable from the console: loop-bound reads
  frameWaitMs ~0 with tapRoundMs high.
- **Near-miss mints reach the operator.** A mint the match service flags as a
  near miss of an existing guest (`nearMiss: {key, cosine, appearanceSim}` —
  measured tonight: face 0.3464 vs the 0.363 threshold with clothing 0.562,
  found only by eye) rides the match tap row verbatim, is counted in
  `nearMissMints` (status, notes, results) and logged. It is a one-click
  merge SUGGESTION for the operator, never behaviour: an impostor pair
  measured face 0.377 with clothing 0.503, so the count moves only when the
  operator's `duplicate` correction arrives through the feedback loop.
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
| POST | `/runs` | `{eventId, placementId?, source:{url\|path}, plannerUrl?, label?, mode?, siteId?, staffId?, exclusionZones?:[{label, points:[[x,y],…]}]}` — zone points normalized 0..1, ≥3 per polygon, 422 otherwise | `{runId, state}` |
| GET | `/runs/{runId}` | — | live local status (frames, unique, manualAdditions, staffCrossings, staffFaceFrames, healedSplits, lockedTrackFolds, distinctTracks, healVetoedByAppearance, healUncertainAppearance, enrolVetoedByAppearance, nearMissMints, excludedByZone, zoneUnmeasured, subCanonShare, feedbackApplied/Rejected, multiFaceFramesSkipped, plannerReportErrors, tapRoundsAbandoned, tapRoundsDeferred, sampleCount, state, error) |
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
The track heal replays tonight's measured bench (mint at 0.3084, same track at
0.69 → folded; plus refusal, window, cosine-floor and staff-skip cases) and
the exclusion zones run end to end (a zoned face never reaches `/match` but
stays visible in the tap; a dims-less frame counts `zoneUnmeasured`; malformed
zones 422 at `POST /runs`). The tap-round economics are pinned end to end: a
2560×1440 source's posted JPEGs decode ≤ 1280 px wide, the duty guard defers
inside K × the measured cost and fires after (mutation-checked), the
every-frame lock-in replay taps once and defers the rest with counting
unharmed, factor 0 restores interval-only, and `tapRoundMs` / `frameWaitMs`
reach the stats flush (a stuttering source shows `frameWaitMs > 0`, a ready
one 0.0). The near-miss replay (mint at 0.3464 vs p00004, clothes 0.562)
checks the flag reaches counter, notes, results and the tap row while
`unique` still steps UP. The torso-appearance tests cover the descriptor
contract on synthetic frames (red vs blue clash < 0.5, identical → 1.0, tiny
crop / dark frame / no person box → None), the `/match` `appearance` field
being sent only when computable, the heal veto replay (clashing frames refuse
the fold; agreeing frames heal exactly as before; knob 0 disables), and the
match-side enrolment veto's visibility. The pure helpers (`taps`, `annotate`,
`feedback`) are unit-tested directly.

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
| `HECO_TAP_DUTY_FACTOR` | `3.0` | Duty-cycle guard: a round fires only after this × the previous round's measured cost has passed since it ENDED (~25% duty at 3 — the every-frame 0.4 fps lock-in is impossible by construction). `0` disables; deferred rounds count `tapRoundsDeferred` |
| `HECO_STAFF_COOLDOWN_S` | `5.0` | A staff member re-seen within this is the SAME crossing |
| `HECO_HEAL_WINDOW_S` | `20.0` | How long after a mint the same track's later match may fold it; `0` disables healing |
| `HECO_HEAL_MIN_COSINE` | `0.45` | Heal evidence floor — above the 0.377 measured impostor ceiling with margin (tonight's heal evidence was 0.69); a cross-key match below it heals nothing |
| `HECO_TRACK_LOCK_MIN_COSINE` | `0.45` | Track identity lock: a verdict matching an existing identity at ≥ this binds the track to it, and a LATER mint on that same track is folded back at once (`lockedTrackFolds` +1). Same floor as the heal, above the 0.377 measured impostor ceiling. **0 = lock off**; it is also off whenever healing is (the lock expires on the heal window) |
| `HECO_HEAL_APPEARANCE_CLASH` | `0.35` | Clothing guard on both folds (heal and lock): below this histogram intersection is a CLEAR clash and the fold is refused as a suspected tracker swap (`healVetoedByAppearance`). **Was 0.50**, which vetoed a probably-correct fold at 0.4991. Reasoned, uncalibrated; **0 = refusal off**; absent descriptors never veto |
| `HECO_HEAL_APPEARANCE_UNSURE` | `0.55` | Upper edge of the UNCERTAIN band: a reading in `[clash, unsure)` lets the fold proceed but counts `healUncertainAppearance`, so an operator can see how often the system acted on weak corroboration. Agreement never loosens a cosine floor (an impostor pair measured clothing 0.747) |
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
