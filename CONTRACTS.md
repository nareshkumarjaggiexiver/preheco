# heco-pipeline — service contracts (v0, POC)

The counting pipeline as Python microservices, reporting every stage into the
site-planner app (the operations console) under an EVENT. This file is the
fixed contract all services build against.

## POC geometry (why the thresholds look low)
Baseline camera: the UNV 2.8 mm fixed (IPC2128LR3-DPF28M-F), mounted at
**2.0 m**, subjects passing **2–3 m** from it — the KB's documented
"close-zone cheat" (docs/KB/hardware/camera/03, §6). Expected face widths
**~64–85 px** — BELOW the production 80/100 px canon, accepted deliberately
for the POC and recorded as such in every report. Quality floor for the POC:
face width ≥ 56 px to enter embedding; flag 56–79 px as "sub-canon".

## Topology (docker-compose)
One shared base image (python:3.12-slim + opencv-python-headless +
onnxruntime). Services are FastAPI apps; the RUNNER orchestrates by HTTP —
no message bus at POC scale (NATS arrives with multi-camera).

| service   | port | job |
| --------- | ---- | --- |
| ingest    | 7101 | RTSP/file → JPEG frames on demand: GET /frame (latest), POST /open {url \| path, owner?, takeover?}, POST /close {owner?} |
| persons   | 7102 | POST /detect {imageB64} → {boxes:[{x,y,w,h,conf}]} — YOLOX-nano ONNX (Apache-2.0) |
| tracker   | 7103 | POST /track {runId, boxes, tMs} → {tracks:[{id,box,ageFrames,hits}]} — own SORT-style IoU+velocity, stateful per runId (POST /reset {runId}, POST /release {runId}) |
| faces     | 7104 | POST /detect {imageB64, within?:[boxes]} → {faces:[{box,landmarks,conf,widthPx,quality,iedPx?,frontality?,sharpness?}]} — YuNet (OpenCV zoo, MIT) |
| embed     | 7105 | POST /embed {imageB64, faces} → {embeddings:[[128]]} — SFace (OpenCV zoo) |
| match     | 7106 | POST /match {runId, embedding, quality?, appearance?} → {personKey, isNew, cosine, galleryN, templateN, templateAdded, appearanceSim, appearanceVetoed, templateId, nearMiss} — gallery in SQLite per runId, cosine threshold 0.363 (SFace paper operating point; POC-tunable via env), several templates per guest, advisory torso-appearance tie-breaker + near-miss mint flag (v2 below; withheld for a `cannot_link` pair since v3) |
| runner    | 7100 | POST /runs {eventId, source, plannerUrl} — drives the loop, batches stats to the planner |

All services: GET /health → {ok, model, version}. Frames as base64 JPEG in
JSON for POC simplicity (shared-memory arrives if profiling demands it —
measure first; do NOT reach for Go/Rust/C++ until a stage is proven the
bottleneck: OpenCV and onnxruntime are already native code).

## Planner ingest (the app side, already scheduled in its schema v4)
POST /api/pipeline/runs                {eventId, placementId?, label?, config?}
PUT  /api/pipeline/runs/:id            {status: ended|failed, notes?}
POST /api/pipeline/runs/:id/stats      {stage, frames, fps, metrics:{name:{count,min,mean,max}}}
POST /api/pipeline/runs/:id/samples    {samples:[{stage,tMs,metrics}]}   (batch ≤200, ~every 2 s)
Stage names: ingest | person-detect | track | face-detect | quality | embed | match | count.
Measured metrics the console charts: personBoxHPx, faceBoxWPx, embedMs, matchCosine.

## Conventions
- Python 3.12; **one venv per service** (`make venv` in each); ruff + pytest.
- Every module has docstrings; every service a README (what, run, test, tune).
- Model weights are DOWNLOADED by `make models` (never committed); pinned URLs
  + sha256 in models.lock. Licences: YOLOX Apache-2.0; YuNet MIT, SFace Apache-2.0
  (per OpenCV-zoo model dirs; models.lock is authoritative) — nothing
  GPL/non-commercial (project hard rule).
- docker-compose.yml at repo root; `runner` reads PLANNER_URL (default
  http://host.docker.internal:8787). Everything runs CPU-only for POC.
- Tests use tiny synthetic images + golden fixtures; no network in tests.


## v1 additions — debug taps, staff whitelist, operator feedback (2026-08-04)

### Debug taps (runner → planner, best-effort, never blocks the loop)
Every ~2 s while running:
- POST {planner}/api/pipeline/runs/:id/frames   multipart form: stage, file
  (JPEG annotated with that stage's output: person boxes / track ids / face
  boxes+quality / match verdict labels; ingest posts the raw frame). Server
  keeps latest + a ring of 4 per stage.
- POST {planner}/api/pipeline/runs/:id/taps     {stage, payload}
  payload = the stage's latest structured output, truncated to what a human
  debugger needs (≤32 KB): boxes, track ids+ages, face quality flags,
  personKey + cosine per match, unique/staff counters.

### Staff whitelist (the professional enrolment flow)
- SITE STAFF STORE: data/staff-<siteId>.db (SQLite; embeddings table:
  staff_id TEXT, vec BLOB float32[128], created_at). Brute-force cosine at
  roster scale; sqlite-vec is the documented growth path past ~5k vectors.
- ENROL MODE: POST runner /runs {mode:'enrol', eventId, staffId, source,
  plannerUrl}. The runner captures faces from the walk-through, keeps the
  best N=5 by quality, appends embeddings to the site staff store, then
  reports sampleCount to the planner (PUT /api/staff/:id — enrolledAt,
  sampleCount). The operator confirms identity in the UI before the next
  person walks.
- COUNT MODE: the matcher checks the STAFF STORE FIRST (threshold as guest
  matching). A staff hit tags the track staff — it stays tracked and visible
  (grey in the console), is excluded from the unique GUEST count, and
  increments a staffCrossings stat. Staff are never silently dropped from
  tracking: suppression would corrupt track association and hide occlusions.

### Operator feedback loop
- Runner polls GET {planner}/api/pipeline/runs/:id/feedback?since=<iso>
  every ~3 s; for each open item it can act on:
    false-match {personKeys:[a,b]}  → split: keep both keys, raise the pair's
                                      internal distance (no auto-merge later)
    duplicate   {personKeys:[a,b]}  → merge: b's embeddings fold into a,
                                      unique count decremented once
    mark-staff  {personKey, staffId?} → move embeddings to the staff store
                                      (staffId if operator picked a roster
                                      member; else an anonymous staff entry),
                                      decrement unique count
    missed / note                    → acknowledged only (audit trail)
  After acting: PUT {planner}/api/feedback/:id {status:'applied'|'rejected'}.

### Planner-side endpoints backing the above (server v6)
- Devices: GET/POST /api/sites/:id/devices, PUT/DELETE /api/devices/:id;
  GET /api/devices/:id/live?stream=<label>&fps=4 — ffmpeg RTSP→MJPEG proxy
  (multipart/x-mixed-replace) so the browser can watch any channel.
- Staff: CRUD under /api/sites/:id/staff; PUT /api/events/:id/staff.
- Control proxy: POST /api/pipeline/control/start|stop → RUNNER_URL.


## v1 addition — staff erasure (consent withdrawal), 2026-08-05

Deleting a staff member in the planner removes the roster row and the consent
record, but their FACE EMBEDDINGS live in the pipeline's per-site staff store
(`data/staff-<siteId>.db`) on the analysis box. Without a signal across the two
repos, a member who withdrew consent keeps being matched as staff at every
future event at that venue — the erasure half of DPDP is simply missing.

The tombstone ledger closes it:

1. PLANNER, on `DELETE /api/staff/:id`: in ONE transaction, insert a row into
   `staff_tombstones` (schema v7 — id, site_id, staff_id, name, deleted_at,
   purged_at NULL) and delete the staff row. The tombstone has no foreign keys
   on purpose: the staff row is already gone and the signal must outlive even a
   site deletion.
2. RUNNER polls `GET /api/staff-tombstones?siteId=` — returns the site's OPEN
   tombstones `[{id, siteId, staffId, deletedAt}]`. Called once BEFORE the
   first frame of a run (so a member deleted while the pipeline was off can
   never match again — this ordering is the guarantee, pinned by test) and then
   every `feedback_poll_s` during the run (so a mid-event withdrawal lands
   within seconds).
3. RUNNER relays the ids to `POST {match}/staff/purge {siteId, staffIds[]}`.
   The match service removes every template keyed by each id from the site
   store and returns `{siteId, removed: {staffId: templatesRemoved}}`. Erasure
   is IDEMPOTENT: an id with no templates reports 0 removed and is success —
   the guarantee is that nothing remains, not that something was found.
4. RUNNER confirms each purge with `PUT /api/staff-tombstones/:id`, which
   stamps `purged_at`. The row is never deleted: a consent withdrawal and the
   proof it was honoured are exactly what an audit asks for. Confirmation is
   idempotent (a second PUT keeps the first stamp).

Every step is best-effort and retried: a planner or match hiccup leaves the
tombstone OPEN, and the next cycle (or the next run's start-up check) tries
again. A dropped confirmation only means the purge re-runs, which is harmless.
The runner reports a `staffPurged` counter in its status.

## v2 — corrections from the 2026-08-05 adversarial review

Eleven confirmed findings, all of which either killed a live run, corrupted a
count, or put a number in the report that did not mean what its name said.
The contract changes they forced are below; the principle behind most of them
is one sentence: **the count is the product, reporting is secondary.**

### The capture slot is exclusive (ingest)

`POST /open` takes `owner` (the runner's run id) and `takeover` (bool).

- An `/open` carrying an `owner` CLAIMS the single capture slot.
- A later `/open` by a DIFFERENT owner is refused **409**, and the detail names
  the run holding it. Previously it silently replaced the live run's source:
  starting a staff enrolment during a gate count made the count run consume the
  enrolment walk-through, with no error anywhere.
- Escapes: the same owner may re-open (idempotent restart); a slot whose
  capture thread has died is claimable; `takeover: true` is the operator's
  explicit seizure. An `/open` with **no** owner keeps the old
  replace-anything behaviour (ad-hoc probes, the smoke script).
- `POST /close {owner?, force?}` releases the slot; owner-checked so a stale
  stop cannot take the camera from the run that holds it now. Idempotent.
- `GET /health` now also returns `owner`.

### Per-run state has a lifecycle (runner, tracker, match)

Every run got a fresh uuid-suffixed planner id, so nothing downstream was ever
cleaned up: one dead SortLite per run resident in the tracker forever, and one
`gallery-<runId>.db` per run on disk forever — each holding real guests' face
embeddings, which is a retention liability, not just disk.

- The runner RELEASES its state at the end of every run (and on failure),
  before the closing `PUT /api/pipeline/runs/:id`: ingest `/close`, tracker
  `/release`, match `/reset` (which deletes the gallery file). All best-effort
  — a run that has produced its number must never fail while tidying up.
- `POST tracker /release {runId}` → `{ok, runId, released}`; idempotent.
  Tracker also evicts runs untouched for `TRACKER_RUN_TTL_S` (default 3600) on
  every `/track`, as the backstop for runs that die without releasing.
  `GET tracker /health` returns `runs` (resident count) so a leak is visible.
- `POST match /gallery/sweep {maxAgeS?}` → `{swept:[runId], maxAgeS}` deletes
  gallery files older than `maxAgeS` (default 24 h). Staff stores are NEVER
  swept — they are meant to persist.

### Enrolment is single-subject, and ranked by recognisability

- A frame contributes a sample only when **exactly one** face passes the
  quality gate. Frames with two or more are skipped and counted in the run
  status as `multiFaceFramesSkipped`.
- An enrolment that captured no single-subject frame **fails** (state
  `failed`, with an error telling the operator to walk the member through
  alone and re-run) rather than quietly reporting zero samples.
- The best-N shortlist is ranked by `iedPx × frontality` (both already emitted
  by faces), falling back to box width only when landmarks are absent.

Why it is worth failing an enrolment: the store is per-SITE and PERSISTENT,
and `staff.enrol` supersedes the member's prior templates. A bystander caught
in the walk-through therefore becomes permanently matched as that staff member
at every future event at the venue and silently excluded from every guest
count, while the real member may stop matching — with no signal that it
happened.

### staffCrossings means crossings

- `staffCrossings` counts **passes**: the same staff member seen again within
  `HECO_STAFF_COOLDOWN_S` (default 5 s) of their last sighting is still the
  same pass. It used to be incremented per matched face per frame, so one
  waiter passing 30 times reported hundreds — a number that moved with the
  frame rate, not with staff behaviour.
- `staffFaceFrames` is the raw per-frame figure, kept because it is genuinely
  useful for debugging recall. Both appear in the run status, in the match tap
  payload, and in the end-of-run notes.

### The feedback ledger tells the truth

The kinds table, revised:

    duplicate   {personKeys:[a,b]}    → merge b into a          (unique −1)
    false-match {personKeys:[a,b]}    → split (do-not-merge)     (unique ±0)
    mark-staff  {personKey, staffId?} → move to the staff store  (unique −1)
                                        REQUIRES a run with siteId
    missed      {note?}               → manual count            (unique +1)
    note                              → acknowledged only (audit trail)

- **mark-staff requires a siteId.** `POST match /mark-staff` without one is
  **400**, and the runner maps it to `invalid` → resolved `rejected`. It used
  to delete the person's templates with nowhere to put them, return `moved>0`,
  and resolve `applied` — after which the person was counted as a brand-new
  guest at their next crossing. The correction silently undid itself.
- **An applied correction is never re-resolved as rejected.** The runner
  remembers the outcome of every correction it has carried out but not yet
  managed to report; a dropped `PUT /api/feedback/:id` is retried with the
  SAME status and the gallery call is NOT repeated. This is also what makes
  the (non-idempotent) `missed` lever safe: +1 means +1.
- **Nothing the runner cannot do is reported `applied`.** An unrecognised kind
  resolves `rejected` (it still closes, so it does not re-poll forever); a
  4xx from the match service settles as `rejected`; a 5xx leaves the item open
  for the next poll.

### The operator can correct an under-count

`POST match /count/manual {runId, note?}` → `{personKey, galleryN, manual:true}`

Under-counting is this pipeline's dominant failure mode (open-set 1:N matching
at a 1:1 verification threshold), and every previously implemented lever moved
the count DOWN. `missed` now mints a person with an `m#####` key and **no
embedding**: they count towards the unique total, can never be matched into,
and stay permanently distinguishable from an automatically detected `p#####`
person. The runner reports `manualAdditions` in its status, in the match tap,
and in the end-of-run notes, so a report can always state how much of the
headline number a human attested by hand. `note` stays audit-only.

### Reporting can never stall or kill the count

- The stats flush is best-effort. It used to raise `PlannerError` straight
  into the frame loop, failing the run — and a restarted run gets a fresh
  planner id and therefore a fresh EMPTY gallery, so every guest already
  counted was counted again. A five-second planner restart corrupted the
  event total. Lost reports are counted in `plannerReportErrors`; stats are
  upserted aggregates, so the next successful flush repairs the gap.
- **Three timeouts, not one.** Stage calls keep `HECO_REQUEST_TIMEOUT_S`
  (30 s, the product); retrying planner calls get `HECO_PLANNER_TIMEOUT_S`
  (5 s); single-shot taps/frames/feedback get `HECO_REPORT_TIMEOUT_S` (2 s).
  "Best effort" used to mean only "errors are swallowed", which does nothing
  about a planner that accepts connections and then answers slowly — and
  ingest's drop-not-queue slot discards every crossing during a stall.
- A whole tap ROUND (5 payloads + 5 JPEGs) additionally shares
  `HECO_TAP_BUDGET_S` (3 s); when it is spent the rest of the round is
  dropped and counted in `tapRoundsAbandoned`.

### Match is opened once, scanned in memory

Every `/match` used to `sqlite3.connect` + run the schema script + `SELECT`
every row + rebuild the numpy matrix from BLOBs — twice with a siteId, three
times on a staff hit, dozens of times a second.

Stores are now opened **once per file** for the life of the process, with the
key list and the `(n, dim)` float32 matrix resident and updated write-through
on every mutation. SQLite is still the durable record (WAL +
`synchronous=NORMAL`; a per-run gallery is deleted at run end anyway and a
staff store can be re-enrolled).

Measured on synthetic galleries (median of 3 runs, per full match call =
staff check over 100 staff templates + gallery match), microseconds:

| gallery | re-sighting before → after | new person before → after |
| ------- | -------------------------- | ------------------------- |
| 50      | 483 → 100 µs (4.8×)        | 3030 → 130 µs (23×)       |
| 200     | 704 → 78 µs (9.0×)         | 3260 → 130 µs (25×)       |
| 500     | 1414 → 86 µs (16×)         | 4340 → 144 µs (30×)       |

The shape matters more than the ratio: the old cost GREW with gallery size
(connect + schema + full-table read + BLOB rebuild every call), the new one is
flat — the scan is a RAM matrix-vector multiply.

Consequence for callers: `gallery.reset` / `sweep` CLOSE the cached connection
before unlinking (an open connection to an unlinked inode keeps answering from
a database nobody can see), and the match service closes every store on
shutdown.

### Run status fields (GET runner /runs/:id)

Added: `staffFaceFrames`, `manualAdditions`, `feedbackRejected`,
`multiFaceFramesSkipped`, `plannerReportErrors`, `tapRoundsAbandoned`.
`staffCrossings` keeps its name and changes its meaning (see above).


## v2 addition — authenticating to the planner (2026-08-05)

The planner binds loopback only by default and REFUSES to listen on any other
address without `HECO_TOKEN` set. A dockerised runner reaches it across the
bridge network (`host.docker.internal`), so that deployment always needs a
token on both sides.

- Set the SAME value as `HECO_TOKEN` for the planner process and the runner
  (docker-compose passes it through).
- The runner sends `Authorization: Bearer <token>` on every planner call: run
  create/update, stats, samples, taps, the multipart frame upload, the feedback
  poll and its status writes, the tombstone poll and its confirmations, and the
  enrolment report. The header rides on the httpx clients so no adapter has to
  know about auth; `PlannerClient(token=…)` also covers the stdlib transport.
- A 401 is a CONFIGURATION error, not a transient. It fails fast with a message
  naming `HECO_TOKEN` rather than being retried — retrying would bury the real
  problem behind a wall of failed attempts.
- With no token configured on either side (loopback development) nothing
  changes.


## v2 addition — a stall is not an end (2026-08-05)

`GET {ingest}/frame` now returns **`ended`** alongside `{tMs, imageB64, w, h,
seq}`. It is the only thing that separates two situations that look identical
to a consumer, because both freeze `seq`:

- a **file** played out with `loop=false` — capture sets `ended`, the count is
  complete;
- a **live camera** that blinked (Wi-Fi dropout, PoE bounce, RTSP reconnect) —
  capture retries forever and NEVER sets `ended`.

The runner records which it was in `RunLoop._end_reason` (`source-ended`,
`source-stalled`, `operator-stopped`) and settles accordingly:

| Reason | Run status | Gallery |
| --- | --- | --- |
| source-ended | `ended` | deleted (transient, per-run) |
| operator-stopped | `ended` | deleted |
| **source-stalled** | **`failed`** | **KEPT** |

A stall keeps the embeddings because they are the only record of who has
already come through the gate: deleting them means a restart counts that whole
room a second time. `match /gallery/sweep` remains the backstop that reclaims
them later. The reason is appended to the run's notes as
`endReason=<reason>` and set on the runner's status.

`source_stall_s` default is now **45 s** (was 5 s, shorter than one RTSP
reconnect over venue Wi-Fi). The planner's own silence detector alarms at 30 s,
so the operator sees a stalled camera before the runner acts on it.

`PUT /api/pipeline/runs/:id` now also carries **`results`** — the structured
final count `{unique, staffCrossings, manualAdditions, frames, matches}` —
which the planner stores in `pipeline_runs.results_json`. The count previously
survived only as prose inside `notes`, so no report could be regenerated from
stored data.


## v2 addition — the runner keeps a log (2026-08-05)

The runner had **no logging at all**, and fourteen places swallowed an
exception to keep a live count alive. The instinct was right — a planner
hiccup must never stop the counting — but the reason went nowhere, so an
operator whose numbers looked wrong at midnight had "it just stopped" as the
entire diagnostic record.

`heco_common.logs` provides `setup_logging(service)` and `RunLog(logger,
run_id)`. Three rules it enforces:

1. **Every line carries `run=<id>`.** A venue runs several gates and a night
   runs several runs; without it the logs cannot be separated back into the
   runs that produced them.
2. **Nothing logs a credential.** `safe()` scrubs `//user:pass@` from anything
   written, and source URLs are logged through `source_label()` anyway.
3. **Swallowed is not silent.** Every best-effort failure now logs at WARNING
   saying what it was attempting.

Level comes from `HECO_LOG_LEVEL` (default INFO). httpx/httpcore/urllib3 are
pinned to WARNING: the runner makes ~11 HTTP calls per frame, so at 15 fps
their INFO chatter is ~165 lines a second and buries everything else.

`setup_logging` uses `basicConfig` WITHOUT `force`, so it installs a handler
when the process has none and stands aside when uvicorn or pytest has already
configured logging.

A normal run now reads:

```
runner: run=run-abc counting from rtsp://192.168.1.64:554/media/video1 (plannerRun=prun-1, site=exiverlabs)
runner: run=run-abc settled failed: unique=3 frames=3 reason=source-stalled (gallery kept for a resume)
```


## v2 addition — the quality gate is composite (2026-08-06)

The gate is the pipeline's **only irreversible discard**: a face it rejects is
never embedded, never matched, and therefore never counted. It was box width
alone, which is the wrong axis in two directions at once.

- Recognition size is honestly read in **inter-eye distance**, the measure FR
  standards specify and the one SFace consumes after alignment. Our 56/80 px
  width floor is only ~24/34 px IED, so the number in the config was not the
  number deciding whether an embedding was usable.
- Size is not the only thing that decides it. A wide face turned side-on has
  little identity-bearing geometry left; a wide face smeared by a guest walking
  past has no detail. `frontality` and `sharpness` see exactly those two
  failures and width sees neither.

`faces /detect` therefore emits three optional signals per face —
`iedPx`, `frontality` and **`sharpness`** (variance of the Laplacian over the
crop, resized to a fixed square so the number is about focus and not about how
close the guest stood). `sharpness` is new: the service documented it and
nothing computed it, so F1's composite had two of its signals, not three.

The runner composes the gate from all four floors:

| env | default | meaning |
| --- | --- | --- |
| `HECO_QUALITY_MIN_PX` | `56.0` | box-width floor (unchanged) |
| `HECO_QUALITY_MIN_IED_PX` | `0.0` | inter-eye-distance floor; 0 = **not armed** |
| `HECO_QUALITY_MIN_FRONTALITY` | `0.0` | 0..1 pose floor; 0 = **not armed** |
| `HECO_QUALITY_MIN_SHARPNESS` | `0.0` | Laplacian-variance floor; 0 = **not armed** |

**Every new floor defaults to unarmed, so the shipped gate is exactly the
width-only gate it has always been.** No floor for the three has been measured
— the eval harness that would price one does not exist yet — and a guessed
floor on the only irreversible discard costs guests off an invoice. Arm them
per camera once they can be priced. `sharpness` in particular is a *within-
camera* ordering: it moves with exposure and contrast, so a value calibrated at
one gate is not transferable to another.

**A signal that could not be measured never rejects a face.** No landmarks
means no `iedPx`; a degenerate crop means no `sharpness`. Absence is UNKNOWN,
not BAD, and under-counting is this pipeline's dominant failure mode. Those
faces are kept and *counted* instead, as `gatedUnmeasured` — an armed floor
that silently evaluates nothing is worse than no floor at all.

**Every rejection reports its reason.** A face carries `gateReason` (the FIRST
floor it failed: width → ied → frontality → sharpness) into:

- the `face-detect` tap payload — per row as `gate`, plus a per-frame
  `gatedBy: {reason: n}` breakdown, and the signals themselves so the console
  can show the evidence beside the verdict;
- the annotated `face-detect` frame — a gated face draws **red whatever its
  width**, labelled `120px frontality`;
- the run status — `gatedByWidth` / `gatedByIed` / `gatedByFrontality` /
  `gatedBySharpness` / `gatedUnmeasured`, and `gateArmed` naming the floors in
  force.

The reporting path READS that verdict and never recomputes it: a second
implementation of the gate in the reporting path is a second gate.

The run row's `config` records `qualityMinPx`, `qualityCanonPx`,
`qualityMinIedPx`, `qualityMinFrontality`, `qualityMinSharpness` and
`gateArmed`, because "which floors produced this number" has to be recoverable
from the record months later, not inferred from today's environment.


## v2 addition — an enrolment leaves a run record (2026-08-06)

Enrolment used to create no `pipeline_run` at all, and that asymmetry was the
defect: **erasure has a permanent ledger** (`staff_tombstones` keeps the row and
its purge confirmation forever, on the argument that a withdrawal and the proof
it was honoured are what an audit asks for) while **the capture that creates the
biometric had none**. A re-enrolment silently overwrote the previous
`enrolled_at`; a *failed* enrolment left zero trace anywhere; and no record said
from which camera, at which event, or over how many frames a staff template was
captured. Consent is only auditable if the capture is.

`mode:'enrol'` now opens a run under the event, exactly like a count run:

- `POST /api/pipeline/runs` with `label` = `enrol <staffId> <source_label>`
  (credential-free — the planner slugs the label into the row's permanent id)
  and `config` = `{source, mode:'enrol', siteId, staffId, enrolBestN, …gate,
  geometry}`.
- `PUT /api/pipeline/runs/:id` on settle with
  `results = {sampleCount, facesSeen, frames, multiFaceFramesSkipped}`,
  `endReason`, and a notes sentence naming the staff id and the source.
- A **failed** enrolment settles the row `failed` with the operator-facing
  reason in the notes. That is the event most worth recording: somebody walked
  a roster member past a camera and the venue now believes they are enrolled
  when they are not.

The row is **best-effort in both directions**, which is the opposite trade from
count mode. There, a failed `create_run` fails the run because the planner id
keys the gallery. Here the deliverable is the staff-store write, the operator
has already walked the member through, and refusing to enrol because the laptop
was rebooting would be the worse outcome — so a planner outage costs the record
only, and the walk-through proceeds as before.

An enrol run releases only ingest's capture slot; it never opened tracker state
or a gallery under that id. `PUT /api/staff/:id` (`enrolledAt`, `sampleCount`)
is unchanged and still the operator-facing confirmation.

Enrolment also applies the **same** quality gate as counting: a template
captured from a face the count would have discarded is a template that will not
match.


## v2 addition — the run registry is bounded, and liveness is liveness (2026-08-06)

`RunManager` held two dicts nothing ever deleted from, so a process that had run
a night of gates retained, per run ever started, a dead `Thread` and a whole
`RunLoop` — and a `RunLoop` pins the last frame's full base64 JPEG.

The same unbounded memory was a correctness bug. "Do I know this run?" was a
bare membership test, and that is the test deciding whether a 409 on ingest's
capture slot means *a sibling is counting on this camera* (fail — seizing would
corrupt that run's count) or *a corpse is holding the slot* (seize it). A run
that settled without its `/close` landing — ingest down or restarting at that
moment — therefore held the camera for the life of the process: the automatic
seizure never fired, `stop` set an event on a thread that had already returned,
and the refusal told the operator to stop a run that had already stopped. Only
a manual ingest restart cleared it.

- The test is now **liveness**: a run counts as live only while its state is
  `starting`, `running` or `enrolling`.
- Settled runs are **reaped** `HECO_RUN_RETENTION_S` (default 600 s) after
  their thread returns, on every `POST /runs` and every status/stop lookup. The
  window keeps `GET /runs/:id` answering for an operator whose run has just
  finished; after that the planner row is the durable record.
- **Thread liveness is the settle signal, not the status dict.** A loop wedged
  mid-frame is still holding the camera whatever its last published status
  said, and reaping it would hand that camera to the next run.

## v2 addition — a guest is several views, not one (match, 2026-08-06)

`/match` used to store a guest ONCE, on the sighting that minted their key, and
compare every later face against that one arbitrary view. The corridor bench,
ground truth **one man walking**, is what that costs: the gallery held **three**
identities for him, pairwise cosines **0.347 / 0.307 / 0.296** against the
0.363 threshold — every pair a near miss, the closest by 0.016. (The quality
gate is not the problem: before it was armed the same measurement was mean
0.172, spread −0.055 to 0.337.)

- An identity now holds up to **`HECO_MATCH_TEMPLATES_PER_PERSON`** (default 5)
  templates. On a match the sighting may be kept as an additional view; at the
  cap the **least distinctive** view is evicted, so a guest accumulates the
  widest spread of views that fits — not their best photographs, and not their
  most recent. Eviction by capture quality was measured and rejected: quality
  is face width, face width is distance, so it collapsed one guest's five
  templates into a two-second window at one distance and counted him twice
  when he reappeared further away.
- A matched sighting is kept only if it clears the threshold by
  `HECO_MATCH_TEMPLATE_CONFIDENCE` (0.05), beats the nearest **rival** identity
  by `HECO_MATCH_TEMPLATE_MARGIN` (0.05), and is below
  `HECO_MATCH_TEMPLATE_MAX_COSINE` (0.90). In order: do not learn from an
  uncertain match; do not store a view that half-belongs to somebody else (it
  becomes a bridge and merges two paying guests); do not spend a capped slot on
  a near-duplicate.
- **The 0.363 threshold is unchanged and stays unchanged.** A sighting must
  clear it against an already-stored template to count as a re-sighting at all.
  Lowering it trades over-counting for under-counting and is blocked on
  impostor data the POC does not yet have.
- **Matching and learning have DIFFERENT bars.** 0.363 decides matched; **0.413**
  (threshold + `HECO_MATCH_TEMPLATE_CONFIDENCE`) decides kept as a template.
  Between the two is a dead band where a sighting is counted as the same guest
  but teaches the gallery nothing, so the identity does not grow towards it.
  Read as "0.363 extends the chain" this is wrong by 0.05, which is the whole
  margin the corridor bench was missing by.
- **A merge is capped too.** The survivor of `POST /merge` inherits both
  identities' templates and is pruned back to `HECO_MATCH_TEMPLATES_PER_PERSON`
  — but by *redundancy*, evicting the least distinctive view, not by quality.
  The quality rule would discard the merged-in views (they are the ones that
  failed to match, which is why a human had to merge them) and the operator
  would re-correct the same guest at every crossing.
- `/match` gains two response fields: **`templateN`** (views the matched
  identity now holds; `null` for a staff hit) and **`templateAdded`**. Both are
  additive — existing consumers are unaffected — and both belong in the match
  debug tap, because a run's counts cannot be read without knowing which
  template policy produced them. `GET /health` reports all four knobs for the
  same reason.
- **Staff are untouched.** Staff templates come only from the operator-
  supervised ENROL MODE walk-through; an unsupervised crossing never enrols
  into `staff-<siteId>.db`, where a mis-tag would outlive the run and poison a
  roster member across every future event at the site.
- **Scope limit, stated because it will be misread otherwise.** Multi-template
  bridges a crossing's extreme poses only when the crossing supplied the
  intermediate poses that connect them. Three views 0.296–0.347 apart and
  nothing between them still count as three people, correctly.

## v2 addition — the track heal and exclusion zones (2026-08-06)

The live bench: **3 real people, counted 5**. Both phantom guests were minted
by mechanisms no threshold can fix, because the measured impostor ceiling on
this camera is **0.377** (two DIFFERENT men's templates scoring above the
0.363 threshold) while same-person misses measured **0.294 / 0.308 / 0.361** —
the impostor and genuine distributions OVERLAP, so **no value of
`HECO_MATCH_THRESHOLD` separates them and 0.363 stays where it is.** The two
fixes act on evidence a threshold cannot see: what the same track did next,
and where in the frame the face was.

### The track heal (runner)

Tonight's p00002 was minted on a RE-ENTRY frame at cosine **0.3084**, and the
SAME physical track matched the correct identity p00001 at **0.69** four
seconds later. The evidence that the mint was junk arrived almost immediately;
the heal is the loop acting on it.

- **Trigger.** A verdict on a track that recently minted a new identity, with
  `isNew=false`, `isStaff=false`, `personKey != mintedKey`,
  `cosine >= HECO_HEAL_MIN_COSINE`, within `HECO_HEAL_WINDOW_S` of the mint.
- **Action.** `POST {match}/merge {runId, keep: matchedKey, drop: mintedKey,
  onlyIfSingleton: true}` — the same endpoint the operator duplicate
  correction uses, with one extra guard (below). On `merged=true`: unique −1,
  `healedSplits` +1, logged with both keys and both cosines. On
  `merged=false`: nothing is counted and the bookkeeping is dropped — **the
  refusal is the system working**, not an error.
- **Knobs.** `HECO_HEAL_WINDOW_S` (default 20.0; 0 disables healing) and
  `HECO_HEAL_MIN_COSINE` (default **0.45** — above the 0.377 measured impostor
  ceiling with margin; tonight's heal evidence was 0.69).
- **Scope, stated so nobody widens it quietly.** The heal never fires INTO a
  staff hit (staff verdicts are skipped; the staff store is
  operator-supervised evidence). The reverse ordering — a track that matches Y
  first and mints X later — is NOT healed: the mint came after the evidence,
  so the evidence says nothing about it.

**`POST /merge` gains `onlyIfSingleton: bool = false`.** When true the merge
is REFUSED (`merged=false`, count unchanged) unless the drop key holds exactly
one template, checked INSIDE the transaction (a template can be enrolled
between the runner's decision and the merge landing). The asymmetry with the
operator flow is the point: the caller asserting "this key is a junk mint"
here is a MACHINE, and a machine's evidence is weaker than an operator's — a
key that has accumulated more templates has been independently re-sighted and
is no longer safely foldable by heuristic. Operator merges are unchanged.

**KNOWN RESIDUAL RISK, documented rather than solved.** A tracker identity
swap — two people crossing paths — can hand a track from person A to person B.
If A's mint is still a singleton inside the window and B then matches at
≥ 0.45, the heal folds A into B: an **under-count of one**. The guards bound
it (singleton-only, the 20 s window, the 0.45 floor, and an operator's
cannot-link split always wins), but they do not eliminate it; the hard fix is
track-quality gating and is out of scope tonight. An operator reading
`healedSplits` beside a count that looks one short should know this is where
to look.

### Exclusion zones (planner UI → runner filter)

Tonight's p00004 was minted from an **87 px face seen THROUGH A FROSTED GLASS
PARTITION** — someone inside an office, not at the gate, scoring 0.3203
against their true owner. No quality floor can reject a face for being in the
wrong PLACE; only the operator knows where the partitions, mirrors and TV
screens are, so the operator draws them.

- **Wire shape.** The planner device config gains
  `exclusionZones: [{label: string, points: [[x, y], ...]}]` — ordered polygon
  vertices, minimum 3, NORMALIZED 0..1 relative to the full frame. The
  planner's control proxy copies `device.config.exclusionZones` into the
  runner run request as the top-level field `exclusionZones`, same shape. The
  runner validates at `POST /runs` (readable 422 on a short polygon, an
  out-of-range coordinate, or a point that is not `[x, y]`).
- **The centre rule.** The runner excludes a face when the CENTRE of its face
  box (pixels) falls inside any polygon after scaling the normalized points by
  the actual frame width/height. Centre, not overlap: a guest walking PAST a
  partition clips the zone and is still counted; a face BEHIND it is not.
- **Excluded faces are never gated, embedded or matched** — the filter runs
  before the quality gate — but they stay VISIBLE: counted in
  `excludedByZone`, flagged `excludedByZone: true` (with the zone's label) in
  the face-detect tap, and drawn magenta inside the translucent magenta
  polygon on the annotated face-detect frame, so the operator can check their
  own drawing against what it is eating.
- **Honesty counter.** Zone points are normalized and need the frame's pixel
  dimensions to scale by; a frame that carries none gets NO exclusion and
  every face that passed untested is counted in `zoneUnmeasured` — the
  `gatedUnmeasured` pattern applied to placement.

### Run status / results fields

Added to `GET runner /runs/:id`, to `results` on the closing
`PUT /api/pipeline/runs/:id`, and to the end-of-run notes as
`healedSplits=N excludedByZone=N`: **`healedSplits`**, **`excludedByZone`**
(plus `zoneUnmeasured` in the status).


### Amendment — the double-mint (2026-08-06 morning bench)

A blurred crossing can mint TWICE in a second: the same woman's consecutive
frames scored 0.29 against each other, so her track minted p00003 and then
p00004 one second apart, settled on p00004 — and the heal, holding ONE
remembered mint per track, had already forgotten p00003. The bookkeeping is
now a per-track LIST (capped at 4): a comfortable match (>= 0.45) folds EVERY
remembered singleton mint of that track except the matched key — including
when the matched key is itself the track's own later mint. Blur at the source
is treated separately (camera shutter floor raised 1/250 -> 1/500); the heal
is what stops one bad second becoming a phantom guest when blur happens
anyway.

## v2 addition — a gallery outlives its run (2026-08-06)

A run's gallery used to be deleted the moment the run settled. The unique
count is an invoice figure, and this destroyed the evidence behind it exactly
when it became one: a venue disputing "5 unique guests" the next morning could
be shown nothing, and on the 2026-08-06 bench the same deletion cost three
diagnoses in one night — each needed rows that no longer existed.

- **The runner no longer calls match `/reset` at settle.** The only reset is
  the run's own start-up initialisation. Camera slot and tracker state are
  still released at settle — those are leases; the gallery is a record.
- **Retention moved to the match service, automated.** A background task
  (started by the service lifespan) sweeps galleries older than
  **`HECO_GALLERY_RETENTION_S`** (default 24 h) on startup and then hourly
  (`HECO_GALLERY_SWEEP_INTERVAL_S`, default 3600). Settled, stalled and
  crashed runs age out on one clock; the old special-casing of stalls
  (keep-on-stall) is gone because keeping is now the rule.
- **`POST /gallery/sweep` remains** for an immediate manual pass.
- **Staff stores are never swept** — they persist by design, under consent,
  until erased through the tombstone flow.
- Retention stays a real control: 24 h covers the morning-after dispute and is
  deliberately not longer, because every gallery file holds real guests' face
  embeddings.

## v2 addition — clothing is an ADVISORY tie-breaker, never evidence of identity (2026-08-06)

Clothing is constant within one event, so a torso-appearance descriptor is
real evidence about whether two sightings seconds apart are the same body.
It is deliberately given NO voice in identity, because the bench measured why
it must not have one: the closest impostor pair on this camera — two
DIFFERENT men at cosine **0.377**, above the 0.363 face threshold — were
**BOTH IN LIGHT SHIRTS**. Any mechanism that let a matching torso rescue a
borderline face match would have merged those two paying guests into one
invoice line, invisibly. Meanwhile the same-person misses (0.294 / 0.308 /
0.361) wore the same clothes in every frame — at a wedding full of light
shirts, a rescue chains strangers together wholesale. So appearance may only
ever **VETO a write** the face pipeline was about to make; it never causes
one.

### The descriptor (computed by the RUNNER; the match service only stores and compares)

- **48 floats**: an L1-normalised **12×4 Hue×Saturation histogram** of the
  torso crop (OpenCV HSV: H 0–179 → 12 bins, S 0–255 → 4 bins). Pixels with
  V < 40 or V > 240 are masked out (shadow / blowout).
- **Torso crop**: within the person box, from the face box's bottom edge down
  to `min(face bottom + 2.5 × face height, person box bottom)`; horizontally
  the person box inset 15% each side.
- **No descriptor** when the crop is under 24 px in either dimension, when
  there is no containing person box, or when fewer than 100 unmasked pixels
  remain. **Absent is not zero** — an absent descriptor disables every
  appearance behaviour for that sighting (the `gatedUnmeasured` /
  `zoneUnmeasured` convention applied to clothing).
- **Similarity**: histogram intersection (sum of element-wise minimums; both
  sides L1-normalised → range 0..1). Partial occlusion only removes mass, so
  it can only lower the score — failing towards "no clash".

### The three v1 uses — and the only three

1. **HEAL VETO (runner).** A track-heal candidate whose remembered torso
   descriptor CLASHES with the current frame's descriptor
   (intersection < `HECO_HEAL_APPEARANCE_CLASH`) is refused — no `/merge` is
   posted. This is a tracker-swap detector: it shrinks the heal's documented
   residual risk (a track handed from person A to person B mid-window folding
   A's mint into B, an under-count of one). An absent descriptor on either
   side means the heal proceeds exactly as before.
2. **ENROLMENT VETO (match).** A matched sighting that `_should_enrol` had
   approved as an extra template is NOT kept when its descriptor clashes
   (`appearanceSim < HECO_MATCH_APPEARANCE_CLASH`) with every descriptor the
   identity has stored (anti-poison: a template enrolled off a tracker swap
   or near-threshold impostor becomes a bridge that silently merges two
   guests later). **The match verdict — isNew / personKey / cosine — is
   never changed by appearance.**
3. **VISIBILITY.** `/match` reports `appearanceSim`; the runner's match tap
   carries it; the run status counts **`healVetoedByAppearance`** and
   **`enrolVetoedByAppearance`**; the end-of-run notes include both.

### Wire and storage

- `POST /match` request gains optional **`appearance`**: exactly 48 floats
  (any other present length is a readable 422 naming the contract).
- The response gains **`appearanceSim`**: `float|null` — best intersection
  against the matched identity's stored descriptors; `null` for staff hits,
  new mints, or when either side lacks a descriptor — and
  **`appearanceVetoed`**: `bool`.
- The gallery's `vectors` table gains a nullable BLOB column **`appearance`**
  (float32[48]), stored beside any template `/match` writes (founding mint
  and enrolled extras). Opening an OLDER gallery/staff file ALTERs the
  column in (SQLite `ADD COLUMN`), so every pre-existing file keeps working;
  its rows read as descriptor-absent, never as zero histograms.
- The descriptor is **never part of the cosine scan** — the resident scan
  matrix stays face-only.
- `GET {match}/health` reports `appearanceClash`.

### Knobs (both UNCALIBRATED — reasoned like the M1 margins, env-tunable so the first labelled torso pairs move them)

| env | default | meaning |
| --- | --- | --- |
| `HECO_HEAL_APPEARANCE_CLASH` (runner) | `0.50` | intersection below this = clash → the heal is refused; **0 disables** |
| `HECO_MATCH_APPEARANCE_CLASH` (match) | `0.50` | intersection below this = clash → the enrolment is refused; **0 disables** |

Empty string means unset for both (the compose `${VAR-}` rendering).

### NEVER — the lines nobody may move without new impostor data

- Never mint, merge, match or count on clothing alone.
- Never let appearance RESCUE a face match (the 0.377 two-white-shirts pair
  is exactly what a rescue merges) — appearance only refuses writes.
- Never touch staff flows: staff identity is operator-attested, the staff
  store gains no appearance rows (mark-staff deliberately leaves the
  descriptor behind with the deleted gallery row), and staff verdicts carry
  `appearanceSim: null`.
- Never treat an absent descriptor as a zero histogram: absent means NO veto,
  anywhere.


## v2 addition — the tap round pays for itself (runner, 2026-08-06)

The debug tap round annotated and JPEG-encoded FIVE full 3840×2160 frames and
uploaded them — **~1–1.5 s per round** on the T440 — and fired whenever
≥ `tap_interval_s` (~2 s) had passed. That trigger has **two stable states**,
both measured: frames at ~0.43 s → a round every ~5th frame → **2.3 fps**;
but ANY transient stall (a docker build ran on the box, both times it
happened) pushes one frame past ~2 s, after which EVERY frame triggers a full
round → ~2.5 s/frame, **locked at 0.4 fps forever**. The stage timings were
identical in both states (ingest 42 ms, persons 76, faces 58, embed 74,
match 4): the stages were innocent, and the untimed round was the entire gap.
Three fixes, each independently sufficient to have caught or killed it:

- **Tap frames are downscaled.** Every annotated tap frame (the raw ingest
  frame included) leaves `annotate.render` at most **1280 px wide** (aspect
  preserved; annotations are drawn at native resolution first, then the
  overlay is resized with `INTER_AREA`). The console displays these small —
  4K tap frames were pure cost, ~9× the encoded pixels for nothing. These
  uploads are the runner's ONLY frame path: there is **no separate keyframe
  path** in this repo, so nothing keeps full resolution and nothing loses
  evidence.
- **The duty-cycle guard.** A round fires only when BOTH hold: (a) the
  interval has passed, AND (b) at least `HECO_TAP_DUTY_FACTOR` (default
  **3**, empty-unset, 0 disables) × the PREVIOUS round's measured cost has
  elapsed since that round ENDED. A 1 s round forces ≥ 3 s of counting before
  the next, bounding the round at ~25% of loop time — and the every-frame
  lock-in becomes **impossible by construction**, because the trigger can
  never outrun the cost it just measured. A postponed round increments
  **`tapRoundsDeferred`** in the run status — a NEW counter, deliberately not
  `tapRoundsAbandoned`: an abandoned round *started* and lost taps to the
  budget; a deferred round never started and lost nothing. The interval clock
  restarts on a deferral, so the counter counts rounds, not frames.
- **The round is finally instrumented.** Each round's wall cost is observed
  as **`tapRoundMs`** on the `count` stage board, and every acquired frame
  observes **`frameWaitMs`** on the `ingest` board — how long the loop stood
  idle because the source had no new frame yet (0.0 when the first poll was
  fresh). Together they split "the source is slow" from "the loop is slow"
  from the console: the death spiral would have shown frameWaitMs ~0 with
  tapRoundMs ~1500 while the frame period read 2500 ms.

## v2 addition — a near-miss mint is flagged to the operator (match + runner, 2026-08-06)

Measured tonight: the same person split at **face cosine 0.3464** against the
0.363 threshold with **clothing intersection 0.562** — new guest p00005,
almost certainly p00004 — and the operator found it BY EYE, because the
signal surfaced nowhere. The mint that near-misses an existing identity while
strongly agreeing on clothing is now flagged at mint time, as a **one-click
merge suggestion**.

- **Wire.** On a MINT (`isNew: true`) whose best face cosine against the
  pre-existing gallery lands in **[`HECO_MATCH_NEARMISS_FLOOR` (default
  0.29, empty-unset, 0 disables) .. threshold)**, the `/match` response gains
  `nearMiss: {key, cosine, appearanceSim}` — the near-missed identity, the
  best score, and the torso intersection against THAT identity's stored
  descriptors (`null` when either side lacks one; absent is not zero).
  Otherwise `nearMiss` is `null`. Matched verdicts and staff hits **never**
  carry it. The floor sits just below the measured same-person misses
  (0.294 / 0.308 / 0.346 / 0.361). `GET /health` reports `nearMissFloor`.
  Match service **0.6.0 → 0.7.0**. *(Extended to two bands with a `basis`
  field in 0.9.0 — see below.)*
- **THE VERDICT IS STILL A MINT — never an automatic merge.** The measured
  reason: an IMPOSTOR pair on the same camera sits at **face 0.377 with
  clothing 0.503** — two different men, near-threshold face AND agreeing
  clothes. Auto-merging on this evidence folds real strangers into one
  invoice line invisibly. The flag is information for a human; **the count
  must not move without the operator clicking** (their click arrives as the
  ordinary `duplicate` feedback → `/merge`).
- **Runner.** The flag rides each match tap row verbatim (`nearMiss` beside
  `appearanceSim`; `null` on unflagged rows), is counted as
  **`nearMissMints`** in the run status, the end-of-run notes
  (`nearMissMints=N`) and the structured `results`, and logs one INFO line
  per flag (both keys, both scores) so the night is greppable.

### The near-miss band becomes TWO bands (match 0.8.0 → 0.9.0)

The 0.29 floor above turned out to be blind to the splits that actually cost
a count. **Bench run 6e1a5d (2026-08-06), ground truth ONE person** who walked
out of frame, came back, and sat down: that one person produced **six tracker
ids**. Three of the extra mints healed back into `p00001`; two survived as
extra guests, and this is how they read against `p00001`:

| survivor | face cosine | clothing intersection | banner before 0.9.0 |
| --- | --- | --- | --- |
| `p00005` | 0.212 | 0.797 | none — 0.212 is under the 0.29 floor |
| `p00006` | 0.228 | 0.875 | none — 0.228 is under the 0.29 floor |

The face signal could not see either split while the v2 clothing descriptor
was reading 0.80–0.88. Seated and turned-away re-entries at this camera's
angles land exactly there. So `nearMiss` now has two entry routes and names
which one spoke.

- **Rider shape gains `basis`.** `nearMiss: {key, cosine, appearanceSim,
  basis}`, same object on both routes.
  - `basis: "face"` — cosine in **[`HECO_MATCH_NEARMISS_FLOOR` ..
    threshold)**. Unchanged behaviour, now labelled. Clothing is *reported*,
    never required, at this range.
  - `basis: "clothing"` — cosine in **[`HECO_MATCH_NEARMISS_WEAK_FLOOR` ..
    `HECO_MATCH_NEARMISS_FLOOR`)** AND `appearanceSim` **is not null** AND
    `appearanceSim >= HECO_MATCH_NEARMISS_CLOTHES`. Absent is not zero: a
    sighting with no torso descriptor never enters this band at all.
- **Consumers must treat a missing/null `basis` as `"face"`.** Rows written by
  a pre-0.9.0 match service carry no `basis` key.
- **New knobs**, both empty-string-unset like every knob in this service, both
  reported by `GET /health` (`nearMissWeakFloor`, `nearMissClothes`):
  - `HECO_MATCH_NEARMISS_WEAK_FLOOR` — default **0.15**; **0 disables the weak
    band only** and leaves the face band exactly as it was.
  - `HECO_MATCH_NEARMISS_CLOTHES` — default **0.78**.
  - `HECO_MATCH_NEARMISS_FLOOR = 0` remains the master switch: the weak band
    is defined as the region *under* that floor, so with no floor there is no
    weak band either.
- **THE HONEST MARGIN, because this is the thinnest evidence in the system.**
  The closest measured impostor pair — two genuinely different men — sat at
  face 0.377 with clothing 0.503, and **another impostor pair reached clothing
  0.747**. The 0.78 bar clears the worst measured impostor by **0.033**. That
  is a hair, not a margin. It is defensible for exactly one reason: a
  near-miss is a **suggestion a human confirms, never a merge** — `isNew`
  stays `true`, the count moves only on an operator click. Stated plainly: **at
  a venue with uniformed staff, a dress code or similar traditional dress this
  band is expected to produce wrong suggestions, and it is the first knob to
  turn off** (`HECO_MATCH_NEARMISS_WEAK_FLOOR=0`). Clothing agreement alone
  never proves identity; here it only buys the operator a look.
- Everything else is unchanged: out-of-band mints carry `nearMiss: null`,
  matched verdicts and staff hits never carry the rider on either basis, and
  a mint can still never carry `overlap`.

## v2 addition — the gallery confesses its own overlaps (2026-08-06)

Multi-template growth can carry two identities INTO each other: a guest
splits early (a seated face, a blurred re-entry, a lens-edge view) and
enrolment then widens both halves until their stored views cross the match
threshold. Measured three times in two days — cross-identity 0.544, 0.452
and 0.376 — every pair one real person, found by the operator reading
cosine matrices by hand.

- On any ENROLMENT whose just-written template lands at-or-above the match
  threshold against a different identity, `/match` returns
  **`overlap`: {key, cosine, appearanceSim}** (same rider shape as
  `nearMiss`). A MINT can never carry it, by construction: on the mint
  branch the nearest rival is bounded by a best that sits below the
  threshold — pinned by test.
- Unlike `nearMiss` it fires regardless of clothing (at-or-above threshold
  is the gallery's own standard of "same person"); the clothing figure rides
  along as evidence for the operator either way.
- An operator's cannot-link split silences the pair at the source, forever.
- The runner forwards the rider through the match tap, counts
  **`galleryOverlaps`** (status, notes, results), and logs one INFO line per
  event. The planner raises a standing amber merge banner on the NEWER
  guest of the pair (keys are monotonic, so newer folds into older),
  wording the clothing agreement or clash honestly. A suggestion, never a
  merge: the count moves only on the operator's click.

Also in this change, display honesty for person boxes: zones filter FACES,
never bodies (a deleted body breaks track continuity for a guest walking
past a partition), but "persons 2" while one body stood inside a zone read
as a bug to the operator. The person-detect tap now carries **`inZone`**
(count + per-box flag) so the console can render "persons 2 · 1 in zone" —
same facts, no ambiguity.


## v3 — the runner trusts the tracker, within limits (2026-08-06)

Bench 6e1a5d, ground truth ONE person who walked out of frame, returned, and
sat down, produced **six tracker ids** (2/5/6/7/8/12). Three mints healed
back; two survived — p00005 (face 0.212 vs p00001, clothing 0.797) and
p00006 (0.228 / 0.875) — because the heal is a CURE that needs a second
comfortable match to arrive, and on a subject the detector keeps losing it
never does. Both also sat under the 0.29 near-miss floor, so no banner fired
either. Three mechanisms answer that, each env-gated because each makes track
identity more authoritative and therefore amplifies the SILENT failure (a
wrong merge under-counts and nobody sees it) if the tracker swaps people.

| knob | default | effect |
| --- | --- | --- |
| `HECO_TRACKER_MAX_AGE` | `30` | frames a track may coast unmatched (was hard-coded 15 = 3.75 s at the bench's 3.97 fps). Treats a symptom: the disease is YOLOX-nano dropping seated bodies. |
| `HECO_TRACK_LOCK_MIN_COSINE` | `0.45` | a verdict matching at or above this BINDS its track to that identity; a later mint on the same track folds immediately. **0 disables.** |
| `HECO_HEAL_APPEARANCE_CLASH` | `0.35` (was 0.50) | below this the torso CLASHES: the fold is refused. |
| `HECO_HEAL_APPEARANCE_UNSURE` | `0.55` | between clash and this the fold PROCEEDS but is counted as weakly corroborated. |

- **The lock is time-scoped, not frame-scoped.** Its claim is "this track was
  following THIS person a moment ago". `_track_for` assigns a face to any
  track whose box contains its centre, so two guests standing close share a
  track id — and processing their faces in order would match the first, bind
  the lock, then fold the SECOND: a real guest erased, with every other guard
  passing. So a fold requires a **strictly later frame** than the binding,
  and a frame that puts two faces on one track **drops that track's lock**.
- **Clothing is a three-band validator, not a cliff.** The old 0.50 hard floor
  vetoed a probably-correct fold at 0.4991. Clash (< 0.35) refuses and counts
  `healVetoedByAppearance`; uncertain (0.35–0.55) proceeds and counts
  `healUncertainAppearance`; above proceeds silently. Absent descriptors
  veto nothing and count nothing.
- **New run counters** (status, end-of-run notes, `results`; absent on older
  runners, and absent is not zero): **`lockedTrackFolds`**,
  **`distinctTracks`**, **`healUncertainAppearance`**.
  `healVetoedByAppearance` now counts refusals from BOTH the heal and the
  lock fold — it is a clothing-refusal counter, not a heal counter.
- **`nearMissMints` counts both near-miss bands** (face and clothing basis);
  the rider's `basis` field is what distinguishes them per suggestion.

## v3 addition — co-presence is machine-asserted cannot-link (runner + match, 2026-08-06)

**The complaint was NOISE, not the count.** Run 05b3b7, ground truth THREE
people, counted THREE — correct — while raising two "likely duplicate" merge
suggestions between people who are demonstrably different:

| banner | face cosine | clothing agreement | reality |
| --- | --- | --- | --- |
| `p00002` "minted from `p00001`" | 0.316 | 0.94 | walked in the main door **together** |
| `p00007` "minted from `p00002`" | 0.360 | 0.57 | stood in the alley **together** |

The tap ledger proves the second outright: **one tap round matched `p00002`
and `p00007` in the SAME FRAME**. Two faces at different positions in one
frame are two different people — about as certain as machine evidence gets —
and the pipeline held that fact and did nothing with it.

**No threshold tuning fixes this.** Both riders sat at 0.316 / 0.360 against
the 0.363 threshold, i.e. *inside* the ordinary face near-miss band, and
clothing (0.94, 0.57) did not save them. The measured same-person misses on
this camera are 0.294 / 0.308 / 0.361 and the closest measured impostor pair
is 0.377: the genuine and impostor face distributions overlap exactly there.
This is not the weak/clothing band misbehaving — genuinely different people
land in the face band at this camera. **Co-presence is an INDEPENDENT signal
and the only certain one available.**

- **Runner → `POST /split`.** For every frame, the distinct **non-staff**
  personKeys among that frame's verdicts are collected; every unordered pair
  in that set is co-present and is posted to `{match}/split {runId, a, b}`
  **once** (already-sent pairs are remembered and never re-sent; the pair
  memory is bounded per run and logs one warning when it fills, after which no
  further pairs are asserted — pairs already recorded keep their constraint).
  Counted as **`coPresenceSplits`** in the run status, the end-of-run notes
  and the structured `results` (absent on older runners; absent is not zero).
- **`HECO_COPRESENCE_SPLIT`** (`env_int`, default **1**, empty-means-unset)
  is the off switch: `0` disables the assertion entirely. It is the ONLY knob
  for this feature — see the suppression note below.
- **Match: a pair under `cannot_link` no longer earns a near-miss rider**, on
  either basis. Match service **0.9.0 → 0.10.0**; no field changed shape,
  some riders simply stop being emitted. A near-miss rider is a QUESTION put
  to the operator; the constraint is the ANSWER already on file — from their
  own `false-match` correction, or now from the runner asserting co-presence.
  Re-asking a settled question is the noise that trains operators to ignore
  banners.
- **`cannot_link` now governs THREE things**, and the count is not one of
  them: `/merge` refuses the pair, the gallery-**overlap** banner is withheld,
  and the **near-miss** rider is withheld. The constraint is deliberately
  becoming the gallery's single record of "known different" — one fact,
  asserted by a human or by the machine, honoured everywhere.
- **The verdict never moves.** `personKey`, `isNew`, `cosine` and `galleryN`
  are identical with and without the constraint (pinned by test): cannot-link
  changes what is SUGGESTED about someone, never who they are.
- **No knob on the suppression itself.** A banner that contradicts a recorded
  fact has no defensible reading, so the suppression is inherent to
  `cannot_link`; the off switch lives at the source (`HECO_COPRESENCE_SPLIT`).
  An operator's own `/split` clicks are unaffected by it, as they should be.
- **The planner needs no change** — a rider that is never emitted renders no
  banner. Its existing standing merge-suggestion UI is untouched.

**What this buys beyond silence.** `cannot_link` also makes `/merge` refuse,
so a track heal or a track-lock fold can no longer fold two **co-present**
people together — the SILENT under-count (a real paying guest erased) that
every guard in this system exists to prevent. Noise reduction and accuracy
protection are the same change.

**THE KNOWN COST, documented rather than hidden.** A person holding a **phone
showing their own face** — or a mirror, or a printed photo — puts one real
person's face in the frame twice, and co-presence will assert they are two
different people. That blocks a fold which was CORRECT: it blocked exactly
such a phone-face heal on the 2026-08-06 bench. The trade is deliberate and
follows the asymmetry this project runs on: a blocked fold **over**-counts,
which is visible and an operator can merge; a wrong fold **under**-counts
silently and nobody ever sees it. Fixed mirrors and screens are handled by
exclusion zones; a hand-held phone is not, and that is the residual.

## v3 addition — one body cannot be in two places (runner + match, 2026-08-07)

**THE MEASUREMENT.** Run `fa8fc3`, 2026-08-07, ground truth **two men**,
reported **one**. Keyframe `kf-577` (t=32957 ms, `persons: 3, matched: 2`)
holds them side by side, both boxed, and both labelled **`p00001`** — matched
at face **0.6874** and **0.6733**. Not borderline: no near-miss band, no
banner, nothing for an operator to notice. By the end of the run that single
key held five templates whose internal cosines ran **0.2223 .. 0.5232** —
mutually impostor-level. One identity had grown to contain two people, and the
run reported one guest.

**Every existing mechanism was structurally blind to it.**

- **Co-presence** constrains two *keys*, and there was only ever one key. The
  matcher collapsed the pair one step before the constraint could exist.
- **The enrolment veto** refuses WRITES, never verdicts, and is charitable by
  construction (BEST intersection over the identity's stored views). Once one
  of the second man's sightings was inside, every later one agreed with *it*
  and nothing clashed again. The run scored `enrolVetoedByAppearance: 0`.
- **The near-miss and overlap banners** fire on MINTS, and a second mint never
  happened.

The clothing evidence to separate them was present the whole time and unused:
the surviving templates cluster {id7, id10, id14} agreeing **0.624 .. 0.817**
and {id13, id16} agreeing **0.668**, with **0.130 .. 0.294** *across* the
divide — two men, cleanly separated, by a signal no mechanism was positioned
to act on.

**THE ASSERTION.** Two faces in ONE frame occupying two DIFFERENT person boxes
cannot be the same guest, whatever the cosine says. This is the same certainty
co-presence already runs on, applied one step earlier — to the case where the
matcher merged the pair before a second key existed to constrain.

- **Runner → `_split_same_key`**, run between the two verdict passes and
  BEFORE co-presence (co-presence needs two keys; this is what produces them).
  Verdicts are grouped by `personKey`; for any key held by more than one
  non-staff verdict, one sighting keeps it and each other must be *disproved*
  to lose it.
- **Who keeps the identity.** A **mint** keeps it outright — that sighting
  brought the key into existence this frame, and its cosine is not comparable
  with a match's (it measures distance to OTHER identities). Among ordinary
  matches, highest cosine keeps it.
- **TWO conditions, both required, before anyone is split.** The two faces sit
  in person boxes that are genuinely different objects (a face with no
  containing box proves nothing and is left alone), AND their torso
  descriptors intersect **below `HECO_SAME_FRAME_CLASH`** (`env_float`,
  default **0.35** — the same CLEAR-CLASH floor `heal_appearance_clash` uses;
  `0` disables the guard). Descriptor missing on either side ⇒ no split:
  absent is not zero, and this deliberately errs towards the under-count.
- **`POST /match` request gains optional `excludeKeys`**: `list[str] | None`,
  identities the caller has PROVEN this face is not. They are masked out of
  the scan before the argmax. The loser is re-asked with the disputed key
  barred, so it resolves against the **rest of the gallery** — landing on an
  already-known guest when one fits, minting only when nobody does. Forcing a
  blind mint instead would split a returning guest every time two people
  happened to share a frame.
- **`POST /match` reply gains `templateId`**: `int | null`, the rowid of the
  template this call enrolled (null when nothing was written, and null on a
  mint — a mint's unit of retraction is the whole key).
- **`POST /template/forget {runId, templateId}` → `{ok, forgotten, galleryN}`**
  retracts exactly that row. **`forgotten: false` is a NORMAL outcome**, not a
  failure: the post-enrolment redundancy prune may already have evicted it, or
  it may be its key's last template, which is refused so no identity is left
  existing-but-unmatchable. `galleryN` is unchanged either way — this removes
  a VIEW of somebody, never somebody.
- **Why the retraction is not optional.** The losing sighting was already
  enrolled into the wrong identity by the very `/match` call that revealed the
  conflict. Leaving it there defeats the whole guard: a template of man B
  living under man A goes on capturing man B in every LATER frame — including
  every frame where he stands ALONE and no second body exists to disprove it a
  second time. That is precisely how one key came to hold five mutually
  impostor-level templates.
- **`sameFrameSplits`** joins the run status, the end-of-run notes and the
  structured `results` (absent on older runners; absent is not zero). It is
  **the only counter in the run that reports a silent UNDER-count being
  caught** — every other mechanism here either suggests something to an
  operator or refuses a fold. A non-zero value means the run counted people
  the matcher alone would have merged away.
- **Tap rows gain `sameFrameSplit`**: `{from, cosine, clothes} | null`. Unlike
  `nearMiss` and `overlap`, which SUGGEST, this records something that already
  happened — the key the face was taken off, the confident match the frame
  overruled, and the torso intersection that corroborated it.

**THE OVER-COUNT RISK, closed by the body test.** A person and their
reflection in a hall mirror, or a face on a hand-held phone screen, also put
two faces in one frame — and clothing alone would sometimes split them: a
phone held at *waist height* gives the second face a torso crop over the
holder's trousers rather than his shirt, which genuinely clashes. The person
box is the arbiter, and it is the only one of the three signals that is about
people. Pinned by test: one box, two faces, clashing crops ⇒ **no split**.

**Match service 0.10.0 → 0.11.0** (new request field, new reply field, new
endpoint; all additive). The runner's verdicts are unchanged in shape.
