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
| faces     | 7104 | POST /detect {imageB64, within?:[boxes]} → {faces:[{box,landmarks,conf,widthPx,quality}]} — YuNet (OpenCV zoo, MIT) |
| embed     | 7105 | POST /embed {imageB64, faces} → {embeddings:[[128]]} — SFace (OpenCV zoo) |
| match     | 7106 | POST /match {runId, embedding, quality?} → {personKey, isNew, cosine, galleryN} — gallery in SQLite per runId, cosine threshold 0.363 (SFace paper operating point; POC-tunable via env) |
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
