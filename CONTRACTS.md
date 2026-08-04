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
| ingest    | 7101 | RTSP/file → JPEG frames on demand: GET /frame (latest), POST /open {url|path} |
| persons   | 7102 | POST /detect {imageB64} → {boxes:[{x,y,w,h,conf}]} — YOLOX-nano ONNX (Apache-2.0) |
| tracker   | 7103 | POST /track {runId, boxes, tMs} → {tracks:[{id,box,ageFrames,hits}]} — own SORT-style IoU+velocity, stateful per runId (POST /reset {runId}) |
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
