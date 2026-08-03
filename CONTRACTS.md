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
