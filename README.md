# heco-pipeline

The HECO counting pipeline as Python microservices: ingest → persons →
tracker → faces → embed → match, conducted by a runner that reports every
stage — counts, face cards, the duplicate-review queue, per-stage statistics —
into the site-planner under an EVENT. It implements stages 1–6 of the design in
the meta repository (`docs/planning/03`, `04`); the planner is stages 7–8.

**The contract is [CONTRACTS.md](./CONTRACTS.md).** Ports, endpoints, stage
names, the planner ingest shapes and every dated addition (v1 → v4, 2026-08-04
→ 2026-10-01) live there; services are built against it exactly. What was
built in September and what it measured is in the meta repo's
`docs/apps/heco-pipeline/changelog-2026-09.md`; the rules a change must pass
are in `docs/planning/16-working-rules-and-field-decisions.md`.

## The gate and the geometry (read before judging a count)

A guest is counted when a face passes the quality gate and matches nobody in
the run's gallery. The console's **Strict** profile is the production gate:
landmarks required, detector confidence ≥ 0.7, face width ≥ **80 px**, IED
≥ 32 px, eye span ≥ 0.32, frontality ≥ 0.65 — and a guest is **created from
≥ 112 px or from the best face of its track** (front-on, then widest, then
sharpest), which is also the face card and the anchor template. Box-level
defaults add the feature-norm floor (18), the half-face balance floor (0.33)
and the nearer-head occlusion gate (0.3).

The placement decides more than the models: on an overview camera 7–11 people
in frame yield 1–3 faces; the JD Grand corridor camera at 4K gave a median
face of 67 px with 27 % over 112 px. The match threshold (`HECO_MATCH_THRESHOLD`
0.363) is still SFace's operating point under an ArcFace 512-d embedder —
measured safe (impostor max 0.34 on a live gallery; a same-person view
re-matches ≥ 0.363 99.7 % of the time) but **not calibrated**; `eval/sweep.py`
on labelled footage is the open item.

## Services

| service | port | job | model / method | device on the demo box |
| ------- | ---- | --- | -------------- | ---------------------- |
| runner  | 7100 | drives the loop; stage overlap, held faces and best-face minting, co-presence and track-presence splits, locks and heals, the durable planner outbox; serves the manifest (`pipeline.json`) on `/health` | — | — |
| ingest  | 7101 | RTSP/file → latest frame on demand; one capture slot per stack | cv2, or an ffmpeg subprocess (`INGEST_DECODER=nvdec`) | NVDEC |
| persons | 7102 | person boxes | **YOLOX-S** 640 (nano and RT-DETRv2-S in the catalog) | TensorRT fp16 |
| tracker | 7103 | own SORT-style IoU + velocity tracks, stateful per run; a 10 s frame-time gap clears them | in-house | CPU |
| faces   | 7104 | face boxes with 5 landmarks, whole frame | **SCRFD-10G** at 1472×832, score ≥ 0.7 (YuNet, SCRFD-2.5G in the catalog) | TensorRT fp16 |
| embed   | 7105 | 512-d embeddings with the raw feature norm; age and sex; head covering on `/headwear` | **ArcFace w600k_r50**; **faceage-dino** (DINOv3 ViT-L/16 + CORAL age head); **SigLIP B/16** zero-shot | TensorRT fp16 (faceage: fp32 compute) |
| match   | 7106 | SQLite gallery, cosine 0.363, up to 5 templates per guest with a best-face anchor; the review queue; operator corrections | numpy | CPU |

`common/` is the shared library (`heco_common`): pydantic schemas for the
inter-service messages, base64 JPEG helpers, config-from-env, run logging, the
ONNX Runtime provider helpers (`HECO_DEVICE`, TensorRT options, engine cache
keyed on the weights), and the planner client with its durable outbox.
`counting/` (`heco_counting`) holds the quality gate, the appearance
descriptors (torso, head, beard, skin) and the configuration the runner stamps
into every run record.

## Layout & conventions

- **Python 3.12, one venv per service** — `make venv` inside any directory,
  or `make venv-all` at the root.
- **ruff-clean** against the single root [`ruff.toml`](./ruff.toml) — `make lint`.
- **pytest per service, no network in tests** — `make test-all`. Tests use
  tiny synthetic videos/images, hand-rolled ONNX graphs and injectable
  transports; the runner's call sequence and every "off is today's pipeline"
  lever are pinned by fixtures that are edited only against a fresh capture.
- **Model weights are never committed.** [`models.lock`](./models.lock) pins
  every fetched weight (sha256 + URL; `make models-all`) and
  [`models-restricted.lock`](./models-restricted.lock) the InsightFace
  weights (`make models-restricted`); `scripts/verify-models.sh` refuses a
  deploy with a row missing. Two files are placed by hand (`genderage.onnx`,
  the SigLIP reader with its prompt bank) — the embed README says where.
- **Licences are recorded per row, not used as a gate.** Since 2026-09-30 a
  model is chosen on measured accuracy and effort; the customer buys what
  needs buying (meta `docs/planning/15`, §9). What ships must run on the
  CUDA/TensorRT providers — an ONNX graph whose ops fall back to CPU is out.
- **Frames travel as base64 JPEG in JSON** on `main`. The
  `shared-memory-transport` branch carries them as tmpfs refs instead
  (measured 1.32× on one camera); it is kept merged with main and is not the
  production path.
- **No Go/Rust/C++ services.** OpenCV, onnxruntime and TensorRT do the heavy
  lifting; the throughput levers of September (stage overlap, NVDEC, TensorRT,
  whole-frame detection) were cheaper than any rewrite and are documented in
  [docs/LEVERS-2026-09-25.md](./docs/LEVERS-2026-09-25.md).
- **Every knob is readable back** on its service's `/health`, and the runner
  stamps the resolved configuration — quality gate as armed, levers, devices,
  models — into the run record. Env knobs reach a container only through a
  `${X-}` passthrough in `docker-compose.yml`; a knob without one does nothing.
- **Commits** are `area: summary`, authored by the human committer, with no
  tool attribution.

## Quickstart

```sh
make venv-all      # one venv per service + common + counting
make test-all      # every suite, offline
make lint          # ruff, single root config
make models-all    # pinned + checksummed weights for the model services
```

Run a single service (example — ingest):

```sh
cd services/ingest
make venv && make run    # uvicorn on the contract port
```

`scripts/run-native-cuda.sh start|stop|status` runs all seven as plain
uvicorns on a machine without docker (the laptop); `scripts/e2e_smoke.py`
runs the plumbing end to end against a fake planner with synthetic frames
(no real faces — it validates orchestration, not accuracy).

## Docker compose

One shared base image carries the heavy native stack once; every service is a
thin layer on top (`docker/service.Dockerfile`). The three model services
have a second, GPU image (`docker/persons-cuda.Dockerfile`, build arg
`BASE=heco-{persons,faces,embed}`, `WITH_TENSORRT=1`) that the `gpumax`
overlay runs. Weights are never baked in — each `services/<name>/models/` is
bind-mounted read-only.

```sh
make models-all                                   # weights on the host
HECO_TRT=1 HECO_HWDEC=1 ./scripts/build-images.sh # base, services, :cuda with TensorRT, :hwdec ingest;
                                                  # refuses under 40 GB free, prunes after itself
./scripts/demo-up.sh a          # camera A on :7100–7106 with the demo profile
./scripts/demo-up.sh b          # camera B on :7200–7206 (docker-compose.camB.yml)
./scripts/demo-up.sh status     # every knob and each stage's device truth
```

The compose files compose: `docker-compose.yml` (the seven services) +
`gpumax.yml` (persons/faces/embed on the `:cuda` images, `gpus: all`) +
`trt.yml` (`HECO_DEVICE=TRT`, the engine cache volume) + `hwdec.yml` (the
ffmpeg ingest image for NVDEC) + `camB.yml` (ports for the second camera).
Image tags are `HECO_IMAGE_TAG` / `HECO_CUDA_TAG` / `HECO_HWDEC_TAG`, so one
box runs `main` and an experiment branch side by side. `scripts/demo-up.sh`
holds the demo configuration — review floor 0.28, light guard off, head
covering on, the age/sex reader with the 15-year gap, the measured lever set
— and `status` prints what the running containers actually hold.

**Two cameras are two stacks**: ingest owns exactly one capture. **Check device
truth after every bring-up**: `/health` on persons, faces and embed must list
`TensorrtExecutionProvider` first (`trt: null` means the engine did not come
up and the stage is running CUDA fp32). Cold TensorRT engine builds happen at
the first `/health` (YOLOX-S ~157 s, SCRFD ~34 s, ArcFace ~35 s, faceage
~11 s) and are cached under `HECO_TRT_CACHE` by the weights' hash.

The planner's address is `PLANNER_URL` (default `http://host.docker.internal:8787`;
on the demo box the planner runs on the box itself at `:8788`). Stage-service
URLs default to compose DNS names and can be overridden per service with
`HECO_<SERVICE>_URL`.

Runs normally start from the planner's console, which sends the quality
profile, the camera's face search region and `lockstep` for footage (every
frame examined). By hand:

```sh
docker cp clip.mp4 heco-pipeline-ingest-1:/media/clip.mp4
curl -X POST localhost:7100/runs -H 'Content-Type: application/json' \
  -d '{"eventId": "evt", "source": {"path": "/media/clip.mp4", "isFile": true, "lockstep": true},
       "quality": {"requireLandmarks": true, "minConf": 0.7, "minPx": 80, "mintMinPx": 112,
                   "bestFaceAnchor": true, "minIedPx": 32, "minFrontality": 0.65, "minEyeSpan": 0.32}}'
curl localhost:7100/runs/<runId>                # live status and counters
curl -X POST localhost:7100/runs/<runId>/stop   # RTSP sources never end alone
```

A run without a `quality` block runs the box's loose defaults and is not
comparable with a console run — diff `config_json` before believing a
before/after number.

## Reporting into the planner

Against `PLANNER_URL` (planner schema v26), per run: `POST /api/pipeline/runs`
and the closing `PUT`, `…/stats` (cumulative per stage, last-write-wins),
`…/samples` (drained deltas), `…/taps` (the per-tick ledger), `…/frames`
(kept moments; forensic stills only when asked), `…/faces` (one card per
guest, replaced only by a face that agrees with it), `…/frame-records`, plus
the feedback and tombstone polls (merge / not the same person / not a guest /
erasure). Since 2026-09-29 the writes are **durable**: a bounded per-run
outbox keeps what the planner was away for and delivers it in order
(`X-Heco-Delivery`, `X-Heco-At`); the planner dedupes and files late rows at
their time. `heco_common.planner.PlannerClient` wraps all of it.

## Documents in this repository

| File | What it is |
| --- | --- |
| [CONTRACTS.md](./CONTRACTS.md) | the normative wire contract and every dated addition |
| [docs/LEVERS-2026-09-25.md](./docs/LEVERS-2026-09-25.md) | every knob of the September levers and the review's evidence: default, what it bought, how to read it back |
| [docs/PIPELINE-REVIEW-2026-09-24.md](./docs/PIPELINE-REVIEW-2026-09-24.md) | the stage-by-stage review on the box, with the Punjab factors |
| [docs/JD-LIVE-CHECKLIST-2026-09-25.md](./docs/JD-LIVE-CHECKLIST-2026-09-25.md) | the operator's live-camera checklist (power-up, IPs, PoE, zoom, GPU clocks) |
| [docs/DEPLOY.md](./docs/DEPLOY.md) | the two deploy lessons: never silence the pull, compare the `build` hash across the fleet |
| `services/*/README.md` | each service's own knobs and models |

## Branches

| Branch | State |
| --- | --- |
| `main` | the demo box's cameras A and B |
| `shared-memory-transport` | frames as tmpfs refs; kept merged with main; the `:7300` stack |
| `tracker-byte` | ByteTrack-style low boxes — parked: one wrong fold on Sharon |
| `contested-reads` | torso reads flagged when a neighbour crosses the band — parked: correct, no review gain |
| `lever/outfit`, `lever/tracklets` | a lower-body descriptor; retroactive tracklet identity — built, not wired |
