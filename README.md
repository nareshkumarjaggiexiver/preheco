# heco-pipeline

The HECO counting pipeline as Python microservices — the POC implementation of
the eight-stage design in `docs/planning/` (stages 03/04), reporting every
stage into the site-planner app under an EVENT.

**The contract is [CONTRACTS.md](./CONTRACTS.md).** Ports, endpoints, stage
names, planner ingest shapes and the POC geometry all live there; services are
built against it exactly.

## POC geometry (read before judging the thresholds)

Baseline camera: UNV 2.8 mm fixed (IPC2128LR3-DPF28M-F) mounted at **2.0 m**,
subjects passing **2–3 m** away — the KB's documented "close-zone cheat".
Expected face widths **~64–85 px**, deliberately below the production
80/100 px canon. POC quality floor: face width **≥ 56 px** to enter embedding,
with **56–79 px flagged "sub-canon"** in every report.

## Services

| service | port | job | owner venv |
| ------- | ---- | --- | ---------- |
| runner  | 7100 | drives the loop, batches stats to the planner | `services/runner` |
| ingest  | 7101 | RTSP/file → latest JPEG frame on demand | `services/ingest` |
| persons | 7102 | YOLOX-nano person boxes (Apache-2.0) | `services/persons` |
| tracker | 7103 | own SORT-style IoU+velocity tracks, stateful per run | `services/tracker` |
| faces   | 7104 | YuNet face detect (OpenCV zoo, Apache-2.0) | `services/faces` |
| embed   | 7105 | SFace 128-d embeddings (OpenCV zoo, Apache-2.0) | `services/embed` |
| match   | 7106 | SQLite gallery, cosine threshold 0.363 | `services/match` |

`common/` is a shared library (`heco_common`) installed **editable** into each
service venv: pydantic schemas for every inter-service message, base64 JPEG
helpers, config-from-env, and the planner client.

## Layout & conventions

- **Python 3.12, one venv per service** — `make venv` inside any directory,
  or `make venv-all` at the root.
- **ruff-clean** against the single root [`ruff.toml`](./ruff.toml) —
  `make lint`.
- **pytest per service, no network in tests** — `make test-all`. Tests use
  tiny synthetic videos/images and injectable transports.
- **Model weights are never committed.** Each model-bearing service has a
  `make models` target that downloads pinned URLs verified by sha256
  (`models.lock`) into its gitignored `models/`.
- **Licence gate:** only Apache-2.0 / MIT / BSD models and libraries — the
  product ships closed-source. YOLOX (Apache-2.0), YuNet + SFace (Apache-2.0
  via the OpenCV zoo). Nothing GPL / AGPL / non-commercial.
- **Frames travel as base64 JPEG in JSON** — POC scale. Shared memory arrives
  only if profiling demands it.
- **No Go/Rust/C++ services for the POC.** OpenCV and onnxruntime are already
  native code doing the heavy lifting; a compiled rewrite of any stage waits
  for profiling evidence that that stage is the bottleneck.

## Quickstart

```sh
make venv-all      # one venv per service + common
make test-all      # every suite, offline
make lint          # ruff, single root config
make models-all    # pinned+checksummed weights for model services
```

Run a single service (example — ingest):

```sh
cd services/ingest
make venv && make run    # uvicorn on the contract port
```

## Docker compose

One shared base image carries the heavy native stack once
(python:3.12-slim + opencv-python-headless + onnxruntime + the FastAPI
layer, ~440 MB); every service is a thin parameterised layer on top
(`docker/service.Dockerfile`, ~40 MB extra). Weights are **never baked into
images** — each model service's `services/<name>/models/` is bind-mounted
read-only.

```sh
# 1. weights on the host (pinned + sha256-checked into services/*/models/)
make models-all

# 2. the shared base image first — compose builds do not order it
docker build -f docker/base.Dockerfile -t heco-pipeline-base:latest .

# 3. the seven service images, then up
docker compose build
docker compose up -d
docker compose ps        # healthchecks: every service polls its own /health
```

The site-planner app runs on the **host** at `:8787`. The runner reaches it
as `http://host.docker.internal:8787` (the compose file adds the
`host-gateway` extra_hosts mapping Linux needs); override with
`PLANNER_URL=… docker compose up -d`. Stage-service URLs default to compose
DNS names (`http://ingest:7101`, …) and can be overridden per service with
`HECO_<SERVICE>_URL`.

Start a run against an event (file sources live on the shared `source-media`
volume mounted at `/media` in the ingest container; RTSP needs no volume):

```sh
docker cp clip.mp4 heco-pipeline-ingest-1:/media/clip.mp4
curl -X POST localhost:7100/runs -H 'Content-Type: application/json' \
  -d '{"eventId": 1, "source": {"path": "/media/clip.mp4"}}'
curl localhost:7100/runs/<runId>          # live status
curl -X POST localhost:7100/runs/<runId>/stop   # RTSP sources never end alone
```

Stats and sampled rows land in the planner under the event every ~2 s; the
run closes with notes carrying the unique count, frames, and sub-canon share.

### End-to-end smoke without docker

`scripts/e2e_smoke.py` launches all seven services as subprocess uvicorns on
the contract ports (real code where present and healthy, built-in stubs
otherwise), generates a synthetic walking-rectangle video, runs it against a
fake in-process planner, and asserts run-created / per-stage stats /
run-ended / unique ≥ 0:

```sh
python3.12 -m venv .e2e-venv && .e2e-venv/bin/pip install -r scripts/requirements-e2e.txt
.e2e-venv/bin/python scripts/e2e_smoke.py
# exercise the full embed+match chain with fabricated faces:
.e2e-venv/bin/python scripts/e2e_smoke.py --force-stubs persons,faces,embed
```

Honest limitation, stated in the script too: synthetic frames contain **no
real faces**, so the smoke validates *plumbing* (orchestration, timing,
aggregation, planner reporting), not accuracy. The first real-face validation
happens at the POC bench with the real camera.

## Reporting into the planner

The planner app (schema v4) accepts, per run:
`POST /api/pipeline/runs`, `PUT /api/pipeline/runs/:id`,
`POST /api/pipeline/runs/:id/stats`, `POST /api/pipeline/runs/:id/samples`
(batch ≤ 200). Stage names:
`ingest | person-detect | track | face-detect | quality | embed | match | count`.
Charted metrics: `personBoxHPx`, `faceBoxWPx`, `embedMs`, `matchCosine`.
`heco_common.planner.PlannerClient` wraps all four calls with retry + batching.
