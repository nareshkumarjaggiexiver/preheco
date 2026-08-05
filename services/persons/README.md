# persons — YOLOX-nano person detection (port 7102)

## What

FastAPI microservice implementing the `persons` row of
[CONTRACTS.md](../../CONTRACTS.md): person-only detection (COCO class 0) on
base64-JPEG frames, YOLOX-nano ONNX on onnxruntime **CPU**. Letterbox
preprocessing and grid-decode + NMS postprocessing are pure, unit-tested
functions (`app/preprocess.py`, `app/postprocess.py`).

**Why nano, not tiny:** the POC geometry (subjects 2–3 m from a 2.0 m mount)
produces large, mostly unoccluded person boxes; nano (~1.1 GFLOPs @416) is
sufficient there and leaves CPU headroom for the rest of the pipeline.
YOLOX-tiny (~5.1 GFLOPs) is a models.lock-row + `PERSONS_MODEL` swap away if
pilot footage shows recall problems — same decode path.

**No compiled rewrite:** per project rule, a Go/Rust/C++ port waits for
profiling evidence. onnxruntime and OpenCV already execute native kernels;
at POC scale Python is orchestration, not the bottleneck.

## Model provenance & licence

| file | source | licence |
| --- | --- | --- |
| `yolox_nano.onnx` | [Megvii YOLOX release asset (tag 0.1.1rc0)](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) | [Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) |

Pinned URL + sha256 live in [`../../models.lock`](../../models.lock);
`make models` downloads into `models/` (gitignored — weights are never
committed). Note: the YOLOX repo has been frozen since Jan 2024 (v0.3.0) —
licence-clean and fine for the POC, disclosed per the design review.

## Run

```sh
make venv     # python3.12 -m venv .venv + pinned requirements
make models   # download + sha256-verify yolox_nano.onnx
make run      # uvicorn on :7102 (PORT=... to override)
```

## API

- `GET /health` → `{ok, model, version}` — `ok:false` until weights exist.
- `POST /detect` `{imageB64, confMin?}` →
  `{boxes: [{x, y, w, h, conf}], inferMs}` — source-image pixels, best first;
  `inferMs` covers letterbox + ONNX run + decode/NMS (JPEG decode excluded).

## Test

```sh
make test   # pure golden tests always run; model tests skip loudly without `make models`
make lint   # ruff
```

Golden decode fixture: `tests/fixtures/decode_golden.json`, generated once by
an independent loop-based reference (`generate_decode_golden.py`).

## Tune

| env | default | meaning |
| --- | --- | --- |
| `PERSONS_CONF_MIN` | `0.30` | default score threshold (request `confMin` overrides per call) |
| `PERSONS_NMS_IOU` | `0.45` | NMS IoU threshold |
| `PERSONS_INPUT_SIZE` | `416` | square network input (letterboxed) |
| `PERSONS_MODEL` | `models/yolox_nano.onnx` | weights path (e.g. a tiny swap) |

## CPU latency — measured on this machine

Measured 2026-08-04 on **11th Gen Intel i5-1135G7 (8 threads), CPU-only**,
Python 3.12.3, onnxruntime 1.28.0, opencv-python-headless 5.0.0.93; 50 runs
after warmup, full `detect()` path (letterbox + inference + decode + NMS):

| input frame | p50 | mean | p95 |
| --- | --- | --- | --- |
| 640×480 | 9.2 ms | 9.5 ms | 11.2 ms |
| 1280×720 | 8.8 ms | 9.0 ms | 9.7 ms |

(Frame size barely matters — the network always runs at 416×416.) Live HTTP
sanity check on a real 548×342 photo returned the expected person boxes at
`inferMs` ≈ 11 ms. These are this-machine numbers, not a product benchmark.


## Threading

`PERSONS_THREADS` (default 8) sets onnxruntime's intra-op thread count; the
inter-op pool is fixed at 1, because one image goes through one graph and there
is nothing to run in parallel between nodes.

It is explicit for two reasons. Left implicit, ORT counts the machine's cores
from `/proc` — 64 on the dual-socket T440 — spawns a pool that size, and PINS
each thread to a chosen CPU. Once the compose file confines the container to one
NUMA node, half of those CPUs are outside its cpuset and every pin fails:

```
pthread_setaffinity_np failed for thread: 93, index: 15, mask: {1, 33, },
error code: 22 ... Specify the number of threads explicitly so the affinity is not set.
```

Second, the graph is small. YOLOX-nano at 416x416 saturates a few cores; past
that the synchronisation costs more than the parallelism returns. The measured
live run used about 2.8 cores while ORT had thirty threads spawned.

The effective value is `min(cpus this process may use, PERSONS_THREADS)`, read
via `sched_getaffinity`, so it follows the cpuset rather than the hardware.
