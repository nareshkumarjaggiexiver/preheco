# faces — YuNet face detection + POC quality flag (port 7104)

## What

FastAPI microservice implementing the `faces` row of
[CONTRACTS.md](../../CONTRACTS.md): YuNet (`cv2.FaceDetectorYN`) face
detection on base64-JPEG frames, returning box + 5-point landmarks (YuNet
order: right eye, left eye, nose tip, right/left mouth corner) + confidence,
plus the POC quality flag. With `within` (person boxes from the persons
service) detection runs per clamped crop and coordinates are mapped back to
frame space — the clamp/offset helpers are pure, unit-tested functions
(`app/mapping.py`).

**Quality flag** (`app/quality.py`, thresholds per the CONTRACTS.md POC
geometry / site-planner canon): the POC camera (UNV 2.8 mm @ 2.0 m mount,
subjects 2–3 m) yields ~64–85 px faces — deliberately below the production
80/100 px canon:

- `ok` — widthPx ≥ 80 (meets production canon)
- `sub-canon` — 56 ≤ widthPx ≤ 79 (POC-accepted, must stay flagged in reports)
- `reject` — widthPx < 56 (below the POC embedding floor)

**Measured signals** (per face, emitted only when measurable): these are what
the runner's composite quality gate is actually built from — the width flag
above is the historical one and the weakest of the four.

| key | what it is | why it is not width |
| --- | --- | --- |
| `iedPx` | inter-eye distance from YuNet's two eye landmarks | the size measure FR standards specify; our 56/80 px width floor is only ~24/34 px IED |
| `frontality` | 0..1, how centred the nose sits between the eyes | a wide face turned side-on has little identity-bearing geometry left |
| `sharpness` | variance of the Laplacian over the face crop, resized to a fixed square (`app/quality.py`) | a wide, square-on face smeared by a walking guest embeds badly and nothing else here can see it |

`sharpness` is **relative, not absolute**: it moves with exposure, contrast and
the camera's own sharpening, so it orders faces *within one camera* and its gate
floor has to be calibrated against that camera's own footage. The resize is not
optional — unnormalised Laplacian variance mostly measures how many pixels the
crop has, which is what `iedPx` already measures honestly.

A signal is **omitted when it cannot be measured** (no landmarks, a degenerate
crop). Absence means unknown, and the runner is required to read it that way:
rejecting a face for a signal nobody measured would drop a guest off an invoice.

## Model provenance & licence

| file | source | licence |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV zoo, face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (URL pinned to a commit) | [MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE) — permissive class allowed by the project licence gate |

Pinned URL + sha256 in [`../../models.lock`](../../models.lock); `make models`
downloads into `models/` (gitignored, never committed). Sizing caveat from the
planning corpus: the shipped ONNX self-reports WIDER FACE Hard ≈ 0.708 — size
capture zones against that, not the paper's 0.81.

## Run

```sh
make venv
make models
make run      # uvicorn on :7104
```

## API

- `GET /health` → `{ok, model, version}`.
- `POST /detect` `{imageB64, within?: [{x, y, w, h, ...}]}` →
  `{faces: [{box, landmarks: [5×[x, y]], conf, widthPx, quality, iedPx?,
  frontality?, sharpness?}], inferMs}`
  — frame coordinates; `within` boxes are clamped, degenerate/outside boxes
  skipped; extra keys on `within` boxes (e.g. `conf`) are ignored.

## Test

```sh
make test   # pure mapping/quality tests always run; model tests skip loudly without `make models`
make lint
```

Synthetic frames prove the endpoint contract, not detector recall — recall
gets measured on pilot footage.

## Tune

| env | default | meaning |
| --- | --- | --- |
| `FACES_SCORE_MIN` | `0.8` | YuNet score threshold |
| `FACES_NMS_IOU` | `0.3` | YuNet NMS threshold |
| `FACES_TOP_K` | `5000` | YuNet top-K before NMS |
| `FACES_CANON_PX` | `80` | quality "ok" boundary (production canon) |
| `FACES_FLOOR_PX` | `56` | POC embedding floor ("sub-canon" lower bound) |
| `FACES_SHARPNESS_NORM_PX` | `64` | square every crop is resized to before the Laplacian |
| `FACES_MODEL` | `models/face_detection_yunet_2023mar.onnx` | weights path |

## CPU latency — measured on this machine

Measured 2026-08-04 on **11th Gen Intel i5-1135G7 (8 threads), CPU-only**,
Python 3.12.3, opencv-python-headless 5.0.0.93; 50 runs after warmup:

| input | p50 | mean | p95 |
| --- | --- | --- | --- |
| 640×480 full frame | 10.0 ms | 10.2 ms | 12.2 ms |
| 200×300 person crop | 2.4 ms | 2.5 ms | 3.1 ms |

YuNet cost scales with input area, so per-person-crop mode (`within`) is the
cheap path at multi-person gates. This-machine numbers, not a benchmark.
