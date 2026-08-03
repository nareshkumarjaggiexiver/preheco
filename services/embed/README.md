# embed — SFace face embedding (port 7105)

## What

FastAPI microservice implementing the `embed` row of
[CONTRACTS.md](../../CONTRACTS.md): SFace (`cv2.FaceRecognizerSF`) turns each
detected face (box + 5-point landmarks from the faces service) into a
128-float identity vector. `alignCrop` warps the face to the model's 112×112
landmark template first — that is the alignment SFace expects; the face-JSON →
15-float alignment row conversion is a pure, unit-tested function
(`app/face_row.py`).

Embeddings are returned raw (not L2-normalised): the match service compares
by cosine (threshold 0.363 per contract), which is scale-invariant.
Callers apply the quality gate — only faces at/above the 56 px POC floor
should be sent here (sub-canon 56–79 px allowed but flagged upstream).

## Model provenance & licence

| file | source | licence |
| --- | --- | --- |
| `face_recognition_sface_2021dec.onnx` | [OpenCV zoo, face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (URL pinned to a commit) | [Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE) — explicit author grant via OpenCV zoo |

Pinned URL + sha256 in [`../../models.lock`](../../models.lock); `make models`
downloads into `models/` (gitignored, never committed). Per the planning
corpus: this is the shipped MobileFaceNet-class model (LFW ≈ 99.40%);
training-data provenance is gray and disclosed as such in product docs.

## Run

```sh
make venv
make models
make run      # uvicorn on :7105
```

## API

- `GET /health` → `{ok, model, version}`.
- `POST /embed` `{imageB64, faces: [{box, landmarks, conf?}]}` →
  `{embeddings: [[128 floats]], alignMs}` — one embedding per face, order
  preserved; `alignMs` is the wall time of the whole align+embed loop.
  Malformed faces (not five [x, y] landmarks) → 400.

## Test

```sh
make test   # pure face_row tests always run; model tests skip loudly without `make models`
make lint
```

SFace aligns+embeds any pixels once given plausible landmarks, so the
model-dependent tests fully exercise the real graph with synthetic frames
(including determinism: same input → identical vector).

## Tune

| env | default | meaning |
| --- | --- | --- |
| `EMBED_MODEL` | `models/face_recognition_sface_2021dec.onnx` | weights path |

(The cosine threshold lives in the match service, not here.)

## CPU latency — measured on this machine

Measured 2026-08-04 on **11th Gen Intel i5-1135G7 (8 threads), CPU-only**,
Python 3.12.3, opencv-python-headless 5.0.0.93; 50 runs after warmup,
align+embed against a 640×480 frame:

| input | p50 | mean | p95 |
| --- | --- | --- | --- |
| 1 face | 6.1–6.7 ms | 6.2–6.9 ms | 7.0–8.3 ms |
| 4 faces (same call) | 25.1–25.4 ms | ~25.7 ms | ~27 ms |

Cost is linear per face (~6.4 ms/face). One outlier run measured the 4-face
batch at ~60 ms (not reproducible across two further runs — likely transient
CPU frequency/thermal state); recorded for honesty. This-machine numbers,
not a benchmark.
