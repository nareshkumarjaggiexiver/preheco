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
by cosine (threshold 0.363 per contract), which is scale-invariant. The raw
feature's L2 norm rides beside each vector (`norms`) — for both families the
magnitude tracks crop quality, so the runner can floor on it
(`HECO_QUALITY_MIN_FEAT_NORM`) once the vector itself is normalised away.
Callers apply the quality gate — only faces at/above the 56 px POC floor
should be sent here (sub-canon 56–79 px allowed but flagged upstream).

### Gender and age (optional)

`app/attributes.py` runs InsightFace's `genderage.onnx` on the same faces,
in the same lock, one session run per face (~1 ms on CPU), and answers
`attributes: [{gender: "M"|"F", genderP, age}]` index-parallel to the
embeddings. It exists because run f0bfc5's review queue put a man beside an
elderly woman at face cosine 0.345 and nothing in the pipeline could say why
that pair was wrong; the match service now sets such pairs aside.

The crop is InsightFace's, not the embedder's: centred on the face BOX,
scaled so 1.5x its longer side fills 96 px, no rotation, fed as RGB 0..255
(the graph normalises itself). Send the detector's TIGHT box — the run's
face cards carry `FACE_CARD_PAD` = 35 % on each side, and fed whole they
read p00049 (an elderly woman) as M 0.74; with the padding undone the same
card reads F 0.99, and on the source frames with the detector boxes she
votes F in 5 of her 6 largest sightings. Faces without eyes in the crop
(p00001, a girl looking straight down) read as an adult male either way:
the net has nothing to read, so the match service weighs gender by template
confidence rather than trusting any one sighting.

Off when no model is configured: `attributes` is then `null` as a whole
(never `[]`), `/health` says `attrModel: null`, and the review line reads
"not measured" — absent is not zero.

## Model provenance & licence

| file | source | licence |
| --- | --- | --- |
| `face_recognition_sface_2021dec.onnx` | [OpenCV zoo, face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (URL pinned to a commit) | [Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE) — explicit author grant via OpenCV zoo |

Pinned URL + sha256 in [`../../models.lock`](../../models.lock); `make models`
downloads into `models/` (gitignored, never committed). Per the planning
corpus: this is the shipped MobileFaceNet-class model (LFW ≈ 99.40%);
training-data provenance is gray and disclosed as such in product docs.

| file | source | licence |
| --- | --- | --- |
| `genderage.onnx` (optional) | InsightFace `buffalo_l` pack ([release v0.7](https://github.com/deepinsight/insightface/releases/tag/v0.7), `buffalo_l.zip`), sha256 `4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb`, 1.3 MB | InsightFace model weights are released for **non-commercial research** — restricted tier (doc 15 §3), same standing as the arcface embedder; not in `models.lock`, never part of a default deploy |

There is no direct download URL for the single file (it ships inside the
zip), so it is placed by hand: unzip `buffalo_l.zip` and copy
`genderage.onnx` into `models/`, or point `EMBED_ATTR_MODEL` at it.

## Run

```sh
make venv
make models
make run      # uvicorn on :7105
```

## API

- `GET /health` → `{ok, model, version, error, device, attrModel, attrError}`.
  `attrModel` is the loaded gender/age file's name or `null`; `attrError`
  is set when a file NAMED by `EMBED_ATTR_MODEL` will not load (`ok` stays
  true — embedding works, the loss is just not silent).
- `POST /embed` `{imageB64, faces: [{box, landmarks, conf?}]}` →
  `{embeddings: [[128 floats]], alignMs, norms: [float],
  attributes: [{gender, genderP, age}] | null, attrMs: float | null}` —
  one embedding per face, order preserved; `norms` is the L2 norm of each
  raw feature; `alignMs` is the wall time of the align+embed loop alone,
  `attrMs` that of the attribute pass (null when off).
  Malformed faces (not five [x, y] landmarks; a box without numeric
  x, y, w, h when the attribute pass is on) → 400.

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
| `EMBED_ATTR_MODEL` | `models/genderage.onnx` if present, else off | gender/age weights path; `off` disables the pass even when the default file exists |
| `HECO_DEVICE` | `CPU` | ORT providers for the arcface family and the attribute pass (`CUDA`, `GPU`, …); the sface family is cv2 and ignores it |

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

The gender/age pass, measured 2026-09-24 on an 8-thread WSL box (a slower
SFace baseline than the table above: 4 faces align+embed p50 ≈ 28 ms),
p50 wall per `/embed` with 4 faces, ON/OFF legs interleaved: pass off
38 ms, pass on 40.5 ms — `attrMs` ≈ 3.5 ms for 4 faces, ~1 ms per face.
That number depends on the attribute session running on ONE intra-op
thread: with ORT's default pool the same call took 69 ms, because the
pool's spin-waiting threads slowed cv2's SFace in the same loop (alignMs
29 → 54 ms) — see `attributes.py`.
