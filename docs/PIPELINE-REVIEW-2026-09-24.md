# Stage-by-stage review — 2026-09-24

Written against the JD Grand and Sharon footage on the .94 box (RTX 4060,
i7-14700K, 23 GB in WSL2), with every number below measured on that hardware
rather than estimated. Read it beside [DEPLOY.md](DEPLOY.md).

The venue population is Punjabi: turbans, caps, tilak, heavy jewellery,
evening decorative lighting. Where that changes a stage's behaviour it is
called out under **Punjab** in that stage's section, because most of it is
invisible in a benchmark and only shows up in a count.

---

## The three findings that matter most

**1. The camera placement is the binding constraint, not the pipeline.**
On the Sharon footage, 7–11 people are in frame and **1–3 faces** are
detected. Roughly 80% of everyone present yields no face at all, because it
is an overview/lobby camera: mounted high, wide, catching people from behind
or in profile at 2–20 m. On the JD footage the median detected face is
**67 px** at full 4K, below the 80 px canon and well below ArcFace's 112 px
native input — only **27% of faces clear 112 px**. No model change recovers
information the optics never captured. Zooming so faces land at 150–200 px
would do more for accuracy than every software change in this document.

**2. The match threshold is SFace's number under an ArcFace embedder.**
`HECO_MATCH_THRESHOLD` is 0.363, the SFace paper's 1:1 operating point, and
the near-miss bands (0.29 face / 0.15 clothing) were measured on SFace 128-d
embeddings at a bench. The install now runs ArcFace 512-d, where cosine sits
on a different scale entirely. Observed consequence: a child and an adult
suggested as the same guest — on the *clothing* band, where the face barely
participates — and a woman suggested as a bearded man. The bands are now
disabled (`HECO_MATCH_NEARMISS_FLOOR=0`); the threshold itself is still
uncalibrated and `eval/sweep.py` is the fix.

**3. The chain is latency-bound, not compute-bound.**
One camera on 4K runs at **3.9 fps** with the GPU near 50% and 26 of 28 CPU
threads idle. Two cameras run at **4.53 and 5.07 fps each** — faster
individually than one camera alone, for 9.6 fps of combined work. Every
stage waits on the one before it; a second stream fills the gaps rather than
competing for them. This is why frame-level pipelining is the remaining
structural win for a single camera, and why two cameras cost far less than
double.

---

## ingest (7101)

**What it does.** One `cv2.VideoCapture`, a single-slot latest-frame buffer,
serving base64 JPEG on demand.

**Measured.** 4K HEVC decode **5.99 ms/frame — a 167 fps ceiling**. JPEG
encode at 4K 13.9 ms. Decode alone could feed eleven cameras at 15 fps.

**NVDEC is not worth it, quantitatively.** The Windows Task Manager shows
Video Decode at 0%, which is true, and tempting. But decode is under 3% of
the frame budget, and the decoded frame would land in GPU memory while the
very next step (JPEG encode) is on the CPU — so a device→host copy of
12–24 MB eats the saving. Expected net: 1–3 ms of a 224 ms frame. It only
pays as part of a GPU-resident frame path, where it comes along for free.

**Upgrade: `INGEST_MAX_WIDTH` is a trap at this geometry.** It halves every
downstream cost (measured: 224 → 115 ms) and finds the same person boxes.
But faces ≥112 px drop from 27% to 15% of detections. Do not enable it until
faces are large enough to afford it.

**Bug risk.** `source_stall_s` is 45 s. On a live RTSP camera a PoE bounce
under that is survived; over it the run ends. Fine for a demo, worth knowing.

---

## persons (7102) — YOLOX-S on CUDA

**Measured.** `detect()` is **8.6 ms** at any input resolution (letterbox
resizes to 640 first, measured 2 ms flat). Over HTTP the service reports
`inferMs` of **31–43 ms** for the same frame.

**OPEN, UNEXPLAINED.** That 4× gap is real and I could not account for it.
Ruled out by measurement: threading (8.1 ms on main, pool, and fresh pool
threads), fresh-buffer allocation (8.66 ms on a freshly decoded 24 MB
frame), box load (idle), lazy session build on a worker thread (8.76 ms),
and CPU fallback (GPU confirmed at 47–58% under service load). It needs
in-process profiling of the uvicorn worker to settle. **If it is
recoverable it is worth ~25 ms/frame on the dominant stage** — the single
largest identified-but-unclaimed win in the pipeline.

**Punjab.** YOLOX is COCO-trained on class 0. Turbans, caps and shawls do
not affect person detection; the measured recall on this footage (7–11 boxes
per frame including background crowd) is not the problem.

---

## tracker (7103) — in-house SORT-lite

**What it does.** Greedy IoU + constant velocity, `max_age=30`,
`min_hits=3`, `iou_min=0.2`.

**Upgrade, ranked highest of the model-level changes.** This is a 2016-era
baseline. ByteTrack, OC-SORT and BoT-SORT are MIT-licensed — no procurement
conversation — and are materially better in exactly these conditions: crowd
surges, occlusion, people stopping to greet. The evidence it matters is
already in the run stats: **190 distinct tracks for 38 unique guests** on one
clip, and `lockedTrackFolds` and `healedSplits` firing repeatedly.

**Punjab.** A baraat is the worst case for greedy IoU: dense, mutually
occluding, irregular motion. Expect track fragmentation to be the dominant
error mode at arrival, not face quality.

---

## faces (7104) — SCRFD-10G on CUDA

**Changed this session: whole-frame detection.** The crop path existed for
*resolution* — letterboxing 4K into a 640 square leaves a 67 px face ~11
network px, under what stride-8 anchors resolve — so the runner cut the
frame into person boxes and paid one inference each. Measured over 15 real
frames:

| | faces | median W | ≥112 px | ms/frame |
|---|---|---|---|---|
| crops (7/frame) | 81 | 71 px | 18 | 138.1 |
| whole frame | 67 | 97 px | **27** | **37.3** |

Four times faster **and half again as many usable faces**, because it does
not depend on the person detector having found somebody to crop. Validated
separately on the Sharon footage: same usable-face count as crops. Enabled
by `HECO_FACES_WHOLE_FRAME=1`.

**Changed this session: score floor 0.5 → 0.7.** SCRFD's family default is
InsightFace's permissive `det_thresh`; YuNet's equivalent was 0.8. At 0.5 the
detector hallucinated a face on the **back of a woman's head** and she became
a counted guest whose best photograph was her hair.

**Calibration warning for whoever tunes the gate next.** A larger network
input finds more faces but FEWER over 112 px (1472×832: 106 faces, 22 over,
versus 640: 67 faces, 27 over). Higher resolution draws tighter boxes, so the
same face measures smaller. **Absolute pixel floors are not comparable across
input sizes.**

**Punjab.** SCRFD is WIDER FACE-trained and handles head coverings well at
the detection stage — turbans and caps did not suppress detection in this
footage. Veils and dupattas that cross the face do, correctly.

---

## quality gate (in the runner)

**Changed this session: a detector-confidence floor (`minConf`).** Every
other gate signal — frontality, eye-span, IED — is computed from the
detector's OWN five landmarks. When the detector invents a face it invents
plausible landmarks with it, so every geometric floor confirms the
invention. `conf` was already on every face and was never read; it is the
only independent witness. Armed at 0.7 via the new **"Strict — detector
confidence"** console profile, with a `gatedByConf` counter.

**The 112 px floor is doing real work and costing real guests.** With
`minPx: 112`, `gatedByWidth` reached 36,491 on one run — correctly rejecting
73% of detections as too small for ArcFace. That is the geometry problem
surfacing as a gate statistic, not a gate that is too strict.

**Punjab.** Frontality (nose centred between the eyes), the landmark
plausibility check and eye-span all use points **below the hairline**, so
turbans, caps and tilak do not affect them. This stage is demographically
neutral. The next one is not.

---

## embed (7105) — ArcFace-R50 512-d on CUDA

**This is where head coverings bite, and it is quantifiable.** The canonical
ArcFace template puts the eye line at y ≈ 51.6 of a 112-px crop — so **46% of
what the embedder sees is above the eyes**: forehead, hair, and whatever
covers the head.

- **Turbans** occupy nearly half the embedding input. ArcFace was trained on
  WebFace600K, overwhelmingly uncovered heads. Two turbaned men share a
  large, visually similar region, which pulls their embeddings *together* →
  false merges → **undercount**, which is the silent direction.
- **Caps** behave the same way.
- **Tilak** is a high-contrast mark in that region; it can shift the same
  person's embedding between sightings as lighting changes across a venue.

**There is no clean fix by cropping.** Tightening the crop to exclude the
forehead breaks the alignment ArcFace was trained against, which degrades
every face. The realistic mitigations are (a) calibrate the threshold on
*this* population, (b) lean on the appearance/clothing signal the matcher
already carries, and (c) when labelling, tag each person turbaned / capped /
veiled / none — if merge errors cluster in one group that is measurable and
actionable, and if they do not, a lot of speculative work is avoided.

**Alignment itself is correct** — Umeyama similarity onto the canonical
5-point template, with a landmark-variance floor that refuses degenerate
sentinels.

---

## match (7106) — cosine gallery

**Uncalibrated, and this is the biggest accuracy risk.** See finding 2. The
near-miss bands are off; the 0.363 threshold remains. `eval/sweep.py` needs
a labels export plus a golden capture with `HECO_GOLDEN_EMBEDDINGS=1` and
will emit a `heco-pack/1` with the whole error curve. **It refuses to propose
a threshold when the distributions do not separate**, which is itself the
answer worth having — overlap beyond repair is an acquisition problem.

**Embeddings are not compatible across the SFace/ArcFace switch.** 128-d vs
512-d: any gallery or staff store written under one is meaningless to the
other and must be re-enrolled. `HECO_EMBEDDING_DIM` and `HECO_EMBEDDER_ID`
travel together.

**Punjab.** The appearance signal includes clothing colour
(`nearMissClothes` 0.78, `appearanceClash` 0.5). At a wedding where many
guests wear similar formal wear, this signal is weaker than at a mixed-dress
venue — and under IR illumination it collapses entirely to monochrome, which
is a strong argument for the camera's warm white light over IR.

---

## Cross-cutting bugs found and fixed this session

| Bug | Effect | Status |
|---|---|---|
| Moment prune deleted forensic frames | forensic ON, 13,605 frames, **372 pictures kept** — empty frame box for everything older | fixed + regression test |
| `HECO_DEVICE` never reached faces/embed | would run SCRFD/ArcFace on CPU while looking correct | fixed in overlay |
| `FACES_SCRFD_SCORE_MIN` not plumbed | setting it looked like it worked and changed nothing | fixed in overlay |
| Shared `cpuset` pins to a 64-thread T440 | docker refuses the container on any smaller box | documented, `HECO_CPUSET=` |
| `PLANNER_PUBLIC_URL` defaults to localhost | split-machine runs accept the start and never report | documented in deploy line |
| Compose merges `ports:` lists | second stack dies claiming the first's port | `!override` |
| `heco-*:cuda` not rebuilt by `compose build` | code changes silently do not reach the running container | documented |

---

## Ranked backlog

1. **Fix the camera geometry.** Free, and larger than everything below.
2. **Calibrate the threshold** (`eval/sweep.py`) — needs labelled footage.
   Until then no count from this profile is trustworthy in absolute terms.
3. **Find the persons 8.6 → 31 ms gap.** ~25 ms/frame on the dominant stage.
4. **Swap the tracker** for ByteTrack/OC-SORT. MIT, no procurement, and the
   190-tracks-for-38-guests number says it matters.
5. **Frame-level pipelining.** The two-camera result proves the headroom is
   there for a single stream.
6. **Shared-memory transport.** ~82 ms/frame, the largest single win, and the
   largest change — it also ends the ability to split stages across machines.
7. RT-DETRv2-S. Deferred deliberately: persons is 13% CPU and the GPU is at
   24–50%; a stronger detector on a stage that is not limiting buys nothing,
   and there is no official ONNX export.
