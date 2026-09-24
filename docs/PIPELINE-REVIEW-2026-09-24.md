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

---

## Duplicate review fixes — 2026-09-24, evening

Run `d02-20260923221223-mp4-f0bfc5`: the shm stack (ports 73xx) on
`D02_20260923221223.mp4`, 3840×2160 at 15 fps, 1005 s, 15,076 frames, Punjab
wedding-hall overview camera. **74 guests counted; 500 pairs in the review
queue**, which is the cap, not the count — 2,701 pairs were considered. The
top of the queue, read in the browser against the face cards:

| # | a | b | face | clothes | what a human sees | what would have settled it |
|---|---|---|---|---|---|---|
| 1 | p00012 | p00065 | 0.357 | 0.15 | red turban, black beard / orange turban, white beard, head down | age (36 vs 63 on the cards) — and the clothes already said no |
| 2 | p00001 | p00047 | 0.347 | 0.78 | small girl, head down / girl, **half a face** behind a pillar | p00047 should not have been embedded at all |
| 3 | p00032 | p00049 | 0.345 | 0.07 | man, blue shirt / elderly woman, glasses | gender |
| 4 | p00014 | p00019 | 0.341 | 0.71 | grey hair, plain white shirt / white hair, **striped** shirt | texture — the colour histogram cannot see a stripe |
| 5 | p00048 | p00052 | 0.338 | 0.72 | girl, yellow top, **back-turned in the same frame** / woman, dark green dress | co-presence — they were on screen together |

And p00002, a girl with a railing across her face, was enrolled as a guest.

### Root causes, each measured

**(a) The shm stack had no review floor.** `HECO_REVIEW_FLOOR` reached the
camera stacks through the passthrough in commit 705a013 and the operator's
shell. The shm tree's `docker-compose.yml` predated that commit — no
passthrough — so `shm-up.sh`'s `HECO_REVIEW_FLOOR=0.28` was exported,
looked set, and never reached the match container, which ran the service
default: 0.15, SFace's number. At 0.15 this run queues 500 (capped); at
0.28 it queues **32**. 468 of the 500 pairs were noise wearing a rank.
`demo-up.sh` now exports 0.28 itself and `status` reads the value back out
of the container, so the next unplumbed knob shows as `·` and not as a
number somebody typed.

**(b) Co-presence only saw faces.** `_assert_co_presence` asserts a
cannot_link between identities whose faces matched in the same frame. A
guest with her back turned has no face, so pair #5 — p00048 back-turned in
the same frame as p00052's face — was never asserted. Replaying the run's
ledger with a track-presence rule (a track bound to an identity once a face
on it matched at ≥ 0.45, presence kept while the tracker reports the id and
the track is not contested at IoU ≥ 0.4): **48 cannot_link pairs against 25
from faces alone**; p00048/p00052 asserted at frame 9283, p00001/p00004 at
687. Of the 32 pairs over 0.28 it removes 2. It is a certain signal with a
small yield; it is on because the two it removes are exactly the kind an
operator cannot judge from two cards.

**(c) The torso descriptor was mostly skin.** `torso_descriptor` cropped
from the face's bottom edge down 2.5 face heights, so the neck and upper
chest — skin, hue bin 1, low saturation — and hair dominated the 48-bin
histogram. Yellow top against dark green dress scored 0.72 because the
crops agreed on the skin, not the cloth. And a 12×3 H×S histogram has no
term for pattern: plain white against striped white is the same histogram.

**(d) Nothing knew gender or age.** Pair #3 is a man and a woman at face
0.345. ArcFace does not encode sex as a separable axis and the queue had no
other witness.

**(e) Nothing used body size.** A child and an adult in one queue, with a
person box on every sighting of both.

### What changed (the wire contract; each file documents its own part)

- **embed** `/embed` answers `norms` (L2 of the raw feature) and
  `attributes` `{gender, genderP, age}` per face, from the InsightFace
  buffalo_l `genderage.onnx` (1×3×96×96 RGB, no normalisation; output
  `[F, M, age/100]`). `EMBED_ATTR_MODEL` names it; unset means
  `models/genderage.onnx` if present, else off, and off is `null` — not
  measured is not nobody. `/health` says `attrModel`.
- **runner** forwards `attributes`, `featNorm` and the sighting's
  containing person box (`body`) to match; `HECO_PRESENCE_SPLIT` (default 1)
  is the track-presence rule above; `HECO_QUALITY_MIN_FEAT_NORM` (default
  0 = off) drops a face after embedding when its raw feature norm is under
  the floor, counted as `gatedBy featnorm` — the one gate that can see a
  railing across a face, because it is the recogniser's own confidence.
- **match** stores gender/age/norm per template and every body box in
  `body_sightings` (with the sighting's face width, `face_w`, and the
  `/match` reply's `bodyId` so the same-frame guard can retract a box logged
  under the wrong key); `/review/duplicates` pairs carry `why` (gender with
  its probability, median age, stature in metres) and the reply carries
  `excluded: {gender, age, stature}`. Set-aside pairs are not shown and not
  written to cannot_link. `appearance` accepts 48 (v2) or 64 (v3) floats;
  comparing the two is `None`, never 0.
- **counting** `torso_descriptor` v3, 64 floats: colour on NON-SKIN pixels
  of a band that starts half a face height BELOW the face (skin-toned
  pixels — YCrCb Cr 133–173, Cb 77–127, at saturation under 150 — weighted
  a quarter, not dropped) weighted 0.9, uniform LBP(8,1) texture 0.07,
  Sobel edge-density 0.03, 12 reserved zeros; each part L1-normalised so
  the intersection stays 0..1. Two None rules beyond the contract's three:
  a face within 0.4 face widths of the person box's side (the band is the
  occluder) and a band whose resized mask keeps no cloth pixel (no
  pattern reading; colour-only would cap every comparison at 0.9).
- **planner** shows one `why` line per pair ("man / woman · ages 41 / 63 ·
  stature 1.71 m / 1.58 m", absent as "not measured", two decimals because
  a one-decimal line printed 1.74 m and 1.66 m as the same) and appends "N
  more pairs were set aside: gender G, age A, stature S" to the summary.
  Default person height in FrameCheck is 1.75 m.

### The knobs

All read by the owning service; empty means unset; **0 turns a signal off**.
`demo-up.sh status` prints what the running containers hold.

| env | service | default | what it does |
|---|---|---|---|
| `HECO_REVIEW_FLOOR` | match | 0.15 in the service, **0.28 from demo-up.sh** | face cosine floor of the review queue |
| `HECO_REVIEW_GENDER_MIN_P` | match | 0.8 | set a pair aside when BOTH identities' template-weighted gender confidence ≥ this and they disagree; one confident side is one opinion, not a disagreement |
| `HECO_REVIEW_AGE_CHILD_MAX` / `HECO_REVIEW_AGE_ADULT_MIN` | match | 12 / 20 | one median age ≤ child, the other ≥ adult; the 8-year dead zone is the model's error on a 40 px face |
| `HECO_REVIEW_STATURE_GAP` | match | 0.2 | abs(ratio a − ratio b) ≥ this — 35 cm on the 1.75 m anchor; the measured adult-adult spread is 0.98–1.08, a child read 0.73 |
| `HECO_REVIEW_STATURE_MIN_N` | match | 8 | standing sightings before a stature is trusted; under it, null |
| `HECO_STATURE_ADULT_M` | match | 1.75 | metres a ratio of 1.0 means — the North Indian adult average (5'9"). Display only; the exclusion compares ratios. (The first cut's service read `HECO_REVIEW_ADULT_M`, so this knob printed as set and changed nothing; the service reads this name since the evening fix, and a test pins it) |
| `HECO_PRESENCE_SPLIT` | runner | 1 | track-presence co-presence |
| `HECO_QUALITY_MIN_FEAT_NORM` | runner | 0 (off) | post-embed floor on the raw feature norm; `gatedBy featnorm` |
| `EMBED_ATTR_MODEL` | embed | `models/genderage.onnx` if mounted, else off | the attribute head |

### Measured on this run's data, before any of it was deployed

**Stature.** Fit over 1,268 standing sightings (h/w ≥ 2, box bottom above
the frame's last 60 px, box top > 5): `h = 0.602 · y_bottom + 300 px`.
Ratios for the five pairs, in metres on the 1.75 m anchor:

| pair | a | b | gap | separable at 0.2? |
|---|---|---|---|---|
| #1 | p00012 1.07 (1.87 m, n=33) | p00065 1.03 (1.79 m, n=26) | 0.04 | no |
| #2 | p00001 0.82 (1.43 m, n=111) | p00047 — no standing sighting (half behind a pillar) | — | not measured |
| #3 | p00032 1.02 (1.78 m, n=30) | p00049 0.93 (1.62 m, n=14) | 0.09 | no |
| #4 | p00014 1.08 (1.90 m, n=20) | p00019 1.01 (1.77 m, n=17) | 0.07 | no |
| #5 | p00048 0.87 (1.52 m, n=19) | p00052 1.04 (1.82 m, n=53) | 0.17 | no — under the gap; track presence takes this one |

Stature separates **none of the five**, and two pairs in the top 32:
p00005/p00009 (1.04 vs 0.73) and p00009/p00071 (0.73 vs 1.03). p00009 is a
child. The gap is set for that distinction and not for 1.62 m against
1.78 m, deliberately: a 0.09 spread is one person mid-stride.

**Gender and age, a single-card probe.** The attribute head run on each
identity's best face card in the scratchpad (the card is already padded,
so this under-fills the model's 1.5× crop — read these as a lower bound on
what the service, cropping from the detector box in the frame, will see):

| id | read | id | read |
|---|---|---|---|
| p00012 | M 1.00, 36 | p00065 | M 0.92, 63 |
| p00001 | **M 0.91, 39** (a small girl, head down) | p00047 | F 0.90, 25 |
| p00032 | M 1.00, 61 | p00049 | **M 0.74, 55** (an elderly woman) |
| p00014 | M 1.00, 48 | p00019 | M 1.00, 69 |
| p00048 | **M 0.77, 38** (a girl) | p00052 | **M 1.00, 41** (a woman in a dress) |
| p00002 | F 0.72, 35 (a girl, railing) | | |

On the cards alone the head is wrong or unsure on every child and on two of
the three women, and would set aside **none of the five**: #3 fails the 0.8
bar on p00049's side, #5 reads two men. Age would separate #1 (36 vs 63) if
the bands were adult-vs-elderly, which they are not. This is the reason the
exclusion needs BOTH sides confident and takes the template-weighted mean
over every enrolled sighting rather than one card: a signal that is unsure
on children and head-down views must not vote from that state. Whether the
run's templates do better than its cards is the first thing to read off
`excluded` on the next run with the head mounted — if they do not, the
gender signal stays a `why` line for the human and the exclusion knob goes
to 0.

**Clothes.** With the v2 descriptor #5 read 0.72 on skin and hair. The v3
band starts half a face below the chin, down-weights skin, and carries a
texture term; the number to watch is #4 (plain against striped, 0.71 under
v2), which the LBP part exists for.

### The evening's second pass — what the review of the change measured, and what moved

Each of these was measured on the run's own data before the code moved.

**The pattern parts were lifting impostors, not separating the targets.**
On the 113 cached crops the texture+edge parts overlap at a median 0.77
ACROSS identities (0.93 within): plain cloth agrees with plain cloth on
weave. At 0.3 of the descriptor they added ~0.2 of constant agreement to
every clear impostor pair (#1 pink check vs white kurta 0.19 → 0.38, above
the runner's 0.35 heal-clash floor; #3 blue shirt vs cream suit 0.02 →
0.27) and moved the two target pairs by nothing — on #4 and #5 the pattern
overlap (0.86 / 0.88) was HIGHER than colour's, because a plain kurta at 4K
has buttons, pocket flaps and folds. Weights are now 0.9 / 0.07 / 0.03:

| pair | v2 (live) | first cut 0.7/0.2/0.1 | shipped 0.9/0.07/0.03 |
|---|---|---|---|
| #1 p00012/p00065 (pink check / white kurta) | 0.242 | 0.381 | **0.254** |
| #3 p00032/p00049 (blue shirt / cream) | 0.156 | 0.268 | **0.096** |
| #4 p00014/p00019 (plain white / striped) | 0.711 | 0.739 | 0.711 |
| #5 p00048/p00052 (yellow top / green dress) | 0.680 | 0.722 | 0.722 |
| #6 p00074/p00075 | 0.241 | 0.326 | **0.158** |
| within-identity mean / worst | 0.849 / 0.672 | 0.874 / 0.728 | 0.864 / 0.692 |

Two plain garments of different colours now floor at 0.10, not 0.30. #4
and #5 are still not separated by clothing — the module docstring says why
(three brightness bins; chromatic brightness ignored) — and #2 now reads
"not measured": p00047's band was the pillar beside her (face 0.17 face
widths from the box edge), and a face within 0.4 widths of the box's side
is no longer a torso reading. A skin-weight RAMP at the window edge was
measured and rejected (every width lowered the pale-yellow top's
self-agreement, 0.69 → 0.65); the saturation cap at S 150 closes the
cliff it was for (saturated orange is inside the Cr/Cb window to V 152).

**Stature could set aside a genuine duplicate.** p00052 (pair #5) had 15
consecutive 205×470 px boxes at seq 9075–9091 — head-to-waist behind a
table, aspect 2.3, so the standing test passed — reading 0.56–0.59 against
1.04–1.07 from her full boxes; a second key minted for her in those frames
would have been set aside against herself at a 0.47 gap. The face width is
already on the wire (`quality`), so it is stored beside the box and a
standing box must be at least **6 face widths** tall (adults p10 10.0 /
median 11.0 / p95 12.4, the smallest child p10 9.9, the waist-up boxes
2.9–4.3). Replayed: 47 of 1,268 standing boxes drop, the fit becomes
`h = 0.463 · y_bottom + 552`, no identity's 8-frame median moves by 0.2,
children stay measured (p00001 0.84). And an identity's boxes must span at
least **two distinct write seconds** — one occlusion's 15 frames are one
second at 15 fps — not `min_n` seconds: the 53 identities with eight
standing boxes spanned a median of four seconds and only three spanned
eight, so a bar of eight would have measured almost nobody.

**Gender: one confident template made a confident identity.** On 594
re-embedded sightings, 8 of 37 identities with two or more reads carried
BOTH a ≥ 0.8 male and a ≥ 0.8 female read (p00048, a girl: eleven M ≥ 0.8
and one F 0.92; p00049, an elderly woman: F 0.84 and M 0.87), and 7 of the
72 keys held a single template — the view a duplicate is minted on is the
view the head flips on. The identity's P(male) is now the template mean
shrunk by two pseudo-votes, `(Σ + 1) / (n + 2)`: one M 1.0 reads 0.67,
three unanimous 0.80 (the bar), five 0.86, p00048's fifteen reads 0.765
(under the bar), and four M 0.97 with one F 0.92 read 0.73 — one dissenting
confident view among five blocks the exclusion.

**Track presence, two guards.** The presence rule could write a permanent
cannot_link between one person's two keys: (a) the detector double-boxes
one body (full + upper; 12 of 2,671 bind frames, IoU 0.07–0.45, under
every dedupe floor), the face lands on the upper box, the bound track on
the full one — two tokens, one body; (b) the nearest-centre tie rule binds
a face to a shorter NEIGHBOUR's box overlapping the head (seq 1012: p00006
bound the plaid-shirt man behind her; 131 of 2,671 bindings sat > 0.3
widths off the track centre) and the stale binding asserted her key on his
body for 200 frames. Now a face binds only where a head sits in a standing
box (centre in the top 35%, within 35% of a width of the middle), and a
bound track whose box contains any keyed face's centre says nothing.
Priced on the replay: 47 of 48 pairs kept (the one lost rested on the
mis-binding), both queue pairs still retired, 246 fewer bindings. The cost
paragraph in the code now states the true price: a wrong cannot_link is
NOT operator-fixable — `/merge` refuses the pair for a human too, nothing
deletes the row, the review queue withholds it — so only a fresh run
clears it. A staff verdict on a bound track unbinds it; a mint never binds
whatever cosine it carries (`isNew` is the guard, not the number); and a
bound track's body joins the fold guard, so a fold into a key present by
track elsewhere in the frame is refused like a fold into a face.

**Version skew.** The runner sends the 64-float descriptor; a match service
still on 0.11.0 refuses it with a 422, and the per-face `/match` call sat
on the frame loop's critical path with no handler — every run would have
failed on its first descriptor-bearing face during a partial rollout. The
runner now re-asks WITHOUT the descriptor on that one 422 (the path an
unmeasurable torso already takes), counts it as `appearanceRefused`, and
the run says "torso not measured" instead of "failed". Rebuild match first
anyway (below).

### Deploying this to the .94 box

Since the evening of 2026-09-24 the box has a GitHub deploy key, and both
trees are clean checkouts that track their branch — **deploy is push here,
`git pull` there**. (Before that the trees were rsync copies over an old
689bbd3 checkout, so their `git status` described a copy, not a branch;
both were reset to their origin branches that evening.)

| tree | branch | stacks |
|---|---|---|
| `~/heco-pipeline-new` | `main` | cameras A (7100–7106) and B (7200–7206), `demo-up.sh` |
| `~/heco-pipeline-shm` | `shared-memory-transport` (main merged in) | shm (7300–7306), `shm-up.sh` |

The attribute model is not in git (InsightFace non-commercial weight, no
direct URL): copy `~/dl/genderage.onnx` into `services/embed/models/` in
both trees once, where the compose bind mount puts it under the embed
service's default path.

**Images are rebuilt per tree, on the box.** Weights are bind-mounted, but
code is baked, so a synced file changes nothing until:

```sh
docker compose build          # runner match ingest tracker, and the PLAIN persons/faces/embed
docker build -f docker/persons-cuda.Dockerfile --build-arg BASE=heco-persons -t heco-persons:cuda .
docker build -f docker/persons-cuda.Dockerfile --build-arg BASE=heco-faces   -t heco-faces:cuda   .
docker build -f docker/persons-cuda.Dockerfile --build-arg BASE=heco-embed   -t heco-embed:cuda   .
./scripts/demo-up.sh          # or ./scripts/shm-up.sh in the shm tree
```

The three `:cuda` images are built FROM the plain ones, so the plain build
comes first, and `docker compose -f … -f docker-compose.gpumax.yml build`
must never be the command (the header of that file says why: it would tag
a CPU image `:cuda`). **Both trees build the same image tags** —
`heco-runner`, `heco-match`, `heco-persons:cuda`, … — so whichever tree
built last is what BOTH stacks get on their next recreate. The box carries
`heco-*:shmcuda` and `heco-shm-base` tags from earlier hand builds; the
committed overlays do not reference them. Rebuild the shm tree last, or
retag, and then read the `build` id from every runner's `/health` (:7100,
:7200, :7300 — DEPLOY.md): identical code is an identical hash, and a
stack that missed the rebuild is the one whose hash differs.

**Rollout order, per tree: match first.** Rebuild and restart `match`
(`/health` must say `"version": "0.12.0"`) BEFORE the runner/counting image
is recreated. The old match refuses the new runner's 64-float torso
descriptor; the runner now degrades that to "torso not measured"
(`appearanceRefused` on the run status, non-zero = the match image on that
stack is stale) rather than failing the run, but a run counted that way
has no clothing evidence at all. The reverse skew (old runner, new match)
is harmless: the new match accepts 48-float descriptors and ignores
nothing it does not know.

Then `demo-up.sh status`: the device-truth block now prints the embed
service's attribute model (absent = every `why` line reads "not measured"
and the gender/age exclusions never fire), and the knob block prints the
review and presence knobs as the containers hold them.
