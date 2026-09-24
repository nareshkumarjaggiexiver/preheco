# `shared-memory-transport` — status, 2026-09-24

**Not ready to merge. The headline measurement is not trustworthy yet, and
this document says why rather than quoting the number that looked good.**

## What it does

Frames stop being JSON. ingest writes each frame once to a tmpfs as raw BGR
and hands the consuming stages its name; they read it out of the page cache.
`heco_common/frameref.py` is the whole mechanism, with 15 tests.

Two levels, both opt-in:

| | env | what it removes |
|---|---|---|
| ref alongside JPEG | `HECO_FRAMES_DIR` on ingest + the three stages | the three JPEG **decodes** |
| ref only | `+ HECO_FRAMES_REF_ONLY=1` on the runner | also the **encode** and the base64/JSON |

`imageB64` still travels unless a caller explicitly declines it, and every
consumer falls back to it whenever a ref does not resolve. Drop the overlay
and the pipeline is byte-for-byte what it was.

## Why files rather than a ring of slots

Each frame is its own file named for its sequence number; old ones are
unlinked once N newer exist. A reader that opened a frame keeps its fd and
POSIX keeps the inode alive until it closes — so retiring a frame mid-read
cannot corrupt it. A ring of reused slots would need leasing, a generation
counter and a fence to buy the same guarantee. There is a test for exactly
this, because "torn reads are impossible" should be checked, not intended.

## Measured

| configuration | fps | note |
|---|---|---|
| main branch | **3.90** | baseline, all stages on CUDA |
| shm, JPEG still sent | **5.14** | **1.32x**, all stages on CUDA — trustworthy |
| shm, ref-only | 2.58 | **NOT COMPARABLE — see below** |

Per-stage, on a real 4K frame: persons 81.1 → 63.9 ms with the ref, which is
exactly the 17.8 ms JPEG decode it removes. `read_frame` of a 24 MB frame
costs 13.4 ms against ~20 ms to decode the JPEG of the same frame.

## Why the ref-only number is not trustworthy

The shm stack's three model stages were running **`CUDA -> CPUExecutionProvider`**
— they had silently fallen back to CPU, which is why they were burning
1000%+ CPU each. So 2.58 fps is a CPU pipeline measured against a GPU one,
not a transport comparison. It says nothing about ref-only either way.

The demo stacks were checked at the same moment and were fine on CUDA; the
fallback was confined to this experiment. **Re-measure ref-only with
`demo-up.sh status` showing CUDA on every shm stage before believing any
number from it.**

Likely cause to investigate first: three stacks (camera A, camera B, shm)
each building CUDA sessions on one 8 GB card.

## Four real bugs this found

Each was invisible to a passing suite and obvious the moment frames moved.

1. **`imageB64: Field(min_length=1)`** refused every ref-only request with a
   422 before anything could look at the ref. The negotiation caught this
   itself and kept the JPEGs — the probe working as intended.
2. **`not body.get("imageB64")` meant end-of-source.** Right when base64 was
   the only carrier; wrong the moment a frame can arrive as a ref. Every
   ref-only run ended at frame zero, state `ended`, error `None` — a complete
   count of nobody, reported as success.
3. **The sweep ran only after a successful write.** A full tmpfs failed the
   write, so the sweep never ran and nothing was retired again: the mount sat
   at 98% holding 21 frames against a keep of 8, paying for a failed 24 MB
   write per frame before falling back to JPEG anyway.
4. **ingest wrote the frame on every poll.** The runner polls at 50 Hz
   waiting for the seq to advance; the same frame was written to tmpfs fifty
   times a second. The JPEG encode had been hiding it — at 13.9 ms it
   throttled the polling, so removing the encode let the waste run free.

Also: `self.log` is a `RunLog` taking one message, but the tests injected
`logging.getLogger`, which accepts `%`-args — so the suite covered code that
raised `TypeError` in production. A double more permissive than the real
thing tests nothing at the boundary it stands in for.

## What it costs

The stages must share a host to benefit. `HECO_PERSONS_URL` exists so a stage
could live on another machine; it still can, it just falls back to JPEG. That
is a one-way door for the deployment shape and should be a deliberate choice.

## To finish this

1. Get the shm stack onto CUDA and re-measure ref-only.
2. Soak it: a full clip, both transports, comparing counts not just fps.
3. Decide the deployment shape question above.
4. Then consider merging — the JPEG-alongside mode (1.32x) is the part that
   is already measured honestly and could land on its own.

## The throughput levers on refs (merge of main, 2026-09-25)

Main brought the levers of 2026-09-24 (docs/LEVERS-2026-09-25.md). Each one
now moves the shared frame exactly as it moves the JPEG:

- **Ingest (L1 gate, live buffer; L6 hardware decode).** With a lever armed
  a frame is written to tmpfs when it is TAKEN, not when it is decoded, so a
  live buffer's backlog stays in ingest's own memory (`INGEST_BUFFER_MB`)
  and a frame the gate skipped or the buffer dropped never touches tmpfs.
  Each frame is written once however often it is polled, keyed by (worker
  generation, seq) — seq restarts at 1 on every /open, and the first cut's
  seq-only key could hand a new run the previous run's frame. `jpeg=0` is
  honoured in lever mode too.
- **Refs are numbered by WRITE, not capture seq.** frameref retires frames
  more than `HECO_FRAMES_KEEP` numbers behind the newest. Numbered by
  capture seq, a still room under the gate (one keepalive per second, 15
  seqs apart at 15 fps) — or any camera outrunning a slow consumer by more
  than the keep — retired the frame the runner was still embedding: a JPEG
  decode with the JPEG alongside, a 400 under ref-only. Numbered by write,
  the keep means "the last N frames handed out" — never fewer than 4:
  overlap + prefetch hold three served frames at once (decided, detecting,
  prefetched), and under ref-only a KEEP of 2 failed the run with "cannot
  read" from embed where serial survived. tmpfs is bounded at KEEP
  frames (+1 in flight), and a run's frames are cleared when its source is
  closed or replaced (`frameref.clear`), instead of lingering until another
  run's writes happened to retire them.
- **Runner (L3 overlap, parallel detect, L4 cadence, L7 region).** Every
  persons and faces POST is built in two helpers (`_post_persons`,
  `_post_faces`) that carry the ref beside the JPEG — the loop, the detect
  worker, the parallel pair, the crops, the whole frame and the operator's
  region alike; embed and enrolment use the same `_pixels(frame)`. A frame
  with no ref sends exactly main's body (the pinned call sequence passes).
- **The ref-only probe no longer loses frames.** It asks ONCE a frame
  exists (the first cut polled twenty times when ingest offered no ref,
  dequeuing twenty frames from a live buffer, or moving a lockstep reader
  twenty frames on, that the run never counted), and its frame is handed to
  the loop as the run's first frame.
- **The runner reads refs too.** Its torso, head and beard descriptors (the
  appearance veto, the heal veto, the review queue's clothing/turban/beard
  evidence), the white balance behind them and the face cards are cut from
  the decoded frame, and a ref-only frame had no JPEG to decode: every one
  of them went unmeasured, silently. docker-compose.shm.yml now mounts the
  frames read-only on the runner, the runner reads the ref when no JPEG
  came, and the ref-only probe requires the runner to read it as well or
  keeps the JPEGs.

Still true under ref-only, and not addressed here: the console's live taps
and a forensic run's frame uploads are cut from the JPEG, so they carry no
picture. Run forensic benches with the JPEG alongside.
