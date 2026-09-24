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
