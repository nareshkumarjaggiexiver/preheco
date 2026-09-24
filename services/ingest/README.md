# ingest (port 7101)

**What.** Turns an RTSP stream or a video file into *latest-frame-on-demand*.
A background capture thread reads the source continuously into a single-slot
buffer; `GET /frame` serves whatever is newest as base64 JPEG.

**Policy: drop, not queue.** Slow consumers skip frames instead of building a
stale backlog — for a live counting pipeline, a frame you could not process
in time is worthless, and an unbounded queue only converts lag into memory
exhaustion. Consumers detect an ended/stalled source by `seq` ceasing to
advance (a file with `loop=false` keeps serving its final frame).

There is no DNN here: `/health` reports `model: "opencv-videocapture"`.

## API

| method | path | body / response |
| ------ | ---- | --------------- |
| POST | `/open` | `{url \| path, loop, isFile?, lockstep?, owner?, takeover?, motionGate?, bufferS?}` — exactly one source; CLAIMS the capture slot |
| POST | `/close` | `{owner?, force?}` — release the slot; owner-checked, idempotent |
| GET | `/frame` | `{tMs, imageB64, w, h, seq, ended}` — 409 nothing open, 503 not ready yet; with a lever armed also `{motion, backlog, skipped, dropped, captured}` |
| GET | `/health` | `{ok, model, version, owner, knobs, capture}` |

`tMs` is milliseconds since the source was opened (monotonic clock).

**One slot, one owner.** This service holds exactly ONE capture worker, so an
`/open` carrying an `owner` (the runner sends its run id) claims it and any
later `/open` by a different owner is refused **409**, naming the holder.
Without that, starting a staff enrolment while a count run was live at the gate
silently replaced the count run's camera and it began counting the enrolment
walk-through — no error anywhere. Escapes: the same owner may re-open
(idempotent restart), a slot whose capture thread has died is claimable, and
`takeover: true` is an explicit seizure. An `/open` with **no** owner keeps the
old replace-anything behaviour, for ad-hoc probes.

## Run

```sh
make venv
make run          # uvicorn on 7101 (PORT=... to override)
# open a file, looped:
curl -s localhost:7101/open -X POST -H 'content-type: application/json' \
     -d '{"path": "/data/clip.mp4", "loop": true}'
curl -s localhost:7101/frame | head -c 200
```

## Test

```sh
make test         # synthetic MJPG clip + in-process client; no network
make lint
```

## Tune

| env | default | meaning |
| --- | ------- | ------- |
| `INGEST_RTSP_TCP` | `1` | Force RTSP-over-TCP (venue WiFi shreds UDP RTP). Set `0` for UDP. |
| `INGEST_FILE_PACE` | `1.0` | File playback speed multiplier: `1.0` = native FPS (a clip behaves like a live camera), `2.0` = double speed, `0` = unpaced (as fast as disk). |
| `INGEST_JPEG_QUALITY` | `85` | JPEG quality for `/frame` responses. |

Read failures on a live stream trigger release → 1 s pause → reopen, forever,
until a new `/open` or shutdown. Opening a source that cannot be opened at
all fails the `/open` call itself with 400.


## Tuning for a 4K camera

Measured on the PowerEdge T440 against the UNV at 3840x2160/20fps: the whole
pipeline ran at **1.59 fps while the box sat 90% idle**, and ingest alone burned
**3.2 cores**. It was decoding every frame the camera sent and converting it to
a 25 MB BGR array, while the consumer took roughly one frame in twelve.

Two knobs, both off by default:

| env | default | what it does |
| --- | --- | --- |
| `INGEST_MAX_WIDTH` | `0` (off) | Caps the longest edge before the frame enters the pipeline. |
| — | — | Decode-on-demand is automatic: while the slot holds an unread frame the loop `grab()`s instead of `read()`ing, skipping the BGR conversion for frames the drop-not-queue slot would discard anyway. |

**`INGEST_MAX_WIDTH` scales face pixels, so choose it against the floor, not by
taste.** On the POC geometry a face measures ~176 px at 4K:

| setting | face px | verdict |
| --- | --- | --- |
| unset (3840) | ~176 | today |
| **1920** | **~88** | **above the 80 px canon — recommended** |
| 1280 | ~59 | above the 56 px floor, below canon |
| the camera's own sub-stream (704x576) | ~47 | **below the floor — unusable** |

Measured saving at 1920, per frame: JPEG encode 21.1 -> 5.5 ms, payload 584 ->
169 KB, and the decode paid at each of the three downstream hops 127.6 -> 27.9
ms. About **112 ms a frame**, against a 629 ms measured budget.

> [!NOTE]
> This is a downscale of the MAIN stream, not a stream swap, because this
> camera's sub-streams are D1 (704x576) and CIF (352x288) — both below the face
> floor. The served frame reports its true `w`/`h`, so the quality gate and the
> taps measure what was actually analysed.


## Levers (L1): the motion gate and the bounded live buffer

Both are OFF by default, and OFF runs the capture loop above untouched: the
same grab-not-retrieve, the same pacing, and a `/frame` body of exactly the
six fields it has always had (the tests pin all three). Arm them in the env,
or per run with `/open`'s `motionGate` / `bufferS` (null = the env's value):

| env | default | meaning |
| --- | --- | --- |
| `INGEST_MOTION_GATE` | `0` | Publish a frame only when something in it moved, or when the keepalive is due. |
| `INGEST_MOTION_MIN_FRAC` | `0.002` | Fraction of the ~1/8-scale luma pixels that must change to count as motion (~260 px of a 480x270 image). |
| `INGEST_MOTION_PIXEL_THR` | `0.08` | How far one small pixel must move, in units of the frame's own luma std. |
| `INGEST_MOTION_KEEPALIVE_S` | `1.0` | The longest a still scene goes unpublished. Footage seconds for a file, wall seconds for a camera. |
| `INGEST_BUFFER_S` | `0` | `0` = the newest-frame slot. `> 0` = a FIFO of up to that many seconds of published frames. |
| `INGEST_BUFFER_MB` | `2048` | Hard cap on the FIFO's pixels (MiB), oldest dropped first. |

**The gate** (app/motion.py) compares each frame's small luma with the last
PUBLISHED frame's, after normalising each by its own mean and std. A global
exposure step or a DJ flash is therefore not motion, while anything local (a
guest, a sweeping spotlight) is. Comparing against the last *published* frame
rather than the previous one means a slow walker builds up change until he is
published, where a frame-to-frame gate could skip him until the keepalive.
Every doubt resolves toward processing: the first frame, and the first after
a reconnect, are always published.

**The buffer** holds published frames instead of letting the newest overwrite
them, for live sources and paced files (a paced file is the bench's stand-in
for a camera). Lockstep never queues: the reader waits for each published
frame to be taken, so nothing is ever dropped. With a lever armed:

* `GET /frame` returns and DEQUEUES the oldest unread frame. With nothing
  unread it repeats the last one (same `seq`), so a poller waits as before.
  **Only the run's own loop may poll it**: a second poller takes frames the
  run then never sees.
* `seq` stays the capture number, so a gap between two served frames is
  exactly the frames skipped or dropped between them.
* `ended` is true only once the file is exhausted AND nothing published is
  left unread, and it never rides on a fresh frame (the runner reads `ended`
  as "no frame").
* Counters on every frame, cumulative and never decreasing: `captured`
  (decoded), `skipped` (not published, no motion; null with the gate off),
  `dropped` (published and discarded unread), and `backlog` (queued behind this
  frame). `motion` is this frame's changed fraction (null when not measured).
  Every decoded frame is accounted for:
  `captured = skipped + published = skipped + served + dropped + pending`.
  GET /health `capture.counters` shows the same numbers plus `backlogMax` and
  `bufferBytes`.

**The cost on the cv2 decoder.** The gate has to look at a frame to skip it,
so in lever mode every frame is retrieved, which is the BGR conversion the
default loop avoids for frames nobody reads. On the laptop that is 37 ms (grab
only) against 72 ms (read) per 4K frame, single loop.
