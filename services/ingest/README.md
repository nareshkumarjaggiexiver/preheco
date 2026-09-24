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
| GET | `/health` | `{ok, model, version, owner, knobs, capture, device, cvThreadsActive}` — `ok` false when a knob is malformed |

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
default loop avoids for frames nobody reads: 55.8 against 33.5 ms of CPU per
captured 4K frame on the .94 box (table below). The NVDEC decoder hands the
gate its Y plane instead, at 30.4.

**Calibrate the gate per placement before arming it.** On the Sharon wedding
entrance the hall behind the door never stops moving: replaying the whole
reference run (D02, 15,076 frames, 78 identities, decision ledger
d02-20260923221223) through the gate gave

| `INGEST_MOTION_MIN_FRAC` | frames skipped | kept-face frames skipped | identities with every verdict frame skipped |
| --- | --- | --- | --- |
| 0.002 (default) | 0.0% (1 frame) | 0.0% | 0 |
| 0.01 | 6.6% | 1.5% | 0 |
| 0.02 | 20.3% | 6.6% | 0 (worst guest keeps 74% of his frames) |
| 0.05 | 43.1% | 20.8% | **1** |

so on that placement the safe gate saves nothing, and a gate that saves
real work loses a guest. A door onto a still corridor is a different story.
`scripts/gate_replay.py` replays any recording (and, given a run's ledger,
prices each threshold in face frames and identities) in one decode pass:

```sh
python scripts/gate_replay.py /clips/D02.mp4 --ledger ledger.jsonl \
    --min-frac 0.002 0.01 0.02 0.05 --hw cuda
```

## L6: hardware decode (`INGEST_DECODER`)

| env | default | meaning |
| --- | --- | --- |
| `INGEST_DECODER` | `cpu` | `cpu` = today's cv2.VideoCapture. `nvdec` / `vaapi` = an ffmpeg subprocess decoding on NVDEC / VA-API. `ffmpeg` = the same subprocess decoding in software. |
| `INGEST_CV_THREADS` | unset | OpenCV's pool size, process-wide. `1` with any lever on a many-core box (below). |

The subprocess writes NV12 (the decoder's own layout, 1.5 bytes a pixel) over
a Unix socketpair. The gate reads the Y plane, a frame waits in the store as
NV12 (so the FIFO holds twice the seconds under the same cap), and cv2
converts it to BGR only when a consumer takes it. Hardware frames stay on the
GPU and come down through `hwdownload`, which refuses a software frame, so
ffmpeg's own quiet CPU fallback fails instead of hiding. Timestamp sync is
passthrough (auto would duplicate or drop frames to fit jittery RTSP
timestamps). stderr is drained for the process's life; stop kills a blocked
read; every exit is reaped.

**A decoder that cannot start falls back to `cpu`, loudly**: a
`[heco-device] ingest requested=nvdec active=['cpu']` line and the reason on
stderr, and GET /health `device: {requested, active, error}`. Check
`device.active` after the first /open of every deploy. A decoder that dies
mid-file stops the source instead of reporting `ended`, so an incomplete count
fails as a stall with its gallery kept.

Deploy (WSL2 box: the overlay carries the /usr/lib/wsl mount, the `video`
driver capability, /dev/dxg and /dev/dri):

```sh
docker compose -f docker-compose.yml build ingest
docker build -f docker/ingest-hwdec.Dockerfile --build-arg BASE=heco-ingest \
  -t heco-ingest:hwdec .
INGEST_DECODER=nvdec INGEST_CV_THREADS=1 docker compose -f docker-compose.yml \
  -f docker-compose.hwdec.yml up -d --force-recreate ingest
```

### Measured on the .94 box (2026-09-24)

`scripts/bench_ingest.py`, one container at a time: the 4K HEVC bench clip
(`bench-sharon-120s.mp4`, the busiest 121 s of the Sharon wedding) paced at
15 fps, unbuffered unless noted, and a runner-like consumer (polls every 20 ms,
holds each fresh frame 200 ms, today's chain). 60 s a row. CPU is the
container's cgroup, so it includes the decoder subprocess.

| decoder | gate | buffer | OpenCV threads | CPU ms / captured frame | cores | captured fps | GET /frame p50 / p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cpu (today's loop) | off | 0 | default | 34.2 | 0.49 | 14.4 | 19.5 / 24.0 |
| cpu | on | 0 | default | 56.4 | 0.85 | 15.0 | 20.4 / 24.2 |
| nvdec | off | 0 | default | 35.8 | 0.53 | 14.9 | 22.2 / 27.8 |
| nvdec | on | 0 | default | 45.4 | 0.68 | 14.9 | 23.4 / 29.4 |
| cpu (today's loop) | off | 0 | 1 | 33.5 | 0.49 | 14.5 | 19.8 / 24.3 |
| cpu | on | 0 | 1 | 55.8 | 0.84 | 15.0 | 20.3 / 25.0 |
| **nvdec** | off | 0 | 1 | **25.5** | 0.38 | 14.8 | 23.3 / 29.6 |
| **nvdec** | on | 0 | 1 | **30.4** | 0.45 | 15.0 | 23.4 / 29.6 |
| nvdec | off | 10 s | 1 | 28.1 | 0.42 | 15.0 | 22.9 / 29.7 |
| cpu | off | 10 s | 1 | 51.5 | 0.77 | 15.0 | 19.4 / 25.5 |
| nvdec, two cameras at once | on | 0 | 1 | 29.0 + 26.5 | 0.83 both | 15.0 each | 23 / 29 |

* The gate published every frame of this clip (skipped 0), as the replay
  above predicts; ~630 of ~900 frames were dropped by the 200 ms consumer
  in every unbuffered row. Today's loop captures 14.4-14.5 fps because it
  sleeps a whole frame period AFTER each decode; lever mode keeps a deadline.
* A repeat poll of an already-served frame costs 3-4.5 ms, not a fresh 4K
  encode (the encode-once cache).
* The 10 s buffer held 150 NV12 frames (1.87 GB) on nvdec, and hit the
  2048 MiB cap at 86 BGR frames (5.7 s) on cpu; a 200 ms consumer cannot keep
  up with 15 fps, so both then dropped oldest-first (483 / 545). The buffer
  carries bursts; it cannot carry a chain that is slower on average.
* NVDEC decodes bit-exactly; BGR against cv2's own decode of the same frames
  differs by 1.06 grey levels on average (p99 2, max 20), all of it cv2's
  NV12 conversion against swscale's.
* Lockstep with a consumer that takes every frame at once: nvdec + gate
  14.9 fps footage at 85.4 ms/frame (OpenCV default threads), today's loop
  14.1 fps at 63.3 ms/frame. Every frame is converted and encoded there.

**Why a socketpair, not a pipe.** The first NVDEC run cost MORE than the cpu
path (56 ms against 34). Decode was 1.3 ms a frame and `hwdownload` 2.9 ms;
the pipe was the rest. Linux charges pipe buffers to the uid that creates
them, container root is host uid 0, and the box's other root processes had
spent uid 0's soft budget (`fs.pipe-user-pages-soft` = 16384 pages), so every
new root pipe got 8 KB (F_GETPIPE_SZ 8192; F_SETPIPE_SZ refused with EPERM)
and a 12 MB frame crossed in ~1,500 round trips. For 300 frames, reader +
ffmpeg CPU per frame: root pipe 43.6 ms at 31 fps max; a dedicated uid with a
1 MiB pipe 23.4 ms; a socketpair 21.6 ms at 217 fps max, as any uid.

**Why `INGEST_CV_THREADS=1`.** OpenCV's default pool is a thread per core. On
the 28-thread box a 4K resize or NV12 conversion spends more waking and
spinning 28 workers than it saves: nvdec + gate went from 45.4 to 30.4 ms of
CPU per frame at the same latency.

### VA-API (the UHD 770) — blocked by the WSL kernel, not by Mesa

Time-boxed tonight (about 7 of 20 minutes); the path stays behind the knob
with the loud fallback. With /dev/dxg, /dev/dri (vgem), /usr/lib/wsl and
`LIBVA_DRIVER_NAME=d3d12` in the container, `vaInitialize` fails ("resource
allocation failed") for every adapter (Intel, NVIDIA, default) on Debian's
Mesa 25.0.7, Ubuntu's 25.2.8 and kisak-mesa 26.2.3, with or without
`MESA_LOADER_DRIVER_OVERRIDE=d3d12`. strace shows why: the VA driver never
opens libd3d12, libdxcore or /dev/dxg at all. It fails in the DRM path first:
it opens `/dev/udmabuf` (ENOENT), falls back to a memfd, queries a DRM cap on
the vgem node and gives up. This kernel (6.18.33.2-microsoft-standard-WSL2)
is built with `# CONFIG_UDMABUF is not set`. WSLg's X11 display fails too
(no DRI3/DRI2). The host side looks ready (the Intel driver package ships
`libigd12dxva64.so`), so the next step is a WSL kernel with `CONFIG_UDMABUF=y`
(then `--device /dev/udmabuf` in the overlay), which needs a WSL restart and
was not attempted with three stacks running.
