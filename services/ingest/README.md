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
| POST | `/open` | `{url \| path, loop, owner?, takeover?}` — exactly one source; CLAIMS the capture slot |
| POST | `/close` | `{owner?, force?}` — release the slot; owner-checked, idempotent |
| GET | `/frame` | `{tMs, imageB64, w, h, seq}` — 409 nothing open, 503 not ready yet |
| GET | `/health` | `{ok, model, version, owner}` |

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
