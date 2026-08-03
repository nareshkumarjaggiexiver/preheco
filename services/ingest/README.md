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
| POST | `/open` | `{url \| path, loop}` — exactly one source; replaces the current one |
| GET | `/frame` | `{tMs, imageB64, w, h, seq}` — 409 nothing open, 503 not ready yet |
| GET | `/health` | `{ok, model, version}` |

`tMs` is milliseconds since the source was opened (monotonic clock).

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
