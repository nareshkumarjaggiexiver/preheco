"""A stand-in ffmpeg for the L6 tests: NV12 frames on stdout, no codec, no GPU.

Driven by the environment (the worker's subprocess inherits it):

* ``FAKE_FFMPEG_MODE`` — ``ok`` (default): header on stderr, FAKE_FRAMES
  frames, exit 0 | ``fail``: a hardware decoder that cannot load, exit 1, no
  frames | ``hang``: says nothing, sleeps until killed | ``die``: FAKE_FRAMES
  frames then exit 1 (a decoder dying mid-file) | ``noisy``: like ok, with
  ~2 MB of warnings on stderr before and between frames | ``forever``: frames
  until killed, FAKE_PERIOD seconds apart (a live camera).
* ``FAKE_FRAMES`` (10), ``FAKE_W`` (64), ``FAKE_H`` (48), ``FAKE_PERIOD`` (0),
  ``FAKE_STATIC`` (0: a bright square moves 8 px a frame; 1: nothing moves).
* ``FAKE_ARGV_LOG`` — append this invocation's argv, one line, to that file.

``-version`` prints a version line, as the real binary does.
"""

import os
import sys
import time

if "-version" in sys.argv:
    print("ffmpeg version 7.1.1-fake Copyright (c) 2000-2025 the FFmpeg developers")
    sys.exit(0)

if os.environ.get("FAKE_ARGV_LOG"):
    with open(os.environ["FAKE_ARGV_LOG"], "a") as log:
        log.write(" ".join(sys.argv[1:]) + "\n")

mode = os.environ.get("FAKE_FFMPEG_MODE", "ok")
frames = int(os.environ.get("FAKE_FRAMES", "10"))
w, h = int(os.environ.get("FAKE_W", "64")), int(os.environ.get("FAKE_H", "48"))
period = float(os.environ.get("FAKE_PERIOD", "0"))
static = os.environ.get("FAKE_STATIC", "0") == "1"
err, out = sys.stderr, sys.stdout.buffer

if mode == "hang":
    time.sleep(3600)
    sys.exit(0)
if mode == "fail":
    err.write("[hevc @ 0x55] Cannot load libnvcuvid.so.1\n")
    err.write("[hevc @ 0x55] Failed setup for format cuda: hwaccel initialisation failed.\n")
    err.write("Impossible to convert between the formats supported by the filter\n")
    sys.exit(1)


def noise():
    """A burst of decoder warnings, as a flaky stream produces."""
    if mode == "noisy":
        err.write("[hevc @ 0x55] Could not find ref with POC 12\n" * 5000)
        err.flush()


noise()
err.write("Input #0, fake, from 'x':\n")
err.write(f"  Stream #0:0: Video: hevc (Main), yuv420p(tv), {w}x{h}, 10 fps\n")
err.write("Output #0, rawvideo, to 'pipe:1':\n")
err.write(
    f"  Stream #0:0: Video: rawvideo (NV12 / 0x3231564E), nv12(tv, progressive), "
    f"{w}x{h}, q=2-31, 368 kb/s, 10 fps, 10 tbn\n"
)
err.flush()


def frame(i):
    """One NV12 frame: dark Y with a bright 16x16 square, grey chroma."""
    y = bytearray(b"\x3c" * (w * h))
    if not static:
        x0 = (i * 8) % max(1, w - 16)
        for row in range(8, min(h, 24)):
            y[row * w + x0: row * w + x0 + 16] = b"\xc8" * 16
    return bytes(y) + b"\x80" * (w * h // 2)


i = 0
try:
    while mode == "forever" or i < frames:
        out.write(frame(i))
        out.flush()
        i += 1
        noise()
        if period:
            time.sleep(period)
except BrokenPipeError:
    sys.exit(0)
sys.exit(1 if mode == "die" else 0)
