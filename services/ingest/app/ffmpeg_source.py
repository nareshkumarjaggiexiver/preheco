"""L6 — decode in an ffmpeg subprocess (NVDEC, VA-API or software) to NV12.

Measured on the CUDA box (300 frames of the 4K HEVC bench clip, CPU time per
frame): CPU decode + BGR 80 ms; CPU decode alone 27 ms; NVDEC decode alone
3.4 ms; NVDEC + ffmpeg's swscale to BGR 42 ms. The BGR CONVERSION, not the
decode, is the dominant CPU cost. So the pipe carries NV12 (the decoder's own
layout, 1.5 bytes a pixel), the motion gate reads its Y plane for free, and the
BGR conversion (cv2, ~7 ms single-threaded at 4K) is paid only for a frame a
consumer actually takes (app.frames.RawFrame).

The frames cross on a Unix SOCKETPAIR, not a pipe. Measured on the .94 box,
300 4K NV12 frames from NVDEC: through a pipe, 31 fps at most and 43.6 ms of
CPU per frame (22.1 in the reader, 21.5 in ffmpeg); through a socketpair,
217 fps and 21.6 ms. The pipe was not slow by nature: Linux charges pipe
buffers to the uid that creates them, container root is host uid 0, and the
box's other root processes had used up uid 0's soft budget
(fs.pipe-user-pages-soft), so every new root pipe got 8 KB and a 12 MB frame
crossed in ~1,500 round trips. F_SETPIPE_SZ is refused (EPERM) in that state.
Socket buffers are not in that accounting at all.

Three rules keep the subprocess honest:

* **No silent software decode.** NVDEC and VA-API frames stay on the GPU
  (``-hwaccel_output_format``) and are downloaded by ``hwdownload``, which
  refuses a software frame. If the hardware decoder cannot start, ffmpeg's own
  quiet fallback to the CPU therefore FAILS instead, and the worker falls back
  to the cv2 path loudly and says so on /health.
* **Every frame, as decoded.** Timestamp sync is passthrough: left on auto,
  rawvideo output is constant-frame-rate, and ffmpeg would duplicate or drop
  frames to fit jittery RTSP timestamps.
* **No zombies, no blocked pipe.** stderr is drained by a thread for the whole
  life of the process; ``close()`` kills and reaps; EOF reaps.
"""

from __future__ import annotations

import contextlib
import re
import socket
import subprocess
import threading
import time
from collections import deque

import numpy as np
from heco_common.logs import safe

from .frames import RawFrame

#: The ffmpeg command prefix. A list so tests can run a fake decoder with
#: ``[sys.executable, "fake_ffmpeg.py"]``.
FFMPEG: list[str] = ["ffmpeg"]
#: VA-API render node (WSL: vgem provides it; the d3d12 driver does the work).
VAAPI_DEVICE = "/dev/dri/renderD128"
#: How long /open waits for the first decoded frame before the decoder is
#: declared dead. A 4K RTSP camera needs a connection, a keyframe (up to one
#: GOP, 1 s here) and a CUDA context — seconds, not tens of them.
OPEN_TIMEOUT_S = 15.0
#: Socket buffer asked for on each end, so a 12 MB frame crosses in a few
#: wake-ups. The kernel caps it at net.core.[rw]mem_max, which only costs speed.
_SOCK_BUF = 4 << 20
#: The output stream line ffmpeg prints once the first frame reaches the muxer:
#: "Stream #0:0: Video: rawvideo (NV12 / 0x3231564E), nv12(tv, ...), 3840x2160, ..."
_GEOM = re.compile(r"Video: rawvideo\b.*?[ ,](\d{2,5})x(\d{2,5})[ ,\[]")


class DecoderError(RuntimeError):
    """The subprocess decoder could not start (no first frame)."""


_VERSION: tuple[int, int] | None = None


def ffmpeg_version() -> tuple[int, int]:
    """(major, minor) of the ffmpeg binary, probed once; (99, 0) if unparseable.

    Two options changed meaning across versions and both matter: timestamp
    passthrough is ``-fps_mode`` from 5.1 (``-vsync`` before), and the RTSP
    socket timeout is ``-timeout`` from 5.0 — where in 4.x ``-timeout`` put
    RTSP into LISTEN mode and the socket timeout was ``-stimeout``. A git
    build prints no number and is assumed modern.
    """
    global _VERSION
    if _VERSION is None:
        try:
            out = subprocess.run(
                [*FFMPEG, "-hide_banner", "-version"],
                capture_output=True, timeout=10, check=False,
            ).stdout.decode("utf-8", "replace")
        except (OSError, subprocess.SubprocessError):
            out = ""
        m = re.search(r"version n?(\d+)\.(\d+)", out)
        _VERSION = (int(m[1]), int(m[2])) if m else (99, 0)
    return _VERSION


def build_command(
    source: str, decoder: str, *, loop: bool = False, rtsp_tcp: bool = True
) -> list[str]:
    """The ffmpeg argv that decodes ``source`` to NV12 frames on stdout."""
    version = ffmpeg_version()
    cmd = [*FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info"]
    if decoder == "nvdec":
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif decoder == "vaapi":
        cmd += [
            "-hwaccel", "vaapi", "-hwaccel_device", VAAPI_DEVICE,
            "-hwaccel_output_format", "vaapi",
        ]
    scheme = source.split("://", 1)[0].lower() if "://" in source else ""
    if scheme in ("rtsp", "rtsps"):
        if rtsp_tcp:
            cmd += ["-rtsp_transport", "tcp"]
        # 10 s without a byte is a dead camera: exit, and the worker reconnects.
        cmd += ["-timeout" if version >= (5, 0) else "-stimeout", "10000000"]
    elif scheme in ("http", "https"):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    if loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", source, "-map", "0:v:0", "-an", "-sn", "-dn"]
    cmd += ["-fps_mode" if version >= (5, 1) else "-vsync", "passthrough"]
    if decoder in ("nvdec", "vaapi"):
        cmd += ["-vf", "hwdownload,format=nv12"]
    cmd += ["-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1"]
    return cmd


class FfmpegSource:
    """One ffmpeg decode subprocess and the NV12 frames it writes.

    The constructor blocks until the FIRST frame has been decoded (or raises
    DecoderError), so /open can refuse, or fall back, immediately — the same
    promise the cv2 path's synchronous open makes.
    """

    def __init__(
        self,
        source: str,
        decoder: str,
        *,
        loop: bool = False,
        rtsp_tcp: bool = True,
        open_timeout_s: float | None = None,
        abort: threading.Event | None = None,
    ) -> None:
        """Start the decoder and wait for its first frame.

        ``abort`` (the worker's stop event) ends that wait early, so stopping
        a worker that is mid-reconnect does not sit out the whole timeout.
        """
        self.decoder = decoder
        self.cmd = build_command(source, decoder, loop=loop, rtsp_tcp=rtsp_tcp)
        self.w: int | None = None
        self.h: int | None = None
        self.frame_bytes = 0
        self.returncode: int | None = None
        self._tail: deque[str] = deque(maxlen=25)
        self._geom_evt = threading.Event()
        self._timed_out = False
        self._sock, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with contextlib.suppress(OSError):
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCK_BUF)
            child.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCK_BUF)
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.DEVNULL,
                stdout=child.fileno(),
                stderr=subprocess.PIPE,
                bufsize=0,
                close_fds=True,
            )
        except OSError as exc:
            self._sock.close()
            raise DecoderError(f"could not start {FFMPEG[0]}: {exc}") from exc
        finally:
            # The child holds its own copy now. Ours must go, or EOF never
            # arrives when ffmpeg exits.
            child.close()
        self._stderr = threading.Thread(
            target=self._drain_stderr, name="ingest-ffmpeg-stderr", daemon=True
        )
        self._stderr.start()
        timeout = OPEN_TIMEOUT_S if open_timeout_s is None else open_timeout_s
        # The watchdog kills a decoder that has not delivered its first frame
        # in time (or whose worker is stopping) — the read below then sees EOF
        # instead of blocking forever.
        opened = threading.Event()
        watchdog = threading.Thread(
            target=self._watch, args=(timeout, abort, opened),
            name="ingest-ffmpeg-open", daemon=True,
        )
        watchdog.start()
        try:
            if not self._geom_evt.wait(timeout):
                self._timed_out = True  # the watchdog may not have fired yet
            if self.w is None or self.h is None:
                raise DecoderError(self._why("no output stream"))
            if self.w % 2 or self.h % 2:
                raise DecoderError(f"odd frame size {self.w}x{self.h} cannot be NV12")
            self.frame_bytes = self.w * self.h * 3 // 2
            first = self._read_frame()
            if first is None:
                raise DecoderError(self._why("no first frame"))
        except DecoderError:
            self.close()
            raise
        finally:
            opened.set()
        self._first: RawFrame | None = first

    # ------------------------------------------------------------- public

    @property
    def pid(self) -> int:
        """The decoder's process id (tests check it was reaped)."""
        return self._proc.pid

    @property
    def clean_eof(self) -> bool:
        """True when the stream ended because the input did, not an error."""
        return self.returncode == 0

    def read(self) -> RawFrame | None:
        """The next frame, or None at EOF / when the process died."""
        if self._first is not None:
            first, self._first = self._first, None
            return first
        return self._read_frame()

    def interrupt(self) -> None:
        """Kill the decoder from ANOTHER thread; a blocked read() returns None."""
        self._kill()
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RD)  # wakes the reader even mid-recv

    def close(self) -> None:
        """Kill (if still running), reap, and close the transport. Idempotent."""
        self._kill()
        with contextlib.suppress(subprocess.TimeoutExpired):  # SIGKILL: belt and braces
            self.returncode = self._proc.wait(timeout=5)
        with contextlib.suppress(OSError):
            self._sock.close()
        with contextlib.suppress(OSError):
            self._proc.stderr.close()
        self._stderr.join(timeout=2)

    def tail(self) -> str:
        """The last lines ffmpeg wrote to stderr, credentials scrubbed."""
        return " | ".join(self._tail)

    # ---------------------------------------------------------- internals

    def _watch(
        self, timeout: float, abort: threading.Event | None, opened: threading.Event
    ) -> None:
        deadline = time.monotonic() + timeout
        while not opened.wait(0.05):
            if abort is not None and abort.is_set():
                self._kill()
                return
            if time.monotonic() >= deadline:
                self._timed_out = True
                self._kill()
                return

    def _kill(self) -> None:
        if self._proc.poll() is None:
            with contextlib.suppress(OSError):
                self._proc.kill()

    def _read_frame(self) -> RawFrame | None:
        buf = np.empty(self.frame_bytes, dtype=np.uint8)
        view = memoryview(buf)
        got = 0
        while got < self.frame_bytes:
            try:
                n = self._sock.recv_into(view[got:])
            except (OSError, ValueError):  # closed under us by close()
                n = 0
            if not n:
                self._reap()
                return None
            got += n
        return RawFrame("nv12", buf, self.w, self.h)

    def _reap(self) -> None:
        """The socket hit EOF: ffmpeg is exiting — collect its exit code."""
        try:
            self.returncode = self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill()
            self.returncode = self._proc.wait()

    def _drain_stderr(self) -> None:
        """Keep stderr flowing (a full pipe would stall ffmpeg) and learn the size."""
        in_output = False
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                line = safe(raw.decode("utf-8", "replace").rstrip())
                if line:
                    self._tail.append(line)
                if self.w is None:
                    if line.startswith("Output #"):
                        in_output = True
                    elif in_output and (m := _GEOM.search(line)):
                        self.w, self.h = int(m[1]), int(m[2])
                        self._geom_evt.set()
        except (OSError, ValueError):
            pass
        finally:
            self._geom_evt.set()  # EOF: wake the constructor, size or no size

    def _why(self, what: str) -> str:
        rc = self._proc.poll()
        if self._timed_out:
            state = "timed out"
        else:
            state = f"exit {rc}" if rc is not None else "still running"
        return f"{self.decoder}: {what} ({state}): {self.tail() or 'no output'}"
