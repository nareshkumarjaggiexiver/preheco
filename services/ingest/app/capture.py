"""Background video capture with a single-slot latest-frame buffer.

Policy: **drop, not queue.** The capture thread reads the source at its own
pace and overwrites one slot with the newest frame; GET /frame always serves
that slot. If a consumer is slower than the source, intermediate frames are
simply never seen — which is exactly right for a live counting pipeline
(stale frames are worthless, and an unbounded queue would trade latency for
memory until the process dies). Consumers detect a stalled/ended source by
``seq`` no longer advancing.

Sources:

* **file** — plays back paced to the file's native FPS (so a clip behaves
  like a live camera); ``loop=True`` restarts at EOF. Pacing can be scaled or
  disabled via INGEST_FILE_PACE (see README "tune").
* **rtsp url** — TCP transport by default (INGEST_RTSP_TCP), because UDP
  RTP loss under event-venue WiFi shreds H.264/H.265 frames. Read failures
  trigger release + reopen with a 1 s pause, forever, until stopped.
"""

import os
import threading
import time

import cv2
import numpy as np
from heco_common.config import env_bool, env_float


class CaptureError(RuntimeError):
    """Raised when a source cannot be opened at all."""


class CaptureWorker(threading.Thread):
    """Owns one cv2.VideoCapture and the single-slot latest-frame buffer.

    The capture is opened synchronously in ``__init__`` so POST /open can
    report an unopenable source immediately; the thread then only reads.
    Stop with ``stop()`` — sets an event, joins, and releases the capture.
    """

    def __init__(self, source: str, is_file: bool, loop: bool = False) -> None:
        """Open the source (raises CaptureError on failure) and prep the slot."""
        super().__init__(name="ingest-capture", daemon=True)
        self.source = source
        self.is_file = is_file
        self.loop = loop
        self.ended = False  # file fully played, loop=False
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[int, int, np.ndarray] | None = None  # (seq, tMs, frame)
        self._pace_s = self._pacing_interval() if is_file else 0.0
        self._cap = self._open()

    # ------------------------------------------------------------- public

    def latest(self) -> tuple[int, int, np.ndarray] | None:
        """Return (seq, tMs, frame) for the newest frame, or None before one.

        The returned array is never mutated afterwards (the reader allocates
        a fresh array per decoded frame), so no copy is taken here.
        """
        with self._lock:
            return self._latest

    def stop(self, join_timeout_s: float = 5.0) -> None:
        """Signal the thread, wait for it, and release the capture."""
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=join_timeout_s)

    # ------------------------------------------------------------ thready

    def run(self) -> None:
        """Read loop: newest frame wins the slot; files pace and loop."""
        t0 = time.monotonic()
        seq = 0
        try:
            while not self._stop_evt.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    if self.is_file and self.loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if self.is_file:
                        self.ended = True
                        return
                    # Live stream hiccup: release, breathe, reopen.
                    self._cap.release()
                    if self._stop_evt.wait(1.0):
                        return
                    try:
                        self._cap = self._open()
                    except CaptureError:
                        continue  # keep retrying until stopped
                    continue
                seq += 1
                t_ms = int((time.monotonic() - t0) * 1000)
                with self._lock:
                    self._latest = (seq, t_ms, frame)
                if self._pace_s and self._stop_evt.wait(self._pace_s):
                    return
        finally:
            self._cap.release()

    # ---------------------------------------------------------- internals

    def _open(self) -> cv2.VideoCapture:
        """Create the VideoCapture; RTSP gets TCP transport unless disabled."""
        if not self.is_file and env_bool("INGEST_RTSP_TCP", True):
            # FFmpeg backend reads this env at open time.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"could not open source: {self.source}")
        return cap

    def _pacing_interval(self) -> float:
        """Seconds to sleep between file frames; 0 disables pacing.

        INGEST_FILE_PACE scales playback speed (1.0 = real time, 2.0 = double
        speed, 0 = as fast as the disk allows). The file's own FPS (fallback
        25) sets the base rate — probed with a throwaway capture because the
        real one is opened after pacing is decided.
        """
        pace = env_float("INGEST_FILE_PACE", 1.0)
        if pace <= 0:
            return 0.0
        probe = cv2.VideoCapture(self.source)
        fps = probe.get(cv2.CAP_PROP_FPS) if probe.isOpened() else 0.0
        probe.release()
        if not fps or fps <= 0:
            fps = 25.0
        return 1.0 / (fps * pace)
