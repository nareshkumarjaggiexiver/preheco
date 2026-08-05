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
from heco_common.config import env_bool, env_float, env_int
from heco_common.logs import safe


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
        # True while the slot holds a frame nobody has taken yet. The loop
        # grabs (cheap) rather than retrieves (expensive) while it is set.
        self._unread = False
        # Longest edge to analyse at, 0 = the camera's own size. See _fit.
        self._max_width = env_int("INGEST_MAX_WIDTH", 0)
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
            self._unread = False
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
                # DECODE ONLY WHAT SOMEBODY WILL READ.
                #
                # `read()` is grab + retrieve, and retrieve is the expensive
                # half: it converts the decoded picture to a BGR numpy array —
                # ~25 MB at 4K. This loop runs at the CAMERA's rate (20 fps on
                # the UNV), while a consumer that is doing real work per frame
                # takes far fewer. Measured on the PowerEdge: ingest burned 3.2
                # cores while the pipeline consumed 1.59 fps, so ~90% of that
                # conversion was thrown away by the drop-not-queue slot.
                #
                # grab() still pulls and decodes the packet — H.264 needs that
                # to keep its reference frames — but skips the conversion. So
                # an unconsumed frame costs the decode and not the copy.
                if self._slot_unread():
                    ok = self._cap.grab()
                    frame = None
                else:
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
                if frame is None:
                    continue  # grabbed only: the slot still holds an unread frame
                t_ms = int((time.monotonic() - t0) * 1000)
                with self._lock:
                    self._latest = (seq, t_ms, self._fit(frame))
                    self._unread = True
                if self._pace_s and self._stop_evt.wait(self._pace_s):
                    return
        finally:
            self._cap.release()

    def _slot_unread(self) -> bool:
        """Is the slot still holding a frame nobody has taken?"""
        with self._lock:
            return self._unread and self._latest is not None

    def _fit(self, frame):
        """Optionally shrink the frame before it enters the pipeline.

        ``INGEST_MAX_WIDTH`` (0 = off, the default: nothing changes unless an
        operator opts in) caps the longest edge. Everything downstream then
        moves and decodes a smaller image — the frame travels to persons,
        faces and embed as base64 JPEG, so halving the width quarters the
        bytes and the decode at every one of those hops.

        THE TRADE, AND IT IS NOT SUBTLE: face pixels scale with the frame.
        On the POC geometry a face measures ~176 px at 3840x2160, so

            INGEST_MAX_WIDTH=1920  ->  ~88 px   (above the 80 px canon)
            INGEST_MAX_WIDTH=1280  ->  ~59 px   (above the 56 px floor only)
             704 px sub-stream     ->  ~47 px   (BELOW the floor — unusable)

        The camera's own sub-streams are D1 and CIF, which is why this exists
        as a downscale of the main stream rather than a stream choice. The
        served frame reports its true w/h, so the quality gate and the taps
        measure what was actually analysed, not what the camera sent.
        """
        if not self._max_width:
            return frame
        h, w = frame.shape[:2]
        if w <= self._max_width:
            return frame
        scale = self._max_width / float(w)
        return cv2.resize(
            frame, (self._max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------- internals

    def _open(self) -> cv2.VideoCapture:
        """Create the VideoCapture; RTSP gets TCP transport unless disabled."""
        if not self.is_file and env_bool("INGEST_RTSP_TCP", True):
            # FFmpeg backend reads this env at open time.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            # NEVER the raw source: it embeds rtsp://user:pass@ and this
            # message travels into the runner's StageError, the planner's
            # permanent run notes, the browser and the export.
            raise CaptureError(f"could not open source: {safe(self.source)}")
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
