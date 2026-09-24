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

Levers (app.config): with every lever off the worker runs the loop below
exactly as it always has. Arm one — the motion gate or the live buffer — and
it runs ``_run_levered`` instead: every frame is retrieved (the gate needs its
pixels), gated (app.motion), published into a :class:`~app.frames.FrameStore`
or counted as skipped, and GET /frame takes frames out of that store with
``take()``. Every frame decoded is accounted for:
``captured == skipped + published`` and
``published == served + dropped + pending``.
"""

import itertools
import os
import threading
import time

import cv2
import numpy as np
from heco_common.config import env_bool, env_float, env_int
from heco_common.logs import safe

from .config import Levers
from .frames import FrameStore, Item, RawFrame, Served
from .motion import MotionGate, small_luma

#: INGEST_BUFFER_MB is in MiB.
_MB = 1024 * 1024


class CaptureError(RuntimeError):
    """Raised when a source cannot be opened at all."""


class CaptureWorker(threading.Thread):
    """Owns one cv2.VideoCapture and the single-slot latest-frame buffer.

    The capture is opened synchronously in ``__init__`` so POST /open can
    report an unopenable source immediately; the thread then only reads.
    Stop with ``stop()`` — sets an event, joins, and releases the capture.
    With a lever armed the slot is a :class:`~app.frames.FrameStore` and
    consumers call ``take()`` (``latest()`` delegates to it).
    """

    #: Distinguishes workers across /open calls: seq restarts at 1 for every
    #: new source, so (generation, seq) is what names one frame.
    _generations = itertools.count(1)

    def __init__(
        self,
        source: str,
        is_file: bool,
        loop: bool = False,
        lockstep: bool = False,
        levers: Levers | None = None,
    ) -> None:
        """Open the source (raises CaptureError on failure) and prep the slot.

        ``levers`` None means all off — today's worker, byte for byte.
        """
        super().__init__(name="ingest-capture", daemon=True)
        self.generation = next(CaptureWorker._generations)
        self.source = source
        self.is_file = is_file
        self.loop = loop
        # EVERY FRAME, for a recording — see OpenSource.lockstep for why this
        # is a file-only idea. Forced off for a live source rather than
        # trusted to the caller: blocking a camera cannot achieve anything
        # except making the reader fall behind the stream, and a mistaken
        # `lockstep: true` on an RTSP url must not be able to wedge a live
        # count.
        self.lockstep = bool(lockstep and is_file)
        self.ended = False  # file fully played, loop=False
        # True while the slot holds a frame nobody has taken yet. The loop
        # grabs (cheap) rather than retrieves (expensive) while it is set —
        # unless lockstep, where it WAITS instead of grabbing, so no frame is
        # ever skipped.
        self._unread = False
        # Signalled whenever a consumer takes the frame in the slot. Lockstep
        # waits on this rather than polling, so a slow consumer costs the
        # reader nothing while it waits.
        self._taken_evt = threading.Event()
        # Longest edge to analyse at, 0 = the camera's own size. See _fit.
        self._max_width = env_int("INGEST_MAX_WIDTH", 0)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[int, int, np.ndarray] | None = None  # (seq, tMs, frame)
        # The file's own FPS, probed once and shared by pacing and the gate.
        self._fps: float | None = None
        self._pace_s = self._pacing_interval() if is_file else 0.0
        self.levers = levers if levers is not None else Levers()
        # LEVER MODE — see _run_levered. Decided once, here: a worker never
        # switches loops mid-stream.
        self.levered = self.levers.armed
        self._gate = (
            MotionGate(
                self.levers.motion_min_frac,
                self.levers.motion_pixel_thr,
                self.levers.motion_keepalive_s,
            )
            if self.levers.motion_gate
            else None
        )
        # Lockstep never queues: the reader waits for each published frame to
        # be taken, so the slot is the whole buffer and nothing is dropped.
        self._store = FrameStore(
            0.0 if self.lockstep else self.levers.buffer_s, self.levers.buffer_mb * _MB
        )
        self._last_item: Item | None = None  # the frame most recently served
        self._exhausted = False  # file read to its end; frames may still wait
        self._counts = {
            "captured": 0, "published": 0, "skipped": 0,
            "dropped": 0, "served": 0, "backlogMax": 0,
        }
        if self.levered and is_file:
            self._probe_fps()  # the gate's footage clock needs it even unpaced
        self._cap = self._open()

    # ------------------------------------------------------------- public

    def latest(self) -> tuple[int, int, np.ndarray] | None:
        """Return (seq, tMs, frame) for the newest frame, or None before one.

        The returned array is never mutated afterwards (the reader allocates
        a fresh array per decoded frame), so no copy is taken here.

        Taking the frame RELEASES a lockstep reader, which is blocked waiting
        for exactly this. Signalled outside the lock: the reader's first act
        on waking is to take that same lock, and holding it while we wake them
        would hand them a lock they immediately have to queue for.
        """
        if self.levered:
            served = self.take()
            return None if served is None else (served.seq, served.t_ms, served.image)
        with self._lock:
            was_unread, self._unread = self._unread, False
            latest = self._latest
        if was_unread:
            self._taken_evt.set()
        return latest

    def take(self) -> Served | None:
        """LEVER MODE: the oldest unread frame, or the last one again.

        A fresh frame leaves the store (and wakes a lockstep reader); with
        nothing unread the last frame served is repeated, so a poller sees the
        same ``seq`` and waits — exactly what the newest-frame slot did.

        ``ended`` is decided HERE, under the same lock as the take, and never
        rides on a fresh frame: the runner treats ``ended`` as "no frame" and
        stops, so a frame handed out with ``ended`` set would be the last one
        of the file thrown away. It is set only once the file is exhausted AND
        nothing published is left unread.
        """
        with self._lock:
            item = self._store.take()
            fresh = item is not None
            if fresh:
                self._last_item = item
                self._counts["served"] += 1
            else:
                item = self._last_item
            backlog = self._store.pending()
            if not fresh and self._exhausted and backlog == 0:
                self.ended = True
            counts = dict(self._counts)
        if fresh:
            self._taken_evt.set()
        if item is None:
            return None
        return Served(
            seq=item.seq,
            t_ms=item.t_ms,
            image=item.frame.bgr(self._fit),
            ended=self.ended and not fresh,
            motion=item.motion,
            backlog=backlog,
            captured=counts["captured"],
            # Absent is not zero: with the gate off nothing was measured, and
            # a 0 would read as "the gate ran and found motion everywhere".
            skipped=counts["skipped"] if self._gate is not None else None,
            dropped=counts["dropped"],
        )

    def describe(self) -> dict:
        """This worker's lever settings and counters, for GET /health."""
        with self._lock:
            counts = dict(self._counts)
            pending, nbytes = self._store.pending(), self._store.bytes
        out = {
            "levered": self.levered,
            "isFile": self.is_file,
            "lockstep": self.lockstep,
            "knobs": self.levers.knobs(),
            "counters": None,
        }
        if self.levered:
            out["counters"] = {
                **counts,
                "skipped": counts["skipped"] if self._gate is not None else None,
                "pending": pending,
                "bufferBytes": nbytes,
                "exhausted": self._exhausted,
            }
        return out

    def stop(self, join_timeout_s: float = 5.0) -> None:
        """Signal the thread, wait for it, and release the capture."""
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=join_timeout_s)

    # ------------------------------------------------------------ thready

    def run(self) -> None:
        """Read loop: newest frame wins the slot; files pace and loop."""
        if self.levered:
            self._run_levered()
            return
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
                    # LOCKSTEP: wait for the consumer instead of skipping past
                    # it. The frame in the slot has not been taken, so reading
                    # the next one would discard it — which is precisely what
                    # this mode exists to prevent. Waiting costs nothing (a
                    # recording has no clock to fall behind) and bounds memory
                    # at exactly one frame, where a buffer would grow forever.
                    if self.lockstep:
                        self._taken_evt.clear()
                        # Re-check under the lock before sleeping: the consumer
                        # may have taken it between _slot_unread and here, and
                        # a wait that misses its wake-up would stall the run.
                        if self._slot_unread() and not self._taken_evt.wait(timeout=1.0):
                            continue  # still unread — loop so stop() is honoured
                        if self._stop_evt.is_set():
                            return
                        continue  # slot free now; decode the next frame
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
                # Only a RETRIEVED frame fills the slot; a grabbed one was
                # decoded to keep the reference chain and then discarded.
                if frame is not None:
                    t_ms = int((time.monotonic() - t0) * 1000)
                    with self._lock:
                        self._latest = (seq, t_ms, self._fit(frame))
                        self._unread = True
                # PACE EVERY FRAME, GRABBED OR RETRIEVED — and this line's
                # placement is the whole fix.
                #
                # It used to sit after a `continue` that skipped it whenever
                # the slot was still unread, i.e. whenever the consumer was
                # slower than the file. That is exactly when pacing matters,
                # so a file was paced only while nothing needed pacing. With
                # any real consumer the loop then span through grab() as fast
                # as the disk allowed: measured on a 60 s / 1803-frame clip,
                # `seq=2, ended=True` after 0.9 s, and runs settling
                # `source-ended` having processed one to three frames. It read
                # as a broken source and was not one.
                #
                # A LIVE source is unaffected: `_pace_s` is 0 for anything not
                # a file (see __init__), so this is a no-op there — the camera
                # does the pacing, and sleeping would make the reader fall
                # behind the stream it is meant to be keeping up with.
                if self._pace_s and self._stop_evt.wait(self._pace_s):
                    return
        finally:
            self._cap.release()

    def _run_levered(self) -> None:
        """LEVER MODE read loop: decode, gate, publish or skip — every frame.

        Differences from the loop above, each on purpose:

        * **Every frame is retrieved.** The gate has to look at a frame to
          skip it, and a buffered frame has to exist to be queued. On the cv2
          decoder that is the BGR conversion the loop above avoids for unread
          frames (~35 ms CPU per 4K frame on the laptop); the ffmpeg decoders
          (INGEST_DECODER, L6) hand the gate a free Y plane instead.
        * **Newest wins the unbuffered slot.** The frame is already retrieved,
          so the unread one it replaces is simply older; the overwrite is
          counted in ``dropped``.
        * **Pacing keeps a deadline** instead of sleeping a fixed interval
          after each frame, so decode time is not added on top of the frame
          period and a paced clip really plays at its native rate.
        """
        t0 = time.monotonic()
        due = t0
        seq = 0
        try:
            while not self._stop_evt.is_set():
                if self.lockstep and self._pending():
                    # Same wait as the loop above: re-check under the lock, and
                    # wake at least once a second so stop() is honoured.
                    self._taken_evt.clear()
                    if self._pending():
                        self._taken_evt.wait(timeout=1.0)
                    continue
                frame = self._next_raw()
                if frame is None:
                    if self._stop_evt.is_set():
                        return
                    if self.is_file and self.loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if self.is_file:
                        with self._lock:
                            self._exhausted = True
                        return
                    # Live stream hiccup: release, breathe, reopen — as above.
                    self._cap.release()
                    if self._stop_evt.wait(1.0):
                        return
                    try:
                        self._cap = self._open()
                    except CaptureError:
                        continue
                    if self._gate is not None:
                        self._gate.reset()  # a new stream: compare with nothing old
                    continue
                seq += 1
                self._admit(seq, t0, frame)
                if self._pace_s:
                    due += self._pace_s
                    delay = due - time.monotonic()
                    if delay > 0 and self._stop_evt.wait(delay):
                        return
        finally:
            self._cap.release()

    def _next_raw(self) -> RawFrame | None:
        """Decode the next frame, or None at EOF / on a read failure."""
        ok, frame = self._cap.read()
        return RawFrame.from_bgr(self._fit(frame)) if ok else None

    def _admit(self, seq: int, t0: float, frame: RawFrame) -> None:
        """Gate one decoded frame, then publish it into the store or skip it."""
        now = time.monotonic()
        t_ms = int((now - t0) * 1000)
        # Footage time for a file — frame number over its FPS, so the gate's
        # keepalive means "a frame per second OF FOOTAGE" whether the run is
        # paced, lockstep or flat out. Wall time for a camera.
        clock_s = seq / (self._fps or 25.0) if self.is_file else now - t0
        publish, motion = True, None
        if self._gate is not None:
            publish, motion = self._gate.decide(small_luma(frame.luma_source()), clock_s)
        with self._lock:
            self._counts["captured"] += 1
            if not publish:
                self._counts["skipped"] += 1
                return
            self._counts["published"] += 1
            self._counts["dropped"] += self._store.put(Item(seq, t_ms, clock_s, frame, motion))
            self._counts["backlogMax"] = max(self._counts["backlogMax"], self._store.pending())

    def _pending(self) -> int:
        """Published frames not yet taken (lever mode)."""
        with self._lock:
            return self._store.pending()

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
        return 1.0 / (self._probe_fps() * pace)

    def _probe_fps(self) -> float:
        """The file's own FPS (fallback 25), probed once with a throwaway capture."""
        if self._fps is None:
            probe = cv2.VideoCapture(self.source)
            fps = probe.get(cv2.CAP_PROP_FPS) if probe.isOpened() else 0.0
            probe.release()
            self._fps = fps if fps and fps > 0 else 25.0
        return self._fps
