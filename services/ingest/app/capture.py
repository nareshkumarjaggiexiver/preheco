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
import sys
import threading
import time

import cv2
import numpy as np
from heco_common.config import env_bool, env_float, env_int
from heco_common.logs import safe
from heco_common.ort import announce_device

from .config import Levers
from .ffmpeg_source import DecoderError, FfmpegSource
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
        # A FILE that stops short of its own length was CUT, not finished:
        # set when a cut could not be resumed, so /health can say why the
        # worker stopped producing. Never set for a file that really ended.
        self.interrupted: str | None = None
        self.resumes = 0
        self._frame_count: int | None = None
        self._pace_s = self._pacing_interval() if is_file else 0.0
        self.levers = levers if levers is not None else Levers()
        # L6 DEVICE TRUTH, served from /health: what was asked for, what
        # actually decodes, and why not when they differ.
        self.decoder = {"requested": self.levers.decoder, "active": "cpu", "error": None}
        self._ff: FfmpegSource | None = None
        if self.levers.decoder != "cpu":
            self._ff = self._start_decoder()
        self._start_error = self.decoder["error"]
        # LEVER MODE — see _run_levered. Decided once, here: a worker never
        # switches loops mid-stream. A hardware decoder that fell back to cpu
        # with no other lever armed IS today's worker, so it runs today's loop.
        self.levered = (
            self.levers.motion_gate or self.levers.buffer_s > 0 or self._ff is not None
        )
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
        # A LIVE source's reconnect record (see _note_down / _note_up):
        # reopenings, and reopenings that failed. Served on /health in lever
        # mode only, so OFF keeps today's /health exactly.
        self._live = {"reconnects": 0, "reconnectFailures": 0}
        self._down_since_fail = 0  # failed reopenings since the source was last up
        # The start-up fallback reason (a hardware decoder that fell back to
        # cpu) survives a reconnect: _note_up restores it, never clears it.
        self._start_error: str | None = None
        if self.levered and is_file:
            self._probe_fps()  # the gate's footage clock needs it even unpaced
        self._cap = self._open() if self._ff is None else None

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
            "decoder": dict(self.decoder),
            "counters": None,
            "resumes": self.resumes,
            "interrupted": self.interrupted,
        }
        if not self.is_file and self.levered:
            with self._lock:
                out["live"] = dict(self._live)
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
        ff = self._ff
        if ff is not None:
            ff.interrupt()  # a read blocked on the pipe returns now
        if self.is_alive():
            self.join(timeout=join_timeout_s)
        elif ff is not None:
            ff.close()  # never started, or already finished: reap it here

    # ------------------------------------------------------------ thready

    def run(self) -> None:
        """Read loop: newest frame wins the slot; files pace and loop."""
        if self.is_file and not self.loop:
            # The file's length, read NOW, while the source is known good: a
            # lookup made at the moment of a cut asks the server that just
            # went away, reads "unknown", and a cut would pass for the end.
            self._expected_frames()
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
                        # A read that fails before the file's own length is a
                        # CUT stream (the planner serving it restarted), not
                        # the end: resume at this frame, or stop producing so
                        # the run fails loudly — never `ended` on half a file.
                        if self._cut_short(seq):
                            if self._resume_at(seq):
                                continue
                            self._give_up(seq)
                            return
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
                    if self.is_file and self._ff is not None and not self._ff.clean_eof:
                        # A decoder that DIED is not the end of the file, and
                        # saying `ended` would settle an incomplete count as a
                        # complete one. Stop producing instead: the runner sees
                        # a stall and fails the run with its gallery kept.
                        self.decoder["error"] = (
                            f"decoder exited {self._ff.returncode} mid-file: {self._ff.tail()}"
                        )
                        sys.stderr.write(f"[heco-device] ingest {self.decoder['error']}\n")
                        return
                    if self.is_file and self.loop and self._ff is None:
                        # (an ffmpeg decoder loops the file itself: -stream_loop)
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if self.is_file and self._cut_short(seq):
                        # Same rule as the loop above. The cv2 decoder can
                        # reopen and seek; an ffmpeg decoder that exited
                        # "cleanly" on a cut HTTP stream is not resumed yet —
                        # it stops producing, so the run fails loudly.
                        if self._ff is None and self._resume_at(seq):
                            continue
                        self._give_up(seq)
                        return
                    if self.is_file:
                        with self._lock:
                            self._exhausted = True
                        return
                    # Live stream hiccup: release, breathe, reopen — as above.
                    self._release_source()
                    if self._stop_evt.wait(1.0):
                        return
                    try:
                        self._reopen_source()
                    except CaptureError as exc:
                        self._note_down(exc)
                        continue
                    self._note_up()
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
            self._release_source()

    def _next_raw(self) -> RawFrame | None:
        """Decode the next frame, or None at EOF / on a read failure."""
        if self._ff is not None:
            return self._ff.read()
        ok, frame = self._cap.read()
        return RawFrame.from_bgr(self._fit(frame)) if ok else None

    def _start_decoder(self) -> FfmpegSource | None:
        """Start the requested subprocess decoder, or fall back to cpu LOUDLY.

        A decoder that cannot start must not take the camera down with it: the
        cv2 path is today's and still works. But it must never be silent
        either — a "hardware decode" benchmark that quietly ran on the CPU is
        exactly the dishonesty the device truth exists to catch — so the
        fallback is announced on stderr and served from /health ``device``.
        """
        requested = self.levers.decoder
        try:
            ff = self._new_ffmpeg()
        except DecoderError as exc:
            self.decoder["error"] = str(exc)
            announce_device("ingest", requested, ["cpu"])
            sys.stderr.write(f"[heco-device] ingest decoder fell back to cpu: {exc}\n")
            return None
        self.decoder["active"] = requested
        announce_device("ingest", requested, [requested])
        return ff

    def _new_ffmpeg(self) -> FfmpegSource:
        return FfmpegSource(
            self.source,
            self.levers.decoder,
            loop=self.is_file and self.loop,
            rtsp_tcp=env_bool("INGEST_RTSP_TCP", True),
            abort=self._stop_evt,
        )

    def _release_source(self) -> None:
        if self._ff is not None:
            self._ff.close()
        elif self._cap is not None:
            self._cap.release()

    def _reopen_source(self) -> None:
        """Reconnect a live source on the decoder it was running (CaptureError on failure).

        A camera that drops is a camera problem, not a decoder one, so the
        reconnect never switches decoders mid-run.
        """
        if self._ff is None:
            self._cap = self._open()
            return
        try:
            self._ff = self._new_ffmpeg()
        except DecoderError as exc:
            raise CaptureError(str(exc)) from exc

    def _note_down(self, exc: Exception) -> None:
        """A live source's reopening failed: say so on /health, and once on stderr.

        Without this a camera whose decoder cannot restart (VRAM pressure while
        TensorRT builds, a driver reset) stayed dark for the rest of the run
        with ``device`` still reading active=nvdec, error=null and nothing
        logged — 40 failed restarts measured, the DecoderError that named the
        cause thrown away — and the runner said only ``source-stalled``.
        """
        reason = f"live source down, reconnecting: {exc}"
        with self._lock:
            self._live["reconnectFailures"] += 1
            self._down_since_fail += 1
            first = self._down_since_fail == 1
            self.decoder["error"] = reason
        if first:
            sys.stderr.write(f"[heco-device] ingest {self.decoder['active']}: {reason}\n")

    def _note_up(self) -> None:
        """A live source reopened: count it, and clear what _note_down said."""
        with self._lock:
            self._live["reconnects"] += 1
            failed, self._down_since_fail = self._down_since_fail, 0
            self.decoder["error"] = self._start_error
        if failed:
            sys.stderr.write(
                f"[heco-device] ingest {self.decoder['active']}: live source back "
                f"after {failed} failed reconnect(s)\n"
            )

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

    #: A file that stops within this many frames of its own length ended;
    #: further from the end it was cut. The last frames of an NVR file can
    #: legitimately fail to decode (a trailing partial GOP), and a lockstep
    #: run's final frame is known to go out with `ended` set.
    CUT_TOLERANCE_FRAMES = 30
    #: Seconds between resume attempts: ~2 minutes in all, enough to ride out
    #: a planner restart (the file is streamed from the planner's disk).
    RESUME_BACKOFF_S = (1, 2, 3, 5, 8, 13, 20, 30, 30)

    def _expected_frames(self) -> int | None:
        """The file's own frame count (container metadata), probed once.

        None when the container does not say — a stream whose length is
        unknown cannot be judged cut, and keeps today's behaviour.
        """
        if self._frame_count is None:
            cap = getattr(self, "_cap", None)
            if cap is not None and hasattr(cap, "get"):
                # The capture already open knows its length: no second
                # connection to the server that is streaming the file.
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            else:  # an ffmpeg decoder holds the stream; ask OpenCV once
                probe = cv2.VideoCapture(self.source)
                n = probe.get(cv2.CAP_PROP_FRAME_COUNT) if probe.isOpened() else 0.0
                probe.release()
            self._frame_count = int(n) if n and n > 0 else 0
        return self._frame_count or None

    def _cut_short(self, frames_read: int) -> bool:
        """Did this file stop well short of its own length?

        THE CASE (2026-09-25). The planner serves footage to ingest over HTTP
        from its own disk; restarting it mid-run cut the stream ("Stream ends
        prematurely at 329869580, should be 629262502"), OpenCV's read simply
        failed, and this worker called that the end of the file: the run
        settled as source-ended with 4709 of 9000 frames counted — a short
        count reported as complete. Run 8b8b87 lost its last 554 frames the
        same way.
        """
        total = self._expected_frames()
        return bool(total) and frames_read < total - self.CUT_TOLERANCE_FRAMES

    def _resume_at(self, frames_read: int) -> bool:
        """Reopen the file and continue from frame ``frames_read``.

        Retries on the backoff above (the source may still be coming back);
        True once a capture is open and positioned, False when stopped or out
        of attempts. The frame sequence stays contiguous: the next frame
        decoded is the one after the last frame this worker delivered.
        """
        for delay in self.RESUME_BACKOFF_S:
            if self._stop_evt.wait(delay):
                return False
            try:
                cap = self._open()
            except CaptureError:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames_read)
            old, self._cap = self._cap, cap
            if old is not None:
                old.release()
            self.resumes += 1
            sys.stderr.write(
                f"[heco-ingest] file source cut at frame {frames_read} of "
                f"{self._expected_frames()}; resumed there (resume #{self.resumes})\n"
            )
            return True
        return False

    def _give_up(self, frames_read: int) -> None:
        """Stop producing WITHOUT claiming the file ended.

        The runner then sees its source stall and fails the run with the
        gallery kept — an incomplete count said out loud, where `ended` would
        have filed it as complete.
        """
        self.interrupted = (
            f"file source cut at frame {frames_read} of {self._expected_frames()} "
            f"and could not be resumed after {len(self.RESUME_BACKOFF_S)} attempts"
        )
        sys.stderr.write(f"[heco-ingest] {self.interrupted}\n")

    def _probe_fps(self) -> float:
        """The file's own FPS (fallback 25), probed once with a throwaway capture."""
        if self._fps is None:
            probe = cv2.VideoCapture(self.source)
            fps = probe.get(cv2.CAP_PROP_FPS) if probe.isOpened() else 0.0
            probe.release()
            self._fps = fps if fps and fps > 0 else 25.0
        return self._fps
