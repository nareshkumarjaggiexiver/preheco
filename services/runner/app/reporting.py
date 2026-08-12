"""The observability plane, running OFF the counting loop.

WHY THIS EXISTS.  Everything the operator watches — the annotated stage
frames, the per-stage taps, the chart stats, the sampled rows, the forensic
pictures — used to be produced INSIDE the frame loop, between counting one
frame and fetching the next.  Measured on the RTX 4060 box with a live 4K
camera that cost 43-96 ms of every frame for tap rounds and another 15-22 ms
for stat flushes: roughly a fifth of a 295 ms frame spent drawing a picture
of the work instead of doing it.

The obvious fix — make the rounds rarer — is the wrong one, and the operator
said so plainly: someone standing at the gate watches faces appear and checks
them against the people actually walking through.  A console that refreshes
every twenty seconds is not a slower console, it is a different (and useless)
product.  So the cost had to leave the loop rather than fire less often:
observability becomes CHEAP, never RARE.

WHAT MOVED, AND THE LINE THAT DECIDES.  This thread runs only work that
REPORTS.  It never writes count state.  Tap rounds, planner flushes, face
cards and forensic frame uploads moved here; ``_maybe_feedback`` and
``_maybe_purge_staff`` deliberately did NOT, because they mutate the gallery
(merges, staff marks, missed-count additions) and the count has exactly one
writer — the loop thread — by design.  That is the same rule the multi-camera
architecture enforces later at the match service; it starts here.

DROP-NOT-QUEUE.  The loop publishes each frame's snapshot into a single slot,
overwriting whatever was there.  A reporter that falls behind therefore skips
frames instead of growing a backlog, and the console always shows the most
recent frame rather than an ever-older one from a queue — the same
latest-wins discipline ingest already uses for camera frames, and for the
same reason: a stale picture is worse than a missing one.

WITH ONE EXCEPTION, which is the whole reason ``_forced`` exists.  A frame
that MINTED a guest is the count-changing event, and the planner only keeps a
keyframe when a match-stage frame arrives.  Run 6cd269 counted three real
people and kept 21 keyframes of which 19 showed nobody, because each guest
was minted inside a window no round coincided with.  Mint frames therefore go
into a small bounded queue that latest-wins cannot swallow, so a guest who
existed for one second still gets their picture.

BOUNDED EVERYWHERE.  Both queues are small and their overflow is counted, not
silent: a reporter that cannot keep up says so in the run's own status
(``tapFramesDropped``, ``forensicFramesDropped``) instead of quietly thinning
the record.  The memory ceiling is the point, and it is worth stating in the
units that actually apply: what is held is the BASE64 TEXT of the frame, not
the JPEG — ~1.9 MB for a 4K frame, a third larger than the ~1.4 MB picture it
encodes.  At the caps below the plane can pin thirteen distinct snapshots
(one pending + four mint + eight forensic), so the true ceiling is ~25 MB.

AND ONE PROMISE THAT CHANGED.  Forensic mode used to upload every processed
frame synchronously, so "a picture of every frame" was literally true.  It is
now best-effort: a planner slower than the camera loses the frames past the
queue's depth.  That is the right trade for a bench setting — the loop must
not stall behind a picture — but it is a weaker guarantee than before, and
``forensicFramesDropped`` is what makes the difference visible rather than
leaving an engineer to infer a gap from a ledger that has none.
"""

import threading
import time
from collections import deque

# One mint frame per guest is the healthy case; four in flight means the
# gallery is minting faster than the planner accepts pictures, and the fifth
# is worth less than bounding the memory.
FORCED_CAP = 4

# ~1.9 MB per 4K frame as base64 text: eight frames is a ~15 MB ceiling on the
# forensic backlog.  Forensic mode is an opt-in bench setting, so a deep queue
# would only trade the operator's memory for pictures they asked to be timely.
FORENSIC_CAP = 8


class Reporter(threading.Thread):
    """Schedules the run loop's reporting work on its own thread.

    Deliberately a SCHEDULER and not a reimplementation: it calls the run
    loop's existing ``_tap_round`` / ``_flush`` / ``_forensic_upload`` methods
    unchanged.  What changed is which thread calls them and how often — the
    payload building, budget shedding and error handling are the same code
    that shipped, so this change is reviewable as "who calls it" rather than
    "what it does".
    """

    def __init__(self, loop, settings) -> None:
        """Wire the reporter to its run loop; call ``start()`` to run it."""
        super().__init__(name="heco-reporter", daemon=True)
        self._loop = loop
        self.s = settings
        # NOT `_stop`: threading.Thread has its own private _stop() METHOD, and
        # shadowing it with an Event breaks join() from inside the stdlib
        # ('Event' object is not callable) — caught by this module's own tests.
        self._stopping = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._pending: dict | None = None
        self._forced: deque = deque()
        self._forensic: deque = deque()
        self._t0 = time.monotonic()
        # Mirrors the loop's own duty guard.  The death spiral it was built
        # for cannot happen now (a slow round no longer delays a frame), but
        # the reporter still shares the box's cores and the planner's
        # bandwidth with the counting it is reporting on, and an unbounded
        # reporter would simply move the theft rather than end it.
        self._round_cost_s = 0.0
        self._round_ended = 0.0

    # ------------------------------------------------------------- publishing

    def publish(self, last: dict, *, minted: bool = False, forensic: bool = False) -> None:
        """Hand the loop's latest frame snapshot to the reporter and return.

        This is the ONLY thing the counting loop pays for observability now:
        one lock, a few assignments and an event set — microseconds against a
        frame measured in hundreds of milliseconds.  ``last`` must be a fresh
        dict the loop will not mutate afterwards (``_remember`` rebinds it
        every frame rather than updating in place), which is what makes
        handing over the reference safe without copying a megabyte.
        """
        with self._lock:
            self._pending = last
            if minted:
                if len(self._forced) >= FORCED_CAP:
                    self._forced.popleft()
                    self._loop._bump("tapFramesDropped")
                self._forced.append(last)
            if forensic:
                if len(self._forensic) >= FORENSIC_CAP:
                    self._forensic.popleft()
                    self._loop._bump("forensicFramesDropped")
                self._forensic.append(last)
        self._wake.set()

    def begin(self, t0: float) -> None:
        """Set the run's start instant, so flushes can report elapsed fps."""
        self._t0 = t0

    # ------------------------------------------------------------------- loop

    def run(self) -> None:
        """Render, flush and upload until stopped; never raise into the run."""
        last_tap = last_flush = time.monotonic()
        while not self._stopping.is_set():
            self._wake.wait(timeout=self.s.reporter_poll_s)
            self._wake.clear()
            now = time.monotonic()

            # Forensic pictures first: they are per-frame and the operator
            # turned them on to see THIS frame, so a queued one ages worst.
            self._drain_forensic()

            if now - last_flush >= self.s.flush_interval_s:
                self._guarded(self._loop._flush, now - self._t0)
                last_flush = now

            forced = self._take_forced()
            due = forced is not None or now - last_tap >= self.s.tap_interval_s
            if due and self._duty_allows(now):
                snapshot = forced if forced is not None else self._take_pending()
                if snapshot is not None:
                    started = time.monotonic()
                    try:
                        self._guarded(self._loop._tap_round, snapshot)
                    finally:
                        self._round_ended = time.monotonic()
                        self._round_cost_s = self._round_ended - started
                        self._loop.board.observe(
                            "count", "tapRoundMs", self._round_cost_s * 1000.0
                        )
                    last_tap = now

    def finish(self, timeout: float = 5.0) -> None:
        """Stop the thread after one last drain, so the tail is not lost.

        A run's final frames are the ones an operator most often goes looking
        for — the last guest through the gate — so stopping simply would
        throw away exactly the wrong pictures.  The drain is bounded by
        ``timeout`` because a planner that has gone away must not hold the
        run's settle open.
        """
        deadline = time.monotonic() + timeout
        self._drain_forensic(deadline=deadline)
        while time.monotonic() < deadline:
            forced = self._take_forced()
            if forced is None:
                break
            self._guarded(self._loop._tap_round, forced)
        self._stopping.set()
        self._wake.set()
        # `is_alive()` rather than an unconditional join: _stop_reporter runs
        # from the run's `finally`, which is reached on paths where the thread
        # was never started (a source that failed to open) — and join() on an
        # unstarted Thread raises, turning a clean failure into a confusing
        # RuntimeError from inside the stdlib.
        if self.is_alive():
            self.join(timeout=max(0.1, deadline - time.monotonic()))

    # ---------------------------------------------------------------- helpers

    def _take_pending(self) -> dict | None:
        """Take the latest published frame, if one arrived since the last."""
        with self._lock:
            pending, self._pending = self._pending, None
            return pending

    def _take_forced(self) -> dict | None:
        """Take the oldest mint frame still owed a keyframe, if any."""
        with self._lock:
            return self._forced.popleft() if self._forced else None

    def _duty_allows(self, now: float) -> bool:
        """True when the last round's cost has been repaid in quiet time."""
        if self._round_cost_s <= 0 or self.s.tap_duty_factor <= 0:
            return True
        if now - self._round_ended >= self.s.tap_duty_factor * self._round_cost_s:
            return True
        self._loop._bump("tapRoundsDeferred")
        return False

    def _drain_forensic(self, deadline: float | None = None) -> None:
        """Upload every queued forensic picture, oldest first."""
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return
            with self._lock:
                if not self._forensic:
                    return
                frame = self._forensic.popleft()
            self._guarded(self._loop._forensic_upload, frame)

    def _guarded(self, fn, *args) -> None:
        """Run one piece of reporting; a failure costs a picture, never a run.

        The loop's own methods already swallow ``PlannerError`` and friends;
        this is the backstop for anything they do not, because an exception
        escaping here kills the thread and takes the console dark for the
        rest of the run with no message anywhere.

        Arguments are passed through rather than closed over: every call site
        is inside the scheduling loop, and a closure would capture the
        variable rather than its value at the moment of the call.
        """
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 — reporting must never stop counting
            self._loop._bump("plannerReportErrors")
