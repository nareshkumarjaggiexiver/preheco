"""The observability plane's own tests: what async reporting must still promise.

test_loop.py and test_loop_v1.py own the CONTENT of a tap round — which
payloads, which boxes, which shedding order.  This file owns the SCHEDULER:
what survives being moved off the counting loop, and what is allowed to be
dropped when the reporter falls behind.  The distinction matters because the
whole point of the change is that reporting may now lag, and a design that
lets it lag has to say precisely what it will never lose.
"""

import threading
import time

from app.config import Settings
from app.reporting import FORCED_CAP, FORENSIC_CAP, Reporter
from app.stats import StatsBoard


class FakeLoop:
    """The reporter's view of a run loop: four calls and a stats board."""

    def __init__(self, *, round_delay_s: float = 0.0, raises: bool = False) -> None:
        self.board = StatsBoard()
        self.rounds: list[dict] = []
        self.flushes: list[float] = []
        self.uploads: list[dict] = []
        self.status: dict[str, int] = {}
        self._round_delay_s = round_delay_s
        self._raises = raises
        self._lock = threading.Lock()

    def _tap_round(self, last: dict) -> None:
        if self._raises:
            raise RuntimeError("planner exploded mid-round")
        if self._round_delay_s:
            time.sleep(self._round_delay_s)
        with self._lock:
            self.rounds.append(last)

    def _flush(self, elapsed_s: float) -> None:
        with self._lock:
            self.flushes.append(elapsed_s)

    def _forensic_upload(self, last: dict) -> None:
        with self._lock:
            self.uploads.append(last)

    def _bump(self, key: str, by: int = 1) -> None:
        with self._lock:
            self.status[key] = self.status.get(key, 0) + by


def settings(**kw) -> Settings:
    """Reporter settings: everything due immediately unless a test says else."""
    return Settings(
        ingest_url="http://ingest:7101", persons_url="http://persons:7102",
        tracker_url="http://tracker:7103", faces_url="http://faces:7104",
        embed_url="http://embed:7105", match_url="http://match:7106",
        planner_url="http://planner:8787",
        **{"tap_interval_s": 0.0, "flush_interval_s": 0.0, "tap_duty_factor": 0.0,
           "reporter_poll_s": 0.005, **kw},
    )


def frame(seq: int) -> dict:
    """A published snapshot, minimal but distinguishable."""
    return {"seq": seq, "image_b64": "", "t_ms": seq * 10}


def wait_until(predicate, timeout: float = 3.0) -> bool:
    """Poll until predicate holds; True if it did, False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_a_mint_frame_is_never_dropped_even_when_the_reporter_is_behind():
    """THE keyframe guarantee: a counted guest always gets their picture.

    Run 6cd269 counted three real people and kept 21 keyframes of which 19
    showed nobody, because each guest was minted inside a window no sampled
    round coincided with.  Drop-not-queue would reproduce that exactly — so
    mint frames ride a queue latest-wins cannot swallow.  Here the reporter is
    made slow (80 ms a round) and flooded with ordinary frames; the minted one
    must still be rendered.
    """
    loop = FakeLoop(round_delay_s=0.08)
    reporter = Reporter(loop, settings())
    reporter.start()
    try:
        for seq in range(20):
            reporter.publish(frame(seq), minted=(seq == 7))
            time.sleep(0.002)
        assert wait_until(lambda: any(f["seq"] == 7 for f in loop.rounds)), (
            "the frame that minted a guest must reach a tap round"
        )
    finally:
        reporter.finish()


def test_ordinary_frames_are_dropped_not_queued():
    """A slow reporter skips frames; it never builds a backlog to work through.

    The console must show the most RECENT frame, not an ever-older one from a
    queue — a stale picture is worse than a missing one, because the operator
    is checking it against people currently walking through the gate.  So a
    flood of 40 frames against an 80 ms round must NOT produce 40 rounds, and
    what does get rendered must be recent.
    """
    loop = FakeLoop(round_delay_s=0.08)
    reporter = Reporter(loop, settings())
    reporter.start()
    try:
        for seq in range(40):
            reporter.publish(frame(seq))
            time.sleep(0.002)
        time.sleep(0.2)
        rendered = list(loop.rounds)
    finally:
        reporter.finish()
    assert len(rendered) < 40, f"a backlog was worked through: {len(rendered)} rounds"
    assert rendered, "some frame must still reach the console"
    # Every rendered frame was the latest at the moment the reporter was free,
    # so the sequence must be strictly increasing — never replayed in arrears.
    seqs = [f["seq"] for f in rendered]
    assert seqs == sorted(seqs), f"frames rendered out of order: {seqs}"


def test_publish_does_not_block_the_loop_on_a_slow_round():
    """The loop's whole cost is the handover, whatever the planner is doing.

    This is the change's entire purpose: with a round taking 200 ms, a
    thousand publishes must still return in microseconds each.  If publish
    ever waited on the reporter the death spiral would simply have moved.
    """
    loop = FakeLoop(round_delay_s=0.2)
    reporter = Reporter(loop, settings())
    reporter.start()
    try:
        started = time.perf_counter()
        for seq in range(1000):
            reporter.publish(frame(seq))
        elapsed = time.perf_counter() - started
    finally:
        reporter.finish()
    assert elapsed < 0.5, f"1000 publishes took {elapsed:.3f}s — publish is blocking"


def test_finish_drains_owed_mint_frames_before_stopping():
    """The last guest through the gate is the one most likely to be disputed.

    Stopping the thread bluntly would discard exactly the frames an operator
    goes looking for afterwards, so finish() drains what is owed first.
    """
    loop = FakeLoop()
    reporter = Reporter(loop, settings(tap_interval_s=999.0))
    # Not started: nothing can have been rendered yet, so what arrives can
    # only have come from the drain.
    reporter.publish(frame(1), minted=True)
    reporter.publish(frame(2), minted=True)
    reporter.finish()
    assert [f["seq"] for f in loop.rounds] == [1, 2], (
        "owed mint frames must be rendered on the way out, oldest first"
    )


def test_forensic_queue_is_bounded_and_says_what_it_dropped():
    """A bounded queue is only honest if its overflow is counted.

    Forensic mode keeps a picture of every frame, ~1.4 MB each at 4K, so the
    queue must have a ceiling — and a thinned record that no counter confesses
    to would have an engineer hunting a gap that nothing reports.
    """
    loop = FakeLoop()
    reporter = Reporter(loop, settings())
    for seq in range(FORENSIC_CAP + 5):
        reporter.publish(frame(seq), forensic=True)
    assert loop.status.get("forensicFramesDropped") == 5, (
        "every dropped forensic picture must be counted"
    )
    reporter.finish()
    assert len(loop.uploads) == FORENSIC_CAP
    assert loop.uploads[0]["seq"] == 5, "the OLDEST pictures are the ones shed"


def test_mint_queue_overflow_is_counted_too():
    """Minting faster than the planner accepts pictures is a fact worth saying."""
    loop = FakeLoop()
    reporter = Reporter(loop, settings(tap_interval_s=999.0))
    for seq in range(FORCED_CAP + 3):
        reporter.publish(frame(seq), minted=True)
    assert loop.status.get("tapFramesDropped") == 3
    reporter.finish()


def test_a_round_that_raises_does_not_kill_the_reporter():
    """One exploding round must not take the console dark for the whole run.

    The loop's own methods swallow PlannerError; this is the backstop for
    everything they do not, because a dead reporter thread is silent — the
    console simply stops updating and nothing anywhere says why.
    """
    loop = FakeLoop(raises=True)
    reporter = Reporter(loop, settings())
    reporter.start()
    try:
        for seq in range(5):
            reporter.publish(frame(seq))
            time.sleep(0.02)
        assert reporter.is_alive(), "the reporter thread must survive a bad round"
        assert wait_until(lambda: loop.status.get("plannerReportErrors", 0) >= 1), (
            "a failed round must be counted, not swallowed silently"
        )
    finally:
        reporter.finish()


def test_flushes_happen_on_their_own_cadence_without_the_loop():
    """Stats keep flowing while the loop does nothing but count."""
    loop = FakeLoop()
    reporter = Reporter(loop, settings(flush_interval_s=0.02))
    reporter.begin(time.monotonic())
    reporter.start()
    try:
        assert wait_until(lambda: len(loop.flushes) >= 3), (
            "the reporter must flush on its own clock, unprompted by frames"
        )
    finally:
        reporter.finish()


def test_the_stats_board_survives_a_new_metric_appearing_mid_snapshot():
    """The specific RuntimeError the board's lock exists to prevent.

    ``observe`` grows the metrics dict via setdefault, and several metric
    names first appear only when the scene produces them — a face metric on
    the first face, an embed metric on the first embedding.  A snapshot
    iterating that dict at the instant a new name arrives raises
    ``RuntimeError: dictionary changed size during iteration`` and kills the
    reporter.  This drives exactly that collision, hard, from two threads.
    """
    board = StatsBoard()
    stop = threading.Event()
    failures: list[BaseException] = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            board.observe("count", f"metric{i}", float(i))  # a NEW key every time
            i += 1

    def reader() -> None:
        try:
            while not stop.is_set():
                board.snapshot(1.0)
        except BaseException as exc:  # noqa: BLE001 — the failure IS the result
            failures.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    time.sleep(0.4)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    assert not failures, f"snapshot raced observe: {failures[0]!r}"


def test_the_duty_guard_delays_a_mint_keyframe_but_never_loses_it():
    """The guard bounds the reporter's share; it does not get to drop guests.

    Deferring pops the mint frame off the queue to decide about it, so the
    obvious implementation silently discards exactly the keyframe the forced
    queue exists to protect. It goes back at the front instead — mint order is
    the order guests were counted, and a register whose pictures arrive
    shuffled is harder to audit than one whose pictures arrive late.
    """
    loop = FakeLoop(round_delay_s=0.15)
    # duty_factor 20 against a 150 ms round = ~3 s of enforced quiet, far
    # longer than this test runs, so every round after the first is deferred.
    reporter = Reporter(loop, settings(tap_duty_factor=20.0))
    reporter.start()
    try:
        reporter.publish(frame(1))                 # first round, arms the guard
        assert wait_until(lambda: len(loop.rounds) >= 1)
        reporter.publish(frame(99), minted=True)   # must be deferred, not dropped
        time.sleep(0.4)
        assert not any(f["seq"] == 99 for f in loop.rounds), (
            "precondition: the guard should be deferring right now"
        )
        assert loop.status.get("tapRoundsDeferred", 0) >= 1
    finally:
        reporter.finish()
    assert any(f["seq"] == 99 for f in loop.rounds), (
        "the deferred mint frame must still be rendered, not discarded"
    )


def test_a_waiting_mint_frame_is_not_counted_as_a_deferred_round():
    """The deferral counter must measure rounds, never the reporter's polls.

    A queued mint frame makes a round OWED on every poll, so counting the duty
    guard's refusal there turns tapRoundsDeferred into a wake-up counter. It
    was measured twice: 153 deferrals against 79 possible rounds on a 848x478
    run, then 1142 against 200 on a 4K run with 37 guests. A counter that
    reads like an emergency during a healthy run is worse than no counter,
    because the next real emergency looks exactly like it.
    """
    loop = FakeLoop(round_delay_s=0.15)
    # ~3 s of enforced quiet after the first round, against a 5 ms poll.
    reporter = Reporter(loop, settings(tap_duty_factor=20.0, tap_interval_s=999.0))
    reporter.start()
    try:
        reporter.publish(frame(1), minted=True)     # first round, arms the guard
        assert wait_until(lambda: len(loop.rounds) >= 1)
        reporter.publish(frame(2), minted=True)     # owed, but the guard says wait
        time.sleep(0.5)                             # ~100 polls at 5 ms
        deferred = loop.status.get("tapRoundsDeferred", 0)
    finally:
        reporter.finish()
    assert deferred == 0, (
        f"a waiting mint frame counted {deferred} deferrals — the counter is "
        "measuring polls, not rounds"
    )
    assert any(f["seq"] == 2 for f in loop.rounds), "and it must still be rendered"
