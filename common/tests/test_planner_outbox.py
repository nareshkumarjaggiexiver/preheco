"""The durable outbox: a planner restart must cost the record nothing.

THE INCIDENT. Best-effort posts swallowed every failure, so the planner's
restart dropped every tap and frame the runner posted meanwhile, and runs were
marked failed "the runner went silent". These tests pin the replacement: a
durable client tries once exactly as before, keeps what the planner was away
for, and delivers it — once, in order, stamped with when it happened — without
ever making the caller wait for a planner that is down.

The fake planner below sits behind BOTH transports and records every attempt
(with the thread that made it) and everything that landed. Outages are
scripted, never timed; the sender's backoff is injected, so nothing here
sleeps for real except where a test is about the real wait.
"""

import random
import re
import threading
import time

import pytest
from heco_common.planner import (
    ACCEPTED,
    AT_HEADER,
    DELIVERY_HEADER,
    DROPPED,
    QUEUED,
    RETRY,
    SENT,
    Outbox,
    OutboxMessage,
    PlannerClient,
    PlannerDeferred,
    PlannerError,
)

ISO_MS = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")


class FakePlanner:
    """Both transports of one fake planner: scripted outages, and what landed.

    ``down`` fails every attempt; ``fail`` fails the next N; ``flaky`` fails
    each attempt with probability ``flaky_p``. A failure is ``failure``: an
    exception CLASS to raise (the planner unreachable, or a timeout) or an
    HTTP status to answer. ``refuse`` maps a URL suffix to a status answered
    while the planner is up — a refusal, not an outage.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.down = False
        self.fail = 0
        self.failure: type[BaseException] | int = OSError
        self.flaky: random.Random | None = None
        self.flaky_p = 0.0
        self.refuse: dict[str, int] = {}
        self.attempts: list[dict] = []
        self.landed: list[dict] = []
        self.next_id = 42

    def _answer(self, record: dict) -> tuple[int, dict]:
        with self.lock:
            self.attempts.append(record)
            failing = self.down or self.fail > 0 or (
                self.flaky is not None and self.flaky.random() < self.flaky_p
            )
            if self.fail > 0:
                self.fail -= 1
            failure = self.failure
            if not failing:
                for suffix, status in self.refuse.items():
                    if record["url"].endswith(suffix):
                        return status, {"error": "refused"}
                self.landed.append(record)
                if record["method"] == "POST" and record["url"].endswith("/api/pipeline/runs"):
                    run_id, self.next_id = self.next_id, self.next_id + 1
                    return 200, {"id": run_id}
                return 200, {"ok": True}
        if isinstance(failure, int):
            return failure, {"error": "planner restarting"}
        raise failure("planner unreachable")

    def json(self, method, url, payload, headers=None):
        """The JSON transport (retrying and best-effort alike)."""
        return self._answer({
            "method": method, "url": url, "body": payload,
            "headers": dict(headers or {}), "thread": threading.current_thread().name,
        })

    def file(self, url, fields, filename, data, content_type, headers=None):
        """The multipart transport (frames, face cards)."""
        return self._answer({
            "method": "POST", "url": url, "fields": dict(fields), "data": data,
            "headers": dict(headers or {}), "thread": threading.current_thread().name,
        })

    def landed_where(self, suffix: str) -> list[dict]:
        """What landed at URLs ending ``suffix``, in the order it landed."""
        return [r for r in self.landed if r["url"].endswith(suffix)]


class Gate:
    """An outbox sleep that parks the sender until released, recording each wait.

    While the sender is parked the queue cannot move, so a test can post into
    it and inspect it without racing the thread that empties it.
    """

    def __init__(self) -> None:
        self.naps: list[float] = []
        self.waiting = threading.Event()
        self._open = threading.Event()

    def __call__(self, seconds: float) -> None:
        """Record the wait asked for, then park until released (5 s at most)."""
        self.naps.append(seconds)
        self.waiting.set()
        self._open.wait(5.0)

    def release(self) -> None:
        """Let the sender through, now and for every later wait."""
        self._open.set()


@pytest.fixture()
def make():
    """Build durable clients against a fake planner; stop every sender afterwards."""
    built: list[PlannerClient] = []
    gates: list[Gate] = []

    def build(planner: FakePlanner, gate: Gate | None = None, **kw) -> PlannerClient:
        kw.setdefault("drain_timeout_s", 5.0)
        pc = PlannerClient(
            "http://planner:8787",
            transport=planner.json,
            file_transport=planner.file,
            durable=True,
            sleep=lambda _s: None,  # the retrying path's backoff, not the outbox's
            outbox_sleep=gate if gate is not None else (lambda _s: None),
            **kw,
        )
        pc.create_run(3)
        built.append(pc)
        if gate is not None:
            gates.append(gate)
        return pc

    yield build
    for gate in gates:
        gate.release()
    for pc in built:
        pc.stop_outbox(timeout=2.0)


def wall_ticks(start: float = 1_790_000_000.0):
    """A wall clock that moves one second per reading, so every stamp differs."""
    t = iter(range(10**6))
    return lambda: start + next(t)


def delivery_numbers(records: list[dict]) -> list[int]:
    """The per-client counter half of each record's X-Heco-Delivery."""
    return [int(r["headers"][DELIVERY_HEADER].split(":")[1]) for r in records]


# ----------------------------------------------------------- durable off


def test_durable_off_is_exactly_todays_client():
    """Off means today: a failed tap is dropped, nothing is queued, no header
    is added and no thread exists. The fakes take exactly the old positional
    arguments, so a headers= keyword would blow up inside them."""
    calls: list[tuple] = []

    def strict_json(method, url, payload):
        calls.append((method, url, payload))
        if url.endswith("/api/pipeline/runs"):
            return 200, {"id": 42}
        return (503, {}) if len(calls) == 2 else (200, {})

    def strict_file(url, fields, filename, data, content_type):
        calls.append((url, fields))
        return 200, {}

    pc = PlannerClient(
        "http://planner:8787", transport=strict_json, file_transport=strict_file
    )
    pc.create_run(3)
    assert pc.durable is False and pc.outbox_stats() is None

    assert pc.post_tap("track", {"tracks": 2}) is False, "the 503 is swallowed, not kept"
    assert pc.post_tap("track", {"tracks": 3}) is True
    assert pc.post_frame("ingest", b"jpeg") is True
    assert pc.post_face_card("p00001", b"jpeg") is True
    assert pc.post_frame_records([{"seq": 1}]) is True
    assert pc.offer_tap("track", {}) == ACCEPTED
    assert len(calls) == 7, "one attempt each, and no retry of the refused tap"
    assert pc.outbox is None, "no queue, so no thread can ever exist"
    assert pc.drain_outbox() is True and pc.stop_outbox() is True


def test_a_durable_client_with_a_healthy_planner_sends_what_it_always_sent(make):
    """Every post lands on its first attempt: the same calls, plus two headers,
    and no thread is ever started."""
    planner = FakePlanner()
    pc = make(planner, wall_clock=wall_ticks())

    assert pc.post_tap("match", {"unique": 4}) is True
    assert pc.post_frame("match", b"jpeg") is True
    assert pc.post_face_card("p00001", b"card") is True
    assert pc.post_frame_records([{"seq": 0}]) is True
    assert pc.outbox._thread is None, "nothing queued, so nothing to send it"
    stats = pc.outbox_stats()
    assert stats["queued"] == 0 and stats["backlog"] == 0 and stats["dropped"] == {}

    posts = planner.landed[1:]
    assert [r["thread"] for r in posts] == ["MainThread"] * 4, "synchronous, as before"
    assert posts[0]["body"] == {"stage": "match", "payload": {"unique": 4}}
    assert [r["headers"][DELIVERY_HEADER] for r in posts] == ["42:1", "42:2", "42:3", "42:4"]
    assert all(ISO_MS.match(r["headers"][AT_HEADER]) for r in posts)


def test_a_durable_client_refuses_a_transport_that_cannot_carry_the_headers():
    """Loudly at construction, not as a message retried forever at 3 a.m."""
    with pytest.raises(TypeError, match="headers="):
        PlannerClient(
            "http://planner:8787", transport=lambda m, u, p: (200, {}), durable=True
        )
    with pytest.raises(TypeError, match="file_transport"):
        PlannerClient(
            "http://planner:8787",
            transport=FakePlanner().json,
            file_transport=lambda u, f, n, d, c: (200, {}),
            durable=True,
        )


# ------------------------------------------------------ outage and recovery


def test_a_planner_that_fails_then_recovers_gets_everything_once_in_order(make):
    """THE INCIDENT, replayed. The planner goes away mid-run and comes back:
    every tap, frame, face card and frame record lands exactly once, in the
    order it was posted, numbered in that order and dated when it was made."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate, wall_clock=wall_ticks())
    planner.down = True

    outcomes = [
        pc.offer_tap("ingest", {"seq": 1}),
        pc.offer_frame("ingest", b"frame-1"),
        pc.offer_tap("match", {"unique": 1}),
        pc.offer_frame("match", b"keyframe-1"),
        pc.offer_face_card("p00001", b"card-1"),
        pc.offer_frame_records([{"seq": 1}, {"seq": 2}]),
        pc.offer_tap("match", {"unique": 2}),
    ]
    assert outcomes == [QUEUED] * 7
    assert pc.post_tap("track", {"late": True}) is False, "True only when accepted"

    stamps = [r["headers"] for r in planner.attempts if r["thread"] == "MainThread"][1:]
    assert len(stamps) == 1, "one synchronous attempt; everything after it queued"
    assert gate.waiting.wait(2.0)

    # The planner comes back — but not at once: three more refusals first.
    planner.down = False
    planner.fail = 3
    gate.release()
    assert pc.drain_outbox(5.0) is True

    posts = planner.landed[1:]
    kinds = [r["url"].rsplit("/", 1)[1] for r in posts]
    assert kinds == ["taps", "frames", "taps", "frames", "faces", "frame-records", "taps", "taps"]
    assert [r["body"]["payload"] for r in posts if r["url"].endswith("/taps")] == [
        {"seq": 1}, {"unique": 1}, {"unique": 2}, {"late": True},
    ]
    assert [r["data"] for r in posts if "data" in r] == [b"frame-1", b"keyframe-1", b"card-1"]
    assert delivery_numbers(posts) == list(range(1, 9)), "numbered in call order"
    assert len({r["headers"][DELIVERY_HEADER] for r in posts}) == 8, "each exactly once"

    # X-Heco-At is the moment of the CALL, identical on every attempt.
    by_delivery: dict[str, set[str]] = {}
    for r in planner.attempts[1:]:
        by_delivery.setdefault(r["headers"][DELIVERY_HEADER], set()).add(r["headers"][AT_HEADER])
    assert all(len(ats) == 1 for ats in by_delivery.values())
    ats = [r["headers"][AT_HEADER] for r in posts]
    assert ats == sorted(ats) and len(set(ats)) == 8, "one stamp per call, in call order"

    assert gate.naps == [0.5, 1.0, 2.0, 4.0], "backoff grows per failure, resets on success"
    stats = pc.outbox_stats()
    assert stats["queued"] == 8 and stats["delivered"] == 8 and stats["backlog"] == 0
    assert stats["deliveredLate"] == 1, "only the head ever failed; the rest waited behind it"
    assert stats["retried"] == 4 and stats["dropped"] == {}
    assert stats["backlogMax"] == 8


def test_while_anything_is_queued_new_posts_join_it_without_an_attempt(make):
    """The order rule, and the reason a planner that is down costs the caller
    nothing: behind a backlog nothing is attempted from the caller's thread."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate)
    planner.down = True
    planner.failure = TimeoutError  # the expensive kind: each attempt a timeout

    assert pc.offer_tap("ingest", {"n": 0}) == QUEUED
    before = len(planner.attempts)
    t0 = time.perf_counter()
    for n in range(1, 50):
        assert pc.offer_tap("ingest", {"n": n}) == QUEUED
    assert len(planner.attempts) == before, "not one attempt while the head waits"
    assert time.perf_counter() - t0 < 1.0
    assert pc.outbox_stats()["backlog"] == 50

    planner.down = False
    gate.release()
    assert pc.drain_outbox(5.0)
    assert [r["body"]["payload"]["n"] for r in planner.landed_where("/taps")] == list(range(50))
    assert {r["thread"] for r in planner.attempts[before:]} == {"heco-outbox"}


def test_a_refusal_is_dropped_and_never_retried(make):
    """A 4xx is the planner answering no. Sending it again gets it refused
    again, so it is counted and dropped — synchronously or from the queue."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate)

    planner.refuse = {"/taps": 422}
    assert pc.offer_tap("match", {"too": "big"}) == DROPPED
    assert len(planner.attempts) == 2, "create_run, then the ONE refused attempt"
    assert pc.outbox_stats()["queued"] == 0 and pc.outbox._thread is None

    # Behind a backlog: the planner is away, then answers 404 for the frames.
    planner.refuse = {}
    planner.down = True
    assert pc.offer_frame_records([{"seq": 1}]) == QUEUED
    assert pc.offer_frame("track", b"frame") == QUEUED
    assert pc.offer_tap("track", {"n": 1}) == QUEUED
    planner.down = False
    planner.refuse = {"/frames": 404}
    gate.release()
    assert pc.drain_outbox(5.0)

    frame_attempts = [r for r in planner.attempts if r["url"].endswith("/frames")]
    assert len(frame_attempts) == 1, "refused once, never retried"
    assert [r["url"].rsplit("/", 1)[1] for r in planner.landed[1:]] == ["frame-records", "taps"]
    stats = pc.outbox_stats()
    assert stats["dropped"] == {"rejected": 2}
    assert stats["delivered"] == 2


def test_a_refused_credential_keeps_its_bookkeeping_and_is_dropped(make):
    """401 on a best-effort message: counted in auth_failures, the token
    flagged, and dropped — a refused credential is configuration, not an
    outage, and retrying it all night buries the problem."""

    class Provider:
        rejected = 0
        forced = 0
        last_error = None

        def mark_rejected(self):
            self.rejected += 1

        def force_refresh(self):
            self.forced += 1

    planner = FakePlanner()
    provider = Provider()
    pc = make(planner, token_provider=provider)
    planner.refuse = {"/faces": 401}
    assert pc.offer_face_card("p00001", b"card") == DROPPED
    assert pc.auth_failures == 1 and provider.rejected == 1
    assert provider.forced == 0, "never refreshed inline on the best-effort path"
    assert pc.outbox_stats()["dropped"] == {"unauthorized": 1}


def test_a_message_the_planner_keeps_failing_is_dropped_so_the_queue_moves_on(make):
    """A 500 is the planner UP and failing this message. Retried like an
    outage, but only max_faults times: one message it cannot take must not
    hold every later one behind it for the rest of the run."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate)
    planner.refuse = {"/faces": 500}

    assert pc.offer_face_card("p00001", b"card") == QUEUED, "a 500 may be transient"
    assert pc.offer_tap("match", {"unique": 1}) == QUEUED
    assert pc.offer_tap("match", {"unique": 2}) == QUEUED
    gate.release()
    assert pc.drain_outbox(5.0)

    faces = [r for r in planner.attempts if r["url"].endswith("/faces")]
    assert len(faces) == pc.outbox.max_faults == 6
    assert gate.naps == [0.5, 1.0, 2.0, 4.0, 8.0], "backed off, then let go — no wait after"
    assert [r["body"]["payload"]["unique"] for r in planner.landed_where("/taps")] == [1, 2]
    assert pc.outbox_stats()["dropped"] == {"failed": 1}


def test_502_503_504_are_the_planner_away_and_never_give_up(make):
    """Unavailable is not failed: a restart behind a proxy answers 502-504,
    and those are retried for as long as the planner is away."""
    planner = FakePlanner()
    gate = Gate()
    gate.release()
    pc = make(planner, gate=gate)
    planner.fail = 20
    planner.failure = 502
    assert pc.offer_tap("match", {"unique": 1}) == QUEUED
    assert pc.drain_outbox(5.0)
    assert pc.outbox_stats()["delivered"] == 1 and pc.outbox_stats()["dropped"] == {}
    assert len(gate.naps) == 20 and max(gate.naps) == 15.0


def test_an_unencodable_payload_is_never_queued(make):
    """A tap that cannot be JSON-encoded can never be sent; queued, it would
    hold everything behind it. Dropped at the door and counted instead."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate)
    planner.down = True

    assert pc.offer_tap("match", {"score": {0.5}}) == DROPPED  # a set: not JSON
    assert pc.offer_tap("match", {"unique": 1}) == QUEUED
    assert pc.offer_tap("match", {"score": object()}) == DROPPED  # behind a backlog too
    planner.down = False
    gate.release()
    assert pc.drain_outbox(5.0)
    assert [r["body"]["payload"] for r in planner.landed_where("/taps")] == [{"unique": 1}]
    assert pc.outbox_stats()["dropped"] == {"unsendable": 2}


def test_5xx_and_timeouts_are_retried_with_capped_exponential_backoff(make):
    """Start 0.5 s, double per failure, cap 15 s — and the SAME head each time."""
    planner = FakePlanner()
    gate = Gate()
    gate.release()  # record the waits without holding the sender
    pc = make(planner, gate=gate)

    script = [503, 500, TimeoutError, 502, 429, 408, OSError]
    state = {"i": 0}
    real = planner._answer

    def scripted(record):
        i = state["i"]
        state["i"] += 1
        if i < len(script):
            planner.attempts.append(record)
            step = script[i]
            if isinstance(step, int):
                return step, {}
            raise step("planner away")
        return real(record)

    planner._answer = scripted
    assert pc.offer_tap("match", {"unique": 9}) == QUEUED
    assert pc.drain_outbox(5.0)

    assert gate.naps == [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 15.0]
    ids = [r["headers"][DELIVERY_HEADER] for r in planner.attempts[1:]]  # [0]: create_run
    assert ids == ["42:1"] * 8, "every attempt was the same message"
    stats = pc.outbox_stats()
    assert stats["retried"] == 7 and stats["delivered"] == 1 and stats["deliveredLate"] == 1


# ----------------------------------------------------------------- the bound


def test_the_byte_bound_sheds_live_ring_pictures_first(make):
    """Over the bound, a new message sheds what matters least: live-ring
    pictures first (they only feed the "latest frame" view), then forensic
    ones; keyframes, taps and face cards survive — and every drop is counted."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate, outbox_max_bytes=1_000_000)
    planner.down = True
    k = 1000

    assert pc.offer_tap("ingest", {"seq": 1}) == QUEUED
    assert gate.waiting.wait(2.0)
    assert pc.offer_frame("ingest", b"i" * 400 * k) == QUEUED  # live ring
    assert pc.offer_frame("match", b"m" * 300 * k) == QUEUED  # keyframe
    assert pc.offer_frame("track", b"t" * 400 * k) == QUEUED  # sheds the ingest frame
    assert pc.outbox_stats()["dropped"] == {"shedLiveFrame": 1}
    forensic = {"forensic": 1, "seq": 7, "tMs": 700}
    assert pc.offer_frame("ingest", b"f" * 300 * k, extra=forensic) == QUEUED  # sheds track
    assert pc.offer_face_card("p00001", b"c" * 450 * k) == QUEUED  # sheds the forensic one
    assert pc.offer_tap("match", {"unique": 1}) == QUEUED
    stats = pc.outbox_stats()
    assert stats["dropped"] == {"shedLiveFrame": 2, "shedFrame": 1}
    assert stats["bytes"] <= 1_000_000 and stats["backlog"] == 4

    planner.down = False
    gate.release()
    assert pc.drain_outbox(5.0)
    kept = [r["url"].rsplit("/", 1)[1] for r in planner.landed[1:]]
    assert kept == ["taps", "frames", "faces", "taps"]
    assert planner.landed[2]["fields"]["stage"] == "match"


def test_past_the_pictures_the_oldest_messages_go_and_an_oversized_one_never_fits(make):
    """With no picture left to shed, the oldest messages make room (never the
    run end); a message larger than the whole bound is dropped itself."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate, outbox_max_bytes=1_000_000)
    planner.down = True
    k = 1000

    assert pc.offer_tap("ingest", {"seq": 1}) == QUEUED
    assert gate.waiting.wait(2.0)
    assert pc.offer_frame("match", b"m" * 300 * k) == QUEUED
    assert pc.offer_face_card("p00001", b"c" * 300 * k) == QUEUED
    assert pc.offer_face_card("p00002", b"d" * 600 * k) == QUEUED  # sheds tap + keyframe
    assert pc.outbox_stats()["dropped"] == {"shedOldest": 2}
    assert pc.offer_frame("match", b"x" * 1_000_001) == DROPPED
    stats = pc.outbox_stats()
    assert stats["dropped"] == {"shedOldest": 2, "overflow": 1}
    assert stats["backlog"] == 2

    planner.down = False
    gate.release()
    assert pc.drain_outbox(5.0)
    assert [r["fields"]["personKey"] for r in planner.landed_where("/faces")] == [
        "p00001", "p00002",
    ]


def test_a_stale_live_frame_is_skipped_at_delivery(make):
    """A live-ring picture delivered late would put an old picture over the
    live view, so past stale_frame_s it is skipped — counted. Keyframes,
    forensic frames and taps are delivered whatever their age."""
    planner = FakePlanner()
    gate = Gate()
    now = [1000.0]
    pc = make(planner, gate=gate, clock=lambda: now[0])
    planner.down = True

    assert pc.offer_frame("ingest", b"old-live") == QUEUED
    assert gate.waiting.wait(2.0)
    assert pc.offer_frame("match", b"old-keyframe") == QUEUED
    assert pc.offer_frame("ingest", b"old-forensic", extra={"forensic": 1}) == QUEUED
    assert pc.offer_tap("match", {"unique": 3}) == QUEUED
    now[0] += 20.0
    assert pc.offer_frame("track", b"young-live") == QUEUED
    now[0] += 11.0  # the first four are 31 s old, the last 11 s

    planner.down = False
    gate.release()
    assert pc.drain_outbox(5.0)
    assert [r.get("data") for r in planner.landed[1:]] == [
        b"old-keyframe", b"old-forensic", None, b"young-live",
    ]
    assert pc.outbox_stats()["dropped"] == {"stale": 1}


# ------------------------------------------------------------------ run end


def test_end_run_drains_the_queue_before_it_closes_the_run(make):
    """The run end is the last word: everything the run reported lands first."""
    planner = FakePlanner()
    pc = make(planner, wall_clock=wall_ticks())
    planner.fail = 3  # the first tap's attempt and two of the sender's

    for n in range(5):
        pc.post_tap("match", {"n": n})
    pc.end_run("ended", notes="done", results={"unique": 5.0})

    posts = planner.landed[1:]
    assert [r["method"] for r in posts] == ["POST"] * 5 + ["PUT"]
    assert [r["body"]["payload"]["n"] for r in posts[:5]] == list(range(5))
    put = posts[-1]
    assert put["body"] == {"status": "ended", "notes": "done", "results": {"unique": 5.0}}
    assert put["headers"][DELIVERY_HEADER] == "42:6", "the run end is numbered too"
    assert pc.outbox_stats()["backlog"] == 0


def test_a_failing_run_end_is_queued_and_delivered_later_by_the_sender(make):
    """The planner is still down when the run ends. The PUT keeps its retrying
    path and still raises — but it is kept, and the sender delivers it once the
    planner answers, after the run loop has long returned."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate, drain_timeout_s=0.2, wall_clock=wall_ticks())
    planner.down = True
    planner.failure = 503

    pc.post_tap("match", {"unique": 7})
    with pytest.raises(PlannerDeferred) as err:
        pc.end_run("ended", notes="counted 7")
    assert isinstance(err.value, PlannerError), "an old caller still catches it"
    assert err.value.retryable and err.value.status == 503
    stats = pc.outbox_stats()
    assert stats["backlog"] == 2, "the tap, then the run end behind it"

    planner.down = False
    gate.release()
    assert pc.drain_outbox(5.0)
    posts = planner.landed[1:]
    assert [r["method"] for r in posts] == ["POST", "PUT"]
    assert posts[1]["body"] == {"status": "ended", "notes": "counted 7"}
    puts = [r for r in planner.attempts if r["method"] == "PUT"]
    assert len(puts) == 4, "three on the retrying path, then the sender's"
    assert len({(r["headers"][DELIVERY_HEADER], r["headers"][AT_HEADER]) for r in puts}) == 1


def test_end_run_can_be_told_the_caller_already_drained(make):
    """A caller that drained the outbox itself (to count what it delivered)
    passes 0, so a planner outage does not cost its settle the wait twice."""
    planner = FakePlanner()
    pc = make(planner, drain_timeout_s=7.0)
    waits: list[float] = []
    real = pc.outbox.drain
    pc.outbox.drain = lambda timeout: (waits.append(timeout), real(timeout))[1]
    pc.end_run("ended")
    pc.end_run("ended", drain_timeout_s=0.0)
    assert waits == [7.0, 0.0]


def test_a_refused_run_end_is_not_queued(make):
    """A 4xx would only be refused again: raised as before, and not kept."""
    planner = FakePlanner()
    pc = make(planner)
    planner.refuse = {"/api/pipeline/runs/42": 409}
    with pytest.raises(PlannerError) as err:
        pc.end_run("ended")
    assert not isinstance(err.value, PlannerDeferred)
    assert err.value.retryable is False and err.value.status == 409
    assert pc.outbox_stats()["queued"] == 0


def test_a_queued_run_end_keeps_its_paths_one_refresh_on_a_401(make):
    """The run end came off the RETRYING path and keeps that path's 401 rule
    in the background: one forced token refresh, one more try."""

    class Provider:
        forced = 0
        rejected = 0
        last_error = None

        def force_refresh(self):
            self.forced += 1

        def mark_rejected(self):
            self.rejected += 1

    planner = FakePlanner()
    gate = Gate()
    provider = Provider()
    pc = make(planner, gate=gate, drain_timeout_s=0.1, token_provider=provider)
    planner.down = True
    with pytest.raises(PlannerDeferred):
        pc.end_run("ended")
    forced_before = provider.forced

    planner.down = False
    planner.fail = 1
    planner.failure = 401
    gate.release()
    assert pc.drain_outbox(5.0)
    assert provider.forced == forced_before + 1
    assert [r["method"] for r in planner.landed[1:]] == ["PUT"]


def test_queued_messages_keep_the_run_they_were_made_for(make):
    """A later create_run on the same client must not re-aim a message queued
    for the previous run: the URL and the delivery id are fixed at the call."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate)
    planner.refuse = {"/taps": 503}  # the taps endpoint is away, runs are not

    assert pc.offer_tap("match", {"run": "first"}) == QUEUED
    assert pc.create_run(4) == 43
    assert pc.offer_tap("match", {"run": "second"}) == QUEUED

    planner.refuse = {}
    gate.release()
    assert pc.drain_outbox(5.0)
    taps = planner.landed_where("/taps")
    assert [(r["url"], r["headers"][DELIVERY_HEADER]) for r in taps] == [
        ("http://planner:8787/api/pipeline/runs/42/taps", "42:1"),
        ("http://planner:8787/api/pipeline/runs/43/taps", "43:2"),
    ]


def test_frame_records_queue_one_message_per_chunk(make):
    """Each chunk is its own message with its own delivery id; a batch whose
    first chunk found the planner away is QUEUED whole — do not re-send it."""
    planner = FakePlanner()
    gate = Gate()
    pc = make(planner, gate=gate)
    planner.down = True
    records = [{"seq": i} for i in range(1200)]
    assert pc.offer_frame_records(records) == QUEUED
    assert pc.outbox_stats()["backlog"] == 3

    planner.down = False
    gate.release()
    assert pc.drain_outbox(5.0)
    chunks = planner.landed_where("/frame-records")
    assert [len(r["body"]["records"]) for r in chunks] == [500, 500, 200]
    assert [s["seq"] for r in chunks for s in r["body"]["records"]] == list(range(1200))
    assert delivery_numbers(chunks) == [1, 2, 3]


# ------------------------------------------------------------ the sender


def test_stop_ends_the_sender_and_the_next_post_resumes_delivery():
    """stop() interrupts a backoff wait (not just the next one), keeps the
    queue, and a later put starts a fresh sender that delivers it in order."""
    sent: list[str] = []
    away = threading.Event()
    away.set()

    def send(msg: OutboxMessage) -> str:
        if away.is_set():
            return RETRY
        sent.append(msg.url)
        return SENT

    box = Outbox(send, backoff_s=30.0)  # only an interrupted wait ends in time
    assert box.put(OutboxMessage(kind="tap", url="/a", headers={}, created=0.0), failed=True)
    sender = box._thread
    assert sender is not None and sender.is_alive()

    t0 = time.perf_counter()
    assert box.stop(timeout=5.0) is True
    assert time.perf_counter() - t0 < 5.0
    assert not sender.is_alive() and box._thread is None
    assert box.stats()["backlog"] == 1, "stopping keeps the queue"

    away.clear()
    box._backoff = 0.0  # the planner is back; skip the 30 s wait the test set up
    assert box.put(OutboxMessage(kind="tap", url="/b", headers={}, created=0.0))
    assert box.drain(5.0) is True
    assert sent == ["/a", "/b"]
    assert box.stop() is True


def test_a_sender_that_meets_an_exception_never_strands_the_queue():
    """send() raising is "the planner is away": retried, never a dead thread."""
    calls = {"n": 0}

    def send(msg: OutboxMessage) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transport adapter bug")
        return SENT

    box = Outbox(send, sleep=lambda _s: None)
    assert box.put(OutboxMessage(kind="tap", url="/a", headers={}, created=0.0))
    assert box.drain(5.0) is True
    assert box.stats()["delivered"] == 1 and box.stats()["retried"] == 2


def test_a_sender_that_cannot_start_strands_nothing(monkeypatch):
    """Out of threads: the message is still kept, and the next drain (or put)
    starts the sender — an unstarted thread must never pass for a running one."""
    real_start = threading.Thread.start
    starts = {"n": 0}

    def first_start_fails(self):
        starts["n"] += 1
        if starts["n"] == 1:
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", first_start_fails)
    sent: list[str] = []
    box = Outbox(lambda msg: sent.append(msg.url) or SENT, sleep=lambda _s: None)
    assert box.put(OutboxMessage(kind="tap", url="/a", headers={}, created=0.0)) is True
    assert box._thread is None and box.stats()["backlog"] == 1
    assert box.drain(5.0) is True
    assert sent == ["/a"]


def test_the_sender_exits_when_idle_so_a_client_holds_no_thread():
    """One client per run, many runs per process: an idle outbox holds nothing."""
    box = Outbox(lambda msg: SENT, sleep=lambda _s: None)
    box.put(OutboxMessage(kind="tap", url="/a", headers={}, created=0.0))
    assert box.drain(5.0)
    deadline = time.monotonic() + 2.0
    while box._thread is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert box._thread is None


def test_concurrent_posting_loses_nothing_and_duplicates_nothing(make):
    """The loop thread and the reporter thread both post. Several threads at
    once against a planner that fails a third of all attempts: every message
    lands exactly once, and each thread's messages in that thread's order."""
    planner = FakePlanner()
    planner.flaky = random.Random(20260929)
    planner.flaky_p = 0.33
    pc = make(planner, outbox_stale_frame_s=None)
    n_per = 60

    def post_taps(t: int) -> None:
        for i in range(n_per):
            planner.failure = OSError if i % 2 else 503
            pc.post_tap("match", {"t": t, "i": i})

    def post_frames(t: int) -> None:
        for i in range(n_per):
            pc.post_frame("track", f"{t}:{i}".encode())

    threads = [threading.Thread(target=post_taps, args=(t,)) for t in range(3)]
    threads += [threading.Thread(target=post_frames, args=(t,)) for t in range(3, 5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(30.0)
    assert pc.drain_outbox(30.0) is True

    taps = [r["body"]["payload"] for r in planner.landed_where("/taps")]
    frames = [r["data"].decode().split(":") for r in planner.landed_where("/frames")]
    assert len(taps) == 3 * n_per and len(frames) == 2 * n_per, "nothing lost"
    assert len({(p["t"], p["i"]) for p in taps}) == 3 * n_per, "nothing duplicated"
    assert len({tuple(f) for f in frames}) == 2 * n_per
    for t in range(3):
        assert [p["i"] for p in taps if p["t"] == t] == list(range(n_per)), "thread order"
    for t in range(3, 5):
        assert [int(i) for tt, i in frames if int(tt) == t] == list(range(n_per))
    landed_ids = [r["headers"][DELIVERY_HEADER] for r in planner.landed[1:]]
    assert len(set(landed_ids)) == len(landed_ids) == 5 * n_per
    stats = pc.outbox_stats()
    assert stats["dropped"] == {} and stats["backlog"] == 0
    assert stats["queued"] > 0, "the flaky planner really did send work to the queue"


# ------------------------------------------------------------- the wire


def test_the_retrying_path_says_whether_the_planner_was_away():
    """PlannerError carries status and retryable, so end_run can tell an
    outage (worth queueing) from a refusal (not)."""
    planner = FakePlanner()
    pc = PlannerClient("http://planner:8787", transport=planner.json, sleep=lambda _s: None)
    planner.down = True
    planner.failure = 503
    with pytest.raises(PlannerError) as err:
        pc.create_run(1)
    assert (err.value.status, err.value.retryable) == (503, True)

    planner.failure = OSError
    with pytest.raises(PlannerError) as err:
        pc.create_run(1)
    assert (err.value.status, err.value.retryable) == (None, True)

    planner.down = False
    for status, retryable in ((429, True), (408, True), (422, False), (404, False)):
        planner.refuse = {"/api/pipeline/runs": status}
        with pytest.raises(PlannerError) as err:
            pc.create_run(1)
        assert (err.value.status, err.value.retryable) == (status, retryable)


def test_the_stdlib_transports_carry_the_durable_headers(monkeypatch):
    """The urllib default and both credential wrappers put the stamp on the wire."""
    import heco_common.planner as planner_module

    seen: list[dict] = []

    class Reply:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        seen.append({k.lower(): v for k, v in req.header_items()})
        return Reply()

    monkeypatch.setattr(planner_module.urllib.request, "urlopen", fake_urlopen)
    stamp = {DELIVERY_HEADER: "42:1", AT_HEADER: "2026-09-29T02:21:05.123Z"}

    class Provider:
        def auth_header(self, *, block=False):
            return {"Authorization": "Bearer minted"}

    planner_module.urllib_transport("POST", "http://p/x", {"a": 1}, headers=stamp)
    planner_module.bearer_urllib_transport("static")("POST", "http://p/x", {}, headers=stamp)
    planner_module.provider_urllib_transport(Provider())("POST", "http://p/x", {}, headers=stamp)
    planner_module.urllib_transport("POST", "http://p/x", {"a": 1})

    for headers in seen[:3]:
        assert headers["x-heco-delivery"] == "42:1"
        assert headers["x-heco-at"] == "2026-09-29T02:21:05.123Z"
    assert seen[1]["authorization"] == "Bearer static"
    assert seen[2]["authorization"] == "Bearer minted"
    assert "x-heco-delivery" not in seen[3], "no stamp unless a durable client sends one"
