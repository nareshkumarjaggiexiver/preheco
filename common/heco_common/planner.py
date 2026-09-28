"""PlannerClient — the pipeline's write side into the site-planner app.

Wraps the four planner ingest endpoints (CONTRACTS.md, mirrored in the
planner's api.js):

    POST /api/pipeline/runs                {eventId, placementId?, label?, config?}
    PUT  /api/pipeline/runs/:id            {status: ended|failed, notes?}
    POST /api/pipeline/runs/:id/stats      {stage, frames, fps, metrics}
    POST /api/pipeline/runs/:id/samples    {samples:[...]} (batch <= 200)

Design points:

* **Injectable transport.** The client speaks through a ``Transport``
  callable ``(method, url, payload) -> (status, body)``. Tests inject a fake;
  production uses the stdlib urllib default — no HTTP library dependency.
* **Retry.** Transport exceptions and 5xx responses retry with exponential
  backoff (``retries`` attempts total); 4xx fail immediately — the payload is
  wrong and retrying will not fix it.
* **Batching.** ``add_sample`` buffers locally; the buffer flushes when it
  reaches ``batch_size`` and always on ``end_run``. Explicit lists posted via
  ``post_samples`` are chunked to the contract's 200-sample cap.

v1 (2026-08-04) adds the best-effort debug/feedback surface: ``post_tap`` and
``post_frame`` (annotated JPEG, multipart) push what each stage produced;
``poll_feedback``/``resolve_feedback`` drive the operator-correction loop; and
``report_enrolment`` confirms a staff enrolment. The taps and feedback calls
are single-shot and swallow failures — CONTRACTS.md requires they never block
or crash the run loop when the planner hiccups.

* **Two transports, because "best effort" is also about time.** Swallowing
  errors stops a hiccup CRASHING the loop; it does nothing about a planner
  that accepts connections and then answers slowly (SQLite lock, event-loop
  stall), which stalls the loop just as effectively and loses every frame the
  drop-not-queue ingest slot overwrites meanwhile. So the single-shot calls
  speak through ``best_effort_transport`` — the caller wires that to a client
  with a much SHORTER timeout than the retrying one. It defaults to
  ``transport``, so a caller that does not care keeps the old behaviour.

* **Durable best-effort writes (opt-in: ``durable=True``).** Swallowing a
  failed tap protected the loop and lost the record: the planner's restart
  dropped every tap and frame the runner posted meanwhile, and runs were
  marked failed "the runner went silent". A durable client still makes ONE
  synchronous attempt exactly as before — same transport, same timeout — but
  a failure that means "the planner is away" (no answer, a timeout,
  502/503/504, 408, 429) parks the message in an in-memory :class:`Outbox`,
  and one background thread delivers the queue IN ORDER once the planner
  answers. While anything is queued, new messages join the queue without an
  attempt: that keeps the order, and it means a planner that is down costs
  the caller nothing at all instead of a timeout per post. (Any other 5xx is
  the planner up and failing THIS message: retried too, but not forever — see
  :data:`FAULT`.) Every durable message carries ``X-Heco-Delivery``
  (``<runId>:<n>``, unique per message) and ``X-Heco-At`` (when the call was
  made), identical on every attempt, so the planner can drop a duplicate and
  date a late arrival. Off, nothing changes: same calls, same transports, no
  headers, no thread.

The client is intentionally synchronous: the runner reports stats every ~2 s,
which never justifies an async dependency at POC scale.
"""

import contextlib
import inspect
import itertools
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .schemas import MetricSummary, PlannerRunCreate, PlannerRunEnd, Sample, StageStats

if TYPE_CHECKING:  # pragma: no cover — typing only
    # Imported for the annotation alone. This client needs three methods from a
    # provider (auth_header, force_refresh, mark_rejected) and nothing else, so
    # it stays structurally typed: a test double is a provider if it behaves
    # like one, with no base class to inherit.
    from .auth import TokenProvider

#: Contract cap on samples per POST.
MAX_SAMPLES_PER_POST = 200

#: (method, url, json-payload-or-None) -> (status_code, decoded-json-body).
#: A DURABLE client also passes ``headers=`` (a dict) on every call it makes
#: through a transport; a non-durable client never passes the keyword, so a
#: three-argument transport keeps working there unchanged.
Transport = Callable[[str, str, dict | None], tuple[int, dict]]

#: (url, form-fields, filename, file-bytes, content-type) -> (status, body).
#: Separate from ``Transport`` because debug frames are multipart, not JSON.
#: The same ``headers=`` rule applies.
FileTransport = Callable[[str, dict, str, bytes, str], tuple[int, dict]]

#: What became of one best-effort message (the ``offer_*`` methods): the
#: planner took it now, a durable client's outbox holds it for later, or it is
#: gone (refused, shed, or never sendable). ``post_*`` is ``offer_* == ACCEPTED``.
ACCEPTED = "accepted"
QUEUED = "queued"
DROPPED = "dropped"
#: A batch answers with its worst part: one dropped chunk makes it DROPPED.
_OUTCOME_RANK = {ACCEPTED: 0, QUEUED: 1, DROPPED: 2}

#: A durable client's default outbox bound, in PAYLOAD bytes: JPEG bytes for a
#: picture, JSON length for everything else. The Python objects holding a JSON
#: payload cost more than its wire length; pictures, which dominate a backlog,
#: are held as the bytes themselves.
OUTBOX_MAX_BYTES = 512 * 1024 * 1024

#: The two headers on every durable message, identical on every attempt.
DELIVERY_HEADER = "X-Heco-Delivery"
AT_HEADER = "X-Heco-At"

# Outbox message kinds. A "frame" is post_frame's multipart picture; the run
# end is the one retrying-path call a durable client queues (see end_run).
TAP = "tap"
FRAME = "frame"
FRAME_RECORDS = "frame-records"
FACE_CARD = "face-card"
RUN_END = "run-end"

# One delivery attempt's verdict (Outbox's ``send``). RETRY is the planner
# AWAY — no answer, a timeout, 502/503/504, 408, 429 — and is retried for as
# long as it takes. FAULT is the planner ANSWERING with a server error (500,
# 501, 505+): retried the same way, but only ``max_faults`` times, because a
# message the planner is up and cannot take would otherwise hold every later
# one behind it for the rest of the run — a total reporting outage made out of
# one bad message, which is worse than the loss this exists to prevent.
# Anything else is a DROP REASON ("rejected", "unauthorized").
SENT = "sent"
RETRY = "retry"
FAULT = "fault"

#: Statuses that mean "come back later" rather than "this failed": the
#: planner (or a proxy in front of it) unavailable, overloaded or timed out.
_AWAY_STATUSES = frozenset({408, 429, 502, 503, 504})


class PlannerError(RuntimeError):
    """Raised when a planner call fails after all retries (or on 4xx).

    ``status`` is the last HTTP status seen (None when the planner never
    answered), and ``retryable`` says whether the failure was the planner
    being AWAY — transport errors, timeouts, 5xx, 408, 429 — rather than it
    refusing the request. Only the first kind is worth handing to an outbox:
    sending a refused request again gets it refused again.
    """

    def __init__(
        self, message: str, *, status: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class PlannerDeferred(PlannerError):
    """A durable client could not deliver the run end now, and QUEUED it.

    Still a :class:`PlannerError`, so a caller that only knows the old
    contract keeps working (it reports a planner that was not told). A caller
    that knows this one can say the truer thing: the end will land, with its
    original ``X-Heco-At``, as soon as the planner answers again.
    """


def _retryable(status: int) -> bool:
    """Is this status the planner being away rather than refusing? 5xx, 408, 429."""
    return status >= 500 or status in (408, 429)


def _iso_utc(epoch_s: float) -> str:
    """ISO-8601 UTC with milliseconds and a Z, e.g. ``2026-09-29T02:21:05.123Z``."""
    stamp = datetime.fromtimestamp(epoch_s, UTC).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def _accepts_headers(fn: Callable) -> bool:
    """Can ``fn`` take a ``headers=`` keyword? An unreadable signature is trusted."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD or (p.name == "headers" and p.kind in named)
        for p in params
    )


def bearer_urllib_transport(token: str) -> "Transport":
    """A stdlib transport that carries ``Authorization: Bearer <token>``.

    The planner refuses to listen beyond loopback without a token, so any
    deployment where the runner is not on the same host needs this. Used
    automatically when a PlannerClient is given a token and no explicit
    transport (the httpx path sets the header on its client instead).
    """

    def transport(
        method: str, url: str, payload: dict | None, headers: dict | None = None
    ) -> tuple[int, dict]:
        return urllib_transport(method, url, payload, token=token, extra_headers=headers)

    return transport


def provider_urllib_transport(provider: "TokenProvider") -> "Transport":
    """A stdlib transport that asks the provider for a header on EVERY call.

    The predecessor computed ``Authorization`` once, when the client was built,
    which is precisely why rotating the credential meant restarting the runner.
    One line moved; a whole class of restart disappeared.
    """

    def transport(
        method: str, url: str, payload: dict | None, headers: dict | None = None
    ) -> tuple[int, dict]:
        header = provider.auth_header(block=True)
        return urllib_transport(method, url, payload, extra_headers={**header, **(headers or {})})

    return transport


def urllib_transport(
    method: str,
    url: str,
    payload: dict | None,
    token: str | None = None,
    extra_headers: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    """Default stdlib transport: JSON in, JSON out, 10 s timeout.

    HTTP error statuses are returned (not raised) so the retry policy in
    PlannerClient owns the decision; network-level failures raise.
    ``headers`` is the durable client's keyword (X-Heco-Delivery/X-Heco-At);
    ``extra_headers`` is the credential wrappers' — both simply ride along.
    """
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            **({"Content-Type": "application/json"} if body else {}),
            **({"Authorization": f"Bearer {token}"} if token else {}),
            **(extra_headers or {}),
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            raw = res.read()
            return res.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"detail": raw.decode(errors="replace")}
        return exc.code, parsed


@dataclass(eq=False)
class OutboxMessage:
    """One durable write: everything needed to send it again exactly as first made.

    The URL is captured when the CALL is made, so a later ``create_run`` on the
    same client cannot re-aim a message queued for the previous run, and the
    headers are stamped then too, so every attempt says the same thing about
    when and which. Compared by identity (``eq=False``): two taps with equal
    bodies are still two messages.
    """

    kind: str
    url: str
    headers: dict[str, str]
    #: The outbox clock when the call was made — how a picture's age is read.
    created: float
    method: str = "POST"
    #: JSON kinds.
    payload: dict | None = None
    #: Multipart kinds: form fields, and the file.
    fields: dict | None = None
    filename: str | None = None
    data: bytes | None = None
    content_type: str | None = None
    stage: str | None = None
    forensic: bool = False
    #: Bytes held against the outbox bound (see :meth:`measure`).
    size: int = 0
    #: Failed attempts so far, the synchronous first one included.
    failures: int = 0
    #: Of those, the ones the planner ANSWERED with a server error (FAULT).
    faults: int = 0
    #: A queued run end has spent its one forced token refresh (see _attempt).
    refreshed: bool = False

    @property
    def stage_frame(self) -> bool:
        """A picture that is not a keyframe: any frame whose stage is not match."""
        return self.kind == FRAME and self.stage != "match"

    @property
    def live_frame(self) -> bool:
        """A picture that only feeds the planner's live "latest frame" ring.

        Not a match frame (the planner keeps a KEYFRAME from those) and not a
        forensic one (the bench copy of one frame's pixels, joined to its
        ledger row). Late, it is worse than useless: it would overwrite the
        live view with an old picture.
        """
        return self.stage_frame and not self.forensic

    def measure(self) -> int:
        """Bytes this message holds against the bound: JPEG bytes, or JSON length.

        Raises TypeError/ValueError for a JSON payload that cannot be encoded
        (a numpy float in a tap, say): a message that can never be sent, and
        must never be queued to hold every later one behind it.
        """
        if self.data is not None:
            fields = sum(len(str(k)) + len(str(v)) for k, v in (self.fields or {}).items())
            return len(self.data) + fields
        return len(json.dumps(self.payload))


#: What the byte bound sheds, cheapest first: (drop reason, which messages).
#: A live-ring picture only feeds the "latest frame" view, a forensic picture
#: is a bench copy, everything else is the record. The run end is never shed —
#: it is tiny, and it is the one write that closes the run.
_SHED_ORDER = (
    ("shedLiveFrame", lambda m: m.live_frame),
    ("shedFrame", lambda m: m.stage_frame),
    ("shedOldest", lambda m: m.kind != RUN_END),
)


class Outbox:
    """A bounded, ordered, in-memory delivery queue with ONE sender thread.

    ``send(msg)`` makes one attempt and returns :data:`SENT`, :data:`RETRY`
    (the planner is away), :data:`FAULT` (it answered with a server error) or
    a drop reason. It belongs to the client, so this class knows nothing about
    HTTP and is testable with a plain function.

    THE ORDER RULE. The head is retried until it lands or is dropped; nothing
    overtakes it. A message counts as queued from :meth:`put` until it is
    settled — including while it is in flight — so :meth:`busy` is true for
    exactly as long as a new message must queue behind it rather than race it.
    The one exception to "until it lands" is a message the planner keeps
    answering with a server error: after ``max_faults`` such answers it is
    dropped as ``failed`` — the planner is up, and a message it cannot take
    must not hold every later one behind it for the rest of the run.

    THE SENDER is started when there is work and exits when the queue is
    empty, so an idle client holds no thread (the runner builds one client per
    run, and a thread per run ever started would be a leak with a nicer name).
    It never raises: an exception in an attempt is a retry, anything else ends
    the thread cleanly and the next :meth:`put` starts another.

    BOUNDED. ``max_bytes`` counts payload bytes. Over it, the new message
    sheds queued live-ring pictures oldest first, then any non-match picture,
    then the oldest messages of any kind but the run end; a message that
    cannot fit even so is dropped itself. At delivery a live-ring picture older
    than ``stale_frame_s`` is skipped (``None`` never skips). Every drop is
    counted by reason — a bound that loses things silently is how a gap in the
    record gets read as a quiet night.
    """

    def __init__(
        self,
        send: Callable[[OutboxMessage], str],
        *,
        max_bytes: int = OUTBOX_MAX_BYTES,
        stale_frame_s: float | None = 30.0,
        backoff_s: float = 0.5,
        backoff_factor: float = 2.0,
        backoff_max_s: float = 15.0,
        max_faults: int = 6,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        name: str = "heco-outbox",
    ) -> None:
        """Configure an empty outbox; no thread starts until something is put.

        ``max_faults`` answers of "server error" to the same message end it
        (six is ~15 s of backoff: a transient hiccup passes, a bug does not).
        ``sleep`` replaces the backoff wait (tests inject one; the default
        waits on the sender's stop event, so :meth:`stop` cuts a backoff
        short). ``clock`` dates messages for the staleness rule.
        """
        self._send = send
        self.max_bytes = max(0, int(max_bytes))
        self.stale_frame_s = stale_frame_s
        self.backoff_s = backoff_s
        self.backoff_factor = backoff_factor
        self.backoff_max_s = backoff_max_s
        self.max_faults = max(1, int(max_faults))
        self._sleep = sleep
        self._clock = clock
        self.name = name
        self._lock = threading.Lock()
        # Signalled whenever a message is settled or shed, and when the sender
        # exits — what drain() waits on.
        self._changed = threading.Condition(self._lock)
        self._queue: deque[OutboxMessage] = deque()
        self._in_flight: OutboxMessage | None = None
        self._thread: threading.Thread | None = None
        self._halt: threading.Event | None = None
        # A put() that found the sender stopping: the stopping sender hands
        # over to a fresh one on its way out instead of stranding the queue.
        self._restart = False
        self._backoff = 0.0
        self._bytes = 0
        self._queued = 0
        self._delivered = 0
        self._delivered_late = 0
        self._retried = 0
        self._dropped: dict[str, int] = {}
        self._backlog_max = 0
        self._bytes_max = 0

    # ------------------------------------------------------------ producers

    def busy(self) -> bool:
        """True while anything is queued or in flight: new messages must queue."""
        with self._lock:
            return bool(self._queue)

    def put(self, msg: OutboxMessage, *, failed: bool = False) -> bool:
        """Queue ``msg`` (making room if needed); False when it had to be dropped.

        ``failed`` says the caller already tried it once and it failed (the
        planner away, or a FAULT already counted in ``msg.faults``): it counts
        as a retry and grows the backoff, so the sender does not hammer a
        planner that has just refused a connection.
        """
        with self._lock:
            if failed:
                msg.failures += 1
                if msg.faults >= self.max_faults:
                    self._drop("failed")
                    return False
                self._retried += 1
                self._backoff = self._next_backoff()
            if not self._make_room(msg.size):
                self._drop("overflow")
                return False
            self._queue.append(msg)
            self._bytes += msg.size
            self._queued += 1
            self._backlog_max = max(self._backlog_max, len(self._queue))
            self._bytes_max = max(self._bytes_max, self._bytes)
            self._ensure_sender()
            return True

    def count_drop(self, reason: str) -> None:
        """Count a message dropped before it ever reached the queue."""
        with self._lock:
            self._drop(reason)

    # -------------------------------------------------------------- control

    def drain(self, timeout: float | None) -> bool:
        """Wait until everything queued is settled; True when the queue is empty.

        Starts the sender if it is not running (after :meth:`stop`, say).
        ``timeout`` is wall seconds; ``None`` waits for as long as it takes.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._lock:
            while self._queue:
                self._ensure_sender()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._changed.wait(remaining)
            return True

    def start(self) -> None:
        """Start the sender if there is work and none is running."""
        with self._lock:
            self._ensure_sender()

    def stop(self, timeout: float | None = 5.0) -> bool:
        """Stop the sender and wait for it; True once it has exited.

        The queue is kept: delivery resumes on the next :meth:`put`,
        :meth:`drain` or :meth:`start`. An attempt already in flight finishes
        (and is settled) first — a transport call cannot be interrupted.
        """
        with self._lock:
            thread, halt = self._thread, self._halt
            if thread is None:
                return True
            halt.set()
            self._restart = False
        if thread is not threading.current_thread():
            thread.join(timeout)
        return not thread.is_alive()

    def stats(self) -> dict:
        """Counters and the backlog now, as one consistent snapshot.

        ``queued`` counts messages that entered the queue; each one ends
        ``delivered`` (``deliveredLate`` when at least one attempt at it had
        failed first), dropped, or still in ``backlog``. ``dropped`` also
        counts messages refused on their first synchronous attempt, which
        never entered the queue. ``retried`` counts attempts that failed with
        the planner away. ``backlogMax``/``bytesMax`` are high-water marks.
        """
        with self._lock:
            return {
                "queued": self._queued,
                "delivered": self._delivered,
                "deliveredLate": self._delivered_late,
                "retried": self._retried,
                "dropped": dict(self._dropped),
                "backlog": len(self._queue),
                "backlogMax": self._backlog_max,
                "bytes": self._bytes,
                "bytesMax": self._bytes_max,
                "bytesLimit": self.max_bytes,
            }

    # --------------------------------------------------------------- sender

    def _run(self, halt: threading.Event) -> None:
        """Deliver the head until the queue is empty or stop() is called.

        Never raises: a thread that dies here strands the queue with nobody
        to notice, which is exactly the silent loss this class exists to end.
        """
        me = threading.current_thread()
        flying: OutboxMessage | None = None
        try:
            while not halt.is_set():
                with self._lock:
                    if not self._queue:
                        # Decided under the lock that saw the queue empty, so a
                        # put() racing this exit starts a fresh sender instead
                        # of trusting one that is leaving.
                        self._retire(me)
                        return
                    delay = self._backoff
                if delay > 0:
                    self._wait(delay, halt)
                    if halt.is_set():
                        return
                with self._lock:
                    flying = self._head()
                    if flying is None:
                        self._retire(me)
                        return
                    self._in_flight = flying
                try:
                    verdict = self._send(flying)
                except Exception:  # noqa: BLE001 — no answer is "the planner is away"
                    verdict = RETRY
                with self._lock:
                    self._in_flight = None
                    self._settle(flying, verdict)
                    flying = None
        except Exception:  # noqa: BLE001 — see the docstring; the finally cleans up
            pass
        finally:
            with self._lock:
                # Only this sender's own marker: a successor may already be
                # sending, and clearing ITS marker would let the bound shed the
                # message it is holding.
                if flying is not None and self._in_flight is flying:
                    self._in_flight = None
                self._retire(me)

    def _wait(self, delay: float, halt: threading.Event) -> None:
        """Back off before the next attempt; stop() cuts the default wait short."""
        if self._sleep is not None:
            self._sleep(delay)
        else:
            halt.wait(delay)

    def _head(self) -> OutboxMessage | None:
        """The message to attempt next, skipping live-ring pictures gone stale.

        Lock held. A stale live picture is dropped rather than delivered: it
        only ever feeds the planner's "latest frame" view, and replaying it late
        would put an old picture over the live one.
        """
        now = self._clock()
        while self._queue:
            msg = self._queue[0]
            if (
                msg.live_frame
                and self.stale_frame_s is not None
                and now - msg.created > self.stale_frame_s
            ):
                self._queue.popleft()
                self._bytes -= msg.size
                self._drop("stale")
                self._changed.notify_all()
                continue
            return msg
        return None

    def _settle(self, msg: OutboxMessage, verdict: str) -> None:
        """Record what one attempt at the head did. Lock held."""
        if verdict == SENT:
            self._remove(msg)
            self._delivered += 1
            if msg.failures:
                self._delivered_late += 1
            self._backoff = 0.0
            self._changed.notify_all()
            return
        msg.failures += 1
        if verdict == FAULT:
            msg.faults += 1
        if verdict == RETRY or (verdict == FAULT and msg.faults < self.max_faults):
            self._retried += 1
            self._backoff = self._next_backoff()
        else:
            # The planner ANSWERED and refused, or kept failing this one
            # message: it is up, so no backoff — and the same request would
            # only fail again, so no retry. The queue behind it moves on.
            self._remove(msg)
            self._drop("failed" if verdict == FAULT else verdict)
            self._backoff = 0.0
        self._changed.notify_all()

    def _retire(self, me: threading.Thread) -> None:
        """The sender ``me`` is leaving; hand over if work arrived meanwhile. Lock held."""
        if self._thread is me:
            self._thread = None
            self._halt = None
            if self._restart:
                self._restart = False
                self._ensure_sender()
        if not self._queue:
            self._backoff = 0.0
        self._changed.notify_all()

    # ------------------------------------------------------------- internals

    def _ensure_sender(self) -> None:
        """Start a sender when there is work and none is running. Lock held."""
        if not self._queue:
            return
        if self._thread is None:
            halt = threading.Event()
            thread = threading.Thread(target=self._run, args=(halt,), name=self.name, daemon=True)
            self._thread, self._halt = thread, halt
            try:
                thread.start()
            except RuntimeError:
                # Out of threads. Left set, an unstarted thread would read as a
                # running sender and strand the queue for good; cleared, the
                # next put() or drain() tries again, and the message is kept.
                self._thread = self._halt = None
        elif self._halt is not None and self._halt.is_set():
            self._restart = True

    def _make_room(self, need: int) -> bool:
        """Shed queued messages, cheapest first, until ``need`` more bytes fit.

        Lock held. The message in flight is never shed: it is being sent, and
        removing it from under the sender would break the order rule for no
        memory gained (its bytes are held until the send returns anyway).
        """
        if need > self.max_bytes:
            return False
        for reason, sheddable in _SHED_ORDER:
            if self._bytes + need <= self.max_bytes:
                return True
            kept: deque[OutboxMessage] = deque()
            for m in self._queue:
                if (
                    self._bytes + need > self.max_bytes
                    and m is not self._in_flight
                    and sheddable(m)
                ):
                    self._bytes -= m.size
                    self._drop(reason)
                else:
                    kept.append(m)
            self._queue = kept
        return self._bytes + need <= self.max_bytes

    def _remove(self, msg: OutboxMessage) -> None:
        """Take a settled message out of the queue. Lock held."""
        if self._queue and self._queue[0] is msg:
            self._queue.popleft()
        else:
            try:
                self._queue.remove(msg)
            except ValueError:
                return
        self._bytes -= msg.size

    def _drop(self, reason: str) -> None:
        """Count one dropped message under ``reason``. Lock held."""
        self._dropped[reason] = self._dropped.get(reason, 0) + 1

    def _next_backoff(self) -> float:
        """The wait after one more failure: start, then x factor, capped. Lock held."""
        if self._backoff <= 0:
            return self.backoff_s
        return min(self.backoff_max_s, self._backoff * self.backoff_factor)


class PlannerClient:
    """Stateful client for one pipeline run's reporting into the planner.

    Typical lifecycle::

        pc = PlannerClient("http://planner:8787")
        pc.create_run(event_id=3, label="gate A POC")
        pc.post_stats("track", frames=1200, fps=14.8, metrics={...})
        pc.add_sample("face-detect", t_ms=52_000, metrics={"faceBoxWPx": 71})
        pc.end_run("ended")
    """

    def __init__(
        self,
        base_url: str,
        transport: Transport | None = None,
        file_transport: FileTransport | None = None,
        retries: int = 3,
        backoff_s: float = 0.2,
        batch_size: int = 100,
        sleep: Callable[[float], None] = time.sleep,
        best_effort_transport: Transport | None = None,
        token: str | None = None,
        token_provider: "TokenProvider | None" = None,
        durable: bool = False,
        outbox_max_bytes: int = OUTBOX_MAX_BYTES,
        outbox_stale_frame_s: float | None = 30.0,
        drain_timeout_s: float = 30.0,
        outbox_sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        """Configure the client; nothing is sent until create_run.

        ``batch_size`` is the local auto-flush threshold for ``add_sample``
        and is clamped to the contract cap of 200.  ``file_transport`` is only
        needed for :meth:`post_frame` (multipart debug frames); without it that
        one method is a no-op and every other call is unaffected.
        ``best_effort_transport`` carries the single-shot taps/feedback traffic
        and should be wired to a short-timeout client; it defaults to
        ``transport``.

        ``token_provider`` is the modern credential: a
        :class:`heco_common.auth.TokenProvider` holding this runner's
        application secret, which mints short-lived tokens and refreshes them
        before they lapse.  Prefer it.  A 401 on the retrying path then means
        "the token in hand was refused", which IS worth one refresh and one
        retry — a rotation should not need a restart.

        The shared secret that used to sit beside it was retired at runbook
        step 7, so a provider is the only way this client is credentialed.

        It is sent as ``Authorization: Bearer`` on every call,
        including the multipart frame upload.  The header is computed AT
        REQUEST TIME, not baked in at construction, which is what makes a
        rotation invisible to a running process.

        ``durable`` turns on the :class:`Outbox` (module docstring): the four
        best-effort posts, and a run end that fails, are kept and delivered in
        order once the planner answers, instead of dropped.  Every transport
        must then accept a ``headers=`` keyword — refused here, loudly, rather
        than as a message retried forever.  ``outbox_max_bytes`` and
        ``outbox_stale_frame_s`` are the outbox's bound and staleness rule;
        ``drain_timeout_s`` is how long :meth:`end_run` waits for the queue to
        empty before it sends the run end.  ``outbox_sleep``, ``clock`` (ages)
        and ``wall_clock`` (the ``X-Heco-At`` stamp) exist for tests.
        """
        self.base_url = base_url.rstrip("/")
        self.token = token or None
        self.token_provider = token_provider
        # Best-effort calls swallow failures by design, which is right for taps
        # and stats — but a 401 is not a hiccup, it is a misconfiguration that
        # silently stops erasure purges and operator corrections from ever
        # being seen. Counted here so the runner can surface it instead of
        # quietly doing nothing all night.
        self.auth_failures = 0
        self._auth_lock = threading.Lock()
        if transport is not None:
            self.transport = transport
        elif token_provider is not None:
            # Read the header per request, so the token this client sends is
            # whatever the provider currently holds — including one minted five
            # seconds ago by a background refresh.
            self.transport = provider_urllib_transport(token_provider)
        elif self.token:
            self.transport = bearer_urllib_transport(self.token)
        else:
            self.transport = urllib_transport
        self.best_effort_transport = best_effort_transport or self.transport
        self.file_transport = file_transport
        self.retries = max(1, retries)
        self.backoff_s = backoff_s
        self.batch_size = min(max(1, batch_size), MAX_SAMPLES_PER_POST)
        self.run_id: int | str | None = None
        self._sleep = sleep
        self._pending: list[Sample] = []
        self.drain_timeout_s = drain_timeout_s
        self._clock = clock
        self._wall_clock = wall_clock
        # X-Heco-Delivery's counter: one per message, never reused by this
        # client, across runs too. next() on a count is atomic under the GIL,
        # and the loop and reporter threads both post.
        self._deliveries = itertools.count(1)
        self.outbox: Outbox | None = None
        if durable:
            for name, fn in (
                ("transport", self.transport),
                ("best_effort_transport", self.best_effort_transport),
                ("file_transport", self.file_transport),
            ):
                if fn is not None and not _accepts_headers(fn):
                    raise TypeError(
                        f"durable=True sends {DELIVERY_HEADER} and {AT_HEADER} on every "
                        f"attempt, so {name} must accept a headers= keyword"
                    )
            self.outbox = Outbox(
                self._attempt,
                max_bytes=outbox_max_bytes,
                stale_frame_s=outbox_stale_frame_s,
                sleep=outbox_sleep,
                clock=clock,
            )

    # ------------------------------------------------------------- calls

    def create_run(
        self,
        event_id: int | str,
        placement_id: int | str | None = None,
        label: str | None = None,
        config: dict | None = None,
    ) -> int | str:
        """Open a run under an event; remembers and returns the run id."""
        payload = PlannerRunCreate(
            eventId=event_id, placementId=placement_id, label=label, config=config
        ).model_dump(exclude_none=True)
        body = self._request("POST", "/api/pipeline/runs", payload)
        run_id = body.get("id")
        if run_id is None:
            raise PlannerError(f"planner run response has no id: {body!r}")
        self.run_id = run_id
        return run_id

    def end_run(
        self,
        status: str = "ended",
        notes: str | None = None,
        results: dict[str, float] | None = None,
        end_reason: str | None = None,
        drain_timeout_s: float | None = None,
    ) -> None:
        """Flush pending samples, then close the run as ended|failed.

        ``results`` gives the count a structured home the planner can read
        back and export; ``notes`` keeps the human sentence beside it.

        DURABLE: the run end is the LAST word, so the outbox is drained first
        (up to ``drain_timeout_s``, by default the client's) — everything the
        run reported lands before the planner is told it is over. A caller
        that has just drained it itself passes 0, or a planner outage costs
        the settle that wait twice. The PUT itself keeps the retrying path
        and raises as before; but when it fails because the planner is away,
        it is also queued, stamped with the moment of this call, and the
        sender keeps trying after the run loop has returned (the process
        lives on). That raises :class:`PlannerDeferred`. A refusal (4xx) is
        not queued: it would only be refused again. Samples buffered with
        :meth:`add_sample` still flush through the retrying path first, as
        before, and a failure there raises before the run end is attempted.
        """
        self.flush()
        payload = PlannerRunEnd(
            status=status, notes=notes, results=results, endReason=end_reason,
        ).model_dump(exclude_none=True)
        run_id = self._require_run()
        path = f"/api/pipeline/runs/{run_id}"
        if self.outbox is None:
            self._request("PUT", path, payload)
            return
        headers = self._stamp(run_id)
        created = self._clock()
        self.outbox.drain(self.drain_timeout_s if drain_timeout_s is None else drain_timeout_s)
        try:
            self._request("PUT", path, payload, headers=headers)
        except PlannerError as exc:
            if not exc.retryable:
                raise
            msg = OutboxMessage(
                kind=RUN_END, method="PUT", url=f"{self.base_url}{path}",
                headers=headers, created=created, payload=payload,
            )
            msg.size = msg.measure()
            if not self.outbox.put(msg, failed=True):
                raise
            raise PlannerDeferred(
                f"{exc} — queued: the run end is delivered as soon as the planner answers",
                status=exc.status,
                retryable=True,
            ) from exc

    def post_stats(
        self,
        stage: str,
        frames: int,
        fps: float,
        metrics: dict[str, MetricSummary | dict] | None = None,
    ) -> None:
        """Upsert one stage's aggregate stats (validated via StageStats)."""
        payload = StageStats(
            stage=stage, frames=frames, fps=fps, metrics=metrics or {}
        ).model_dump()
        self._request("POST", f"/api/pipeline/runs/{self._require_run()}/stats", payload)

    def add_sample(self, stage: str, t_ms: int, metrics: dict[str, float]) -> None:
        """Buffer one sample; auto-flush when the buffer hits batch_size."""
        self._pending.append(Sample(stage=stage, tMs=t_ms, metrics=metrics))
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """POST any buffered samples now (no-op when the buffer is empty)."""
        if self._pending:
            pending, self._pending = self._pending, []
            self.post_samples(pending)

    def post_samples(self, samples: list[Sample]) -> None:
        """POST an explicit sample list, chunked to the 200-sample cap."""
        run_id = self._require_run()
        for i in range(0, len(samples), MAX_SAMPLES_PER_POST):
            chunk = samples[i : i + MAX_SAMPLES_PER_POST]
            payload = {"samples": [s.model_dump() for s in chunk]}
            self._request("POST", f"/api/pipeline/runs/{run_id}/samples", payload)

    # ------------------------------------------------- debug taps (v1)

    # Every best-effort post has an ``offer_*`` twin that says WHAT BECAME of
    # the message — ACCEPTED, QUEUED or DROPPED — and ``post_*`` is simply
    # ``offer_* == ACCEPTED``, so "True only when the planner accepted it"
    # stays true in both modes. A caller that re-sends on False needs the twin
    # under a durable client: re-sending a QUEUED message would deliver it
    # twice and, every flush, grow the queue by a copy of what it already holds.

    def post_tap(self, stage: str, payload: dict) -> bool:
        """Best-effort POST of one stage's structured debug payload.

        Debug taps must never block or crash the run loop on a planner hiccup
        (CONTRACTS.md), so this is a single attempt with no retry and swallows
        every failure, returning True only when the planner accepted it.
        Durable, a tap the planner was away for is queued (and False).
        """
        return self.offer_tap(stage, payload) == ACCEPTED

    def offer_tap(self, stage: str, payload: dict) -> str:
        """:meth:`post_tap`, answering ACCEPTED / QUEUED / DROPPED."""
        run_id = self._require_run()
        path = f"/api/pipeline/runs/{run_id}/taps"
        body = {"stage": stage, "payload": payload}
        if self.outbox is None:
            ok, _ = self._send_once("POST", path, body)
            return ACCEPTED if ok else DROPPED
        return self._offer(self._json_message(TAP, run_id, path, body, stage=stage))

    def post_frame(
        self,
        stage: str,
        jpeg: bytes,
        filename: str = "frame.jpg",
        content_type: str = "image/jpeg",
        extra: dict | None = None,
    ) -> bool:
        """Best-effort multipart POST of one stage's annotated JPEG frame.

        No-op (returns False) when the client was built without a
        ``file_transport``.  Like :meth:`post_tap`, a single attempt that
        swallows failures so the loop is never held up by frame uploads.
        """
        return self.offer_frame(stage, jpeg, filename, content_type, extra) == ACCEPTED

    def offer_frame(
        self,
        stage: str,
        jpeg: bytes,
        filename: str = "frame.jpg",
        content_type: str = "image/jpeg",
        extra: dict | None = None,
    ) -> str:
        """:meth:`post_frame`, answering ACCEPTED / QUEUED / DROPPED.

        ``extra["forensic"]`` marks the bench copy of one frame's pixels; any
        other non-match frame only feeds the planner's live ring, and is the
        first thing a durable client's bound sheds (see :class:`Outbox`).
        """
        if self.file_transport is None:
            return DROPPED
        run_id = self._require_run()
        url = f"{self.base_url}/api/pipeline/runs/{run_id}/frames"
        # Multipart text fields ride beside the stage name; the forensic path
        # uses them to hand the planner seq/tMs and the NATIVE dimensions the
        # overlay scaling needs. None-valued fields are simply not sent.
        fields = {"stage": stage}
        for k, v in (extra or {}).items():
            if v is not None:
                fields[k] = str(v)
        if self.outbox is not None:
            return self._offer(
                OutboxMessage(
                    kind=FRAME, url=url, headers=self._stamp(run_id), created=self._clock(),
                    fields=fields, filename=filename, data=jpeg, content_type=content_type,
                    stage=stage, forensic=bool((extra or {}).get("forensic")),
                )
            )
        try:
            status, _ = self.file_transport(url, fields, filename, jpeg, content_type)
        except Exception:  # noqa: BLE001 — best-effort, never propagate
            return DROPPED
        return ACCEPTED if self._note_upload_status(status) else DROPPED

    #: Frame records per POST. A flush at 4 fps carries ~8, so this only binds
    #: after an outage — and a 4000-record backlog in ONE body would be ~5.6 MB
    #: against the planner's JSON limit, which would fail the whole batch
    #: instead of the part that did not fit.
    FRAME_RECORD_CHUNK = 500

    def post_frame_records(self, records: list[dict]) -> bool:
        """Best-effort batch POST of per-frame decision records.

        THE LEDGER'S ONE RULE IS BATCHING. This is called from the stats flush,
        not from the frame loop: at 4 fps a POST per frame is 4 network round
        trips a second inside the loop, which is exactly the shape that
        produced the 0.4 fps death spiral the tap duty guard exists to prevent.

        Returns True only when EVERY chunk landed. A partial success returns
        False and the caller re-queues the whole batch, which can duplicate
        records the planner already holds — the planner therefore upserts on
        (run, seq), so a replayed frame overwrites itself rather than appearing
        twice. Losing order or inventing frames would both be worse than
        writing the same truth again.
        """
        return self.offer_frame_records(records) == ACCEPTED

    def offer_frame_records(self, records: list[dict]) -> str:
        """:meth:`post_frame_records`, answering ACCEPTED / QUEUED / DROPPED.

        One message per chunk; the batch answers with its WORST chunk, so
        DROPPED means some chunk is gone (re-sending the batch is the caller's
        old recourse and the planner's upsert makes it safe) and QUEUED means
        none is gone but not all have landed yet — do NOT re-send.
        """
        if not records:
            return ACCEPTED
        try:
            run_id = self._require_run()
        except Exception:  # noqa: BLE001 — no run open: nothing to file against
            return DROPPED
        path = f"/api/pipeline/runs/{run_id}/frame-records"
        outcome = ACCEPTED
        for i in range(0, len(records), self.FRAME_RECORD_CHUNK):
            body = {"records": records[i : i + self.FRAME_RECORD_CHUNK]}
            if self.outbox is None:
                sent, _ = self._send_once("POST", path, body)
                chunk = ACCEPTED if sent else DROPPED
            else:
                chunk = self._offer(self._json_message(FRAME_RECORDS, run_id, path, body))
            outcome = max(outcome, chunk, key=_OUTCOME_RANK.__getitem__)
        return outcome

    def post_face_card(self, person_key: str, jpeg: bytes) -> bool:
        """Best-effort multipart POST of ONE guest's face card.

        A card is that guest's face cut out of the frame, so the register can
        show WHO an identity is rather than which crowded doorway they were
        standing in (operator, 2026-08-07: "sometimes all images of that guest
        come with other people ... it is hard to find who it is").

        Same contract as :meth:`post_frame` — one attempt, failures swallowed,
        no retry. A missing card costs a thumbnail; a loop held up by an upload
        costs frames, and frames are how guests are counted.
        """
        return self.offer_face_card(person_key, jpeg) == ACCEPTED

    def offer_face_card(self, person_key: str, jpeg: bytes) -> str:
        """:meth:`post_face_card`, answering ACCEPTED / QUEUED / DROPPED."""
        if self.file_transport is None:
            return DROPPED
        try:
            # _require_run() is caught, unlike post_frame's. This is called
            # from the tap round, whose caller has no except clause, so an
            # exception here would end the run — and a run must never die over
            # a thumbnail. "No run open" is simply "no card".
            run_id = self._require_run()
        except Exception:  # noqa: BLE001 — best-effort, never propagate
            return DROPPED
        url = f"{self.base_url}/api/pipeline/runs/{run_id}/faces"
        fields, filename = {"personKey": person_key}, f"{person_key}.jpg"
        if self.outbox is not None:
            return self._offer(
                OutboxMessage(
                    kind=FACE_CARD, url=url, headers=self._stamp(run_id), created=self._clock(),
                    fields=fields, filename=filename, data=jpeg, content_type="image/jpeg",
                )
            )
        try:
            status, _ = self.file_transport(url, fields, filename, jpeg, "image/jpeg")
        except Exception:  # noqa: BLE001 — best-effort, never propagate
            return DROPPED
        return ACCEPTED if self._note_upload_status(status) else DROPPED

    # ---------------------------------------- operator feedback (v1)

    def poll_feedback(self, since: str | None = None) -> list[dict]:
        """Return the run's open feedback items (best-effort; [] on any failure).

        Polls ``GET /api/pipeline/runs/:id/feedback?since=<iso>``.  Accepts the
        planner returning either a bare list or an object wrapping the items
        under ``feedback``/``items`` so the two sides can settle that detail
        without breaking this client.
        """
        path = f"/api/pipeline/runs/{self._require_run()}/feedback"
        if since:
            path += "?" + urllib.parse.urlencode({"since": since})
        ok, body = self._send_once("GET", path, None)
        if not ok:
            return []
        if isinstance(body, list):
            return body
        items = body.get("feedback") or body.get("items") or []
        return items if isinstance(items, list) else []

    def resolve_feedback(self, feedback_id: int | str, status: str) -> bool:
        """Best-effort ``PUT /api/feedback/:id {status}`` after acting on an item.

        A dropped status update is harmless: the item stays open and is
        re-applied next poll, and the corrections are idempotent (a second
        merge of an already-folded pair is a no-op), so no retry is needed.
        """
        ok, _ = self._send_once("PUT", f"/api/feedback/{feedback_id}", {"status": status})
        return ok

    # ------------------------------------------- staff erasure (v1)

    def staff_tombstones(self, site_id: str) -> list[dict]:
        """Return the site's open erasure tombstones (best-effort; [] on failure).

        ``GET /api/staff-tombstones?siteId=`` — each item
        ``{id, siteId, staffId, deletedAt}`` names a deleted roster member whose
        face templates must be purged from the site staff store.  Best-effort
        like the feedback poll: a planner hiccup returns [] and the next cycle
        retries, because the tombstone stays open until confirmed.
        """
        path = "/api/staff-tombstones?" + urllib.parse.urlencode({"siteId": site_id})
        ok, body = self._send_once("GET", path, None)
        return body if ok and isinstance(body, list) else []

    def confirm_tombstone(self, tombstone_id: str) -> bool:
        """Confirm a purge: ``PUT /api/staff-tombstones/:id`` stamps purged_at.

        A dropped confirmation is harmless — the tombstone stays open and the
        purge re-runs next cycle; erasure is idempotent.
        """
        ok, _ = self._send_once("PUT", f"/api/staff-tombstones/{tombstone_id}", None)
        return ok

    # ------------------------------------------- staff enrolment (v1)

    def report_enrolment(self, staff_id: int | str, enrolled_at: str, sample_count: int) -> dict:
        """Report a completed staff enrolment: ``PUT /api/staff/:id``.

        Uses the retrying transport (the samples are already stored pipeline
        side; this is the operator-facing confirmation, worth a retry).  Raises
        PlannerError after exhausting retries — the enrol flow catches it so a
        planner outage cannot fail the enrolment itself.
        """
        return self._request(
            "PUT", f"/api/staff/{staff_id}",
            {"enrolledAt": enrolled_at, "sampleCount": sample_count},
        )

    # --------------------------------------------- durable delivery

    @property
    def durable(self) -> bool:
        """True when this client keeps the writes the planner was away for."""
        return self.outbox is not None

    def outbox_stats(self) -> dict | None:
        """The outbox's counters and backlog (:meth:`Outbox.stats`); None when not durable."""
        return None if self.outbox is None else self.outbox.stats()

    def drain_outbox(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout`` (default ``drain_timeout_s``) for the outbox to
        empty; True once it has. A non-durable client has nothing to wait for."""
        if self.outbox is None:
            return True
        return self.outbox.drain(self.drain_timeout_s if timeout is None else timeout)

    def stop_outbox(self, timeout: float | None = 5.0) -> bool:
        """Stop the outbox's sender, keeping the queue; True once it has exited."""
        return True if self.outbox is None else self.outbox.stop(timeout)

    def _offer(self, msg: OutboxMessage) -> str:
        """The durable path for one best-effort message: now, queued, or dropped.

        Something queued: join the queue WITHOUT an attempt. That is the order
        rule, and it is why a planner that is down costs the caller nothing —
        not a timeout per post. Nothing queued: one synchronous attempt through
        the same transport, with the same timeout, a non-durable client uses;
        if the planner was away, the message is queued instead of lost.
        """
        outbox = self.outbox
        try:
            if outbox.busy():
                return self._enqueue(msg, failed=False)
            verdict = self._attempt(msg)
            if verdict == SENT:
                return ACCEPTED
            if verdict in (RETRY, FAULT):
                if verdict == FAULT:
                    msg.faults += 1
                return self._enqueue(msg, failed=True)
            outbox.count_drop(verdict)
            return DROPPED
        except Exception:  # noqa: BLE001 — a best-effort post never raises into its caller
            outbox.count_drop("error")
            return DROPPED

    def _enqueue(self, msg: OutboxMessage, *, failed: bool) -> str:
        """Hand a message to the outbox: QUEUED, or DROPPED when it cannot be.

        A JSON payload that cannot be encoded never reaches the queue: it could
        never be sent, and at the head it would hold everything behind it.
        """
        try:
            msg.size = msg.measure()
        except (TypeError, ValueError):
            self.outbox.count_drop("unsendable")
            return DROPPED
        return QUEUED if self.outbox.put(msg, failed=failed) else DROPPED

    def _attempt(self, msg: OutboxMessage) -> str:
        """One try at a durable message: SENT, RETRY, FAULT or a drop reason.

        Never raises. The best-effort transports, exactly as the synchronous
        path uses them, plus the message's two headers. No answer, a timeout,
        502/503/504, 408 and 429 are RETRY (the planner away); any other 5xx is
        FAULT (it answered, and failed — retried, but not forever); any other
        4xx is dropped as "rejected". A queued run end keeps the retrying
        transport it was first sent through, and that path's 401 rule: one
        forced token refresh, then one more try. For everything else a 401
        keeps the best-effort bookkeeping — counted, the token flagged — and is
        dropped: a refused credential is configuration, not an outage.
        """
        try:
            if msg.data is not None:
                if self.file_transport is None:  # removed after the message was made
                    return "rejected"
                status, _ = self.file_transport(
                    msg.url, msg.fields, msg.filename, msg.data, msg.content_type,
                    headers=msg.headers,
                )
            else:
                send = self.transport if msg.kind == RUN_END else self.best_effort_transport
                status, _ = send(msg.method, msg.url, msg.payload, headers=msg.headers)
        except Exception:  # noqa: BLE001 — no answer: the planner is away
            return RETRY
        if status == 401:
            self._note_auth_failure()
            if msg.kind == RUN_END and self.token_provider is not None and not msg.refreshed:
                msg.refreshed = True
                # A failed mint is recorded by the provider itself; the retry
                # then goes out with the token in hand, as _request's would.
                with contextlib.suppress(Exception):
                    self.token_provider.force_refresh()
                return RETRY
            return "unauthorized"
        if status < 400:
            return SENT
        if status in _AWAY_STATUSES:
            return RETRY
        return FAULT if status >= 500 else "rejected"

    def _json_message(
        self, kind: str, run_id: int | str, path: str, body: dict, stage: str | None = None
    ) -> OutboxMessage:
        """A durable JSON message made NOW: its URL, stamp and age fixed here."""
        return OutboxMessage(
            kind=kind, url=f"{self.base_url}{path}", headers=self._stamp(run_id),
            created=self._clock(), payload=body, stage=stage,
        )

    def _stamp(self, run_id: int | str) -> dict[str, str]:
        """The two durable headers for a message made now, for run ``run_id``.

        ``X-Heco-Delivery`` is unique per message (the planner may drop a
        repeat: a reply lost after the planner had stored the message means it
        is sent again). ``X-Heco-At`` is when the CALL was made, so a message
        delivered minutes late still says when it happened.
        """
        return {
            DELIVERY_HEADER: f"{run_id}:{next(self._deliveries)}",
            AT_HEADER: _iso_utc(self._wall_clock()),
        }

    # ---------------------------------------------------------- internals

    def _send_once(self, method: str, path: str, payload: dict | None) -> tuple[bool, dict]:
        """One best-effort attempt that never raises: (accepted, body).

        Uses ``best_effort_transport`` so a hung planner costs the caller that
        transport's (short) timeout once, rather than the retrying budget.
        """
        url = f"{self.base_url}{path}"
        try:
            status, body = self.best_effort_transport(method, url, payload)
        except Exception:  # noqa: BLE001 — best-effort callers want no exceptions
            return False, {}
        if status == 401:
            # Counts the failure and flags the token; does NOT refresh inline.
            # These callers are best-effort because their latency budget is
            # the frame loop's, and a token round-trip here would cost the
            # very frames the tap was reporting on. The next ordinary read
            # picks the refresh up.
            self._note_auth_failure()
        return status < 400, (body or {})

    def _note_upload_status(self, status: int) -> bool:
        """Record a multipart result the same way ``_send_once`` records a JSON
        one, and return whether it was accepted.

        THE GAP THIS CLOSES: the two multipart uploads — debug frames and face
        cards — swallowed their statuses whole, so a runner whose credential had
        been refused went on posting frames into a 401 all night with
        ``auth_failures`` sitting at zero and nothing in any log. The JSON paths
        had counted 401s since v1; these two never did.
        """
        if status == 401:
            self._note_auth_failure()
        return status < 400

    def _note_auth_failure(self) -> None:
        """Count one refused credential and tell the token provider.

        LOCKED because the runner reports from two threads now: the counting
        loop polls feedback and confirms tombstones while the reporter thread
        posts taps, frames and face cards, and ``+= 1`` is a read-modify-write.
        Lost increments would make ``plannerAuthFailures`` under-report exactly
        when it matters — a credential refused on every upload is the case this
        counter exists to surface, and "3" instead of "300" reads like a
        transient blip rather than an outage.
        """
        with self._auth_lock:
            self.auth_failures += 1
        if self.token_provider is not None:
            self.token_provider.mark_rejected()

    def _require_run(self) -> int | str:
        """Return the current run id or fail: reporting needs create_run first."""
        if self.run_id is None:
            raise PlannerError("no run open — call create_run() first")
        return self.run_id

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Send with retry: 5xx/errors back off and retry, 4xx fail fast.

        A 401 is its own case. With a token provider it means "the token in
        hand was refused" — a rotation, or a clock — which earns exactly ONE
        forced refresh and ONE retry, so a rotation never needs a restart. Once
        that retry also comes back 401 the problem is configuration, and
        retrying further only buries it, so it fails loudly and names the two
        variables to check.

        ``headers`` (the durable run end's stamp) is passed to the transport
        only when given, so every other call reaches it exactly as before. The
        raised PlannerError says whether the planner was away (``retryable``).
        """
        url = f"{self.base_url}{path}"
        last: str = "no attempt made"
        last_status: int | None = None
        refreshed = False
        for attempt in range(self.retries):
            try:
                if headers:
                    status, body = self.transport(method, url, payload, headers=headers)
                else:
                    status, body = self.transport(method, url, payload)
            except Exception as exc:  # noqa: BLE001 — any transport failure retries
                last = f"transport error: {exc}"
                last_status = None
            else:
                if status < 400:
                    return body
                last = f"HTTP {status}: {body!r}"
                last_status = status
                if status == 401:
                    if self.token_provider is not None and not refreshed:
                        refreshed = True
                        self.token_provider.force_refresh()
                        continue  # immediately, with no backoff: nothing is hung
                    raise PlannerError(self._auth_failure_message(method, url), status=401)
                if status < 500:
                    raise PlannerError(
                        f"{method} {url} rejected — {last}",
                        status=status,
                        retryable=_retryable(status),
                    )
            if attempt < self.retries - 1:
                self._sleep(self.backoff_s * (2**attempt))
        raise PlannerError(
            f"{method} {url} failed after {self.retries} attempts — {last}",
            status=last_status,
            retryable=last_status is None or _retryable(last_status),
        )

    def _auth_failure_message(self, method: str, url: str) -> str:
        """Say what to fix. This message is read by someone at a venue with a
        pipeline that will not report, so it names variables, not concepts."""
        if self.token_provider is not None:
            why = self.token_provider.last_error
            return (
                f"{method} {url} refused after refreshing this runner's token. "
                + (f"The auth service said: {why} " if why else "")
                + "Check HECO_APP_ID and HECO_APP_SECRET on the runner, and that "
                "HECO_AUTH_URL points at the same auth service the planner verifies "
                "against."
            )
        return (
            f"{method} {url} refused: the planner requires a token and this "
            f"runner {'sent the wrong one' if self.token else 'sent none'}. "
            "Give this runner HECO_APP_ID/HECO_APP_SECRET for an application "
            "registered with the auth service."
        )
