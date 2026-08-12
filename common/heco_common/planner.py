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

The client is intentionally synchronous: the runner reports stats every ~2 s,
which never justifies an async dependency at POC scale.
"""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
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
Transport = Callable[[str, str, dict | None], tuple[int, dict]]

#: (url, form-fields, filename, file-bytes, content-type) -> (status, body).
#: Separate from ``Transport`` because debug frames are multipart, not JSON.
FileTransport = Callable[[str, dict, str, bytes, str], tuple[int, dict]]


class PlannerError(RuntimeError):
    """Raised when a planner call fails after all retries (or on 4xx)."""


def bearer_urllib_transport(token: str) -> "Transport":
    """A stdlib transport that carries ``Authorization: Bearer <token>``.

    The planner refuses to listen beyond loopback without a token, so any
    deployment where the runner is not on the same host needs this. Used
    automatically when a PlannerClient is given a token and no explicit
    transport (the httpx path sets the header on its client instead).
    """

    def transport(method: str, url: str, payload: dict | None) -> tuple[int, dict]:
        return urllib_transport(method, url, payload, token=token)

    return transport


def provider_urllib_transport(provider: "TokenProvider") -> "Transport":
    """A stdlib transport that asks the provider for a header on EVERY call.

    The predecessor computed ``Authorization`` once, when the client was built,
    which is precisely why rotating the credential meant restarting the runner.
    One line moved; a whole class of restart disappeared.
    """

    def transport(method: str, url: str, payload: dict | None) -> tuple[int, dict]:
        header = provider.auth_header(block=True)
        return urllib_transport(method, url, payload, extra_headers=header)

    return transport


def urllib_transport(
    method: str,
    url: str,
    payload: dict | None,
    token: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    """Default stdlib transport: JSON in, JSON out, 10 s timeout.

    HTTP error statuses are returned (not raised) so the retry policy in
    PlannerClient owns the decision; network-level failures raise.
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
    ) -> None:
        """Flush pending samples, then close the run as ended|failed.

        ``results`` gives the count a structured home the planner can read
        back and export; ``notes`` keeps the human sentence beside it.
        """
        self.flush()
        payload = PlannerRunEnd(
            status=status, notes=notes, results=results, endReason=end_reason,
        ).model_dump(exclude_none=True)
        self._request("PUT", f"/api/pipeline/runs/{self._require_run()}", payload)

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

    def post_tap(self, stage: str, payload: dict) -> bool:
        """Best-effort POST of one stage's structured debug payload.

        Debug taps must never block or crash the run loop on a planner hiccup
        (CONTRACTS.md), so this is a single attempt with no retry and swallows
        every failure, returning True only when the planner accepted it.
        """
        ok, _ = self._send_once(
            "POST", f"/api/pipeline/runs/{self._require_run()}/taps",
            {"stage": stage, "payload": payload},
        )
        return ok

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
        if self.file_transport is None:
            return False
        url = f"{self.base_url}/api/pipeline/runs/{self._require_run()}/frames"
        # Multipart text fields ride beside the stage name; the forensic path
        # uses them to hand the planner seq/tMs and the NATIVE dimensions the
        # overlay scaling needs. None-valued fields are simply not sent.
        fields = {"stage": stage}
        for k, v in (extra or {}).items():
            if v is not None:
                fields[k] = str(v)
        try:
            status, _ = self.file_transport(url, fields, filename, jpeg, content_type)
        except Exception:  # noqa: BLE001 — best-effort, never propagate
            return False
        return self._note_upload_status(status)

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
        if not records:
            return True
        try:
            run_id = self._require_run()
        except Exception:  # noqa: BLE001 — no run open: nothing to file against
            return False
        ok = True
        for i in range(0, len(records), self.FRAME_RECORD_CHUNK):
            chunk = records[i : i + self.FRAME_RECORD_CHUNK]
            sent, _ = self._send_once(
                "POST", f"/api/pipeline/runs/{run_id}/frame-records",
                {"records": chunk},
            )
            ok = ok and sent
        return ok

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
        if self.file_transport is None:
            return False
        try:
            # _require_run() is INSIDE the try, unlike post_frame's. This is
            # called from the tap round, whose caller has no except clause, so
            # an exception here would end the run — and a run must never die
            # over a thumbnail. "No run open" is simply "no card".
            url = f"{self.base_url}/api/pipeline/runs/{self._require_run()}/faces"
            status, _ = self.file_transport(
                url, {"personKey": person_key}, f"{person_key}.jpg", jpeg, "image/jpeg"
            )
        except Exception:  # noqa: BLE001 — best-effort, never propagate
            return False
        return self._note_upload_status(status)

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

    def _request(self, method: str, path: str, payload: dict | None) -> dict:
        """Send with retry: 5xx/errors back off and retry, 4xx fail fast.

        A 401 is its own case. With a token provider it means "the token in
        hand was refused" — a rotation, or a clock — which earns exactly ONE
        forced refresh and ONE retry, so a rotation never needs a restart. Once
        that retry also comes back 401 the problem is configuration, and
        retrying further only buries it, so it fails loudly and names the two
        variables to check.
        """
        url = f"{self.base_url}{path}"
        last: str = "no attempt made"
        refreshed = False
        for attempt in range(self.retries):
            try:
                status, body = self.transport(method, url, payload)
            except Exception as exc:  # noqa: BLE001 — any transport failure retries
                last = f"transport error: {exc}"
            else:
                if status < 400:
                    return body
                last = f"HTTP {status}: {body!r}"
                if status == 401:
                    if self.token_provider is not None and not refreshed:
                        refreshed = True
                        self.token_provider.force_refresh()
                        continue  # immediately, with no backoff: nothing is hung
                    raise PlannerError(self._auth_failure_message(method, url))
                if status < 500:
                    raise PlannerError(f"{method} {url} rejected — {last}")
            if attempt < self.retries - 1:
                self._sleep(self.backoff_s * (2**attempt))
        raise PlannerError(f"{method} {url} failed after {self.retries} attempts — {last}")

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
