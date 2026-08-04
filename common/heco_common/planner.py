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

The client is intentionally synchronous: the runner reports stats every ~2 s,
which never justifies an async dependency at POC scale.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from .schemas import MetricSummary, PlannerRunCreate, PlannerRunEnd, Sample, StageStats

#: Contract cap on samples per POST.
MAX_SAMPLES_PER_POST = 200

#: (method, url, json-payload-or-None) -> (status_code, decoded-json-body).
Transport = Callable[[str, str, dict | None], tuple[int, dict]]

#: (url, form-fields, filename, file-bytes, content-type) -> (status, body).
#: Separate from ``Transport`` because debug frames are multipart, not JSON.
FileTransport = Callable[[str, dict, str, bytes, str], tuple[int, dict]]


class PlannerError(RuntimeError):
    """Raised when a planner call fails after all retries (or on 4xx)."""


def urllib_transport(method: str, url: str, payload: dict | None) -> tuple[int, dict]:
    """Default stdlib transport: JSON in, JSON out, 10 s timeout.

    HTTP error statuses are returned (not raised) so the retry policy in
    PlannerClient owns the decision; network-level failures raise.
    """
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
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
    ) -> None:
        """Configure the client; nothing is sent until create_run.

        ``batch_size`` is the local auto-flush threshold for ``add_sample``
        and is clamped to the contract cap of 200.  ``file_transport`` is only
        needed for :meth:`post_frame` (multipart debug frames); without it that
        one method is a no-op and every other call is unaffected.
        """
        self.base_url = base_url.rstrip("/")
        self.transport = transport or urllib_transport
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

    def end_run(self, status: str = "ended", notes: str | None = None) -> None:
        """Flush pending samples, then close the run as ended|failed."""
        self.flush()
        payload = PlannerRunEnd(status=status, notes=notes).model_dump(exclude_none=True)
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
    ) -> bool:
        """Best-effort multipart POST of one stage's annotated JPEG frame.

        No-op (returns False) when the client was built without a
        ``file_transport``.  Like :meth:`post_tap`, a single attempt that
        swallows failures so the loop is never held up by frame uploads.
        """
        if self.file_transport is None:
            return False
        url = f"{self.base_url}/api/pipeline/runs/{self._require_run()}/frames"
        try:
            status, _ = self.file_transport(url, {"stage": stage}, filename, jpeg, content_type)
        except Exception:  # noqa: BLE001 — best-effort, never propagate
            return False
        return status < 400

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
        """One transport attempt that never raises: (accepted, body)."""
        url = f"{self.base_url}{path}"
        try:
            status, body = self.transport(method, url, payload)
        except Exception:  # noqa: BLE001 — best-effort callers want no exceptions
            return False, {}
        return status < 400, (body or {})

    def _require_run(self) -> int | str:
        """Return the current run id or fail: reporting needs create_run first."""
        if self.run_id is None:
            raise PlannerError("no run open — call create_run() first")
        return self.run_id

    def _request(self, method: str, path: str, payload: dict | None) -> dict:
        """Send with retry: 5xx/errors back off and retry, 4xx fail fast."""
        url = f"{self.base_url}{path}"
        last: str = "no attempt made"
        for attempt in range(self.retries):
            try:
                status, body = self.transport(method, url, payload)
            except Exception as exc:  # noqa: BLE001 — any transport failure retries
                last = f"transport error: {exc}"
            else:
                if status < 400:
                    return body
                last = f"HTTP {status}: {body!r}"
                if status < 500:
                    raise PlannerError(f"{method} {url} rejected — {last}")
            if attempt < self.retries - 1:
                self._sleep(self.backoff_s * (2**attempt))
        raise PlannerError(f"{method} {url} failed after {self.retries} attempts — {last}")
