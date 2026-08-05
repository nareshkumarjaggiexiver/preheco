"""Thin HTTP readers for the runner, the planner and ingest.

WHY THESE ARE THIN. Everything the harness needs already exists on the wire:
the runner starts and reports a run, and the planner is the system of record
for how it ended and what each stage saw. There is nothing to invent here, so
these classes are three endpoints each and no cleverness.

WHY THEY SHARE ONE TRANSPORT SHAPE. ``heco_common.planner`` already defines a
``(method, url, payload) -> (status, body)`` callable and already knows how to
carry the planner's bearer token — the planner refuses to listen beyond
loopback without it. Reusing that transport means the harness has no HTTP
dependency of its own and no second implementation of the auth header, and
tests inject a fake by passing a callable.
"""

from collections.abc import Callable

from heco_common.planner import Transport, bearer_urllib_transport, urllib_transport


class EvalHttpError(RuntimeError):
    """A call the harness cannot proceed without was refused."""


def _default_transport(token: str | None = None) -> Transport:
    """The stdlib transport, with the bearer header when a token is set."""
    return bearer_urllib_transport(token) if token else urllib_transport


class RunnerClient:
    """The runner service (CONTRACTS.md port 7100): start, poll, stop.

    ``stop`` is deliberately the only mutating call besides ``start``, and the
    harness only ever passes it an id it created itself — see eval.run. This
    class has no way to enumerate other people's runs, which is the cheapest
    possible guarantee that the harness cannot touch a live production run.
    """

    def __init__(self, base_url: str, transport: Transport | None = None) -> None:
        """Bind to one runner base URL, optionally with an injected transport."""
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _default_transport()

    def start_run(
        self,
        event_id: str,
        path: str,
        label: str,
        site_id: str | None = None,
        placement_id: str | None = None,
    ) -> str:
        """POST /runs against a FILE source; returns the runner's run id.

        ``loop`` is false and stays false: a looping file source never ends,
        so the run could only ever settle as stalled or operator-stopped —
        neither of which is a scorable, complete count.
        """
        body: dict = {
            "eventId": event_id,
            "source": {"path": path, "loop": False},
            "label": label,
        }
        if site_id:
            body["siteId"] = site_id
        if placement_id:
            body["placementId"] = placement_id
        status, reply = self.transport("POST", f"{self.base_url}/runs", body)
        if status >= 400 or not reply.get("runId"):
            raise EvalHttpError(f"runner refused to start a run (HTTP {status}): {reply!r}")
        return str(reply["runId"])

    def status(self, run_id: str) -> dict:
        """GET /runs/:id — the runner's live view of one run."""
        status, reply = self.transport("GET", f"{self.base_url}/runs/{run_id}", None)
        if status >= 400:
            raise EvalHttpError(f"runner status for {run_id} failed (HTTP {status}): {reply!r}")
        return reply

    def stop(self, run_id: str) -> bool:
        """POST /runs/:id/stop — used only on runs this harness started."""
        status, _ = self.transport("POST", f"{self.base_url}/runs/{run_id}/stop", None)
        return status < 400


class PlannerReader:
    """Read side of the planner — the durable record a score is taken from."""

    def __init__(
        self, base_url: str, token: str | None = None, transport: Transport | None = None
    ) -> None:
        """Bind to the planner, carrying its bearer token when one is set."""
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _default_transport(token)

    def get_run(self, run_id: str) -> dict:
        """GET /api/pipeline/runs/:id — the run row plus its stage stats.

        This is the document :mod:`eval.metrics` scores: it carries
        ``endReason`` (the scorability filter), ``results`` (the count),
        ``config`` (the gate that produced it) and ``stats`` (the funnel).
        """
        url = f"{self.base_url}/api/pipeline/runs/{run_id}"
        status, reply = self.transport("GET", url, None)
        if status >= 400:
            raise EvalHttpError(f"planner run {run_id} unreadable (HTTP {status}): {reply!r}")
        return reply


class IngestProbe:
    """Asks ingest who owns its single capture slot, and nothing else.

    The slot is exclusive. If a real gate is running right now, starting an
    eval clip would either be refused or — worse — seize the camera from a
    live count. So the harness asks first and refuses to start at all.
    """

    def __init__(self, base_url: str, transport: Transport | None = None) -> None:
        """Bind to the ingest service (CONTRACTS.md port 7101)."""
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _default_transport()

    def slot_owner(self) -> str | None:
        """The run id holding the capture slot, or None if free/unreachable.

        Unreachable is reported as None on purpose: this is a courtesy check
        on top of the runner's own 409 handling, and a probe that cannot
        connect must not be the reason a whole evaluation refuses to start.
        """
        try:
            status, body = self.transport("GET", f"{self.base_url}/health", None)
        except Exception:  # noqa: BLE001 — a courtesy probe never raises
            return None
        if status >= 400 or not isinstance(body, dict):
            return None
        owner = body.get("owner")
        return str(owner) if owner else None


#: Signature of a sleep function, so tests can drive the poll loop instantly.
Sleeper = Callable[[float], None]
