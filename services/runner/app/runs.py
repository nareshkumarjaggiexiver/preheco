"""Run registry: creates RunLoops, threads them, answers status queries."""

import threading
import uuid

import httpx
from heco_common.planner import PlannerClient

from .config import Settings
from .loop import RunLoop, httpx_file_transport, httpx_transport


class RunManager:
    """Holds every run of this runner process, live and finished."""

    def __init__(self, settings: Settings) -> None:
        """Create an empty registry bound to one Settings snapshot."""
        self.settings = settings
        self._runs: dict[str, RunLoop] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(self, request: dict) -> str:
        """Spawn a RunLoop thread for a validated POST /runs body; returns runId.

        THREE http clients, deliberately: the stage client keeps the generous
        timeout (a stage call is the product), the planner client a short one
        (reporting), and the best-effort client a shorter one still.  They used
        to be one 30 s client, so a planner that accepted connections and then
        wedged could freeze the frame loop for minutes per tick while ingest's
        drop-not-queue slot threw away every crossing.
        """
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        settings = self.settings
        planner_url = request.get("plannerUrl") or settings.planner_url
        client = httpx.Client(timeout=settings.request_timeout_s)
        # The planner's token rides on the CLIENT, so both the JSON transports
        # and the multipart frame upload carry it without each adapter having
        # to know about auth. Absent when the planner is loopback-only, which
        # is the default and needs no token.
        # Passed only when there IS a token, so the loopback default keeps the
        # plain two-argument construction (which test doubles rely on).
        auth = (
            {"headers": {"Authorization": f"Bearer {settings.planner_token}"}}
            if settings.planner_token
            else {}
        )
        planner_http = httpx.Client(timeout=settings.planner_timeout_s, **auth)
        report_http = httpx.Client(timeout=settings.report_timeout_s, **auth)
        planner = PlannerClient(
            planner_url,
            transport=httpx_transport(planner_http),
            best_effort_transport=httpx_transport(report_http),
            file_transport=httpx_file_transport(report_http),
            token=settings.planner_token,
        )
        loop = RunLoop(run_id, request, settings, client, planner)
        thread = threading.Thread(target=loop.run, name=run_id, daemon=True)
        with self._lock:
            self._runs[run_id] = loop
            self._threads[run_id] = thread
        thread.start()
        return run_id

    def get(self, run_id: str) -> dict | None:
        """Return the live status dict for a run, or None if unknown."""
        with self._lock:
            loop = self._runs.get(run_id)
        return loop.status() if loop else None

    def stop(self, run_id: str) -> bool:
        """Signal a run to stop; True if the run exists."""
        with self._lock:
            loop = self._runs.get(run_id)
        if not loop:
            return False
        loop.stop()
        return True
