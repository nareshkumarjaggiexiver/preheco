"""Run registry: creates RunLoops, threads them, answers status queries.

The registry is BOUNDED.  It used to be a pair of dicts nothing ever deleted
from, so a process that had run a night of gates retained, per run ever
started, a dead Thread object and a whole RunLoop — and a RunLoop pins
``self._last``, which holds the last frame's full base64 JPEG.  Hundreds of
kilobytes per settled run, plus a ``_matching`` scan that walked every run the
process had ever seen on every console poll.

The same unbounded memory was also a correctness bug.  "Do I know this run?"
was a membership test, so a run that settled hours ago still read as a live
sibling — and that is the test that decides whether a 409 on ingest's capture
slot is a live camera (fail) or a corpse (seize).  A settled run whose /close
never landed therefore held the camera for the life of the process, while the
error told the operator to stop a run that had already stopped.  Both are fixed
by the same change: ask about LIVENESS, and reap what has settled.
"""

import threading
import time
import uuid

import httpx
from heco_common.auth import TokenProvider
from heco_common.planner import PlannerClient

from .config import Settings
from .loop import (
    RunLoop,
    auth_for,
    build_token_provider,
    httpx_file_transport,
    httpx_transport,
    probe_models,
)

#: States in which a run is still using its downstream resources (the camera
#: above all).  Anything else has finished, however it finished.
LIVE_STATES = frozenset({"starting", "running", "enrolling"})


class ModelSwapInProgress(RuntimeError):
    """The swap window and a run start collided; whoever came second gets this.

    Raised by :meth:`RunManager.start` when /models/apply holds the window,
    and by :meth:`RunManager.begin_swap` when another apply already does.
    The route maps it to a 409 — non-blocking on purpose, so the second
    caller SEES the collision instead of silently last-writer-winning.
    """


class RunManager:
    """Holds this runner process's live runs, and recently settled ones."""

    def __init__(self, settings: Settings, token_provider: TokenProvider | None = None) -> None:
        """Create an empty registry bound to one Settings snapshot.

        ONE token provider for the whole process, built here and shared by every
        run: the credential belongs to this runner, not to a run, and a provider
        per run would mint a token per run and lose the refresh schedule every
        time a run ended.
        """
        self.settings = settings
        self.token_provider = token_provider or build_token_provider(settings)
        self._runs: dict[str, RunLoop] = {}
        self._threads: dict[str, threading.Thread] = {}
        # run_id -> monotonic time we FIRST observed its thread finished.
        # Reaping is timed from here rather than from the loop's own settle,
        # because the status dict has no settled-at field and adding one would
        # put a clock in the hot path for a housekeeping job's benefit.
        self._settled_at: dict[str, float] = {}
        # True while /models/apply owns the box (begin_swap .. end_swap).
        # Guarded by _lock, the SAME lock the run registry uses — that
        # sharing is the mechanism: either a swap sees the registered live
        # thread and refuses, or a run start sees the in-flight swap and
        # refuses; no interleaving admits both, so a run's model stamp
        # always executes in a swap-free window (review finding, 2026-08-26:
        # the one-shot live_run_ids() read left a 60-94 s window in which a
        # run could start mid-swap and stamp falsified models).
        self._swapping = False
        self._lock = threading.Lock()

    def probe_client(self) -> "httpx.Client":
        """A short-lived credentialed client for probe/apply orchestration.

        The caller closes it. Same identity a run would use, so the swap
        passes an armed sibling gate exactly when a run would.
        """
        return httpx.Client(timeout=30.0, **auth_for(self.settings, self.token_provider))

    def probe_live_models(self) -> dict:
        """The live model per stage, for the POST /runs profile gate.

        A short-lived client with the runner's own credential: the probe must
        see exactly what a run would see, including through an armed gate.
        """
        client = httpx.Client(timeout=2.0, **auth_for(self.settings, self.token_provider))
        try:
            return probe_models(client, self.settings)
        finally:
            client.close()

    def begin_swap(self) -> list[str]:
        """Claim the box for a model swap, or say why it cannot be claimed.

        Returns the LIVE run ids when any run thread is alive — the caller
        refuses with them (a swap under a running count would falsify that
        run's own stamp).  Raises :class:`ModelSwapInProgress` when another
        apply already holds the window: that refusal IS the concurrent-apply
        serialization — the second operator gets a 409 naming the collision
        rather than two applies interleaving per-service swaps into a model
        combination neither of them asked for.  Returns ``[]`` when the
        window is claimed; the caller MUST pair it with :meth:`end_swap`
        (try/finally), or the box refuses runs forever.
        """
        with self._lock:
            live = [rid for rid, t in self._threads.items() if t.is_alive()]
            if live:
                return live
            if self._swapping:
                raise ModelSwapInProgress(
                    "a model apply is already in flight on this install — "
                    "retry when it settles"
                )
            self._swapping = True
            return []

    def end_swap(self) -> None:
        """Release the swap window claimed by :meth:`begin_swap`."""
        with self._lock:
            self._swapping = False

    def start(self, request: dict) -> str:
        """Spawn a RunLoop thread for a validated POST /runs body; returns runId.

        Refuses with :class:`ModelSwapInProgress` while /models/apply holds
        the swap window — checked inside the SAME locked block that registers
        and starts the thread, so the check and the registration are one
        atomic step against :meth:`begin_swap` (see its docstring for why the
        shared lock is the whole point).

        THREE http clients, deliberately: the stage client keeps the generous
        timeout (a stage call is the product), the planner client a short one
        (reporting), and the best-effort client a shorter one still.  They used
        to be one 30 s client, so a planner that accepted connections and then
        wedged could freeze the frame loop for minutes per tick while ingest's
        drop-not-queue slot threw away every crossing.
        """
        self.reap()  # a new run is the natural moment to take the bins out
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        settings = self.settings
        planner_url = request.get("plannerUrl") or settings.planner_url
        # The STAGE client carries the same credential as the planner clients
        # below (runbook step 8): once the sibling services arm their inbound
        # gate (HECO_REQUIRE_AUTH), every /open, /detect, /track and /match
        # this runner makes must present it. Unconfigured, auth_for is {} and
        # nothing changes.
        auth = auth_for(settings, self.token_provider)
        client = httpx.Client(timeout=settings.request_timeout_s, **auth)
        # WHERE THE CREDENTIAL RIDES, and why it moved.
        #
        # It used to be a static header baked into the httpx client at
        # construction — which is exactly why rotating the secret meant
        # restarting this process, and why a botched restart order cost four
        # interruptions in one day. httpx.Auth is consulted per request, so the
        # token these clients send is whatever the provider currently holds,
        # including one minted seconds ago by a background refresh.
        #
        # It still rides on the CLIENT rather than in each call, so the JSON
        # transports and the multipart frame upload all carry it without any
        # adapter having to know about auth.
        planner_http = httpx.Client(timeout=settings.planner_timeout_s, **auth)
        report_http = httpx.Client(timeout=settings.report_timeout_s, **auth)
        planner = PlannerClient(
            planner_url,
            transport=httpx_transport(planner_http),
            best_effort_transport=httpx_transport(report_http),
            file_transport=httpx_file_transport(report_http),
            token_provider=self.token_provider,
        )
        loop = RunLoop(run_id, request, settings, client, planner, is_live_run=self._is_live)
        thread = threading.Thread(target=loop.run, name=run_id, daemon=True)
        try:
            with self._lock:
                if self._swapping:
                    raise ModelSwapInProgress(
                        "a model swap is in progress — retry when it settles"
                    )
                self._runs[run_id] = loop
                self._threads[run_id] = thread
                # Started INSIDE the lock: a reaper reads liveness off the thread,
                # and a registered-but-not-yet-started thread reports not-alive.
                # The reaper takes this same lock, so it cannot observe that gap.
                thread.start()
        except ModelSwapInProgress:
            # The loop never ran, so nothing will ever close its clients — do
            # it here or every refused start leaks three sockets' keep-alives.
            client.close()
            planner_http.close()
            report_http.close()
            raise
        return run_id

    def live_run_ids(self) -> list[str]:
        """Runs whose threads are still counting — the hot-swap veto list.

        A model swap under a live count would falsify the run's own stamp
        (taken at start) mid-flight. /models/apply no longer reads this
        directly — :meth:`begin_swap` computes the same list under the lock
        so the read and the window claim are one atomic step — but the
        console's diagnostics still ask it, and the sentence naming the runs
        to stop is built from it either way.
        """
        with self._lock:
            return [rid for rid, t in self._threads.items() if t.is_alive()]

    def reap(self, now: float | None = None) -> list[str]:
        """Evict runs that settled more than ``run_retention_s`` ago.

        Returns the ids evicted (for logging and tests).  A settled run is kept
        for the retention window so an operator whose run has just finished can
        still read its final status from ``GET /runs/:id``; after that the
        planner row is the durable record and holding a whole RunLoop — with
        the last frame's base64 JPEG inside it — buys nothing.

        Thread liveness is the settle signal rather than the status state,
        because it is the property that actually matters here: a thread that
        has returned cannot be holding a camera, whatever its last status said,
        and a loop wedged mid-frame must NOT be reaped just because it looks
        finished.
        """
        now = time.monotonic() if now is None else now
        retention = self.settings.run_retention_s
        with self._lock:
            for run_id in list(self._runs):
                thread = self._threads.get(run_id)
                if thread is not None and thread.is_alive():
                    self._settled_at.pop(run_id, None)
                else:
                    self._settled_at.setdefault(run_id, now)
            reaped = [
                run_id
                for run_id, at in self._settled_at.items()
                if now - at >= retention
            ]
            for run_id in reaped:
                self._runs.pop(run_id, None)
                self._threads.pop(run_id, None)
                self._settled_at.pop(run_id, None)
        return reaped

    def _is_live(self, run_id: str) -> bool:
        """Is run_id a run of this process that is STILL RUNNING?

        LIVENESS, not membership — the distinction is the whole point.  A loop
        uses this to decide whether ingest's 409 means "a sibling is using the
        camera" (fail the start; seizing would corrupt that run's count) or "a
        corpse is holding the slot" (seize it).  Answering "yes, I have heard
        of it" for a run that settled hours ago wedged the camera permanently:
        the automatic seizure never fired, ``manager.stop`` set an event on a
        thread that had already returned, and the only escape was restarting
        ingest by hand — while the refusal told the operator to stop a run that
        was already stopped.
        """
        with self._lock:
            loop = self._runs.get(run_id)
        # status() takes the loop's own lock, so it is called OUTSIDE ours.
        return loop is not None and loop.status().get("state") in LIVE_STATES

    def _matching(self, run_id: str) -> list[RunLoop]:
        """Every loop answering to run_id — the runner's own id OR the
        planner's row id.

        The planner console only ever holds its row id (this runner creates
        that row and reports under it), so status and stop must answer to
        both; with only the memory key, every stop from the console 404'd
        as "unknown run" while the loop kept counting.

        A LIST, deliberately: more than one loop can end up bound to one
        planner row (observed live — a double start), and resolving "the
        first match" let an already-ended loop mask a live sibling, so the
        console's stop stopped nothing while the pipeline kept pulling
        frames.
        """
        self.reap()
        with self._lock:
            exact = self._runs.get(run_id)
            if exact:
                return [exact]
            loops = list(self._runs.values())
        # Each status() takes that loop's own lock; taking them while holding
        # the registry lock would nest two lock families for no reason.
        return [loop for loop in loops if loop.status().get("plannerRunId") == run_id]

    def get(self, run_id: str) -> dict | None:
        """Return the live status dict for a run, or None if unknown.

        When several loops share the id, the LIVE one answers — the console
        is asking about the run it can still affect, not the corpse.
        """
        loops = self._matching(run_id)
        if not loops:
            return None
        live = [x for x in loops if x.status().get("state") == "running"]
        return (live[-1] if live else loops[-1]).status()

    def stop(self, run_id: str) -> bool:
        """Signal EVERY loop answering to run_id; True if any exists."""
        loops = self._matching(run_id)
        for loop in loops:
            loop.stop()
        return bool(loops)
