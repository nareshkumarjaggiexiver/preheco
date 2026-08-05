"""RunManager registry: the id-answering contract.

The console (via the planner's control proxy) can only ever name a run by the
PLANNER row id — the runner created that row and reports under it. The
registry therefore answers get/stop by its own memory key AND by plannerRunId,
otherwise every stop from the console is "unknown run" while the loop keeps
counting. Stub loops keep this a pure registry test: no threads, no network.
"""

from app.config import Settings
from app.runs import RunManager


class StubLoop:
    """Just enough of RunLoop for the registry: status() and stop()."""

    def __init__(self, planner_run_id):
        self.planner_run_id = planner_run_id
        self.stopped = False

    def status(self):
        """The one field the registry reads, shaped like RunLoop.status()."""
        return {"plannerRunId": self.planner_run_id, "state": "running"}

    def stop(self):
        """Record the signal the registry is expected to deliver."""
        self.stopped = True


def make_manager_with(runs):
    """A RunManager with stub loops injected under given memory keys."""
    mgr = RunManager(Settings())
    mgr._runs.update(runs)
    return mgr


def test_get_and_stop_answer_to_both_ids():
    """One loop, addressable by the memory key and by its planner row id."""
    loop = StubLoop("prun-slug-from-label")
    mgr = make_manager_with({"run-abc12345": loop})

    # The runner's own id — the id POST /runs returned to a curl caller.
    assert mgr.get("run-abc12345")["state"] == "running"
    # The planner row id — the only id the console ever holds.
    assert mgr.get("prun-slug-from-label")["state"] == "running"

    assert mgr.stop("prun-slug-from-label") is True
    assert loop.stopped is True


def test_unknown_id_still_refused():
    """An id matching neither key nor plannerRunId stays a 404."""
    mgr = make_manager_with({"run-abc12345": StubLoop("prun-1")})
    assert mgr.get("prun-nope") is None
    assert mgr.stop("prun-nope") is False


def test_memory_key_wins_over_a_colliding_planner_id():
    """The exact memory key outranks another run's colliding plannerRunId."""
    # Pathological but cheap to pin: if a runner id ever equalled another
    # run's plannerRunId, the exact memory key must win.
    a = StubLoop("run-bbb22222")
    b = StubLoop("prun-2")
    mgr = make_manager_with({"run-aaa11111": a, "run-bbb22222": b})
    assert mgr.stop("run-bbb22222") is True
    assert b.stopped is True
    assert a.stopped is False
