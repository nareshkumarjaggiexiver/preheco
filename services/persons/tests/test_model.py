

def test_thread_count_respects_the_cpuset_and_caps_itself(monkeypatch):
    """Pinning the container to one NUMA node broke ORT's own thread pinning.

    Left implicit, ORT counts the machine's cores (64 on the dual-socket
    T440), spawns that many threads and pins each to a chosen CPU — including
    CPUs on the socket the container no longer owns, which fails EINVAL and
    spams the log on every start. The number must come from what the process
    may actually use, and be capped so a small graph is not oversubscribed.
    """
    import os

    from app.model import _default_threads

    monkeypatch.delenv("PERSONS_THREADS", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(32)))
    assert _default_threads() == 8, "capped, not one-per-core"

    # A tighter cpuset wins over the cap — never ask for CPUs we do not have.
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 2})
    assert _default_threads() == 2

    # And an operator can still say.
    monkeypatch.setenv("PERSONS_THREADS", "16")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(32)))
    assert _default_threads() == 16
