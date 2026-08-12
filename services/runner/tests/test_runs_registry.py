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

    def __init__(self, planner_run_id, state="running"):
        self.planner_run_id = planner_run_id
        self.state = state
        self.stopped = False

    def status(self):
        """The fields the registry reads, shaped like RunLoop.status()."""
        return {"plannerRunId": self.planner_run_id, "state": self.state}

    def stop(self):
        """Record the signal the registry is expected to deliver."""
        self.stopped = True


class DeadThread:
    """A thread handle that has already returned (a settled run)."""

    def is_alive(self):
        """Stand in for threading.Thread.is_alive on a finished run."""
        return False


def make_manager_with(runs, threads=None, **settings_kw):
    """A RunManager with stub loops injected under given memory keys."""
    mgr = RunManager(Settings(**settings_kw))
    mgr._runs.update(runs)
    mgr._threads.update(threads or {})
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


def test_shared_planner_id_stops_every_loop_and_answers_with_the_live_one():
    """Two loops bound to ONE planner row (a double start, observed live):
    stop must signal both — the first-match answer let an ended loop mask
    the live one, so the console's stop stopped nothing — and status must
    describe the loop the operator can still affect."""
    ended = StubLoop("prun-1", state="ended")
    alive = StubLoop("prun-1", state="running")
    mgr = make_manager_with({"run-aaaaaaaa": ended, "run-bbbbbbbb": alive})

    assert mgr.get("prun-1")["state"] == "running", "the live loop answers, not the corpse"
    assert mgr.stop("prun-1") is True
    assert ended.stopped is True
    assert alive.stopped is True


# ------------------------------------- liveness, not membership (point C)


def test_a_settled_run_is_not_a_live_sibling():
    """The test that decides whether a camera can be taken back.

    Regression: this was a bare `run_id in self._runs`, so a run that settled
    hours ago still read as a live sibling. That is the test a loop uses to
    tell ingest's 409 "a sibling is counting on this camera" (fail, correctly
    — seizing would corrupt that run's count) from "a corpse is holding the
    slot" (seize it). A run whose /close never landed — ingest down or
    restarting at the moment it settled — therefore held the camera for the
    life of the runner process: the automatic seizure never fired, `stop` set
    an event on a thread that had already returned, and the refusal told the
    operator to stop a run that had already stopped. Only a manual ingest
    restart cleared it.
    """
    mgr = make_manager_with({
        "run-live": StubLoop("prun-1", state="running"),
        "run-starting": StubLoop("prun-2", state="starting"),
        "run-enrolling": StubLoop("prun-3", state="enrolling"),
        "run-ended": StubLoop("prun-4", state="ended"),
        "run-failed": StubLoop("prun-5", state="failed"),
    })
    assert mgr._is_live("run-live") is True
    assert mgr._is_live("run-starting") is True, "a run mid-startup owns its camera"
    assert mgr._is_live("run-enrolling") is True
    assert mgr._is_live("run-ended") is False, "a settled run holds nothing"
    assert mgr._is_live("run-failed") is False
    assert mgr._is_live("run-never-heard-of") is False


def test_settled_runs_are_reaped_so_the_registry_stays_bounded():
    """Nothing ever deleted from _runs/_threads, and a RunLoop is not small.

    Each retained loop pins `_last` — the last frame's full base64 JPEG — so a
    venue running gates all night accumulated hundreds of kilobytes per
    finished run, and `_matching` walked every run the process had ever seen
    on every console poll.
    """
    mgr = make_manager_with(
        {"run-old": StubLoop("prun-1", state="ended"),
         "run-live": StubLoop("prun-2", state="running")},
        threads={"run-old": DeadThread()},   # settled: its thread has returned
        run_retention_s=0.0,                 # retention window already expired
    )
    # A live run has no thread handle here, so pin the survivor on the loop
    # that IS still threaded rather than on the accident of an empty dict.
    mgr._threads["run-live"] = type("Alive", (), {"is_alive": lambda self: True})()

    assert mgr.reap() == ["run-old"]
    assert set(mgr._runs) == {"run-live"}
    assert set(mgr._threads) == {"run-live"}
    assert mgr._settled_at == {}


def test_a_settled_run_survives_its_retention_window():
    """An operator whose run has just finished must still be able to read its
    final status from GET /runs/:id — the reap is housekeeping, not a race."""
    mgr = make_manager_with(
        {"run-just-done": StubLoop("prun-1", state="ended")},
        threads={"run-just-done": DeadThread()},
        run_retention_s=600.0,
    )
    assert mgr.reap() == []
    assert mgr.get("run-just-done")["state"] == "ended"
    # ...and it goes once the window has passed (monotonic clock injected).
    assert mgr.reap(now=mgr._settled_at["run-just-done"] + 601.0) == ["run-just-done"]
    assert mgr.get("run-just-done") is None


def test_a_wedged_run_is_never_reaped_even_if_its_status_looks_settled():
    """Thread liveness is the settle signal, not the status dict.

    A loop blocked mid-frame is still holding the camera whatever its last
    published status said, and reaping it would hand that camera to the next
    run — two loops, one stream, a silently wrong invoice.
    """
    alive = type("Alive", (), {"is_alive": lambda self: True})()
    mgr = make_manager_with(
        {"run-wedged": StubLoop("prun-1", state="ended")},
        threads={"run-wedged": alive},
        run_retention_s=0.0,
    )
    assert mgr.reap() == []
    assert "run-wedged" in mgr._runs


def test_inbound_auth_gate_refuses_the_open_lan_when_armed(monkeypatch):
    """HECO_REQUIRE_AUTH=1 turns the LAN door off (runbook step 8).

    No credential -> 401 with the machine-readable code; a token signed by the
    auth service passes; /health stays open for the compose healthcheck. The
    gate sits ahead of routing, so an unknown path proves both halves: 401
    without a credential, 404 — the router's own answer — with one. Every
    other test in this file runs unarmed and is untouched.

    The key set is PINNED here (HECO_JWKS_JSON) so the check is hermetic: no
    network, no Worker, no clock beyond the token's own expiry.
    """
    import base64
    import json as _json
    import time as _time

    from app.main import app
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from fastapi.testclient import TestClient

    def b64(raw):
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    key = Ed25519PrivateKey.generate()
    jwks = {"keys": [{
        "kty": "OKP", "crv": "Ed25519", "kid": "k-test", "alg": "EdDSA", "use": "sig",
        "x": b64(key.public_key().public_bytes_raw()),
    }]}
    head = b64(_json.dumps({"alg": "EdDSA", "kid": "k-test", "typ": "JWT"}).encode())
    body = b64(_json.dumps({
        "iss": "https://auth.test", "aud": "heco-planner",
        "iat": int(_time.time()), "exp": int(_time.time()) + 3600,
    }).encode())
    token = f"{head}.{body}.{b64(key.sign(f'{head}.{body}'.encode()))}"

    monkeypatch.setenv("HECO_REQUIRE_AUTH", "1")
    monkeypatch.setenv("HECO_JWKS_JSON", _json.dumps(jwks))
    monkeypatch.setenv("HECO_AUTH_ISSUER", "https://auth.test")
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    refused = client.get("/gate-probe")
    assert refused.status_code == 401
    assert refused.json()["code"] == "auth"

    allowed = client.get("/gate-probe", headers={"Authorization": f"Bearer {token}"})
    assert allowed.status_code == 404, "a valid credential reaches the router itself"


# ------------------------------------------------------------------ build id


def test_health_reports_a_build_id_that_changes_with_the_source():
    """The question a deploy raises is 'is this box running what I think'.

    VERSION has read "0.1.0" on every box through every deploy this pipeline
    has ever had, so it cannot answer it. Two of the four boxes are not git
    checkouts, so a sha would not be askable there — and a sha would not
    notice a file edited in place on the box either. So the build id hashes
    the SOURCE.
    """
    from app.main import BUILD_ID, _build_id

    assert BUILD_ID != "unknown", "the build id must compute in a normal tree"
    assert len(BUILD_ID) == 12 and all(c in "0123456789abcdef" for c in BUILD_ID)
    assert _build_id() == BUILD_ID, "it must be stable across calls"


def test_the_build_id_is_deterministic_and_content_sensitive(tmp_path, monkeypatch):
    """Same content -> same id; one changed byte -> different id.

    Both halves matter. Without determinism two identical boxes would report
    different builds and every deploy check would cry wolf; without content
    sensitivity a stale container would report the same build as a fresh one,
    which is the failure this exists to catch.
    """
    import hashlib
    from pathlib import Path

    def digest_of(root: Path) -> str:
        d = hashlib.sha256()
        for p in sorted(root.rglob("*.py")):
            d.update(str(p.relative_to(root)).encode())
            d.update(p.read_bytes())
        return d.hexdigest()[:12]

    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "m.py").write_text("x = 1\n")
    assert digest_of(a) == digest_of(b), "identical trees must agree"

    (b / "pkg" / "m.py").write_text("x = 2\n")
    assert digest_of(a) != digest_of(b), "one changed byte must be visible"
