"""TokenProvider: refresh before expiry, never block the loop, never lie.

Every test here drives a fake clock and a fake transport, so the whole 48-hour
lifecycle runs in microseconds and nothing touches a socket.

    cd common && .venv/bin/pytest tests/test_auth.py
"""

from __future__ import annotations

import json
import stat
import threading
import time

import pytest
from heco_common.auth import RETRY_FLOOR_S, TokenProvider


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        """The current fake time."""
        return self.t

    def advance(self, seconds: float) -> None:
        """Move the fake clock forward."""
        self.t += seconds


class FakeAuth:
    """Counts mints, and can be told to fail or to answer slowly."""

    def __init__(self, lifetime: float = 48 * 3600) -> None:
        self.lifetime = lifetime
        self.calls = 0
        self.status = 200
        self.error: Exception | None = None
        self.gate: threading.Event | None = None
        self.seen: list[dict] = []

    def __call__(self, url: str, payload: dict) -> tuple[int, dict]:
        """Answer one /token call, honouring the current failure settings."""
        self.calls += 1
        self.seen.append(payload)
        if self.gate is not None:
            self.gate.wait(timeout=5)
        if self.error is not None:
            raise self.error
        if self.status >= 400:
            return self.status, {"code": "invalid_client", "error": "no"}
        return 200, {
            "access_token": f"tok-{self.calls}",
            "token_type": "Bearer",
            "expires_in": self.lifetime,
            "scope": "planner:report",
        }


def provider(auth: FakeAuth, clock: Clock, **kw) -> TokenProvider:
    """A provider wired to the fake service and the fake clock."""
    kw.setdefault("jitter_s", 0)  # deterministic unless a test asks otherwise
    return TokenProvider(
        "http://auth.example", "app-1", "sekrit", transport=auth, now=clock, **kw
    )


def test_first_call_mints_and_sends_the_application_secret():
    """The credential on the wire is the application secret, and nothing else."""
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    assert p.token() == "tok-1"
    assert auth.seen == [{"app_id": "app-1", "app_secret": "sekrit"}]
    assert p.expires_at == clock.t + 48 * 3600


def test_the_token_is_reused_until_its_refresh_moment():
    """An unexpired token is a valid credential; re-minting one is pure waste."""
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    p.token()
    for _ in range(50):
        clock.advance(60)
        assert p.token() == "tok-1"
    assert auth.calls == 1, "an unexpired token must not be re-minted"


def test_refresh_happens_at_half_life_so_a_full_day_of_runway_is_always_in_hand():
    """THE NUMBER THAT MATTERS. Whenever the internet fails, the token in hand has at least 24
    h left — longer than any event.
    """
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    p.token()
    clock.advance(24 * 3600 - 1)
    assert p.token() == "tok-1" and auth.calls == 1

    clock.advance(2)  # now past half-life
    p.token()
    _settle(p)
    assert auth.calls == 2
    assert p.token() == "tok-2"
    assert p.expires_at - clock.t == pytest.approx(48 * 3600, abs=2)


def test_a_short_lived_token_refreshes_at_ITS_half_life_not_immediately():
    """The lead is 24 h by default. Applied unclamped to a 60 s test token it lands in the
    past, so the token is due for refresh the instant it is minted and EVERY call re-mints —
    a token loop, not a token.
    """
    auth, clock = FakeAuth(lifetime=60), Clock()
    p = provider(auth, clock)
    p.token()
    for _ in range(5):
        clock.advance(1)
        p.token()
        _settle(p)
    assert auth.calls == 1, "a freshly minted short token must not be due already"

    clock.advance(31)  # past half of 60 s, still valid
    p.token()
    _settle(p)
    assert auth.calls == 2, "and it does refresh at ITS half-life"


def test_a_refresh_that_is_due_does_not_make_the_caller_wait():
    """Property 1: no reporting call ever blocks on a refresh. The frame loop reports every ~2
    s; a thread parked on a socket is lost frames.
    """
    auth, clock = FakeAuth(), Clock()
    auth.gate = threading.Event()  # the auth service will not answer yet
    p = provider(auth, clock)
    auth.gate.set()
    p.token()  # first mint, blocking, as it must be

    auth.gate.clear()
    clock.advance(25 * 3600)  # refresh is now due
    assert p.token() == "tok-1", "the good token comes back immediately"
    auth.gate.set()
    _settle(p)
    assert p.token() == "tok-2"


def test_eight_threads_discovering_expiry_together_mint_once():
    """Property 2. Every tap noticing at the same millisecond must not become eight HTTP calls."""
    auth, clock = FakeAuth(), Clock()
    auth.gate = threading.Event()
    p = provider(auth, clock)

    start = threading.Barrier(8)
    out: list[str | None] = [None] * 8

    def ask(i: int) -> None:
        start.wait()
        out[i] = p.token()

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    auth.gate.set()
    for t in threads:
        t.join(timeout=5)

    assert auth.calls == 1, f"single-flight failed: {auth.calls} mints"
    assert out == ["tok-1"] * 8


def test_an_unreachable_auth_service_keeps_the_token_it_already_has():
    """Refusing to work because a refresh failed would trade a real outage for a hypothetical
    one. The token is still valid; use it.
    """
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    p.token()

    auth.error = OSError("network is unreachable")
    clock.advance(25 * 3600)
    assert p.token() == "tok-1"
    _settle(p)
    assert p.refresh_failures >= 1
    assert "could not reach the auth service" in (p.last_error or "")
    assert p.token() == "tok-1", "still reporting on the token in hand"


def test_a_failing_service_is_not_hammered_once_per_call():
    """A down auth service must not turn every reporting call into a failed request."""
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    p.token()
    auth.error = OSError("down")
    clock.advance(25 * 3600)

    for _ in range(20):
        p.token()
        _settle(p)
    assert auth.calls == 2, "one failed attempt, then the retry floor holds"

    clock.advance(RETRY_FLOOR_S + 1)
    p.token()
    _settle(p)
    assert auth.calls == 3, "and it tries again once the floor has passed"


def test_a_rejected_application_says_which_variables_to_check():
    """Read at a venue by someone whose pipeline will not report: name variables."""
    auth, clock = FakeAuth(), Clock()
    auth.status = 401
    p = provider(auth, clock)
    assert p.token() is None
    assert "HECO_APP_ID" in (p.last_error or "")
    assert "HECO_APP_SECRET" in (p.last_error or "")


def test_no_token_and_no_service_is_an_empty_header_not_an_exception():
    """The best-effort paths call this. An unauthenticated request that comes back 401 is
    counted and visible; an exception here would end a run.
    """
    auth, clock = FakeAuth(), Clock()
    auth.error = OSError("down")
    p = provider(auth, clock)
    assert p.auth_header() == {}
    assert p.auth_header(block=True) == {}


def test_a_non_blocking_caller_still_acquires_a_first_token():
    """THE BUG THIS PINS, found by the runner's integration test: a non-blocking read used to
    return None *and do nothing about it*, so a provider whose first caller did not wait
    never acquired a token at all. Every request went out unauthenticated, all night, with
    no explanation.
    """
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    assert p.token(block=False) is None, "it does not wait"
    _settle(p)
    assert auth.calls == 1, "but it did go and get one"
    assert p.token(block=False) == "tok-1"


def test_blocking_waits_for_the_first_token_and_never_for_a_refresh():
    """The distinction TokenAuth relies on. Without a token there is nothing to send; with one,
    a refresh must never hold the frame loop.
    """
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    assert p.token(block=True) == "tok-1", "waits when there is nothing"

    auth.gate = threading.Event()  # the next mint will hang
    clock.advance(25 * 3600)
    assert p.token(block=True) == "tok-1", "does NOT wait when a good token is in hand"
    auth.gate.set()
    _settle(p)


def test_auth_header_is_non_blocking_by_default():
    """The best-effort paths ask for a header; they must never wait for one."""
    auth, clock = FakeAuth(), Clock()
    auth.gate = threading.Event()  # would hang a blocking caller
    p = provider(auth, clock)
    assert p.auth_header() == {}, "must not wait on the auth service"
    auth.gate.set()


def test_force_refresh_replaces_a_token_that_was_refused():
    """The ordinary rotation case: refused, re-minted, carrying on."""
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    assert p.token() == "tok-1"
    assert p.force_refresh() == "tok-2"
    assert p.token() == "tok-2"


def test_force_refresh_that_cannot_reach_the_service_keeps_the_old_token():
    """A runner holding nothing reports nothing. If the mint fails — the same outage that would
    explain a lot of 401s — keeping the refused token costs one more refused request and
    preserves the chance the refusal was a blip.
    """
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    p.token()
    auth.error = OSError("down")
    assert p.force_refresh() == "tok-1"
    assert p.refresh_failures == 1


def test_force_refresh_is_not_throttled_by_the_retry_floor():
    """The floor throttles POLLING. This is a response to a live refusal, and making a rotation
    wait a minute would strand a run for that minute.
    """
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    p.token()
    auth.error = OSError("down")
    p.force_refresh()
    assert auth.calls == 2

    auth.error = None
    assert p.force_refresh() == "tok-3", "no waiting on the floor"
    assert auth.calls == 3


def test_a_due_refresh_spawns_ONE_thread_however_many_callers_notice(monkeypatch):
    """Not just one mint — one THREAD. With a 2 s report cadence, dozens of callers can notice
    the same due refresh before it lands, and a thread per caller is a thread pile-up inside
    the frame loop's process.
    """
    import heco_common.auth as auth_module

    spawned = []

    class CountingThread(threading.Thread):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            spawned.append(self)

    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    p.token()

    gate = threading.Event()
    auth.gate = gate
    monkeypatch.setattr(auth_module.threading, "Thread", CountingThread)
    clock.advance(25 * 3600)
    for _ in range(20):
        assert p.token() == "tok-1"
    gate.set()
    for t in spawned:
        t.join(timeout=5)

    assert len(spawned) == 1, f"{len(spawned)} refresh threads for one due refresh"


def test_mark_rejected_costs_nothing_now_and_refreshes_on_the_next_read():
    """The best-effort contract: flag it, never spend a round-trip inline."""
    auth, clock = FakeAuth(), Clock()
    p = provider(auth, clock)
    p.token()
    before = auth.calls
    p.mark_rejected()
    assert auth.calls == before, "mark_rejected must not touch the network"

    p.token()
    _settle(p)
    assert auth.calls == before + 1


def test_jitter_spreads_a_fleet_and_stays_inside_its_bounds():
    """Every runner in a fleet was started by the same deploy and would otherwise wake to
    refresh in the same second.
    """
    seen = []

    def rng(lo: float, hi: float) -> float:
        seen.append((lo, hi))
        return hi  # the extreme, to prove the bound

    auth, clock = FakeAuth(), Clock()
    p = TokenProvider(
        "http://auth.example", "a", "b", transport=auth, now=clock, jitter_s=300, rng=rng
    )
    p.token()
    assert seen == [(0.0, 300)]
    # Jitter moves the refresh EARLIER, never later: a fleet spreads out ahead
    # of the deadline. Pushed the other way it eats into the outage budget the
    # 24-hour lead exists to guarantee.
    assert p._refresh_at == pytest.approx(p.expires_at - 24 * 3600 - 300)


def test_a_service_that_answers_without_a_lifetime_is_refused():
    """A token with no expiry is not a short-lived token; refuse it outright."""
    class NoLifetime(FakeAuth):
        def __call__(self, url, payload):
            self.calls += 1
            return 200, {"access_token": "x", "expires_in": 0}

    clock = Clock()
    p = provider(NoLifetime(), clock)
    assert p.token() is None
    assert "no usable lifetime" in (p.last_error or "")


def _settle(p: TokenProvider) -> None:
    """Wait for any background refresh to finish. Threads are daemons started
    inside token(); tests need them landed before asserting on counts."""
    for t in threading.enumerate():
        if t.name == "heco-token-refresh":
            t.join(timeout=5)


# ---------------------------------------------------- the offline restart

def _stub_transport(token="tok-1", lifetime=48 * 3600, calls=None):
    def transport(url, payload):
        if calls is not None:
            calls.append(url)
        return 200, {"access_token": token, "expires_in": lifetime}
    return transport


def test_a_minted_token_is_written_to_disk(tmp_path):
    """The cache is what makes an offline restart possible at all."""
    cache = tmp_path / "token.json"
    p = TokenProvider("http://auth", "runner", "sekrit",
                      transport=_stub_transport(), jitter_s=0, cache_path=str(cache))
    assert p.token() == "tok-1"
    assert cache.exists()

    written = json.loads(cache.read_text())
    assert written["access_token"] == "tok-1"
    assert written["app_id"] == "runner"
    assert written["expires_at"] > time.time()
    # A live bearer token must not be readable by every process on the box.
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600


def test_a_restart_with_no_network_reuses_the_cached_token(tmp_path):
    """THE POINT OF THE WHOLE FEATURE.

    First process mints and writes. Second process starts with an auth service
    it cannot reach — and still has a credential, so the night still reports.
    """
    cache = tmp_path / "token.json"
    first = TokenProvider("http://auth", "runner", "sekrit",
                          transport=_stub_transport(), jitter_s=0, cache_path=str(cache))
    assert first.token() == "tok-1"

    def dead(url, payload):
        raise OSError("no route to host")

    second = TokenProvider("http://auth", "runner", "sekrit",
                           transport=dead, jitter_s=0, cache_path=str(cache))
    assert second.loaded_from_cache is True
    assert second.token(block=False) == "tok-1"
    assert second.auth_header() == {"Authorization": "Bearer tok-1"}


def test_an_expired_cache_is_not_adopted(tmp_path):
    """A dead token is worse than none: it would be refused and read as a
    broken auth service rather than a stale file."""
    cache = tmp_path / "token.json"
    cache.write_text(json.dumps({
        "app_id": "runner", "access_token": "stale", "expires_at": time.time() - 10,
    }))
    calls = []
    p = TokenProvider("http://auth", "runner", "sekrit",
                      transport=_stub_transport(calls=calls), jitter_s=0, cache_path=str(cache))
    assert p.loaded_from_cache is False
    assert p.token() == "tok-1"
    assert len(calls) == 1, "it had to mint, because the cache was dead"


def test_a_cache_from_a_different_application_is_ignored(tmp_path):
    """Two applications sharing a volume must not adopt each other's tokens."""
    cache = tmp_path / "token.json"
    cache.write_text(json.dumps({
        "app_id": "eval", "access_token": "not-mine", "expires_at": time.time() + 9999,
    }))
    p = TokenProvider("http://auth", "runner", "sekrit",
                      transport=_stub_transport(), jitter_s=0, cache_path=str(cache))
    assert p.loaded_from_cache is False
    assert p.token() == "tok-1"


@pytest.mark.parametrize("junk", ["", "{", "null", '{"access_token": 5}', '[]'])
def test_a_corrupt_cache_degrades_to_minting(tmp_path, junk):
    """Unreadable is not fatal: boot must never die on a bad cache file."""
    cache = tmp_path / "token.json"
    cache.write_text(junk)
    p = TokenProvider("http://auth", "runner", "sekrit",
                      transport=_stub_transport(), jitter_s=0, cache_path=str(cache))
    assert p.loaded_from_cache is False
    assert p.token() == "tok-1"


def test_an_unwritable_cache_never_stops_the_runner(tmp_path, capsys):
    """A housekeeping failure must not become an outage."""
    p = TokenProvider("http://auth", "runner", "sekrit",
                      transport=_stub_transport(), jitter_s=0,
                      cache_path="/proc/definitely/not/writable/token.json")
    assert p.token() == "tok-1"
    # ...and the complaint must not carry the token itself.
    assert "tok-1" not in capsys.readouterr().out


def test_persistence_is_opt_in(tmp_path):
    """No cache_path, no file: writing a bearer token to disk is a decision."""
    p = TokenProvider("http://auth", "runner", "sekrit", transport=_stub_transport(), jitter_s=0)
    assert p.token() == "tok-1"
    assert list(tmp_path.iterdir()) == []
