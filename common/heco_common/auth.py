"""TokenProvider — the runner's standing credential, refreshed before it lapses.

WHAT THIS REPLACES. The runner used to carry one shared secret, baked into its
container environment and read once at construction. Rotating it meant editing
a compose file and restarting two processes in the right order; getting the
order wrong cost four separate interruptions in a single day, and one copy was
scraped out of ``/proc`` by a tool that had no business reading it. A secret
that can only be changed by restarting the thing that uses it is a secret that
does not get changed.

Now the runner holds an *application secret* and exchanges it for a short-lived
token from the auth service. The secret is long-lived and stays put; the token
turns over on its own; nothing restarts.

THE NUMBER THAT MATTERS. Tokens live 48 h and this refreshes at half-life, so
the token in hand always has **at least 24 h** left on it. That is not a
security preference — it is an outage budget. A venue's internet can fail at
the worst possible moment on an event night and the runner still has a full day
of valid reporting, which is longer than any event. If the auth service cannot
be reached, the current token is KEPT and used; refusing to work because a
refresh failed would be trading a real outage for a hypothetical one.

Three properties this file exists to guarantee, in order of how much they cost
when absent:

1. **No reporting call ever blocks on a refresh.** The runner reports every
   ~2 s from the frame loop. A thread that stalls for a network round-trip
   stalls the count. So a refresh happens on ONE thread while everyone else
   keeps using the token already in hand (stale-while-revalidate), and only a
   caller with no token at all waits.
2. **One refresh, not N.** Eight taps discovering an expiry in the same
   millisecond must produce one HTTP call. Single-flight, under a lock.
3. **Never raise into the loop.** Failures are counted and readable
   (``refresh_failures``, ``last_error``) so the runner can say so out loud;
   they are never thrown at a caller who was trying to post a frame.

Stdlib only, like planner.py — the pipeline's dependency budget is spent on
models, not on HTTP libraries.
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

#: Refresh this far ahead of expiry. Half of the auth service's 48 h machine
#: token: see the outage-budget argument above.
DEFAULT_REFRESH_AHEAD_S = 24 * 60 * 60

#: Spread over the refresh moment. Every runner in a fleet was started by the
#: same deploy and would otherwise wake to refresh in the same second.
DEFAULT_JITTER_S = 300

#: A refresh that fails is not retried faster than this. Without a floor, a
#: down auth service turns every reporting call into a failed HTTP request.
RETRY_FLOOR_S = 60


class AuthError(RuntimeError):
    """Raised only by :meth:`TokenProvider.token` when there is nothing to hand
    back — no cached token and no way to mint one. Callers on the best-effort
    paths never see it; they ask for ``auth_header`` instead."""


#: (url, payload) -> (status, decoded-body). Injectable so tests need no socket.
TokenTransport = Callable[[str, dict], tuple[int, dict]]


def urllib_token_transport(url: str, payload: dict, timeout: float = 10.0) -> tuple[int, dict]:
    """Default transport: JSON in, JSON out. HTTP errors are RETURNED, not
    raised, so the provider decides what a 403 means; network failures raise."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read()
            return res.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"detail": raw.decode(errors="replace")}
        return exc.code, parsed


class TokenProvider:
    """Holds an application secret; hands out a valid bearer token.

    Thread-safe. Build one per process and share it::

        provider = TokenProvider(auth_url, app_id, app_secret)
        client = PlannerClient(planner_url, token_provider=provider)
    """

    def __init__(
        self,
        auth_url: str,
        app_id: str,
        app_secret: str,
        *,
        transport: TokenTransport | None = None,
        refresh_ahead_s: float = DEFAULT_REFRESH_AHEAD_S,
        jitter_s: float = DEFAULT_JITTER_S,
        now: Callable[[], float] = time.time,
        rng: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.auth_url = auth_url.rstrip("/")
        self.app_id = app_id
        self.app_secret = app_secret
        self._transport = transport or urllib_token_transport
        self._refresh_ahead_s = max(0.0, refresh_ahead_s)
        self._jitter_s = max(0.0, jitter_s)
        self._now = now
        self._rng = rng

        self._lock = threading.Lock()
        self._refreshing = threading.Lock()
        self._token: str | None = None
        self._expires_at: float = 0.0
        #: When to START trying — expiry minus the lead, jittered, computed once
        #: per token so every caller agrees and the jitter does not re-roll.
        self._refresh_at: float = 0.0
        self._next_attempt_at: float = 0.0

        #: Observable. The runner surfaces these; a silent auth failure is how
        #: a night of frames goes unreported with nothing in the log.
        self.refresh_failures = 0
        self.last_error: str | None = None
        self.mints = 0

    # ------------------------------------------------------------- reading

    @property
    def expires_at(self) -> float:
        """Epoch seconds at which the current token stops being accepted."""
        return self._expires_at

    def token(self, *, block: bool = True) -> str | None:
        """The token to send.

        Returns the cached token whenever one is usable — including while a
        refresh is due or failing — because a token with time left on it is
        still a valid credential and the alternative is not reporting.

        ``block=False`` never waits: it returns what is in hand, possibly None.
        That is what the best-effort paths use, so a tap can never be the thing
        that stalls the frame loop.
        """
        with self._lock:
            cached = self._token
            usable = cached is not None and self._now() < self._expires_at
            due = self._now() >= self._refresh_at

        if usable and not due:
            return cached
        if usable:
            # Stale-while-revalidate: hand back the good token immediately and
            # let ONE thread go and get the next one.
            self._refresh_in_background()
            return cached
        if not block:
            # Nothing usable and the caller will not wait. Start the mint
            # anyway: without this the provider NEVER acquires a first token —
            # every call returns None, every request goes out unauthenticated,
            # and the runner reports nothing all night with no explanation.
            self._refresh_in_background()
            return cached  # may be None; the caller asked not to wait
        self._refresh_now()
        with self._lock:
            return self._token

    def auth_header(self, *, block: bool = False) -> dict[str, str]:
        """``{"Authorization": "Bearer …"}``, or ``{}`` when there is nothing.

        Note what ``block=True`` does and does not do. It waits ONLY when there
        is no usable token at all — the case where not waiting guarantees a
        failed request. It never waits for a *refresh*, because a token with
        time left on it is a valid credential and the frame loop must not stop
        for one. Both branches are decided in :meth:`token`.

        With ``block=False`` an empty dict means the request goes out
        unauthenticated and comes back 401, which is counted and visible.
        """
        tok = self.token(block=block)
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    # ------------------------------------------------------------- writing

    def force_refresh(self) -> str | None:
        """Mint another token now, synchronously, and return it.

        Called after a 401 on a path that can afford one retry: the token was
        rejected, so its remaining lifetime is a lie — most likely this runner's
        application was rotated, or a clock is wrong.

        The refused token is NOT discarded up front, and that ordering is
        deliberate. If the mint fails — the same venue outage that would explain
        a lot of 401s — clearing it first would leave the runner holding nothing
        at all, and a runner with nothing reports nothing. Keeping it costs one
        more refused request and preserves the chance that the refusal was a
        blip. An over-report starts an argument; a silent under-report is just
        wrong, and this project takes the argument every time.

        The retry floor is cleared too: this is a response to a live refusal,
        not the polling loop the floor exists to throttle.
        """
        with self._lock:
            self._refresh_at = 0.0
            self._next_attempt_at = 0.0
        self._refresh_now()
        with self._lock:
            return self._token

    def mark_rejected(self) -> None:
        """Note that a token was refused, WITHOUT touching the network.

        For best-effort callers. They must not spend a refresh inline — the
        latency is the whole reason they are best-effort — but they also must
        not keep sending a credential that has already been refused all night.
        Flagging it means the next ordinary read refreshes.
        """
        with self._lock:
            self._refresh_at = 0.0

    # ------------------------------------------------------------ internals

    def _refresh_in_background(self) -> None:
        if self._refreshing.locked():
            return  # someone is already on it — property 2
        threading.Thread(target=self._refresh_now, name="heco-token-refresh", daemon=True).start()

    def _refresh_now(self) -> None:
        """Mint a token. Single-flight: concurrent callers wait for the one in
        progress and then use its result rather than each minting their own."""
        with self._refreshing:
            with self._lock:
                # Whoever held the lock may have just succeeded on our behalf.
                if self._token is not None and self._now() < self._refresh_at:
                    return
                if self._now() < self._next_attempt_at:
                    return  # a recent attempt failed; do not hammer
            self._mint()

    def _mint(self) -> None:
        try:
            status, body = self._transport(
                f"{self.auth_url}/token",
                {"app_id": self.app_id, "app_secret": self.app_secret},
            )
        except Exception as exc:  # noqa: BLE001 — a refresh never raises at a caller
            self._fail(f"could not reach the auth service: {exc}")
            return

        if status >= 400 or not body.get("access_token"):
            detail = body.get("error") or f"HTTP {status}"
            # 401/403 is a CONFIGURATION problem — a wrong secret or a revoked
            # application — and no amount of retrying fixes it. Say which.
            hint = (
                " — check HECO_APP_ID and HECO_APP_SECRET against the auth service"
                if status in (401, 403)
                else ""
            )
            self._fail(f"the auth service refused this application: {detail}{hint}")
            return

        # OAuth-shaped: expires_in is SECONDS FROM NOW, so there is no date
        # format to parse and no clock to agree on beyond our own.
        try:
            lifetime = float(body.get("expires_in") or 0)
        except (TypeError, ValueError):
            lifetime = 0.0
        if lifetime <= 0:
            self._fail(f"the auth service returned no usable lifetime: {body.get('expires_in')!r}")
            return

        now = self._now()
        with self._lock:
            self._token = str(body["access_token"])
            self._expires_at = now + lifetime
            # Refresh at half-life or at the configured lead, whichever comes
            # first — a short-TTL token must not be scheduled to refresh after
            # it has already expired.
            lead = min(self._refresh_ahead_s, lifetime / 2)
            jitter = self._rng(0.0, self._jitter_s) if self._jitter_s else 0.0
            self._refresh_at = max(now, self._expires_at - lead - jitter)
            self._next_attempt_at = 0.0
            self.last_error = None
            self.mints += 1

    def _fail(self, message: str) -> None:
        with self._lock:
            self.refresh_failures += 1
            self.last_error = message
            self._next_attempt_at = self._now() + RETRY_FLOOR_S
