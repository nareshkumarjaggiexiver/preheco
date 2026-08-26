"""Inbound token verification for the pipeline services — runbook step 8.

The planner has verified heco-auth's Ed25519 tokens since the migration; the
pipeline services accepted anything on the LAN. This module is the missing
half: a verifier the services can hold that answers one question — *is this
bearer token a live credential minted by our auth service?* — without any
request-path call to that service.

THE CONSTRAINT IS THE PLANNER'S, INHERITED WHOLE: **a venue must keep
counting when its internet is down.** So verification is local (Ed25519
public keys, cached), and the key set survives a restart without the network:

    keys, in the order they are trusted at boot:
      1. ``HECO_JWKS_JSON``   — a pinned key set in the environment. Garbage
                                here raises AT CONSTRUCTION: a misconfigured
                                pin must fail the deploy, not every request.
      2. the disk cache       — whatever the last successful fetch wrote
                                (``HECO_JWKS_CACHE``); an offline restart
                                boots trusting what it trusted yesterday.
      3. the network          — {auth_url}/.well-known/jwks.json, fetched
                                lazily (never at construction, which must not
                                block a service boot on a WAN), re-fetched
                                six-hourly IN THE BACKGROUND and on an
                                unknown ``kid`` inline, floored at one fetch
                                a minute so junk tokens bearing random kids
                                cannot make this service hammer the Worker.

Checks, and why each exists:
  - ``alg`` must be exactly ``EdDSA`` — read from OUR expectation, never the
    token's word for it; accepting the header's choice is the classic
    algorithm-confusion hole ("alg":"none", HS256-with-public-key).
  - signature over ``header.payload`` with the ``kid``'s Ed25519 key.
  - ``exp`` required and in the future, ``nbf`` honoured if present, both
    with 60 s of tolerance — LAN boxes have no NTP discipline between them,
    and a token refused because two clocks disagree by four seconds is an
    outage with no findable cause (the planner's constant, same argument).
  - ``iss`` pinned to the auth service's URL. Exact match, fail closed.
  - ``aud`` must contain ``heco-planner`` — the ONE audience heco-auth mints.
    The runner, the planner and eval all present the same lab-wide credential
    to every internal surface; a per-service audience would triple the number
    of applications to rotate for no attacker this threat model contains.
    ``scope`` is not enforced HERE — verification answers "who is this",
    and the claims (scope included) ride back to the caller. Authorization
    lives one layer up: the bearer gate holds a per-path scope table
    (gate_auth.SCOPE_RULES) so control surfaces like the runner's
    ``POST /models/apply`` can demand ``planner:operate`` while the report
    paths keep accepting the fleet's ``planner:report`` machine tokens.
    The 2026-08 review showed why the split matters: without it, any box's
    reporting credential could hot-swap models on an armed install.

Only stdlib + ``cryptography`` (the Ed25519 primitive; hand-rolling
signature verification is how auth modules become CVEs).
"""

from __future__ import annotations

import base64
import binascii
import json
import threading
import time
import urllib.request
from collections.abc import Callable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

#: The one audience heco-auth mints (see apps/heco-auth/src/tokens.js).
AUDIENCE = "heco-planner"

#: Clock slack between LAN boxes with no shared NTP discipline.
CLOCK_TOLERANCE_S = 60

#: Scheduled re-fetch cadence. Keys change about never; this exists so a key
#: published today is trusted well before it first signs (rotation step 2).
JWKS_REFRESH_S = 6 * 60 * 60

#: Floor between unknown-kid fetches — the anti-hammer guard, measured from
#: the last kid-TRIGGERED fetch only, so a rotation published a minute after
#: boot is adopted when its first token arrives, not six hours later.
JWKS_MIN_INTERVAL_S = 60

#: How long a JWKS fetch may take. Nothing on a frame path ever waits on
#: this: scheduled refreshes run on their own thread, and the only inline
#: fetch is for a kid this verifier has never seen — a request that would
#: fail anyway without the key.
FETCH_TIMEOUT_S = 5.0


class TokenRefused(Exception):
    """One refused token, with the reason a log line can carry.

    The reason names the CHECK that failed, never the expected value: an
    attacker probing with junk learns "bad issuer", not which issuer to forge.
    """


#: (url, timeout) -> decoded JSON body. Injectable so tests need no socket.
JwksFetch = Callable[[str, float], dict]


def _urllib_fetch(url: str, timeout: float = FETCH_TIMEOUT_S) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as res:  # noqa: S310 — https URL from config
        return json.loads(res.read())


def _b64url(data: str) -> bytes:
    """Decode base64url with the padding JWTs strip; raise ValueError on junk."""
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"not base64url: {exc}") from exc


def _keys_from_jwks(jwks: dict) -> dict[str, Ed25519PublicKey]:
    """kid -> public key, keeping only the Ed25519 signing keys we can use.

    Unusable entries are SKIPPED, not fatal: a JWKS that one day carries an
    RSA key for something else must not take the working Ed25519 keys with it.
    A retired key simply stops being published and drops out on refresh.
    """
    out: dict[str, Ed25519PublicKey] = {}
    for key in jwks.get("keys") or []:
        if not isinstance(key, dict):
            continue
        if key.get("kty") != "OKP" or key.get("crv") != "Ed25519":
            continue
        kid, x = key.get("kid"), key.get("x")
        if not kid or not x:
            continue
        try:
            out[str(kid)] = Ed25519PublicKey.from_public_bytes(_b64url(str(x)))
        except (ValueError, TypeError):
            continue  # one malformed key must not poison the set
    return out


class JwksVerifier:
    """Holds the key set; answers verify(token) locally. Thread-safe."""

    def __init__(
        self,
        auth_url: str | None = None,
        *,
        issuer: str | None = None,
        audience: str = AUDIENCE,
        pinned_jwks: str | dict | None = None,
        cache_path: str | None = None,
        fetch: JwksFetch | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.auth_url = auth_url.rstrip("/") if auth_url else None
        #: iss defaults to the auth url itself — heco-auth sets AUTH_ISSUER to
        #: its own public address, so one env serves both roles unless split.
        self.issuer = issuer or self.auth_url
        self.audience = audience
        self.cache_path = cache_path
        self._fetch = fetch or _urllib_fetch
        self._now = now

        self._lock = threading.Lock()
        self._refreshing = threading.Lock()
        self._keys: dict[str, Ed25519PublicKey] = {}
        self._refresh_at = 0.0  # next scheduled fetch; 0 = fetch when first needed
        self._kid_fetch_floor = 0.0
        #: When a fetch was last ATTEMPTED (success or not). The floor below
        #: makes concurrent refresh paths — the scheduled thread and an
        #: unknown-kid request racing at boot — collapse into one HTTP call,
        #: with the loser reusing the winner's result.
        self._last_attempt_at = float("-inf")

        if pinned_jwks:
            parsed = json.loads(pinned_jwks) if isinstance(pinned_jwks, str) else pinned_jwks
            keys = _keys_from_jwks(parsed)
            if not keys:
                # A pin that yields no usable key is a broken deploy, and a
                # broken deploy should break AT deploy.
                raise ValueError("HECO_JWKS_JSON is set but contains no usable Ed25519 key")
            self._keys = keys
            # A pinned set is authoritative by choice: no scheduled fetches.
            self._refresh_at = float("inf")
        elif cache_path:
            try:
                with open(cache_path, encoding="utf-8") as fh:
                    self._keys = _keys_from_jwks(json.load(fh))
            except (OSError, ValueError):
                pass  # no cache yet, or a torn write: the network leg covers it

    # ------------------------------------------------------------ verifying

    def verify(self, token: str) -> dict:
        """Return the claims of a valid token; raise TokenRefused otherwise."""
        self._maybe_scheduled_refresh()

        parts = token.split(".")
        if len(parts) != 3:
            raise TokenRefused("not a JWT")
        raw_header, raw_payload, raw_sig = parts
        try:
            header = json.loads(_b64url(raw_header))
            claims = json.loads(_b64url(raw_payload))
            signature = _b64url(raw_sig)
        except (ValueError, json.JSONDecodeError) as exc:
            raise TokenRefused(f"undecodable token: {exc}") from exc

        # OUR algorithm, never the token's suggestion.
        if header.get("alg") != "EdDSA":
            raise TokenRefused("wrong algorithm")

        kid = header.get("kid")
        if not kid:
            raise TokenRefused("no kid")
        key = self._key_for(str(kid))
        if key is None:
            raise TokenRefused("unknown signing key")

        try:
            key.verify(signature, f"{raw_header}.{raw_payload}".encode())
        except InvalidSignature as exc:
            raise TokenRefused("bad signature") from exc

        now = self._now()
        exp = claims.get("exp")
        if not isinstance(exp, (int, float)):
            raise TokenRefused("no expiry")
        if now > exp + CLOCK_TOLERANCE_S:
            raise TokenRefused("expired")
        nbf = claims.get("nbf")
        if isinstance(nbf, (int, float)) and now < nbf - CLOCK_TOLERANCE_S:
            raise TokenRefused("not yet valid")

        if self.issuer and claims.get("iss") != self.issuer:
            raise TokenRefused("bad issuer")
        aud = claims.get("aud")
        auds = aud if isinstance(aud, list) else [aud]
        if self.audience not in auds:
            raise TokenRefused("bad audience")

        return claims

    # ------------------------------------------------------------- key set

    def _key_for(self, kid: str) -> Ed25519PublicKey | None:
        with self._lock:
            key = self._keys.get(kid)
            if key is not None:
                return key
            # Unknown kid: the one case worth an INLINE fetch (the request
            # fails anyway without the key), floored so junk kids cannot turn
            # this service into a weapon against the Worker.
            if self._now() < self._kid_fetch_floor:
                return None
            self._kid_fetch_floor = self._now() + JWKS_MIN_INTERVAL_S
        self._refresh_now()
        with self._lock:
            return self._keys.get(kid)

    def _maybe_scheduled_refresh(self) -> None:
        """Six-hourly re-fetch, in the background — never on a frame path.

        Only once keys EXIST: the schedule's job is picking up a rotation
        early, and an empty set is the unknown-kid path's problem — inline,
        blocking, deterministic. Spawning a thread for the first-ever fetch
        would just race that path for the same HTTP call.
        """
        with self._lock:
            due = bool(self._keys) and self._now() >= self._refresh_at
        if not due or self._refreshing.locked():
            return
        threading.Thread(target=self._refresh_now, name="heco-jwks-refresh", daemon=True).start()

    def _refresh_now(self) -> None:
        if not self.auth_url:
            return
        with self._refreshing:
            with self._lock:
                # Whoever held the refresh lock may have just fetched on our
                # behalf; going again inside the floor would double every
                # boot-time fetch and hammer the Worker on failure.
                if self._now() - self._last_attempt_at < JWKS_MIN_INTERVAL_S:
                    return
                self._last_attempt_at = self._now()
            try:
                jwks = self._fetch(f"{self.auth_url}/.well-known/jwks.json", FETCH_TIMEOUT_S)
            except Exception:  # noqa: BLE001 — an unreachable Worker must not raise into a request
                with self._lock:
                    # Keep what we have; try again on the next schedule tick.
                    self._refresh_at = self._now() + JWKS_MIN_INTERVAL_S
                return
            keys = _keys_from_jwks(jwks if isinstance(jwks, dict) else {})
            with self._lock:
                if keys:
                    self._keys = keys
                self._refresh_at = self._now() + JWKS_REFRESH_S
            if keys and self.cache_path:
                try:
                    with open(self.cache_path, "w", encoding="utf-8") as fh:
                        json.dump(jwks, fh)
                except OSError:
                    pass  # the cache is an optimisation for offline boots, not a duty
