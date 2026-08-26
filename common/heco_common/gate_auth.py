"""The inbound bearer gate every pipeline service installs — runbook step 8.

One middleware, seven services, armed by ONE switch:

    HECO_REQUIRE_AUTH=1     refuse inbound requests without a credential
    (unset / 0)             today's behaviour, bit for bit — open

The switch exists because these are exactly the paths that brick a lab when
auth goes wrong mid-event. Arming is a deliberate act after the credentials
are proven to flow (the runbook's verify-then-advance habit); disarming is
one env change and a `docker compose up -d`, which is the whole rollback.

ONE kind of credential is accepted: an Ed25519 token minted by heco-auth,
verified LOCALLY against the cached JWKS (heco_common.verify — offline-safe,
nothing on a request path calls the Worker). The shared secret that briefly
sat beside it was retired at runbook step 7; there is no long-lived string
that opens these services, which is the whole point of the exercise.

``/health`` stays open: the compose healthchecks poll it unauthenticated,
the deploy runbook verifies versions through it, and it holds nothing an
attacker wants beyond "alive" (ingest's `owner` run-id included — a run id
is not a credential).

Pure ASGI on purpose: heco-common does not depend on FastAPI, and a raw
middleware is testable with a fake scope and no web framework. Environment
is read PER REQUEST, not at install: a service module imported once by a
test session can still be armed and disarmed per test, and the cost is two
dict lookups.
"""

from __future__ import annotations

import json
import os

from .verify import JwksVerifier, TokenRefused

#: Paths that never require a credential. Exact match.
OPEN_PATHS = ("/health",)

#: Control paths that demand MORE than a valid token: (method, path) → the
#: scope the token must carry. Report paths stay open to the fleet's
#: ``planner:report`` machine tokens — that is what they exist for — but a
#: hot-swap is an operator's act, and the 2026-08 review showed the gap:
#: with no scope check, ANY box's reporting credential (including a lost
#: one, until its rotation) could switch models on an armed install. The
#: table is consulted only when the gate is armed; unarmed boxes are open
#: LAN exactly as before.
#:
#: HECO_SCOPE_ENFORCE=0 suspends the table on an armed box — a TEMPORARY,
#: per-box bridge for exactly one situation: the box armed before its
#: planner held an operate-scoped credential (scopes are immutable in
#: hecoa, so the upgrade is a new application + a .env change, not a
#: rotation). The default is ON because a security rule that ships
#: disarmed is a policy that does not exist; the runbook's step is
#: register-verify-delete-the-line, never "leave it".
SCOPE_RULES: dict[tuple[str, str], str] = {
    ("POST", "/models/apply"): "planner:operate",
}

TRUTHY = {"1", "true", "yes", "on"}


def _refusal_body(reason: str) -> bytes:
    # The planner's error dialect: `code` for machines, the sentence for a
    # human reading a log at midnight.
    return json.dumps({
        "detail": f"this pipeline service requires a bearer token ({reason}) — "
        "mint one from the auth service, or unset HECO_REQUIRE_AUTH to reopen the LAN",
        "code": "auth",
    }).encode()


def _scope_refusal_body(required: str) -> bytes:
    # 403, not 401: the credential is real, its rights are not enough. The
    # sentence names the exact scope so the fix is a registration, not an
    # afternoon of guessing.
    return json.dumps({
        "detail": f"this path requires the {required} scope — the token was "
        "accepted but carries only reporting rights; register the calling "
        "application with the extra scope in heco-auth and mint a new token",
        "code": "scope",
    }).encode()


class BearerGate:
    """ASGI middleware: pass /health and non-HTTP traffic; gate the rest."""

    def __init__(
        self, app, *, open_paths: tuple[str, ...] = OPEN_PATHS, environ=os.environ,
    ) -> None:
        self.app = app
        self.open_paths = open_paths
        self._environ = environ
        # The verifier is rebuilt only when the env that shapes it changes —
        # which is never in production and constantly in tests.
        self._verifier: JwksVerifier | None = None
        self._verifier_env: tuple | None = None

    # ------------------------------------------------------------- plumbing

    def _armed(self) -> bool:
        return self._environ.get("HECO_REQUIRE_AUTH", "").strip().lower() in TRUTHY

    def _scopes_enforced(self) -> bool:
        # Default ON: only the literal opt-out suspends the table (see the
        # SCOPE_RULES comment for the one legitimate reason it exists).
        return self._environ.get("HECO_SCOPE_ENFORCE", "1").strip().lower() in TRUTHY

    def _current_verifier(self) -> JwksVerifier | None:
        env = (
            self._environ.get("HECO_AUTH_URL", "").strip(),
            self._environ.get("HECO_AUTH_ISSUER", "").strip(),
            self._environ.get("HECO_JWKS_JSON", "").strip(),
            self._environ.get("HECO_JWKS_CACHE", "").strip(),
        )
        if env == self._verifier_env:
            return self._verifier
        auth_url, issuer, pinned, cache = env
        verifier = None
        if auth_url or pinned:
            # A bad pin raises here, surfacing on the FIRST gated request
            # as a 401 naming configuration — never silently fail-open.
            try:
                verifier = JwksVerifier(
                    auth_url or None,
                    issuer=issuer or None,
                    pinned_jwks=pinned or None,
                    cache_path=cache or None,
                )
            except ValueError:
                verifier = None
        self._verifier, self._verifier_env = verifier, env
        return verifier

    def _verified_claims(self, token: str) -> dict | None:
        """The token's claims when it verifies, None when it does not."""
        verifier = self._current_verifier()
        if verifier is not None:
            try:
                return verifier.verify(token)
            except TokenRefused:
                return None
        return None

    # ----------------------------------------------------------------- ASGI

    async def __call__(self, scope, receive, send):
        """Gate one ASGI event: pass, or answer 401 without calling the app."""
        if scope["type"] != "http" or not self._armed() or scope.get("path") in self.open_paths:
            await self.app(scope, receive, send)
            return

        token = ""
        for name, value in scope.get("headers") or []:
            if name == b"authorization":
                text = value.decode("latin-1")
                if text[:7].lower() == "bearer ":
                    token = text[7:].strip()
                break

        if not token:
            await self._refuse(send, "no credential was sent")
            return
        claims = self._verified_claims(token)
        if claims is None:
            await self._refuse(send, "the credential was refused")
            return
        required = SCOPE_RULES.get((scope.get("method", ""), scope.get("path", "")))
        if (
            required
            and self._scopes_enforced()
            and required not in str(claims.get("scope", "")).split()
        ):
            await self._refuse_scope(send, required)
            return
        await self.app(scope, receive, send)

    async def _refuse(self, send, reason: str) -> None:
        await self._answer(send, 401, _refusal_body(reason))

    async def _refuse_scope(self, send, required: str) -> None:
        await self._answer(send, 403, _scope_refusal_body(required))

    async def _answer(self, send, status: int, body: bytes) -> None:
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def install_bearer_gate(app) -> None:
    """The one line each service's main.py calls after building its app."""
    app.add_middleware(BearerGate)
