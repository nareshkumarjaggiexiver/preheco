"""The inbound bearer gate every pipeline service installs — runbook step 8.

One middleware, seven services, armed by ONE switch:

    HECO_REQUIRE_AUTH=1     refuse inbound requests without a credential
    (unset / 0)             today's behaviour, bit for bit — open

The switch exists because these are exactly the paths that brick a lab when
auth goes wrong mid-event. Arming is a deliberate act after the credentials
are proven to flow (the runbook's verify-then-advance habit); disarming is
one env change and a `docker compose up -d`, which is the whole rollback.

Two credentials are accepted while both exist, mirroring the planner's own
migration so the same rollout order works here:

  - an Ed25519 token minted by heco-auth, verified LOCALLY against the cached
    JWKS (heco_common.verify — offline-safe, nothing on a request path calls
    the Worker);
  - the legacy shared secret (``HECO_TOKEN``), compared constant-time — the
    dual-accept leg that lets a lab arm the gate before every caller has an
    application registered, and that dies at runbook step 7 with the rest of
    the shared-secret era.

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

import hmac
import json
import os

from .verify import JwksVerifier, TokenRefused

#: Paths that never require a credential. Exact match.
OPEN_PATHS = ("/health",)

TRUTHY = {"1", "true", "yes", "on"}


def _refusal_body(reason: str) -> bytes:
    # The planner's error dialect: `code` for machines, the sentence for a
    # human reading a log at midnight.
    return json.dumps({
        "detail": f"this pipeline service requires a bearer token ({reason}) — "
        "mint one from the auth service, or unset HECO_REQUIRE_AUTH to reopen the LAN",
        "code": "auth",
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

    def _static_token(self) -> str:
        return self._environ.get("HECO_TOKEN", "").strip()

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

    def _credential_ok(self, token: str) -> bool:
        static = self._static_token()
        if static and hmac.compare_digest(token.encode(), static.encode()):
            return True
        verifier = self._current_verifier()
        if verifier is not None:
            try:
                verifier.verify(token)
                return True
            except TokenRefused:
                return False
        return False

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
        if not self._credential_ok(token):
            await self._refuse(send, "the credential was refused")
            return
        await self.app(scope, receive, send)

    async def _refuse(self, send, reason: str) -> None:
        body = _refusal_body(reason)
        await send({
            "type": "http.response.start",
            "status": 401,
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
