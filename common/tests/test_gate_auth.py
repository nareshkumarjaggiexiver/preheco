"""BearerGate: the seven services' front door, driven as raw ASGI.

No web framework here on purpose — heco-common does not depend on one, and
the gate is plain ASGI so a fake scope and two coroutines are the whole
harness. What FastAPI adds (routing, validation) is downstream of the gate
and irrelevant to whether the door opens.
"""

import asyncio
import json

from conftest import SigningKit
from heco_common.gate_auth import BearerGate

ISS = "https://auth.test"


async def _downstream(scope, receive, send):
    """The app behind the gate: answers 200 to anything it is allowed to see."""
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b'{"ok": true}'})


def call(gate, path="/detect", headers=None, scope_type="http"):
    """Drive the gate once; return (status, body, response_headers)."""
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": scope_type,
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    asyncio.run(gate(scope, receive, send))
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    resp_headers = next(
        (m.get("headers", []) for m in sent if m["type"] == "http.response.start"), [],
    )
    return status, body, resp_headers


def gate_with(env):
    """A gate whose whole environment is the given dict — nothing global."""
    return BearerGate(_downstream, environ=env)


def test_unarmed_is_todays_behaviour_bit_for_bit():
    """Until HECO_REQUIRE_AUTH is set, the gate must change NOTHING."""
    g = gate_with({})
    assert call(g)[0] == 200
    g2 = gate_with({"HECO_REQUIRE_AUTH": "0", "HECO_TOKEN": "sekrit"})
    assert call(g2)[0] == 200, "a configured but unarmed gate changes nothing"


def test_armed_refuses_the_open_lan():
    """Armed, no credential: 401 with the machine-readable code."""
    g = gate_with({"HECO_REQUIRE_AUTH": "1", "HECO_TOKEN": "sekrit"})
    status, body, headers = call(g)
    assert status == 401
    refusal = json.loads(body)
    assert refusal["code"] == "auth", "the code is what a machine switches on"
    assert (b"www-authenticate", b"Bearer") in [(k, v) for k, v in headers]


def test_health_stays_open_for_the_compose_healthchecks():
    """Compose polls /health unauthenticated; the deploy runbook reads it too."""
    g = gate_with({"HECO_REQUIRE_AUTH": "1", "HECO_TOKEN": "sekrit"})
    assert call(g, path="/health")[0] == 200


def test_the_legacy_shared_secret_is_the_dual_accept_leg():
    """HECO_TOKEN works while it exists, constant-time, Bearer scheme only."""
    g = gate_with({"HECO_REQUIRE_AUTH": "1", "HECO_TOKEN": "sekrit"})
    assert call(g, headers={"authorization": "Bearer sekrit"})[0] == 200
    assert call(g, headers={"authorization": "Bearer wrong"})[0] == 401
    assert call(g, headers={"authorization": "Bearer s"})[0] == 401, (
        "a short token must not crash the compare"
    )
    assert call(g, headers={"authorization": "Basic sekrit"})[0] == 401, (
        "only the Bearer scheme is a credential"
    )


def test_a_worker_minted_token_is_accepted_via_the_pinned_jwks(kit):
    """The JWT leg: a valid Ed25519 token passes, an expired one does not."""
    env = {
        "HECO_REQUIRE_AUTH": "1",
        "HECO_JWKS_JSON": json.dumps(kit.jwks),
        "HECO_AUTH_ISSUER": ISS,
    }
    g = gate_with(env)
    good = kit.mint(iss=ISS)
    assert call(g, headers={"authorization": f"Bearer {good}"})[0] == 200
    expired = kit.mint(iss=ISS, exp_in=-3600)
    assert call(g, headers={"authorization": f"Bearer {expired}"})[0] == 401


def test_both_legs_may_coexist_during_the_migration(kit):
    """Dual-accept is what lets the rollout arm before every caller migrates."""
    env = {
        "HECO_REQUIRE_AUTH": "1",
        "HECO_TOKEN": "sekrit",
        "HECO_JWKS_JSON": json.dumps(kit.jwks),
        "HECO_AUTH_ISSUER": ISS,
    }
    g = gate_with(env)
    assert call(g, headers={"authorization": "Bearer sekrit"})[0] == 200
    assert call(g, headers={"authorization": f"Bearer {kit.mint(iss=ISS)}"})[0] == 200


def test_an_env_change_rebuilds_the_verifier(kit):
    """Tests arm and disarm per test; rotation swaps key sets. Both work
    because the gate reads its environment per request, not at install."""
    env = {
        "HECO_REQUIRE_AUTH": "1",
        "HECO_JWKS_JSON": json.dumps(kit.jwks),
        "HECO_AUTH_ISSUER": ISS,
    }
    g = gate_with(env)
    token = kit.mint(iss=ISS)
    assert call(g, headers={"authorization": f"Bearer {token}"})[0] == 200

    env["HECO_JWKS_JSON"] = json.dumps(SigningKit(kid="k2").jwks)
    assert call(g, headers={"authorization": f"Bearer {token}"})[0] == 401, (
        "the old key set is gone with the env"
    )


def test_a_garbage_pin_fails_closed_never_open(kit):
    """A pin that parses to no usable key refuses tokens; it never opens."""
    env = {"HECO_REQUIRE_AUTH": "1", "HECO_JWKS_JSON": '{"keys": []}', "HECO_AUTH_ISSUER": ISS}
    g = gate_with(env)
    assert call(g, headers={"authorization": f"Bearer {kit.mint(iss=ISS)}"})[0] == 401


def test_armed_with_nothing_configured_refuses_everything(kit):
    """No static token, no JWKS source: fail closed and say so — never a
    silently open door that reads as armed in the compose file."""
    g = gate_with({"HECO_REQUIRE_AUTH": "1"})
    assert call(g, headers={"authorization": f"Bearer {kit.mint(iss=ISS)}"})[0] == 401


def test_non_http_traffic_passes_untouched():
    """Lifespan events must reach the app or no service ever finishes booting."""
    ran = []

    async def downstream(scope, receive, send):
        ran.append(scope["type"])

    g = BearerGate(downstream, environ={"HECO_REQUIRE_AUTH": "1"})

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        pass

    asyncio.run(g({"type": "lifespan", "path": "/"}, receive, send))
    assert ran == ["lifespan"], "the gate guards HTTP, not the app lifecycle"
