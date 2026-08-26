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


def call(gate, path="/detect", headers=None, scope_type="http", method="GET"):
    """Drive the gate once; return (status, body, response_headers)."""
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": scope_type,
        "path": path,
        "method": method,
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
    g2 = gate_with({"HECO_REQUIRE_AUTH": "0", "HECO_AUTH_ISSUER": ISS})
    assert call(g2)[0] == 200, "a configured but unarmed gate changes nothing"


def test_armed_refuses_the_open_lan():
    """Armed, no credential: 401 with the machine-readable code."""
    g = gate_with({"HECO_REQUIRE_AUTH": "1", "HECO_AUTH_ISSUER": ISS})
    status, body, headers = call(g)
    assert status == 401
    refusal = json.loads(body)
    assert refusal["code"] == "auth", "the code is what a machine switches on"
    assert (b"www-authenticate", b"Bearer") in [(k, v) for k, v in headers]


def test_health_stays_open_for_the_compose_healthchecks():
    """Compose polls /health unauthenticated; the deploy runbook reads it too."""
    g = gate_with({"HECO_REQUIRE_AUTH": "1", "HECO_AUTH_ISSUER": ISS})
    assert call(g, path="/health")[0] == 200


def test_the_retired_shared_secret_opens_nothing(kit):
    """Runbook step 7: a static string is not a credential here any more.

    HECO_TOKEN is set in the environment on purpose — the gate must ignore it
    entirely rather than quietly honour a variable that happens to be present.
    """
    g = gate_with({
        "HECO_REQUIRE_AUTH": "1",
        "HECO_TOKEN": "sekrit",
        "HECO_JWKS_JSON": json.dumps(kit.jwks),
        "HECO_AUTH_ISSUER": ISS,
    })
    assert call(g, headers={"authorization": "Bearer sekrit"})[0] == 401
    # ...while a signed token is unaffected.
    assert call(g, headers={"authorization": f"Bearer {kit.mint(iss=ISS)}"})[0] == 200
    assert call(g, headers={"authorization": "Basic sekrit"})[0] == 401


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


def test_a_report_token_cannot_reach_the_hot_swap(kit):
    """The 2026-08 review's confused deputy: every box holds a
    ``planner:report`` machine token, and without a scope check any of them
    (a lost one included) could switch models on an armed install. The gate's
    SCOPE_RULES table demands ``planner:operate`` on POST /models/apply —
    403 with its own code, because the credential is real and the fix is a
    registration, not a rotation."""
    env = {
        "HECO_REQUIRE_AUTH": "1",
        "HECO_JWKS_JSON": json.dumps(kit.jwks),
        "HECO_AUTH_ISSUER": ISS,
    }
    g = gate_with(env)
    report = kit.mint(iss=ISS, extra_claims={"scope": "planner:report"})
    auth = {"authorization": f"Bearer {report}"}

    status, body, _ = call(g, path="/models/apply", method="POST", headers=auth)
    assert status == 403
    refusal = json.loads(body)
    assert refusal["code"] == "scope"
    assert "planner:operate" in refusal["detail"], "the sentence names the fix"

    # The same report token keeps its day job everywhere else...
    assert call(g, path="/runs", method="POST", headers=auth)[0] == 200
    assert call(g, path="/models/apply", method="GET", headers=auth)[0] == 200, (
        "the rule is (method, path) — reading models is reporting"
    )
    # ...and a token that carries the operate scope opens the door.
    operate = kit.mint(iss=ISS, extra_claims={"scope": "planner:report planner:operate"})
    status, _, _ = call(
        g, path="/models/apply", method="POST",
        headers={"authorization": f"Bearer {operate}"},
    )
    assert status == 200

    # A scopeless token (pre-scope mints) is refused on the control path too:
    # absence of rights is not rights.
    bare = kit.mint(iss=ISS)
    assert call(g, path="/models/apply", method="POST",
                headers={"authorization": f"Bearer {bare}"})[0] == 403


def test_scope_rules_are_dormant_while_unarmed(kit):
    """Unarmed is unarmed: the scope table must not smuggle in enforcement."""
    g = gate_with({})
    assert call(g, path="/models/apply", method="POST")[0] == 200


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
