"""JwksVerifier: every acceptance is earned, every refusal is named.

The list of forgeries mirrors apps/heco-auth/test/authz.test.js — the same
attacks, refused at the OTHER end of the wire, because a verifier that has
never met a forgery in a test meets its first one in production.
"""

import json
import time

import pytest
from conftest import SigningKit
from heco_common.verify import (
    CLOCK_TOLERANCE_S,
    JWKS_MIN_INTERVAL_S,
    JwksVerifier,
    TokenRefused,
)

ISS = "https://auth.test"


def make_verifier(kit, **kw):
    """A verifier pinned to the kit's key set — the no-network configuration."""
    kw.setdefault("issuer", ISS)
    kw.setdefault("pinned_jwks", json.dumps(kit.jwks))
    return JwksVerifier(None, **kw)


def test_a_valid_token_verifies_and_returns_its_claims(kit):
    """The happy path hands back the claims, which carry sub for a log line."""
    v = make_verifier(kit)
    claims = v.verify(
        kit.mint(iss=ISS, extra_claims={"sub": "app:runner", "scope": "planner:report"}),
    )
    assert claims["sub"] == "app:runner"
    assert claims["iss"] == ISS


def test_the_audience_may_be_a_list_and_must_contain_ours(kit):
    """aud is a string today; a list tomorrow must not fail open OR closed."""
    v = make_verifier(kit)
    assert v.verify(kit.mint(iss=ISS, aud=["something-else", "heco-planner"]))
    with pytest.raises(TokenRefused, match="audience"):
        v.verify(kit.mint(iss=ISS, aud=["something-else"]))


def test_algorithm_is_ours_never_the_tokens_suggestion(kit):
    """alg:none and HS256 are the classic confusion attacks; both die on the
    same check, which reads OUR expectation and not the header."""
    v = make_verifier(kit)
    with pytest.raises(TokenRefused, match="algorithm"):
        v.verify(kit.mint(iss=ISS, header_override={"alg": "none"}))
    with pytest.raises(TokenRefused, match="algorithm"):
        v.verify(kit.mint(iss=ISS, header_override={"alg": "HS256"}))


def test_a_signature_from_the_wrong_key_is_refused(kit):
    """Same kid, different key — the literal forgery the signature check is for."""
    imposter = SigningKit(kid=kit.kid)
    v = make_verifier(kit)
    with pytest.raises(TokenRefused, match="signature"):
        v.verify(kit.mint(iss=ISS, signer=imposter))


def test_expiry_is_required_enforced_and_clock_tolerant(kit):
    """exp is mandatory; refusal starts past the LAN clock tolerance, not at
    the instant — two boxes disagreeing by seconds is not an attack."""
    v = make_verifier(kit)
    with pytest.raises(TokenRefused, match="expired"):
        v.verify(kit.mint(iss=ISS, exp_in=-(CLOCK_TOLERANCE_S + 10)))
    assert v.verify(kit.mint(iss=ISS, exp_in=-(CLOCK_TOLERANCE_S - 30)))
    with pytest.raises(TokenRefused, match="expiry"):
        v.verify(kit.mint(iss=ISS, extra_claims={"exp": None}))


def test_nbf_is_honoured_with_the_same_tolerance(kit):
    """A token from a slightly-fast clock works; one from the future does not."""
    v = make_verifier(kit)
    with pytest.raises(TokenRefused, match="not yet valid"):
        v.verify(kit.mint(iss=ISS, extra_claims={"nbf": time.time() + CLOCK_TOLERANCE_S + 60}))
    assert v.verify(kit.mint(iss=ISS, extra_claims={"nbf": time.time() + CLOCK_TOLERANCE_S - 30}))


def test_issuer_is_pinned_exactly(kit):
    """A lookalike issuer is a different authority, full stop."""
    v = make_verifier(kit)
    with pytest.raises(TokenRefused, match="issuer"):
        v.verify(kit.mint(iss="https://auth.test.evil.example"))


def test_junk_is_refused_never_crashed(kit):
    """Whatever lands in an Authorization header must come back 401, not 500."""
    v = make_verifier(kit)
    for junk in ["", "a", "a.b", "a.b.c", "?.?.?", "e30.e30.e30"]:
        with pytest.raises(TokenRefused):
            v.verify(junk)


def test_a_token_without_a_kid_is_refused(kit):
    """No kid means no key to check against — refused, never guessed."""
    v = make_verifier(kit)
    with pytest.raises(TokenRefused, match="kid"):
        v.verify(kit.mint(iss=ISS, header_override={"kid": None}))


def test_unknown_kid_fetches_once_then_respects_the_floor(kit):
    """A rotation's first token triggers ONE fetch; a flood of junk kids does
    not turn this verifier into a weapon against the auth service."""
    clock = {"t": 1000.0}
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return kit.jwks

    v = JwksVerifier("https://auth.test", issuer=ISS, fetch=fetch, now=lambda: clock["t"])
    assert v.verify(kit.mint(iss=ISS, now=clock["t"]))
    assert calls == ["https://auth.test/.well-known/jwks.json"]

    other = SigningKit(kid="k-unknown")
    with pytest.raises(TokenRefused, match="unknown signing key"):
        v.verify(other.mint(iss=ISS, now=clock["t"]))
    assert len(calls) == 1, "a second unknown kid inside the floor must not re-fetch"

    clock["t"] += JWKS_MIN_INTERVAL_S + 1
    with pytest.raises(TokenRefused, match="unknown signing key"):
        v.verify(other.mint(iss=ISS, now=clock["t"]))
    assert len(calls) == 2, "past the floor the unknown kid may try again"


def test_an_unreachable_auth_service_keeps_the_keys_in_hand(kit):
    """The offline constraint: fetch failures never clear the key set."""
    state = {"fail": False}

    def fetch(url, timeout):
        if state["fail"]:
            raise OSError("no route to host")
        return kit.jwks

    clock = {"t": 1000.0}
    v = JwksVerifier("https://auth.test", issuer=ISS, fetch=fetch, now=lambda: clock["t"])
    assert v.verify(kit.mint(iss=ISS, now=clock["t"]))

    state["fail"] = True
    clock["t"] += 10
    assert v.verify(kit.mint(iss=ISS, now=clock["t"])), "verification is local; no WAN on this path"


def test_disk_cache_round_trip_covers_the_offline_restart(kit, tmp_path):
    """A restart with no WAN boots trusting what it trusted yesterday."""
    cache = tmp_path / "jwks.json"
    fetched = JwksVerifier(
        "https://auth.test", issuer=ISS, cache_path=str(cache),
        fetch=lambda url, t: kit.jwks, now=lambda: 1000.0,
    )
    assert fetched.verify(kit.mint(iss=ISS, now=1000.0))
    assert cache.exists(), "a successful fetch writes the cache"

    def dead_fetch(url, t):
        """The WAN is down; only the cache can answer."""
        raise OSError("offline")

    reborn = JwksVerifier(
        "https://auth.test", issuer=ISS, cache_path=str(cache),
        fetch=dead_fetch, now=lambda: 2000.0,
    )
    assert reborn.verify(kit.mint(iss=ISS, now=2000.0))


def test_a_pinned_key_set_with_no_usable_key_fails_the_deploy(kit):
    """A garbage pin must break at construction, not on every request."""
    with pytest.raises(ValueError, match="no usable Ed25519 key"):
        JwksVerifier(None, issuer=ISS, pinned_jwks='{"keys": [{"kty": "RSA", "kid": "r1"}]}')


def test_a_jwks_with_one_bad_entry_keeps_the_good_keys(kit):
    """One torn key in a published set must not take the working keys down."""
    polluted = {
        "keys": [{"kty": "OKP", "crv": "Ed25519", "kid": "torn", "x": "!!!"}, *kit.jwks["keys"]],
    }
    v = JwksVerifier(None, issuer=ISS, pinned_jwks=json.dumps(polluted))
    assert v.verify(kit.mint(iss=ISS))
