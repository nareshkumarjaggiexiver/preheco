"""Shared fixtures for the inbound-auth tests: a real Ed25519 keypair and a
JWT factory that signs with it — the genuine article, not a mock, because the
verifier under test must refuse real forgeries, and a forgery is only real if
the signatures are.
"""

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def b64url(data: bytes) -> str:
    """Base64url without padding — the JWT segment encoding."""
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


class SigningKit:
    """One Ed25519 keypair, its JWKS, and a token mint bound to them."""

    def __init__(self, kid: str = "k-test") -> None:
        """Generate the pair and publish the public half as a JWKS dict."""
        self.kid = kid
        self.private = Ed25519PrivateKey.generate()
        x = self.private.public_key().public_bytes_raw()
        self.jwks = {
            "keys": [
                {
                    "kty": "OKP", "crv": "Ed25519", "kid": kid,
                    "x": b64url(x), "alg": "EdDSA", "use": "sig",
                }
            ]
        }

    def mint(
        self,
        *,
        iss: str = "https://auth.test",
        aud="heco-planner",
        exp_in: float = 3600,
        now: float | None = None,
        extra_claims: dict | None = None,
        header_override: dict | None = None,
        signer: "SigningKit | None" = None,
    ) -> str:
        """A signed token; every knob exists so a test can break one thing."""
        t = time.time() if now is None else now
        header = {"alg": "EdDSA", "kid": self.kid, "typ": "JWT", **(header_override or {})}
        claims = {
            "iss": iss, "aud": aud, "iat": int(t), "exp": int(t + exp_in),
            **(extra_claims or {}),
        }
        head = b64url(json.dumps(header).encode())
        body = b64url(json.dumps(claims).encode())
        signing_input = f"{head}.{body}"
        key = (signer or self).private
        return f"{signing_input}.{b64url(key.sign(signing_input.encode()))}"


@pytest.fixture()
def kit() -> SigningKit:
    """A fresh keypair per test — no cross-test key reuse to hide behind."""
    return SigningKit()
