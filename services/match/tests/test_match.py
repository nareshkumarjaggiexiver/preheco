"""Unit tests for the match service using synthetic embeddings.

Synthetic identities are clustered gaussians on the unit sphere: each person
is a random unit "centroid", and each sighting is centroid + small gaussian
noise, re-normalised.  Sightings of one person land close (high cosine),
different people land far — the same geometry SFace promises, without weights.
No network access: FastAPI's TestClient drives the ASGI app in-process.
"""

import math

import numpy as np
import pytest
from app import config, main
from fastapi.testclient import TestClient

RNG = np.random.default_rng(42)


def _unit(v: np.ndarray) -> list[float]:
    """Normalise to unit length and return as a JSON-friendly list."""
    return (v / np.linalg.norm(v)).astype(float).tolist()


def person_centroid() -> np.ndarray:
    """A synthetic identity: a random point on the 128-d unit sphere."""
    return RNG.normal(size=128)


def sighting(centroid: np.ndarray, noise: float = 0.05) -> list[float]:
    """One face capture of an identity: the centroid plus small gaussian noise."""
    return _unit(centroid + RNG.normal(scale=noise, size=128))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with the gallery data dir redirected to a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    with TestClient(main.app) as c:
        yield c


def _match(client, run_id, embedding, quality=85.0):
    res = client.post(
        "/match", json={"runId": run_id, "embedding": embedding, "quality": quality}
    )
    assert res.status_code == 200, res.text
    return res.json()


def test_health(client):
    """Health reports the gallery policy and the active threshold."""
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["model"] == "cosine-gallery-sqlite"
    assert body["threshold"] == pytest.approx(config.DEFAULT_THRESHOLD)


def test_new_person_path(client):
    """Distinct gaussian clusters each get a fresh personKey and grow the count."""
    keys = set()
    for i in range(5):
        out = _match(client, "run-a", sighting(person_centroid()))
        assert out["isNew"] is True
        assert out["galleryN"] == i + 1
        keys.add(out["personKey"])
    assert len(keys) == 5  # five people, five keys


def test_repeat_match(client):
    """Noisy re-sightings of one centroid resolve to the same personKey."""
    c = person_centroid()
    first = _match(client, "run-b", sighting(c))
    assert first["isNew"] is True
    for _ in range(4):
        again = _match(client, "run-b", sighting(c))
        assert again["isNew"] is False
        assert again["personKey"] == first["personKey"]
        assert again["cosine"] > config.DEFAULT_THRESHOLD
        assert again["galleryN"] == 1


def test_threshold_boundary(client):
    """Cosine exactly at / just above the threshold matches; just below is new.

    Constructs query vectors with an exact cosine against the stored one:
    q = cos*e1 + sin*e2 for orthonormal e1, e2.
    """
    t = config.DEFAULT_THRESHOLD
    e1 = np.zeros(128)
    e1[0] = 1.0
    e2 = np.zeros(128)
    e2[1] = 1.0

    def at_cosine(c: float) -> list[float]:
        return _unit(c * e1 + math.sqrt(1.0 - c * c) * e2)

    seed = _match(client, "run-c", e1.tolist())
    assert seed["isNew"] is True

    exact = _match(client, "run-c", at_cosine(t))
    assert exact["isNew"] is False, ">= threshold must match"
    assert exact["cosine"] == pytest.approx(t, abs=1e-5)

    above = _match(client, "run-c", at_cosine(t + 0.01))
    assert above["isNew"] is False

    below = _match(client, "run-c", at_cosine(t - 0.01))
    assert below["isNew"] is True
    assert below["cosine"] == pytest.approx(t - 0.01, abs=1e-5)


def test_sub_canon_tagged_but_matched(client):
    """A 64 px face (POC close-zone) is matched normally, only tagged."""
    c = person_centroid()
    first = _match(client, "run-d", sighting(c), quality=64.0)
    assert first["isNew"] is True and first["subCanon"] is True
    again = _match(client, "run-d", sighting(c), quality=85.0)
    assert again["isNew"] is False and again["subCanon"] is False
    assert again["personKey"] == first["personKey"]


def test_reset_clears_gallery(client):
    """After /reset the same face is new again and the count restarts."""
    c = person_centroid()
    _match(client, "run-e", sighting(c))
    assert client.post("/reset", json={"runId": "run-e"}).status_code == 200
    out = _match(client, "run-e", sighting(c))
    assert out["isNew"] is True
    assert out["galleryN"] == 1


def test_runs_are_isolated(client):
    """Galleries are per run: the same person in two runs is new in each."""
    c = person_centroid()
    a = _match(client, "run-f1", sighting(c))
    b = _match(client, "run-f2", sighting(c))
    assert a["isNew"] and b["isNew"]


def test_bad_run_id_rejected(client):
    """Path-traversal runIds never reach the filesystem."""
    res = client.post("/reset", json={"runId": "../evil"})
    assert res.status_code == 422
