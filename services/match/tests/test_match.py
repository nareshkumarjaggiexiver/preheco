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


# ----------------------------- the near-miss flag on a mint (v2, 2026-08-06)
#
# Measured tonight: the same person split at face cosine 0.3464 against the
# 0.363 threshold with clothing intersection 0.562 — new guest p00005, likely
# p00004 — and the operator found it by eye because the signal surfaced
# nowhere.  A mint whose best cosine lands in [0.29 .. threshold) now carries
# nearMiss: {key, cosine, appearanceSim}.  It is a SUGGESTION to the operator:
# an IMPOSTOR pair on the same camera sits at face 0.377 with clothing 0.503,
# so the verdict stays a mint and the count never moves without a human click.


def _e(i: int) -> np.ndarray:
    """The i-th standard basis vector in 128-d (an orthonormal anchor)."""
    v = np.zeros(128)
    v[i] = 1.0
    return v


def _at_cosine(c: float, ortho: int = 1) -> list[float]:
    """A unit query at an exact cosine against _e(0): c*e0 + sqrt(1-c^2)*e_ortho.

    ``ortho`` picks the orthogonal axis; two probes fired into the SAME
    gallery must use different axes, or their large orthogonal components
    make them near-duplicates of each other (~0.9998) and the second probe
    matches the first probe's mint instead of testing the seed.
    """
    return _unit(c * _e(0) + math.sqrt(1.0 - c * c) * _e(ortho))


def _desc(weights: dict[int, float]) -> list[float]:
    """An L1-normalised 48-bin torso descriptor with mass in the given bins."""
    d = [0.0] * 48
    for i, w in weights.items():
        d[i] = w
    return d


def _match_full(client, run_id, embedding, appearance=None, quality=85.0):
    body = {"runId": run_id, "embedding": embedding, "quality": quality}
    if appearance is not None:
        body["appearance"] = appearance
    res = client.post("/match", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_in_band_mint_carries_the_near_miss_replaying_tonight(client):
    """THE REPLAY: a mint at 0.3464 with torso intersection 0.562 is flagged.

    The seed identity stores a solid-colour torso; the minting sighting shares
    0.562 of its histogram mass with it — the measured pair.  The response
    must name the near-missed key, the score, and the torso agreement, while
    the verdict stays a MINT (isNew true, fresh key): the flag is the
    operator's one-click-merge cue, never a merge.
    """
    seed_desc = _desc({0: 1.0})
    probe_desc = _desc({0: 0.562, 1: 0.438})
    seed = _match_full(client, "run-nm", _e(0).tolist(), appearance=seed_desc)
    assert seed["isNew"] is True and seed["nearMiss"] is None  # empty gallery

    out = _match_full(client, "run-nm", _at_cosine(0.3464), appearance=probe_desc)
    assert out["isNew"] is True, "the verdict is STILL a mint — never an auto-merge"
    assert out["personKey"] != seed["personKey"]
    assert out["galleryN"] == 2, "the count moved UP; only an operator click folds it"
    nm = out["nearMiss"]
    assert nm is not None
    assert nm["key"] == seed["personKey"]
    assert nm["cosine"] == pytest.approx(0.3464, abs=1e-5)
    assert nm["cosine"] == pytest.approx(out["cosine"], abs=1e-9)
    assert nm["appearanceSim"] == pytest.approx(0.562, abs=1e-5)


def test_out_of_band_mint_has_null_near_miss(client):
    """A mint clearly below the 0.29 floor is a genuinely new face: no flag.

    Flagging every mint would train the operator to ignore the cue; the band
    exists because every split we have watched happen (0.294/0.308/0.346/
    0.361) sits above 0.29 and true strangers mostly do not.
    """
    _match_full(client, "run-nm2", _e(0).tolist())
    out = _match_full(client, "run-nm2", _at_cosine(0.20))
    assert out["isNew"] is True
    assert out["nearMiss"] is None


def test_near_miss_band_edges(client):
    """Just above the floor flags; just below does not; the threshold matches.

    Each probe carries its own orthogonal axis (see _at_cosine) so the three
    tests all measure against the SEED, not against each other's mints.
    """
    seed = _match_full(client, "run-nm3", _e(0).tolist())
    below = _match_full(client, "run-nm3", _at_cosine(0.28, ortho=1))
    assert below["isNew"] is True and below["nearMiss"] is None
    inside = _match_full(client, "run-nm3", _at_cosine(0.30, ortho=2))
    assert inside["isNew"] is True and inside["nearMiss"] is not None
    assert inside["nearMiss"]["key"] == seed["personKey"]
    assert inside["nearMiss"]["cosine"] == pytest.approx(0.30, abs=1e-5)
    # At the threshold the verdict is a MATCH, and matches never carry the flag.
    at_t = _match_full(client, "run-nm3", _at_cosine(config.DEFAULT_THRESHOLD, ortho=3))
    assert at_t["isNew"] is False and at_t["nearMiss"] is None


def test_near_miss_appearance_sim_is_null_when_either_side_lacks_one(client):
    """Absent is not zero: a missing descriptor reports None, never a clash.

    Both directions: a seed stored without a torso descriptor, and a probe
    arriving without one against a seed that has one.
    """
    _match_full(client, "run-nm4", _e(0).tolist())  # seeded descriptor-less
    out = _match_full(
        client, "run-nm4", _at_cosine(0.32), appearance=_desc({3: 1.0})
    )
    assert out["nearMiss"] is not None
    assert out["nearMiss"]["appearanceSim"] is None

    _match_full(client, "run-nm5", _e(0).tolist(), appearance=_desc({3: 1.0}))
    out = _match_full(client, "run-nm5", _at_cosine(0.32))  # probe descriptor-less
    assert out["nearMiss"] is not None
    assert out["nearMiss"]["appearanceSim"] is None


def test_matched_verdicts_never_carry_near_miss(client):
    """A comfortable re-sighting is not a mint; the field is present and null."""
    _match_full(client, "run-nm6", _e(0).tolist())
    out = _match_full(client, "run-nm6", _at_cosine(0.75))
    assert out["isNew"] is False
    assert "nearMiss" in out and out["nearMiss"] is None


def test_staff_hits_never_carry_near_miss(client):
    """A staff hit is not a mint either — no flag, whatever the gallery holds."""
    samples = [{"embedding": _e(7).tolist(), "quality": 80.0}]
    res = client.post(
        "/staff/enrol", json={"siteId": "site-nm", "staffId": "st-9", "samples": samples}
    )
    assert res.status_code == 200, res.text
    body = {
        "runId": "run-nm7", "embedding": _e(7).tolist(),
        "quality": 85.0, "siteId": "site-nm",
    }
    out = client.post("/match", json=body).json()
    assert out["isStaff"] is True
    assert out["nearMiss"] is None


def test_near_miss_floor_zero_disables_and_health_reports_it(client, monkeypatch):
    """0 is the off switch (config convention) and /health names the band.

    An empty env value means unset (the compose ``${VAR-}`` rendering) and
    keeps the 0.29 default — the parse rule every knob in this service obeys.
    """
    assert client.get("/health").json()["nearMissFloor"] == pytest.approx(0.29)

    monkeypatch.setenv("HECO_MATCH_NEARMISS_FLOOR", "0")
    _match_full(client, "run-nm8", _e(0).tolist())
    out = _match_full(client, "run-nm8", _at_cosine(0.3464))
    assert out["isNew"] is True
    assert out["nearMiss"] is None, "floor 0 must disable the flag entirely"
    assert client.get("/health").json()["nearMissFloor"] == 0.0

    monkeypatch.setenv("HECO_MATCH_NEARMISS_FLOOR", "")
    assert config.nearmiss_floor() == pytest.approx(0.29), "empty means unset"
