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


# ------------------- the WEAK (clothing) near-miss band (v2 two-signal, 0.9.0)
#
# Bench run 6e1a5d (2026-08-06), ground truth ONE person who walked out of
# frame, came back, and sat down.  That one person produced SIX tracker ids;
# three mints healed back, and TWO survived as extra guests:
#
#     p00005: face 0.212 vs p00001, clothing 0.797
#     p00006: face 0.228 vs p00001, clothing 0.875
#
# Both sat BELOW the 0.29 face floor, so no banner fired and the operator was
# never asked — while the torso descriptor was reading 0.80-0.88.  The weak
# band flags exactly that shape: cosine in [0.15 .. 0.29) AND clothing >= 0.78.
#
# 0.78 clears the worst measured IMPOSTOR clothing reading (0.747, two
# genuinely different men) by 0.033.  That hair is why the rider is a
# suggestion a human confirms, never a merge — pinned below by asserting the
# verdict and the count on every one of these.


def test_weak_band_replays_bench_6e1a5d_p00005(client):
    """THE REPLAY: face 0.212 with clothing 0.797 is flagged on basis clothing.

    The face signal alone could not see this split (0.212 is nowhere near the
    0.29 floor); the clothing signal was shouting.  The rider must name the
    identity, the face score, the torso agreement, and that it is speaking on
    CLOTHING evidence — while the verdict stays a mint and the count moves up,
    because clothing never merges anyone (impostors measured at 0.747).
    """
    seed = _match_full(client, "run-wb", _e(0).tolist(), appearance=_desc({0: 1.0}))
    out = _match_full(
        client, "run-wb", _at_cosine(0.212), appearance=_desc({0: 0.797, 1: 0.203})
    )
    assert out["isNew"] is True, "the verdict is STILL a mint — clothing never merges"
    assert out["personKey"] != seed["personKey"]
    assert out["galleryN"] == 2, "the count moved UP; only an operator click folds it"
    nm = out["nearMiss"]
    assert nm is not None, "0.212 with clothing 0.797 was invisible before 0.9.0"
    assert nm["basis"] == "clothing"
    assert nm["key"] == seed["personKey"]
    assert nm["cosine"] == pytest.approx(0.212, abs=1e-5)
    assert nm["appearanceSim"] == pytest.approx(0.797, abs=1e-5)


def test_weak_band_replays_bench_6e1a5d_p00006(client):
    """The second survivor: face 0.228 with clothing 0.875, same person, flagged."""
    seed = _match_full(client, "run-wb2", _e(0).tolist(), appearance=_desc({0: 1.0}))
    out = _match_full(
        client, "run-wb2", _at_cosine(0.228, ortho=2), appearance=_desc({0: 0.875, 1: 0.125})
    )
    assert out["isNew"] is True and out["galleryN"] == 2
    nm = out["nearMiss"]
    assert nm is not None and nm["basis"] == "clothing"
    assert nm["key"] == seed["personKey"]
    assert nm["cosine"] == pytest.approx(0.228, abs=1e-5)
    assert nm["appearanceSim"] == pytest.approx(0.875, abs=1e-5)


def test_weak_band_stays_silent_when_the_clothes_disagree(client):
    """Same face score, clothing 0.60: no rider — the bar is the whole guard.

    Without the clothes bar the weak band would flag every mint above 0.15,
    which at a wedding is most of them, and a banner that fires constantly
    trains the operator to click it away.
    """
    _match_full(client, "run-wb3", _e(0).tolist(), appearance=_desc({0: 1.0}))
    out = _match_full(
        client, "run-wb3", _at_cosine(0.212), appearance=_desc({0: 0.60, 1: 0.40})
    )
    assert out["isNew"] is True
    assert out["nearMiss"] is None, "0.60 clothing is below the 0.78 bar"


def test_weak_band_needs_a_descriptor_absent_is_not_zero(client):
    """No descriptor on either side means no weak-band rider — never an assumed one.

    Absent is not zero and absent is not agreement: a sighting with no person
    box (or an old gallery row) simply has no clothing evidence, so the weak
    band — which rests on nothing else — must not speak.  Both directions.
    """
    # Probe has no descriptor.
    _match_full(client, "run-wb4", _e(0).tolist(), appearance=_desc({0: 1.0}))
    out = _match_full(client, "run-wb4", _at_cosine(0.212))
    assert out["isNew"] is True and out["nearMiss"] is None

    # Seed has no descriptor.
    _match_full(client, "run-wb5", _e(0).tolist())
    out = _match_full(
        client, "run-wb5", _at_cosine(0.212), appearance=_desc({0: 0.797, 1: 0.203})
    )
    assert out["isNew"] is True and out["nearMiss"] is None


def test_face_band_is_labelled_face_whatever_the_clothes_say(client):
    """A mint at 0.31 is the ORIGINAL band: basis "face", clothing irrelevant.

    Above the 0.29 floor the face is the evidence and the clothing figure only
    rides along for the operator's judgement — so a clashing torso must not
    suppress the rider, and a missing one must not either.  This is also the
    old rider shape, now merely labelled.
    """
    seed = _match_full(client, "run-wb6", _e(0).tolist(), appearance=_desc({0: 1.0}))

    clash = _match_full(
        client, "run-wb6", _at_cosine(0.31), appearance=_desc({0: 0.10, 5: 0.90})
    )
    assert clash["nearMiss"]["basis"] == "face"
    assert clash["nearMiss"]["key"] == seed["personKey"]
    assert clash["nearMiss"]["appearanceSim"] == pytest.approx(0.10, abs=1e-5), (
        "the clothing figure is reported, never required, above the face floor"
    )

    bare = _match_full(client, "run-wb6", _at_cosine(0.31, ortho=2))
    assert bare["nearMiss"]["basis"] == "face"
    assert bare["nearMiss"]["appearanceSim"] is None


def test_near_miss_band_edges_are_inclusive_at_every_floor(client, monkeypatch):
    """Both floors and the clothes bar are inclusive; just under each is silent.

    The knobs are moved to binary-exact values (0.25 / 0.125 / 0.75) on
    purpose: templates and descriptors are stored as float32, so a value
    authored as 0.29 or 0.78 can land a fraction BELOW itself after the round
    trip (float32(0.78) == 0.7799999713897705).  Testing the comparison at
    values every float represents exactly measures the band's inclusivity
    rather than the storage format's rounding.
    """
    monkeypatch.setenv("HECO_MATCH_NEARMISS_FLOOR", "0.25")
    monkeypatch.setenv("HECO_MATCH_NEARMISS_WEAK_FLOOR", "0.125")
    monkeypatch.setenv("HECO_MATCH_NEARMISS_CLOTHES", "0.75")
    seed = _match_full(client, "run-wb7", _e(0).tolist(), appearance=_desc({0: 1.0}))

    # Exactly at the FACE floor: in band, on face evidence.
    at_floor = _match_full(client, "run-wb7", _at_cosine(0.25, ortho=1))
    assert at_floor["nearMiss"] is not None
    assert at_floor["nearMiss"]["basis"] == "face"
    assert at_floor["nearMiss"]["key"] == seed["personKey"]

    # Exactly at the WEAK floor, with clothing exactly at the bar: in band.
    at_weak = _match_full(
        client, "run-wb7", _at_cosine(0.125, ortho=2), appearance=_desc({0: 0.75, 1: 0.25})
    )
    assert at_weak["nearMiss"] is not None
    assert at_weak["nearMiss"]["basis"] == "clothing"
    assert at_weak["nearMiss"]["appearanceSim"] == pytest.approx(0.75, abs=1e-6)

    # Below the weak floor: perfect clothing agreement cannot rescue it.
    below = _match_full(
        client, "run-wb7", _at_cosine(0.0625, ortho=3), appearance=_desc({0: 1.0})
    )
    assert below["nearMiss"] is None, "under the weak floor the face says nothing at all"

    # Inside the weak band but a hair under the clothes bar: silent.
    under_bar = _match_full(
        client, "run-wb7", _at_cosine(0.1875, ortho=4), appearance=_desc({0: 0.74, 1: 0.26})
    )
    assert under_bar["nearMiss"] is None


def test_weak_floor_zero_disables_only_the_clothing_band(client, monkeypatch):
    """0 is the off switch for the weak band alone — the face band survives it.

    This is the knob to reach for at a venue with uniformed staff, a dress
    code or similar traditional dress, where torso agreement stops carrying
    information about identity and the weak band would suggest the wrong
    person all night.  Turning it off must not cost the face band.
    """
    monkeypatch.setenv("HECO_MATCH_NEARMISS_WEAK_FLOOR", "0")
    _match_full(client, "run-wb8", _e(0).tolist(), appearance=_desc({0: 1.0}))

    weak = _match_full(
        client, "run-wb8", _at_cosine(0.212), appearance=_desc({0: 0.797, 1: 0.203})
    )
    assert weak["nearMiss"] is None, "weak floor 0 must silence the clothing band"

    face = _match_full(
        client, "run-wb8", _at_cosine(0.31, ortho=2), appearance=_desc({0: 0.797, 1: 0.203})
    )
    assert face["nearMiss"] is not None and face["nearMiss"]["basis"] == "face"
    assert client.get("/health").json()["nearMissWeakFloor"] == 0.0


def test_floor_zero_disables_the_weak_band_too(client, monkeypatch):
    """The face floor is the master switch: 0 silences BOTH bands.

    The weak band is defined as the region UNDER that floor, so with no floor
    there is no region — one env var still turns the whole feature off.
    """
    monkeypatch.setenv("HECO_MATCH_NEARMISS_FLOOR", "0")
    _match_full(client, "run-wb9", _e(0).tolist(), appearance=_desc({0: 1.0}))
    out = _match_full(
        client, "run-wb9", _at_cosine(0.212), appearance=_desc({0: 0.797, 1: 0.203})
    )
    assert out["isNew"] is True and out["nearMiss"] is None


def test_matched_verdict_never_carries_a_clothing_rider(client):
    """A re-sighting in identical clothes is a MATCH, not a suggestion.

    The rider is a mint-only object on both bases; a matched verdict already
    resolved to one guest and has nothing to suggest.
    """
    outfit = _desc({0: 1.0})
    _match_full(client, "run-wb10", _e(0).tolist(), appearance=outfit)
    out = _match_full(client, "run-wb10", _at_cosine(0.75), appearance=outfit)
    assert out["isNew"] is False
    assert out["nearMiss"] is None


def test_staff_hit_never_carries_a_clothing_rider(client):
    """Staff identity is operator-attested; no clothing evidence is consulted."""
    samples = [{"embedding": _e(7).tolist(), "quality": 80.0}]
    res = client.post(
        "/staff/enrol", json={"siteId": "site-wb", "staffId": "st-11", "samples": samples}
    )
    assert res.status_code == 200, res.text
    out = client.post(
        "/match",
        json={
            "runId": "run-wb11", "embedding": _e(7).tolist(), "quality": 85.0,
            "siteId": "site-wb", "appearance": _desc({0: 1.0}),
        },
    ).json()
    assert out["isStaff"] is True
    assert out["nearMiss"] is None


def test_health_reports_both_weak_band_knobs(client, monkeypatch):
    """A disputed suggestion must be traceable to the exact bar that made it.

    0.78 sits 0.033 above the measured impostor clothing reading of 0.747, so
    "which bar was this run using" is a question that will be asked.
    """
    body = client.get("/health").json()
    assert body["version"] == "0.11.0"
    assert body["nearMissWeakFloor"] == pytest.approx(config.DEFAULT_NEARMISS_WEAK_FLOOR)
    assert body["nearMissClothes"] == pytest.approx(config.DEFAULT_NEARMISS_CLOTHES)
    assert pytest.approx(0.15) == config.DEFAULT_NEARMISS_WEAK_FLOOR
    assert pytest.approx(0.78) == config.DEFAULT_NEARMISS_CLOTHES

    monkeypatch.setenv("HECO_MATCH_NEARMISS_WEAK_FLOOR", "")
    monkeypatch.setenv("HECO_MATCH_NEARMISS_CLOTHES", "")
    assert config.nearmiss_weak_floor() == pytest.approx(0.15), "empty means unset"
    assert config.nearmiss_clothes() == pytest.approx(0.78), "empty means unset"


# ------------- a SETTLED pair is never asked about again (v3, 2026-08-06)
#
# Run 05b3b7, ground truth THREE people, counted THREE — the count was right
# and the banners were wrong:
#
#     p00002: "minted 0.316 from p00001", clothing agreement 0.94
#     p00007: "minted 0.360 from p00002", clothing agreement 0.57
#
# p00001 and p00002 came in through the main door TOGETHER; p00007 (the
# operator) stood in the alley with p00002, and one tap round matched p00002
# and p00007 IN THE SAME FRAME.  Two faces at different positions in one frame
# are two people — the pipeline held the proof and argued with itself anyway.
#
# No threshold move fixes it: 0.316 and 0.360 sit inside the ordinary face
# band against a 0.363 threshold, the same-person misses on this camera are
# 0.294/0.308/0.361 and the closest impostor pair is 0.377 — the distributions
# overlap right there, and clothing (0.94, 0.57) did not separate them either.
# So the answer is an independent signal recorded through the primitive that
# already exists: the runner posts co-presence to /split, and cannot_link —
# which already refuses /merge and silences the overlap banner — now silences
# the near-miss rider too.  A rider is a QUESTION; the constraint is the
# ANSWER already on file.


def _split(client, run_id, a, b):
    """Record the do-not-merge constraint the operator (or co-presence) asserts."""
    res = client.post("/split", json={"runId": run_id, "a": a, "b": b})
    assert res.status_code == 200, res.text
    return res.json()


def test_settled_pair_carries_no_face_basis_rider(client):
    """THE 05b3b7 REPLAY: a face-band mint against a known-different key is silent.

    Same geometry as the original face-band replay (0.3464 against the seed),
    with one difference — the pair is already recorded as two people.  The
    verdict is untouched (still a mint, still a fresh key, count still up);
    only the suggestion disappears, because it has already been answered.
    """
    seed = _match_full(client, "run-cl1", _e(0).tolist(), appearance=_desc({0: 1.0}))
    assert seed["personKey"] == "p00001"
    _split(client, "run-cl1", "p00001", "p00002")

    out = _match_full(
        client, "run-cl1", _at_cosine(0.3464), appearance=_desc({0: 0.94, 1: 0.06})
    )
    assert out["personKey"] == "p00002"
    assert out["isNew"] is True, "the verdict is untouched — cannot-link is not a merge"
    assert out["galleryN"] == 2, "the count is untouched too"
    assert out["cosine"] == pytest.approx(0.3464, abs=1e-5)
    assert out["nearMiss"] is None, (
        "0.3464 with clothing 0.94 is exactly the 05b3b7 banner — and the pair "
        "is already on file as two different people, so there is nothing to ask"
    )


def test_settled_pair_carries_no_clothing_basis_rider(client):
    """The weak band is silenced by the same constraint, on the same reasoning.

    The clothing basis is the thinnest evidence in the system (its bar clears
    the worst measured impostor by 0.033), so it is the LAST thing that should
    keep arguing with a recorded fact.
    """
    seed = _match_full(client, "run-cl2", _e(0).tolist(), appearance=_desc({0: 1.0}))
    assert seed["personKey"] == "p00001"
    _split(client, "run-cl2", "p00001", "p00002")

    out = _match_full(
        client, "run-cl2", _at_cosine(0.212), appearance=_desc({0: 0.797, 1: 0.203})
    )
    assert out["personKey"] == "p00002"
    assert out["isNew"] is True and out["galleryN"] == 2
    assert out["nearMiss"] is None, "bench 6e1a5d's clothing rider, settled and silent"


def test_the_suppression_is_per_pair_not_a_global_gag(client):
    """One settled pair silences that pair ONLY — every other pair still speaks.

    Two ways of being unrelated are checked: a LATER mint that near-misses the
    same constrained identity (the constraint is about a pair, not about
    p00001), and a mint that near-misses an entirely different seed.  A
    constraint that gagged a key wholesale would hide the real duplicates this
    banner exists to catch, which is the failure the near-miss band was added
    to fix in the first place.
    """
    a = _match_full(client, "run-cl3", _e(0).tolist())
    b = _match_full(client, "run-cl3", _e(5).tolist())
    assert (a["personKey"], b["personKey"]) == ("p00001", "p00002")
    _split(client, "run-cl3", "p00001", "p00003")

    settled = _match_full(client, "run-cl3", _at_cosine(0.3464, ortho=1))
    assert settled["personKey"] == "p00003"
    assert settled["nearMiss"] is None

    # A different mint, the SAME near-missed identity: unconstrained, so asked.
    other_mint = _match_full(client, "run-cl3", _at_cosine(0.3464, ortho=2))
    assert other_mint["personKey"] == "p00004"
    assert other_mint["nearMiss"] is not None
    assert other_mint["nearMiss"]["key"] == "p00001"
    assert other_mint["nearMiss"]["basis"] == "face"

    # An entirely unrelated pair: also asked.
    elsewhere = _match_full(
        client, "run-cl3", _unit(0.3464 * _e(5) + math.sqrt(1 - 0.3464**2) * _e(3))
    )
    assert elsewhere["personKey"] == "p00005"
    assert elsewhere["nearMiss"] is not None
    assert elsewhere["nearMiss"]["key"] == "p00002"


def test_cannot_link_changes_the_suggestion_and_nothing_else(client):
    """The verdict is IDENTICAL with and without the constraint, field for field.

    cannot-link must never change WHO someone is — only what is suggested
    about them.  Two runs are driven with byte-identical inputs and differ in
    exactly one thing: one has the constraint recorded first.  Every other
    field of the /match response must match, including `cosine` (the rider's
    suppression must not be implemented by hiding the score) and `galleryN`
    (a suppressed suggestion must never quietly move the count).
    """
    probe = _at_cosine(0.3464)
    outfit, worn = _desc({0: 1.0}), _desc({0: 0.94, 1: 0.06})

    _match_full(client, "run-cl4", _e(0).tolist(), appearance=outfit)
    loud = _match_full(client, "run-cl4", probe, appearance=worn)

    _match_full(client, "run-cl5", _e(0).tolist(), appearance=outfit)
    _split(client, "run-cl5", "p00001", "p00002")
    quiet = _match_full(client, "run-cl5", probe, appearance=worn)

    assert loud["nearMiss"] is not None and quiet["nearMiss"] is None
    assert {k: v for k, v in loud.items() if k != "nearMiss"} == {
        k: v for k, v in quiet.items() if k != "nearMiss"
    }


# ------------------------------------------------------- gallery overlap flag


def test_enrolment_that_creates_overlap_is_flagged(client):
    """THE PATTERN, measured three times before it had a name: a split pair's
    identities GROW into each other via multi-template enrolment until their
    templates cross the match threshold (0.544 / 0.452 / 0.376 on consecutive
    benches, every pair one real person by the operator's own eyes) — and the
    gallery held that evidence with nothing surfacing it.

    Deliberate geometry.  A founded at e0; B minted at 0.30 vs A (safely below
    the 0.363 threshold).  Then a sighting X with cos(X,A)=0.45 and
    cos(X,B)=0.60 arrives: it matches B (argmax), clears every enrolment gate
    (0.413 <= 0.60 < 0.90; margin over rival A = 0.15 >= 0.05) — and the
    template just written sits 0.45 vs A, above the threshold.  B now holds a
    view the gallery could not tell from A, and the write comes back flagged.
    """
    a = _match_full(client, "run-ov", _e(0).tolist(), appearance=_desc({0: 1.0}))
    b = _match_full(client, "run-ov", _at_cosine(0.30), appearance=_desc({1: 1.0}))
    assert b["isNew"] is True and b["overlap"] is None, "0.30 apart: two people, no flag"

    # X's clothes must agree with B's (its own identity) or the appearance
    # veto refuses the very write this test needs — the first draft of this
    # test dressed X in the RIVAL's clothes and the anti-poison veto blocked
    # the enrolment, which is the layering working exactly as designed.
    x = _unit(0.45 * _e(0) + 0.487 * _e(1) + math.sqrt(1 - 0.45**2 - 0.487**2) * _e(2))
    out = _match_full(client, "run-ov", x, appearance=_desc({0: 0.2, 1: 0.8}))
    assert out["personKey"] == b["personKey"] and out["isNew"] is False
    assert out["templateAdded"] is True, "the write is what creates the overlap"
    ov = out["overlap"]
    assert ov is not None, "an above-threshold rival to the new template must flag"
    assert ov["key"] == a["personKey"]
    assert ov["cosine"] == pytest.approx(0.45, abs=1e-5)
    assert ov["appearanceSim"] == pytest.approx(0.2, abs=1e-5), (
        "torso vs the RIVAL's stored descriptor"
    )
    assert out["galleryN"] == 2, "information, never behaviour: the count is untouched"


def test_a_mint_never_carries_overlap(client):
    """Overlap is an enrolment phenomenon, impossible on a mint by construction.

    On the mint branch the best pre-write cosine is below the threshold, and
    the nearest rival to the founding view is bounded by that same best — so
    the flag cannot fire, and the code no longer even asks.  Pinned here so a
    refactor that reorders the write and the search gets caught: a mint in the
    near-miss band must carry nearMiss and never overlap.
    """
    _match_full(client, "run-ov2", _e(0).tolist())
    out = _match_full(client, "run-ov2", _at_cosine(0.30, ortho=2))
    assert out["isNew"] is True
    assert out["nearMiss"] is not None, "0.30 is inside the near-miss band"
    assert out["overlap"] is None


def test_overlap_respects_an_operator_split(client):
    """cannot_link silences the overlap flag for good: the operator has ruled.

    Two identities held apart by a split are EXPECTED to sit close; a banner
    that keeps arguing with the operator's explicit decision is noise, and
    noise trains operators to ignore banners.  Identical geometry to the
    flagged test above — the ONLY difference is the split — so this pair
    proves the suppression rather than a quirk of the vectors.
    """
    a = _match_full(client, "run-ov3", _e(0).tolist())
    b = _match_full(client, "run-ov3", _at_cosine(0.30))
    assert b["isNew"] is True
    r = client.post("/split", json={"runId": "run-ov3", "a": a["personKey"], "b": b["personKey"]})
    assert r.status_code == 200
    x = _unit(0.45 * _e(0) + 0.487 * _e(1) + math.sqrt(1 - 0.45**2 - 0.487**2) * _e(2))
    out = _match_full(client, "run-ov3", x)
    assert out["personKey"] == b["personKey"] and out["templateAdded"] is True
    assert out["overlap"] is None, "split pairs must never re-raise the banner"


def test_band_floors_mean_what_they_say_at_the_boundary(client, monkeypatch):
    """A knob set to a decimal an operator would type must fire AT that value.

    Embeddings round-trip through float32, so a true cosine of 0.29 reads back
    as 0.28999999165534973 and a hard `>=` dropped it out of the band — the
    failure is silent (a missed suggestion, never a wrong one), which is how
    it survived unexamined on the face floor. The weak band's safety margin is
    0.033 wide, so the boundary has to be honest. The existing edge test uses
    binary-exact values (0.25/0.75) and cannot catch this by construction.
    """
    monkeypatch.setenv("HECO_MATCH_NEARMISS_FLOOR", "0.29")
    _match_full(client, "run-eps", _e(0).tolist())
    out = _match_full(client, "run-eps", _at_cosine(0.29, ortho=1))
    assert out["isNew"] is True
    assert out["nearMiss"] is not None, "0.29 against a floor of 0.29 must flag"
    assert out["nearMiss"]["basis"] == "face"


# ------------------------- same-frame guard: excludeKeys + template retraction
#
# Run fa8fc3 (2026-08-07, ground truth TWO men, counted ONE) drove both of
# these.  Frame kf-577 held the two of them side by side, both matched to
# p00001 at face 0.6874 and 0.6733; by the end that one key held five
# templates whose internal cosines ran 0.2223..0.5232 — impostor-level against
# each other, one identity containing two people.  The matcher could not have
# seen it: /match is asked about ONE embedding at a time and never sees the
# frame.  The runner can, so it needs two things the service did not offer —
# a way to re-ask with the proven-wrong identity off the table, and a way to
# take back the template that identity was just given.


def test_exclude_keys_resolves_against_the_rest_of_the_gallery(client):
    """A barred key is skipped and the RUNNER-UP is returned, not a blind mint.

    This is the whole reason exclusion beats "force a mint": the second body
    in the frame is often a guest already in the gallery, and minting them
    blindly would split a returning guest every time two people share a frame.
    """
    alice, bob = person_centroid(), person_centroid()
    a = client.post("/match", json={"runId": "r", "embedding": sighting(alice)}).json()
    b = client.post("/match", json={"runId": "r", "embedding": sighting(bob)}).json()
    assert a["isNew"] and b["isNew"] and a["personKey"] != b["personKey"]

    again = client.post(
        "/match", json={"runId": "r", "embedding": sighting(alice)}
    ).json()
    assert again["personKey"] == a["personKey"], "unbarred, alice is alice"

    barred = client.post(
        "/match",
        json={
            "runId": "r",
            "embedding": sighting(alice),
            "excludeKeys": [a["personKey"]],
        },
    ).json()
    assert barred["personKey"] != a["personKey"], "the barred key is off the table"


def test_exclude_keys_mints_when_nobody_else_is_close(client):
    """Bar the only identity in the gallery and the sighting has to mint."""
    alice = person_centroid()
    a = client.post("/match", json={"runId": "r", "embedding": sighting(alice)}).json()

    barred = client.post(
        "/match",
        json={
            "runId": "r",
            "embedding": sighting(alice),
            "excludeKeys": [a["personKey"]],
        },
    ).json()
    assert barred["isNew"] is True
    assert barred["personKey"] != a["personKey"]
    assert barred["galleryN"] == 2, "the second man is now a guest in his own right"


def test_exclude_keys_absent_or_empty_changes_nothing(client):
    """The field is opt-in: no excludeKeys and an empty list both match normally."""
    alice = person_centroid()
    a = client.post("/match", json={"runId": "r", "embedding": sighting(alice)}).json()
    for payload in ({}, {"excludeKeys": []}, {"excludeKeys": None}):
        got = client.post(
            "/match", json={"runId": "r", "embedding": sighting(alice), **payload}
        ).json()
        assert got["personKey"] == a["personKey"], payload


def view_at(centroid: np.ndarray, target_cosine: float) -> list[float]:
    """A sighting at an EXACT cosine from the identity — a genuinely new view.

    ``sighting()`` lands around 0.99, which the redundancy bar
    (``template_max_cosine`` 0.90) rejects on purpose: a view identical to one
    already stored teaches the gallery nothing.  Enrolment tests therefore need
    a view that is comfortably the same person and comfortably not a duplicate.
    """
    c = centroid / np.linalg.norm(centroid)
    r = RNG.normal(size=128)
    r = r - r.dot(c) * c
    r = r / np.linalg.norm(r)
    return _unit(target_cosine * c + math.sqrt(1.0 - target_cosine**2) * r)


def test_match_reply_carries_the_rowid_of_the_template_it_enrolled(client, monkeypatch):
    """templateId names the exact row, so a retraction cannot hit the wrong one."""
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    alice = person_centroid()
    first = client.post(
        "/match", json={"runId": "r", "embedding": sighting(alice), "quality": 90.0}
    ).json()
    assert first["isNew"] is True
    assert first["templateId"] is None, "a mint is retracted as a whole key, not a row"

    second = client.post(
        "/match",
        json={"runId": "r", "embedding": view_at(alice, 0.75), "quality": 90.0},
    ).json()
    assert second["isNew"] is False
    assert second["templateAdded"] is True
    assert isinstance(second["templateId"], int)

    third = client.post(
        "/match", json={"runId": "r", "embedding": sighting(alice), "quality": 90.0}
    ).json()
    assert third["templateAdded"] is False, "0.99 is a duplicate view, not a new one"
    assert third["templateId"] is None, "nothing written, nothing to retract"


def test_forget_template_removes_exactly_that_view_and_not_the_guest(client, monkeypatch):
    """The retraction takes back a VIEW of somebody, never somebody.

    Without this the same-frame split fixes only the frame in hand: man B's
    template stays under man A and goes on capturing him in every later frame
    where he stands ALONE and there is no second body to disprove it.  That is
    exactly how fa8fc3 held two men under one key.
    """
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    alice = person_centroid()
    client.post("/match", json={"runId": "r", "embedding": sighting(alice), "quality": 90.0})
    enrolled = client.post(
        "/match", json={"runId": "r", "embedding": view_at(alice, 0.75), "quality": 90.0}
    ).json()
    assert enrolled["templateAdded"] is True
    assert enrolled["templateN"] == 2

    got = client.post(
        "/template/forget", json={"runId": "r", "templateId": enrolled["templateId"]}
    ).json()
    assert got["forgotten"] is True
    assert got["galleryN"] == 1, "the guest still exists — one fewer view of them"

    back = client.post(
        "/match", json={"runId": "r", "embedding": sighting(alice), "quality": 90.0}
    ).json()
    assert back["personKey"] == enrolled["personKey"], "still recognisable"
    assert back["templateN"] == 1, "the retracted view is really gone"


def test_forget_template_refuses_a_keys_last_template(client):
    """A key with no vectors would exist in the count and be unmatchable forever."""
    from app import store as store_mod

    alice = person_centroid()
    minted = client.post(
        "/match", json={"runId": "r", "embedding": sighting(alice)}
    ).json()
    assert minted["isNew"] is True

    with store_mod.VectorStore(
        main.gallery.db_path(main.config.data_dir(), "r")
    ) as s:
        row_id = s.conn.execute("SELECT id FROM vectors").fetchone()[0]

    got = client.post(
        "/template/forget", json={"runId": "r", "templateId": row_id}
    ).json()
    assert got["forgotten"] is False, "refused — this is the guest's only view"
    assert got["galleryN"] == 1
    after = client.post("/match", json={"runId": "r", "embedding": sighting(alice)}).json()
    assert after["personKey"] == minted["personKey"], "still recognisable"


def test_forget_template_on_an_absent_row_is_a_normal_false(client):
    """The redundancy prune may already have evicted the row — not an error."""
    alice = person_centroid()
    client.post("/match", json={"runId": "r", "embedding": sighting(alice)})
    got = client.post("/template/forget", json={"runId": "r", "templateId": 999999}).json()
    assert got == {"ok": True, "forgotten": False, "galleryN": 1}


# ------------------------- the duplicate review list (run 0f5c6d)
#
# THE MEASUREMENT.  Run 0f5c6d, 2026-08-07, ground truth THREE people,
# reported FOUR.  One man — at the back of the room by the glass door, phone
# to his ear — was minted twice: p00002 and p00003 scored 0.2117 against each
# other, a same-person miss far below the 0.363 threshold.  Nothing could act
# on it automatically, because nothing separated it from a real new arrival:
# the duplicate pair agreed on CLOTHING at 0.587 while two genuinely different
# men in the same run agreed at 0.538.  Five hundredths.  A threshold in that
# gap is a coin toss wearing a number, and getting it wrong merges a paying
# guest — the failure nobody ever sees.
#
# So this feature decides nothing.  It hands a human a short ranked list.


def review(client, run="r", **kw):
    """Ask one run's gallery which identity pairs a human should look at."""
    return client.post("/review/duplicates", json={"runId": run, **kw}).json()


def band_pair(client, run, base, appearance_a=None, appearance_b=None, cosine=0.22):
    """Two keys of one run whose mutual face score sits inside the review band.

    ``excludeKeys`` is what makes the second one a separate key at all — the
    same door the runner's same-frame guard uses — so a test can build the
    exact geometry run 0f5c6d produced without needing two real faces.
    """
    body = {"runId": run, "embedding": sighting(base), "quality": 90.0}
    if appearance_a is not None:
        body["appearance"] = appearance_a
    ka = client.post("/match", json=body).json()["personKey"]
    body = {
        "runId": run, "embedding": view_at(base, cosine), "quality": 90.0,
        "excludeKeys": [ka],
    }
    if appearance_b is not None:
        body["appearance"] = appearance_b
    kb = client.post("/match", json=body).json()["personKey"]
    return ka, kb


def hist(**bins):
    """An L1-normalised 48-bin descriptor with named mass — intersections by
    construction, so a test states the reading it means to test."""
    d = [0.0] * 48
    for i, m in bins.items():
        d[int(i[1:])] = m
    assert abs(sum(d) - 1.0) < 1e-9, "a descriptor is L1-normalised"
    return d


def test_review_asks_only_about_pairs_the_gallery_has_not_settled(client):
    """A cannot_link pair never appears — that question is already answered.

    The constraint is written by co-presence (two faces at different positions
    in one frame) or by the operator's own false-match click. Re-asking a
    settled question is exactly the noise that teaches operators to ignore
    banners, and in the motivating run this alone removed both pairs involving
    p00001 — the only guest anyone was ever certain about.
    """
    ka, kb = band_pair(client, "r", person_centroid())

    before = review(client)
    assert [frozenset((p["a"], p["b"])) for p in before["pairs"]] == [frozenset((ka, kb))]

    client.post("/split", json={"runId": "r", "a": ka, "b": kb})

    after = review(client)
    assert after["pairs"] == [], "settled, so no longer asked"
    assert after["considered"] == before["considered"], "still examined, just not asked"


def test_review_ignores_pairs_outside_the_band(client):
    """At/above threshold the gallery already calls them one person; far below,
    they are not similar in any measurable sense. Neither needs a human."""
    a = person_centroid()
    ka = client.post("/match", json={"runId": "r", "embedding": sighting(a)}).json()["personKey"]
    twin = client.post(
        "/match", json={"runId": "r", "embedding": sighting(a), "excludeKeys": [ka]}
    ).json()["personKey"]
    assert twin != ka
    assert review(client)["pairs"] == [], "0.99 is not a question, it is an answer"

    client.post("/match", json={"runId": "r2", "embedding": sighting(person_centroid())})
    client.post("/match", json={"runId": "r2", "embedding": sighting(person_centroid())})
    assert review(client, run="r2")["pairs"] == [], "orthogonal strangers are not a question"


def test_clothing_ranks_the_queue_and_never_removes_anyone(client):
    """THE 0.587/0.538 RULE, pinned with the measured numbers.

    In run 0f5c6d the DUPLICATE pair agreed on clothing at 0.587 and two
    genuinely DIFFERENT men agreed at 0.538. Both readings are reproduced
    here. The stronger agreement must LEAD the queue, and the weaker must
    still BE in it: any clothing bar between those two numbers would have put
    the real duplicate and an unrelated pair on opposite sides of a coin toss,
    and the losing side of that toss is a merged guest nobody ever sees.
    """
    dup_a = hist(b0=0.587, b1=0.413)
    dup_b = hist(b0=0.587, b2=0.413)          # intersects dup_a at 0.587
    oth_a = hist(b3=0.538, b4=0.462)
    oth_b = hist(b3=0.538, b5=0.462)          # intersects oth_a at 0.538, dup_* at 0

    dup = band_pair(client, "r", person_centroid(), dup_a, dup_b)
    oth = band_pair(client, "r", person_centroid(), oth_a, oth_b)

    got = review(client)
    ranked = [(frozenset((p["a"], p["b"])), p["clothes"]) for p in got["pairs"]]
    assert ranked[0][0] == frozenset(dup)
    assert ranked[0][1] == pytest.approx(0.587, abs=1e-3)
    assert frozenset(oth) in [r[0] for r in ranked], "the weaker pair is still asked about"
    assert dict(ranked)[frozenset(oth)] == pytest.approx(0.538, abs=1e-3)


def test_an_unmeasured_torso_still_gets_reviewed(client):
    """Absent is not zero — a guest nobody could measure must not fall off the
    list silently. It ranks behind measured pairs, it does not vanish."""
    base = person_centroid()
    k1 = client.post(
        "/match",
        json={"runId": "r", "embedding": sighting(base), "quality": 90.0,
              "appearance": [1.0 / 48] * 48},
    ).json()["personKey"]
    k2 = client.post(
        "/match",
        json={"runId": "r", "embedding": view_at(base, 0.22), "quality": 90.0,
              "excludeKeys": [k1]},
    ).json()["personKey"]
    got = review(client)
    assert len(got["pairs"]) == 1
    p = got["pairs"][0]
    assert {p["a"], p["b"]} == {k1, k2}
    assert p["clothes"] is None, "one side had no descriptor: not measured, not 0.0"


def test_review_states_what_it_dropped(client):
    """A truncated queue that looks complete is how a real duplicate goes unread."""
    base = person_centroid()
    keys = []
    for t in (0.20, 0.21, 0.22, 0.23):
        e = sighting(base) if not keys else view_at(base, t)
        keys.append(client.post(
            "/match", json={"runId": "r", "embedding": e, "excludeKeys": keys}
        ).json()["personKey"])
    full = review(client)
    capped = review(client, limit=1)
    assert capped["returned"] == 1
    assert capped["dropped"] == len(full["pairs"]) - 1
    assert capped["considered"] == full["considered"], "the cap bounds the ANSWER, not the work"


def test_review_writes_nothing(client):
    """Read-only: it asks a question, it does not act on one."""
    a = person_centroid()
    client.post("/match", json={"runId": "r", "embedding": sighting(a), "quality": 90.0})
    client.post("/match", json={"runId": "r", "embedding": view_at(a, 0.22), "quality": 90.0})
    before = client.post("/match", json={"runId": "r", "embedding": sighting(a)}).json()
    review(client)
    review(client)
    after = client.post("/match", json={"runId": "r", "embedding": sighting(a)}).json()
    assert after["galleryN"] == before["galleryN"]
    assert after["personKey"] == before["personKey"]


def test_review_on_an_empty_or_unknown_run_is_an_empty_queue(client):
    """No gallery, no question — not a 500."""
    got = review(client, run="never-ran")
    assert got == {
        "runId": "never-ran", "threshold": config.DEFAULT_THRESHOLD,
        "pairs": [], "considered": 0, "returned": 0, "dropped": 0,
    }


def test_review_rejects_an_unsafe_run_id(client):
    """Same 422 as every other run-scoped call — a runId is part of a filename."""
    assert client.post(
        "/review/duplicates", json={"runId": "../../etc/passwd"}
    ).status_code == 422
