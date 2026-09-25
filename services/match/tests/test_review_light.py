"""The review's light guard (match 0.15.0, 2026-09-25).

A duplicate is minted exactly when a face fails to match, and a change of
light is one cause — so a colour set-aside (clothes, head, beard) must not
hide the pair whose two identities were read under two different lights.
Replayed on run f0bfc5's crops: an 8% light shift set a genuine duplicate
aside, a stage wash seven or eight, and frame white balance cannot see a
light that falls on one person.  Each sighting now carries the face's skin
reading (the cheek's median log(R/G), log(B/G)); when two identities' skin
differs by more than HECO_REVIEW_LIGHT_TOL the colour reasons are held back
and the pair stays in the ranked queue, saying so.

Pinned: the wire (2 finite floats or absent), the body-log column and its
migration, the guard holding colours but never sex, age or stature, its off
switch, unmeasured skin meaning today's rule, and the knobs on /health and
in compose.
"""

import itertools
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from app import config, gallery, main, store
from fastapi.testclient import TestClient

DIM = 128
BODY = {"h": 1200.0, "w": 400.0, "yBottom": 1500.0, "frameH": 2160}
REPO = Path(__file__).resolve().parents[3]
#: A warm hall's skin, and the same skin under an 8% warmer light.
SKIN = [0.30, -0.25]
SKIN_WARM8 = [0.30 + math.log(1.08), -0.25 + math.log(0.92)]


def _e(i: int) -> np.ndarray:
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def hub() -> list[float]:
    """The identity every in-band pair is measured against."""
    return [float(x) for x in _e(0)]


def spoke(i: int, cosine: float = 0.30) -> list[float]:
    """A probe at an EXACT cosine from the hub along its own axis."""
    v = cosine * _e(0) + math.sqrt(1.0 - cosine * cosine) * _e(i)
    return [float(x) for x in v]


def torso(bin_: int) -> list[float]:
    """A v3 torso whose colour sits in one bin (plus texture and edges)."""
    v = [0.0] * 64
    v[bin_], v[39], v[49] = 0.9, 0.07, 0.03
    return v


WHITE_SHIRT, BLUE_SHIRT = torso(38), torso(24)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with gallery data redirected to a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    store.close_all_stores()
    with TestClient(main.app) as c:
        yield c
    store.close_all_stores()


@pytest.fixture()
def ticking(monkeypatch):
    """A store clock that moves one second per write."""
    t0 = datetime(2026, 9, 25, 21, 0, 0, tzinfo=UTC)
    counter = itertools.count()
    monkeypatch.setattr(
        store, "_now", lambda: (t0 + timedelta(seconds=next(counter))).isoformat()
    )


def sighted(client, emb, appearance, skin, n=3, run="r"):
    """One identity seen ``n`` times in one garment under one light; its key."""
    key = None
    for _ in range(n):
        payload = {"runId": run, "embedding": emb, "quality": 90.0, "body": BODY,
                   "appearance": appearance}
        if skin is not None:
            payload["skin"] = skin
        res = client.post("/match", json=payload)
        assert res.status_code == 200, res.text
        key = res.json()["personKey"]
    return key


def review(client, run="r"):
    """The run's review queue."""
    res = client.post("/review/duplicates", json={"runId": run})
    assert res.status_code == 200, res.text
    return res.json()


# ------------------------------------------------------------- the wire


def test_match_takes_skin_and_refuses_the_wrong_shape(client):
    """Two finite log ratios or absent; anything else names the contract."""
    ok = client.post("/match", json={"runId": "r", "embedding": hub(), "skin": SKIN})
    assert ok.status_code == 200
    for bad in ([0.1], [0.1, 0.2, 0.3], [0.1, 1e9]):
        res = client.post("/match", json={"runId": "r", "embedding": hub(), "skin": bad})
        assert res.status_code == 422 and "skin" in res.text, bad
    # NaN never reaches the wire as JSON; the validator refuses it all the same.
    with pytest.raises(ValueError, match="skin"):
        main.MatchRequest(runId="r", embedding=hub(), skin=[float("nan"), 0.0])


def test_the_body_row_logs_skin_and_a_legacy_table_migrates(client, tmp_path):
    """Per sighting, beside the head and beard; an older body table gets the
    column ALTERed in and its rows read None."""
    sighted(client, hub(), WHITE_SHIRT, SKIN, n=1)
    sighted(client, spoke(1), WHITE_SHIRT, None, n=1)
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.reading():
        ev = s.sighting_evidence()
    assert ev[0].skin.tolist() == pytest.approx(SKIN) and ev[1].skin is None

    path = tmp_path / "gallery-legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE body_sightings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL,"
        " h REAL NOT NULL, w REAL NOT NULL, y_bottom REAL NOT NULL, frame_h INTEGER NOT NULL,"
        " created_at TEXT NOT NULL, face_w REAL, appearance BLOB, head BLOB, beard BLOB);"
    )
    conn.commit()
    conn.close()
    with store.VectorStore(path) as legacy:
        cols = {row[1] for row in legacy.conn.execute("PRAGMA table_info(body_sightings)")}
        assert "skin" in cols


# ------------------------------------------------------------- the guard


def test_same_light_the_clothes_still_set_the_pair_aside(client, ticking):
    """Both read under one light: the colour rule applies as before."""
    sighted(client, hub(), WHITE_SHIRT, SKIN)
    sighted(client, spoke(1), BLUE_SHIRT, SKIN)
    got = review(client)
    assert got["excluded"]["clothes"] == 1 and got["keptByLight"] == 0
    (row,) = got["setAside"]
    assert row["reasons"] == ["clothes"]
    assert row["why"]["light"]["shift"] == pytest.approx(0.0)
    assert row["why"]["light"]["held"] == []


def test_a_light_change_holds_the_colour_set_aside_back(client, ticking):
    """The late identity read under an 8% warmer light: its clothes may be the
    light's, so the pair is ASKED — in the queue, the held reason named."""
    sighted(client, hub(), WHITE_SHIRT, SKIN)
    sighted(client, spoke(1), BLUE_SHIRT, SKIN_WARM8)
    got = review(client)
    assert got["excluded"] == {
        "gender": 0, "age": 0, "stature": 0, "clothes": 0, "head": 0, "beard": 0,
        "headwear": 0}
    assert got["setAside"] == [] and got["keptByLight"] == 1
    (row,) = got["pairs"]
    light = row["why"]["light"]
    assert light["held"] == ["clothes"]
    assert light["shift"] == pytest.approx(-math.log(0.92), abs=1e-6), "the larger ratio gap"
    assert light["shift"] > config.DEFAULT_REVIEW_LIGHT_TOL


def test_the_guard_is_off_at_zero_and_absent_skin_is_todays_rule(client, ticking, monkeypatch):
    """HECO_REVIEW_LIGHT_TOL=0: set aside again. One side unmeasured (a runner
    from before 0.15.0): no guard — the rule as it was."""
    sighted(client, hub(), WHITE_SHIRT, SKIN)
    sighted(client, spoke(1), BLUE_SHIRT, SKIN_WARM8)
    monkeypatch.setenv("HECO_REVIEW_LIGHT_TOL", "0")
    got = review(client)
    assert got["excluded"]["clothes"] == 1 and got["keptByLight"] == 0
    monkeypatch.delenv("HECO_REVIEW_LIGHT_TOL")
    sighted(client, spoke(2), BLUE_SHIRT, None)  # no skin on this identity
    got = review(client)
    unmeasured = [r for r in got["setAside"] if r["why"]["light"]["shift"] is None]
    assert len(unmeasured) == 1 and unmeasured[0]["reasons"] == ["clothes"]


def test_the_guard_never_holds_back_sex_age_or_stature(client, ticking):
    """Those are not colours: a child against an adult stays set aside under
    any light, and is not counted as kept by the light."""
    for emb, shirt, skin, age in ((hub(), WHITE_SHIRT, SKIN, 8.0),
                                  (spoke(1), BLUE_SHIRT, SKIN_WARM8, 40.0)):
        for _ in range(3):
            client.post("/match", json={
                "runId": "r", "embedding": emb, "quality": 90.0, "body": BODY,
                "appearance": shirt, "skin": skin,
                "attributes": {"gender": "M", "genderP": 0.6, "age": age}})
    got = review(client)
    (row,) = got["setAside"]
    assert row["reasons"] == ["age"], row["reasons"]
    assert row["why"]["light"]["held"] == ["clothes"]
    assert got["keptByLight"] == 0


def test_health_and_compose_carry_the_new_knobs(client, monkeypatch):
    """Readable beside the queue they shaped, and passed into the container."""
    body = client.get("/health").json()
    assert body["reviewLightTol"] == pytest.approx(config.DEFAULT_REVIEW_LIGHT_TOL) == 0.07
    assert body["reviewBeardPale"] is False
    monkeypatch.setenv("HECO_REVIEW_LIGHT_TOL", "")
    monkeypatch.setenv("HECO_REVIEW_BEARD_PALE", "")
    assert config.review_light_tol() == pytest.approx(0.07)
    assert config.review_beard_pale() is False
    compose = (REPO / "docker-compose.yml").read_text()
    for name in ("HECO_REVIEW_LIGHT_TOL", "HECO_REVIEW_BEARD_PALE"):
        assert f"{name}: ${{{name}-}}" in compose, f"{name} has no compose passthrough"
