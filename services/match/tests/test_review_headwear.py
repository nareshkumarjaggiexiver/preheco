"""The head-covering reads and the review rule that may act on them (match 0.16.0).

Run 8b8b87's review pair p00005/p00009 put a Sikh man in a sky-blue turban
beside a bare-headed man, and nothing could say why: the colour head rule
compares headwear against headwear only, because a dark turban and black hair
are one colour.  The embed service now reads the head of every mint and
template enrolment with SigLIP B/16 (two views, 4 class logits each), the
runner writes the 8 logits onto the sighting's body row, and the review:

* reports ``why.headwear`` {a, b, nA, nB, turbanA, bareA, turbanB, bareB} on
  EVERY row, whether or not the rule is on;
* sets a pair aside (reason ``headwear``) only with HECO_REVIEW_HEADWEAR=1,
  only between two confidently male identities, only turban (>= N confident
  reads, none bare) against bare (the reverse) — never a dupatta, a cap or an
  unsure head against anything;
* is not a colour: the light guard never holds it back;
* writes nothing: no cannot_link, no merge.

Geometry as in test_review_why: a hub on e0, spokes at cosine 0.30.
"""

import itertools
import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from app import config, gallery, main, store
from app.headwear import (
    HEADWEAR_DIM,
    HeadwearTally,
    headwear_apart,
    headwear_label,
    headwear_tallies,
    read_call,
)
from fastapi.testclient import TestClient

DIM = 128
FRAME_H = 2160
BODY = {"h": 1200.0, "w": 400.0, "yBottom": 1500.0, "frameH": FRAME_H}
MODEL = "b05840caadfa+0123456789ab"


def _e(i: int) -> np.ndarray:
    """The i-th standard basis vector."""
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def hub() -> np.ndarray:
    """The identity every in-band pair is measured against."""
    return _e(0)


def spoke(i: int, cosine: float = 0.30) -> np.ndarray:
    """A probe at an EXACT cosine from the hub along its own axis."""
    return cosine * _e(0) + math.sqrt(1.0 - cosine * cosine) * _e(i)


def logits(loose: list[float], tight: list[float]) -> list[float]:
    """8 logits from two views' class probabilities (log p IS a logit up to a
    per-view constant, which the softmax removes)."""
    return [float(math.log(p)) for p in (*loose, *tight)]


#: Class order on the wire: turban, bare, dupatta_or_scarf, cap_or_hat.
TURBAN = logits([0.95, 0.03, 0.01, 0.01], [0.90, 0.06, 0.02, 0.02])
BARE = logits([0.02, 0.96, 0.01, 0.01], [0.05, 0.90, 0.03, 0.02])
DUPATTA = logits([0.02, 0.02, 0.95, 0.01], [0.02, 0.03, 0.94, 0.01])
CAP = logits([0.02, 0.02, 0.01, 0.95], [0.03, 0.02, 0.01, 0.94])
#: The views disagree (the loose view sees a neighbour's turban in its corner).
UNSURE = logits([0.90, 0.08, 0.01, 0.01], [0.10, 0.85, 0.03, 0.02])
#: Turban in both views but the tight view only 0.70 sure: under the 0.80 bar.
WEAK_TURBAN = logits([0.95, 0.03, 0.01, 0.01], [0.70, 0.20, 0.05, 0.05])


def excluded(**counts) -> dict:
    """The reply's ``excluded`` block: every signal at zero but those named."""
    return {
        "gender": 0, "age": 0, "stature": 0, "clothes": 0, "head": 0, "beard": 0, "headwear": 0,
        **counts,
    }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with gallery data redirected to a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    store.close_all_stores()
    with TestClient(main.app) as c:
        yield c
    store.close_all_stores()


def attrs(gender: str, p: float = 0.97, age: float = 35.0) -> dict:
    """One face's attribute reading as the wire carries it."""
    return {"gender": gender, "genderP": p, "age": age}


def match(client, v, attributes=None, body=BODY, skin=None, run="r") -> dict:
    """POST /match; a body by default, so the reply carries a bodyId."""
    payload = {"runId": run, "embedding": [float(x) for x in v], "quality": 90.0}
    if attributes is not None:
        payload["attributes"] = attributes
    if body is not None:
        payload["body"] = body
    if skin is not None:
        payload["skin"] = skin
    res = client.post("/match", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


def write(client, body_id, reading, model=MODEL, run="r"):
    """POST /body-sightings/headwear."""
    return client.post("/body-sightings/headwear", json={
        "runId": run, "bodyId": body_id, "headwear": reading, "model": model,
    })


def person(client, tmp_path, v, reads, gender="M", p=0.97, run="r", skin=None) -> str:
    """One identity: minted with a sex reading, settled to a confident sex
    (four agreeing templates: (4 * 0.97 + 1) / 6 = 0.81 over the 0.8 bar),
    then one sighting per head-covering read, each read written by bodyId.
    ``gender=None`` leaves the identity's sex unmeasured."""
    first = match(client, v, None if gender is None else attrs(gender, p), skin=skin, run=run)
    key = first["personKey"]
    if gender is not None:
        s = store.open_store(gallery.db_path(tmp_path, run))
        with s.transaction():
            for _ in range(3):
                s.add(key, v, quality=70.0, attributes=attrs(gender, p))
    body_ids = [first["bodyId"]] + [
        match(client, v, skin=skin, run=run)["bodyId"] for _ in range(max(0, len(reads) - 1))
    ]
    for body_id, reading in zip(body_ids, reads, strict=False):
        res = write(client, body_id, reading, run=run)
        assert res.status_code == 200 and res.json() == {"ok": True, "written": True}, res.text
    return key


def review(client, run="r", **kw):
    """The run's review queue."""
    res = client.post("/review/duplicates", json={"runId": run, **kw})
    assert res.status_code == 200, res.text
    return res.json()


def rows(report) -> list[dict]:
    """Every row, queued or set aside."""
    return report["pairs"] + report["setAside"]


def row_of(report, a, b) -> dict:
    """The row of one unordered pair."""
    (row,) = [r for r in rows(report) if {r["a"], r["b"]} == {a, b}]
    return row


def side(row, key) -> str:
    """'A' when ``key`` is the row's a, else 'B'."""
    return "A" if row["a"] == key else "B"


@pytest.fixture()
def rule_on(monkeypatch):
    """HECO_REVIEW_HEADWEAR=1: the rule acts."""
    monkeypatch.setenv("HECO_REVIEW_HEADWEAR", "1")


# ------------------------------------------------------------- the pure call


def test_a_read_is_confident_only_when_both_views_agree_over_the_bar():
    """The reference reader's rule: both views argmax the class, the SMALLER
    of the two probabilities at the class's bar (turban 0.80, bare 0.50)."""
    assert read_call(TURBAN, 0.8, 0.5) == "turban"
    assert read_call(BARE, 0.8, 0.5) == "bare"
    assert read_call(UNSURE, 0.8, 0.5) == "unsure", "the views disagree"
    assert read_call(WEAK_TURBAN, 0.8, 0.5) == "unsure", "tight view 0.70 < 0.80"
    assert read_call(WEAK_TURBAN, 0.7, 0.5) == "turban", "the bar is config, not stored"
    assert read_call(DUPATTA, 0.8, 0.5) == "unsure", "dupatta has no confident call"
    assert read_call(CAP, 0.8, 0.5) == "unsure", "caps come off: no confident call"
    # bare at exactly the bar is confident (>=), a hair under is not
    at_bar = logits([0.25, 0.5, 0.125, 0.125], [0.25, 0.5, 0.125, 0.125])
    assert read_call(at_bar, 0.8, 0.5) == "bare"
    assert read_call(at_bar, 0.8, 0.5001) == "unsure"
    # logits are softmax-invariant to a per-view shift, and huge ones do not overflow
    shifted = [x + (400.0 if i < 4 else -300.0) for i, x in enumerate(TURBAN)]
    assert read_call(shifted, 0.8, 0.5) == "turban"
    assert read_call([float("nan")] * 8, 0.8, 0.5) == "unsure"
    assert read_call([0.0] * 7, 0.8, 0.5) == "unsure"


def test_labels_need_n_unanimous_confident_reads():
    """turban / bare need N confident reads and NONE of the other; both is mixed."""
    assert headwear_label(None, 2) is None
    assert headwear_label(HeadwearTally(0, 0, 0), 2) is None
    assert headwear_label(HeadwearTally(11, 11, 0), 2) == "turban"
    assert headwear_label(HeadwearTally(7, 0, 2), 2) == "bare"
    assert headwear_label(HeadwearTally(7, 0, 2), 3) == "unsure", "N = 3 loses the boy"
    assert headwear_label(HeadwearTally(5, 1, 0), 2) == "unsure"
    assert headwear_label(HeadwearTally(5, 1, 0), 1) == "turban"
    assert headwear_label(HeadwearTally(9, 4, 1), 2) == "mixed"
    assert headwear_label(HeadwearTally(3, 0, 0), 2) == "unsure"


def test_apart_is_turban_against_bare_between_two_confident_men_only():
    """Every guard of the rule, on the pure function."""
    t, b = HeadwearTally(11, 11, 0), HeadwearTally(7, 0, 7)
    men = (("M", 0.9), ("M", 0.85))
    assert headwear_apart(t, b, *men, 2, 0.8) is True
    assert headwear_apart(b, t, *men, 2, 0.8) is True, "order-free"
    assert headwear_apart(t, t, *men, 2, 0.8) is False, "turban vs turban: the colour rule's"
    assert headwear_apart(b, b, *men, 2, 0.8) is False
    assert headwear_apart(t, HeadwearTally(7, 1, 6), *men, 2, 0.8) is False, "mixed"
    assert headwear_apart(t, HeadwearTally(5, 0, 1), *men, 2, 0.8) is False, "one bare read"
    assert headwear_apart(t, None, *men, 2, 0.8) is False, "absent is not zero"
    assert headwear_apart(t, b, ("M", 0.9), ("F", 0.9), 2, 0.8) is False, "a woman"
    assert headwear_apart(t, b, ("M", 0.9), ("M", 0.79), 2, 0.8) is False, "unsure sex"
    assert headwear_apart(t, b, ("M", 0.9), (None, None), 2, 0.8) is False, "sex unread"
    assert headwear_apart(t, b, *men, 0, 0.8) is False, "N 0 is off"
    assert headwear_apart(t, b, *men, 2, 0.0) is False, "no gender bar, no male gate, no rule"


# ------------------------------------------------------------- storage


def test_the_write_lands_on_the_row_and_the_first_write_stamps_the_gallery(client, tmp_path):
    """The logits ride on the /match call's own body row, as float32[8]."""
    m = match(client, hub())
    assert write(client, m["bodyId"], TURBAN).json() == {"ok": True, "written": True}
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.reading():
        (ev,) = s.sighting_evidence()
        assert ev.key == m["personKey"] and ev.headwear.dtype == np.float32
        assert ev.headwear.tolist() == pytest.approx(TURBAN, abs=1e-6)
        assert ev.appearance is None and ev.head is None, "absent stays absent"
        assert s.headwear_stamp() == MODEL


def test_a_gone_row_or_an_unknown_run_is_written_false_and_creates_nothing(client, tmp_path):
    """Retracted by the same-frame guard, or no such run: ordinary, not an error."""
    m = match(client, hub())
    client.post("/template/forget", json={"runId": "r", "bodyId": m["bodyId"]})
    assert write(client, m["bodyId"], BARE).json() == {"ok": True, "written": False}
    assert write(client, 99, BARE).json()["written"] is False
    assert write(client, 1, BARE, run="never-ran").json() == {"ok": True, "written": False}
    assert not gallery.db_path(tmp_path, "never-ran").exists(), "no empty gallery conjured"


def test_the_wire_is_8_finite_logits_a_model_and_a_row(client):
    """Anything else names the contract (422), and a bad runId is a 422 too."""
    m = match(client, hub())
    for bad in ({"headwear": [0.0] * 7}, {"headwear": [0.0] * 9}, {"model": ""}, {"bodyId": 0}):
        payload = {"runId": "r", "bodyId": m["bodyId"], "headwear": TURBAN, "model": MODEL, **bad}
        assert client.post("/body-sightings/headwear", json=payload).status_code == 422, bad
    # 1e400 is valid JSON that parses to infinity: refused as a 422 with a
    # message, never a 500 from trying to echo the infinity back.
    text = json.dumps(
        {"runId": "r", "bodyId": m["bodyId"], "headwear": [0.0] * 7 + ["INF"], "model": MODEL}
    ).replace('"INF"', "1e400")
    res = client.post("/body-sightings/headwear", content=text,
                      headers={"content-type": "application/json"})
    assert res.status_code == 422 and "FINITE" in res.json()["detail"]
    assert write(client, 1, TURBAN, run="../etc").status_code == 422


def test_another_models_logits_are_refused_409_and_write_nothing(client, tmp_path):
    """Logits from another graph or prompt set are on another scale."""
    m1, m2 = match(client, hub()), match(client, hub())
    assert write(client, m1["bodyId"], TURBAN).json()["written"] is True
    res = write(client, m2["bodyId"], BARE, model="ffffffffffff+ffffffffffff")
    assert res.status_code == 409
    assert MODEL in res.json()["detail"] and "ffffffffffff" in res.json()["detail"]
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.reading():
        assert [e.headwear is not None for e in s.sighting_evidence()] == [True]
        assert s.headwear_stamp() == MODEL


def test_an_existing_gallery_migrates_and_its_rows_read_null(tmp_path):
    """A 0.15.x gallery (body table with skin, no headwear) opens, gains the
    column by ADD COLUMN, and every old row reads None: not measured."""
    path = tmp_path / "gallery-old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE body_sightings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL,"
        " h REAL NOT NULL, w REAL NOT NULL, y_bottom REAL NOT NULL, frame_h INTEGER NOT NULL,"
        " created_at TEXT NOT NULL, face_w REAL, appearance BLOB, head BLOB, beard BLOB,"
        " skin BLOB);"
    )
    conn.execute(
        "INSERT INTO body_sightings (key, h, w, y_bottom, frame_h, created_at, skin)"
        " VALUES ('p00001', 1200, 400, 1500, 2160, '2026-09-25T10:00:00+00:00', ?)",
        (np.array([0.3, -0.2], np.float32).tobytes(),),
    )
    conn.commit()
    conn.close()
    with store.VectorStore(path) as s:
        cols = [row[1] for row in s.conn.execute("PRAGMA table_info(body_sightings)")]
        assert cols[-1] == "headwear", "added last, like every column before it"
        (row,) = s.sighting_evidence()
        assert row.headwear is None and row.skin.size == 2
        assert s.headwear_stamp() is None
        assert s.set_body_headwear(1, BARE, MODEL) is True
    with store.VectorStore(path) as s:
        (row,) = s.sighting_evidence()
        assert row.headwear.size == HEADWEAR_DIM


def test_a_fresh_table_has_the_column_order_the_migration_produces(tmp_path):
    """CREATE and ALTER agree, so no gallery's columns ever read in another order."""
    with store.VectorStore(tmp_path / "fresh.db") as s:
        cols = [row[1] for row in s.conn.execute("PRAGMA table_info(body_sightings)")]
    added = [name for name, _ in store._BODY_COLUMNS_ADDED]
    assert cols[-len(added):] == added


def test_the_reads_move_with_a_merge_and_leave_with_mark_staff(tmp_path):
    """They are the row's: re-keyed by merge, deleted with the person."""
    with store.VectorStore(tmp_path / "s.db") as s:
        s.add("p00001", hub())
        s.add("p00002", spoke(1))
        b1 = s.add_body_sighting("p00001", 1200.0, 400.0, 1500.0, FRAME_H)
        b2 = s.add_body_sighting("p00002", 1200.0, 400.0, 1500.0, FRAME_H)
        s.set_body_headwear(b1, TURBAN, MODEL)
        s.set_body_headwear(b2, TURBAN, MODEL)
        assert s.merge("p00001", "p00002") is True
        tallies = headwear_tallies(s.sighting_evidence(), 0.8, 0.5)
        assert tallies == {"p00001": HeadwearTally(2, 2, 0)}
        s.remove("p00001")
        assert s.sighting_evidence() == []


# ------------------------------------------------------------- the review


def test_why_headwear_rides_every_row_and_log_only_sets_nothing_aside(client, tmp_path):
    """The default: turban against bare between two men is REPORTED, still asked."""
    kt = person(client, tmp_path, hub(), [TURBAN] * 3)
    kb = person(client, tmp_path, spoke(1), [BARE] * 2 + [UNSURE])
    k_none = person(client, tmp_path, spoke(2), [])
    got = review(client)
    assert config.review_headwear() is False
    assert got["excluded"] == excluded() and got["setAside"] == []
    assert len(got["pairs"]) == 2, "turban against bare between two men: reported, still asked"
    row = row_of(got, kt, kb)
    t, b = side(row, kt), side(row, kb)
    assert row["why"]["headwear"] == {
        t.lower(): "turban", b.lower(): "bare",
        f"n{t}": 3, f"n{b}": 3,
        f"turban{t}": 3, f"bare{t}": 0, f"turban{b}": 0, f"bare{b}": 2,
    }
    unread = row_of(got, kt, k_none)
    u = side(unread, k_none)
    assert unread["why"]["headwear"][u.lower()] is None, "never read: null, not 'bare'"
    assert unread["why"]["headwear"][f"n{u}"] == 0


def test_with_the_rule_on_only_the_turban_against_bare_pair_is_set_aside(
    client, tmp_path, rule_on
):
    """Five spokes around a turbaned hub; only the bare man leaves the queue."""
    kt = person(client, tmp_path, hub(), [TURBAN] * 3)
    kb = person(client, tmp_path, spoke(1), [BARE] * 2)
    k_dup = person(client, tmp_path, spoke(2), [DUPATTA] * 4)
    k_cap = person(client, tmp_path, spoke(3), [CAP] * 4)
    k_uns = person(client, tmp_path, spoke(4), [UNSURE, WEAK_TURBAN, UNSURE])
    k_tur = person(client, tmp_path, spoke(5), [TURBAN] * 3)
    got = review(client)
    assert got["excluded"] == excluded(headwear=1)
    (aside,) = got["setAside"]
    assert {aside["a"], aside["b"]} == {kt, kb} and aside["reasons"] == ["headwear"]
    queued = {frozenset((r["a"], r["b"])) for r in got["pairs"]}
    assert queued == {frozenset((kt, k)) for k in (k_dup, k_cap, k_uns, k_tur)}, (
        "never dupatta, cap or unsure against anything; turban vs turban is the colour rule's"
    )


def test_a_woman_an_unsure_sex_or_an_unread_sex_is_never_set_aside(client, tmp_path, rule_on):
    """The male gate: a dupatta comes and goes, and genderage cannot age this
    camera's children — so only two confident men."""
    kt = person(client, tmp_path, hub(), [TURBAN] * 3)
    k_woman = person(client, tmp_path, spoke(1), [BARE] * 3, gender="F")
    k_unsure = person(client, tmp_path, spoke(2), [BARE] * 3, p=0.6)
    k_unread = person(client, tmp_path, spoke(3), [BARE] * 3, gender=None)
    got = review(client)
    assert got["excluded"] == excluded(gender=1), "the woman goes on sex, not on headwear"
    (aside,) = got["setAside"]
    assert {aside["a"], aside["b"]} == {kt, k_woman} and aside["reasons"] == ["gender"]
    assert {frozenset((r["a"], r["b"])) for r in got["pairs"]} == {
        frozenset((kt, k_unsure)), frozenset((kt, k_unread))}


def test_one_contrary_read_or_too_few_reads_keeps_the_pair_asked(
    client, tmp_path, rule_on, monkeypatch
):
    """Unanimity and N: one clean bare read cancels a turban identity (a
    neighbour's turban in the crop is the one observed false turban), and one
    confident read is not two — until MIN_N says one is enough."""
    kb = person(client, tmp_path, hub(), [BARE] * 3)
    k_mixed = person(client, tmp_path, spoke(1), [TURBAN] * 4 + [BARE])
    k_once = person(client, tmp_path, spoke(2), [TURBAN, UNSURE])
    got = review(client)
    assert got["excluded"] == excluded() and got["setAside"] == []
    mixed = row_of(got, kb, k_mixed)
    assert mixed["why"]["headwear"][side(mixed, k_mixed).lower()] == "mixed"
    once = row_of(got, kb, k_once)
    assert once["why"]["headwear"][side(once, k_once).lower()] == "unsure", "1 read < N = 2"
    monkeypatch.setenv("HECO_REVIEW_HEADWEAR_MIN_N", "1")
    got = review(client)
    assert got["excluded"] == excluded(headwear=1)
    (aside,) = got["setAside"]
    assert {aside["a"], aside["b"]} == {kb, k_once}, "one read is enough at N = 1; mixed never"


def test_the_switch_min_n_and_the_gender_bar_are_all_off_switches(
    client, tmp_path, monkeypatch
):
    """HECO_REVIEW_HEADWEAR 0/empty, MIN_N 0, GENDER_MIN_P 0: nothing set aside."""
    kt = person(client, tmp_path, hub(), [TURBAN] * 3)
    kb = person(client, tmp_path, spoke(1), [BARE] * 3)
    for env in ({"HECO_REVIEW_HEADWEAR": ""}, {"HECO_REVIEW_HEADWEAR": "0"},
                {"HECO_REVIEW_HEADWEAR": "1", "HECO_REVIEW_HEADWEAR_MIN_N": "0"},
                {"HECO_REVIEW_HEADWEAR": "1", "HECO_REVIEW_GENDER_MIN_P": "0"}):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        got = review(client)
        assert got["excluded"] == excluded() and got["setAside"] == [], env
        for k in env:
            monkeypatch.delenv(k)
    monkeypatch.setenv("HECO_REVIEW_HEADWEAR", "1")
    (aside,) = review(client)["setAside"]
    assert {aside["a"], aside["b"]} == {kt, kb}


def test_the_bars_are_config_the_logits_never_change(client, tmp_path, rule_on, monkeypatch):
    """Raising the turban bar over the reads' own probability un-calls them."""
    kt = person(client, tmp_path, hub(), [TURBAN] * 3)
    person(client, tmp_path, spoke(1), [BARE] * 3)
    assert review(client)["excluded"]["headwear"] == 1
    monkeypatch.setenv("HECO_REVIEW_HEADWEAR_TURBAN_P", "0.95")
    got = review(client)
    assert got["excluded"]["headwear"] == 0
    (row,) = got["pairs"]
    assert row["why"]["headwear"][f"turban{side(row, kt)}"] == 0, "the tight view read 0.90"


def test_the_light_guard_never_holds_headwear_back(client, tmp_path, rule_on, monkeypatch):
    """Two men read under different light: a colour reason would be held,
    the head covering is not a colour and still sets the pair aside."""
    monkeypatch.setenv("HECO_REVIEW_LIGHT_TOL", "0.07")
    warm = [0.40, -0.10]
    kt = person(client, tmp_path, hub(), [TURBAN] * 3, skin=[0.30, -0.20])
    kb = person(client, tmp_path, spoke(1), [BARE] * 3, skin=warm)
    got = review(client)
    (aside,) = got["setAside"]
    assert {aside["a"], aside["b"]} == {kt, kb}
    assert aside["reasons"] == ["headwear"]
    assert aside["why"]["light"]["shift"] > config.review_light_tol()
    assert got["keptByLight"] == 0 and got["excluded"] == excluded(headwear=1)


@pytest.fixture()
def ticking(monkeypatch):
    """A store clock that moves one second per write (the colour rules' 2 s span)."""
    t0 = datetime(2026, 9, 25, 21, 0, 0, tzinfo=UTC)
    counter = itertools.count()
    monkeypatch.setattr(
        store, "_now", lambda: (t0 + timedelta(seconds=next(counter))).isoformat()
    )


def clothed(client, tmp_path, v, reads, torso, skin) -> str:
    """A confident man whose every sighting also carries a v3 torso and a skin
    reading, and whose head-covering reads are written by bodyId."""
    key, body_ids = None, []
    for i in range(max(3, len(reads))):
        payload = {"runId": "r", "embedding": [float(x) for x in v], "quality": 90.0,
                   "body": BODY, "appearance": torso, "skin": skin}
        if i == 0:
            payload["attributes"] = attrs("M")
        res = client.post("/match", json=payload)
        assert res.status_code == 200, res.text
        key = key or res.json()["personKey"]
        body_ids.append(res.json()["bodyId"])
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.transaction():
        for _ in range(3):
            s.add(key, v, quality=70.0, attributes=attrs("M"))
    for body_id, reading in zip(body_ids, reads, strict=False):
        assert write(client, body_id, reading).json()["written"] is True
    return key


def test_a_colour_reason_held_by_light_does_not_hold_the_headwear_reason(
    client, tmp_path, rule_on, monkeypatch, ticking
):
    """Same pair, also a clear clothing clash: clothes held, headwear stands,
    and the pair is not counted as kept by the light (it was not kept)."""
    monkeypatch.setenv("HECO_REVIEW_LIGHT_TOL", "0.07")
    red = [1.0 / 3 if i in (0, 1, 2) else 0.0 for i in range(64)]
    blue = [1.0 / 3 if i in (15, 16, 17) else 0.0 for i in range(64)]
    kt = clothed(client, tmp_path, hub(), [TURBAN] * 3, red, [0.30, -0.20])
    kb = clothed(client, tmp_path, spoke(1), [BARE] * 3, blue, [0.40, -0.10])
    got = review(client)
    (aside,) = got["setAside"]
    assert {aside["a"], aside["b"]} == {kt, kb}
    assert aside["why"]["light"]["held"] == ["clothes"], "the colour reason IS held"
    assert aside["reasons"] == ["headwear"]
    assert got["keptByLight"] == 0 and got["excluded"] == excluded(headwear=1)
    # ...and with the light guard off both speak, colour first, headwear counted last.
    monkeypatch.setenv("HECO_REVIEW_LIGHT_TOL", "0")
    (aside,) = review(client)["setAside"]
    assert aside["reasons"] == ["clothes", "headwear"]


def test_nothing_is_written_to_cannot_link_and_the_pair_stays_mergeable(
    client, tmp_path, rule_on
):
    """A set-aside is a question not asked, never a fact recorded."""
    kt = person(client, tmp_path, hub(), [TURBAN] * 3)
    kb = person(client, tmp_path, spoke(1), [BARE] * 3)
    assert review(client)["excluded"]["headwear"] == 1
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.reading():
        assert s.cannot_link(kt, kb) is False
    merged = client.post("/merge", json={"runId": "r", "keep": kt, "drop": kb}).json()
    assert merged["merged"] is True


def test_health_reports_the_headwear_policy_and_empty_means_unset(client, monkeypatch):
    """Log-only by default, readable next to the queue it produced."""
    body = client.get("/health").json()
    assert body["reviewHeadwear"] is False
    assert body["reviewHeadwearMinN"] == 2
    assert body["reviewHeadwearTurbanP"] == pytest.approx(0.80)
    assert body["reviewHeadwearBareP"] == pytest.approx(0.50)
    for name in ("HECO_REVIEW_HEADWEAR", "HECO_REVIEW_HEADWEAR_MIN_N",
                 "HECO_REVIEW_HEADWEAR_TURBAN_P", "HECO_REVIEW_HEADWEAR_BARE_P"):
        monkeypatch.setenv(name, "")
    assert config.review_headwear() is False
    assert config.review_headwear_min_n() == 2
    monkeypatch.setenv("HECO_REVIEW_HEADWEAR", "1")
    monkeypatch.setenv("HECO_REVIEW_HEADWEAR_MIN_N", "-3")
    assert client.get("/health").json()["reviewHeadwear"] is True
    assert config.review_headwear_min_n() == 0, "clamped: a negative N is off, not a wildcard"
