"""The review queue's clothing set-aside (2026-09-24, night).

THE ASK, in the user's words: "cosine for cloths is not good; keep cloth
colour, pattern style and compare".  The torso descriptor already RANKED the
queue; it may now also SET A PAIR ASIDE, on both identities' own testimony:
each must wear ONE garment across its sightings (at least three v3 torso
reads over two seconds, median pairwise intersection >= 0.6), and even the
best reading of one against the other must clash (< 0.35).  Measured on run
f0bfc5: own reads agree at a median 0.90, one person split across a time
gap at 0.77-0.97, the queue's different-people pairs at 0.10-0.42.

What these tests pin:

* both sides needed — an unmeasured side, a side under three reads or two
  seconds, a side whose own reads disagree, is never a reason;
* v2 (48-float) torsos never count, absent is not zero;
* the evidence is the BODY LOG's — every sighting's torso, retracted with
  the row;
* setting aside writes nothing: no cannot_link, and the operator's /merge
  still works;
* HECO_REVIEW_CLOTHES_CLASH=0 is the off switch and off is the old code
  path — the body log is never even read.

Geometry as in test_review_why: a HUB identity on e0 and SPOKES at cosine
0.30 from it on their own axes, so every (hub, spoke) pair is in the review
band and no two spokes are.
"""

import itertools
import math
import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from app import config, gallery, main, store
from app.appearance import best_cross, self_agreement, spread
from fastapi.testclient import TestClient

DIM = 128
FRAME_H = 2160
BODY = {"h": 1200.0, "w": 400.0, "yBottom": 1500.0, "frameH": FRAME_H}


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


def torso(bins: dict[int, float]) -> list[float]:
    """A v3 torso: colour spread over ``bins`` (weight 0.9), plain weave.

    Two plain garments of different colours share the pattern parts and
    floor at 0.10, exactly as the real descriptor does.
    """
    v = np.zeros(64)
    total = float(sum(bins.values()))
    for b, w in bins.items():
        v[b] = 0.9 * w / total
    v[39], v[49] = 0.07, 0.03
    return [float(x) for x in v]


RED = torso({0: 1.0})
BLUE = torso({24: 1.0})
#: Half red, half blue: agrees with RED at 0.55 — not a clash.
RED_BLUE = torso({0: 1.0, 24: 1.0})


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


@pytest.fixture()
def ticking(monkeypatch):
    """A store clock that moves one second per write."""
    t0 = datetime(2026, 9, 24, 21, 0, 0, tzinfo=UTC)
    counter = itertools.count()
    monkeypatch.setattr(
        store, "_now", lambda: (t0 + timedelta(seconds=next(counter))).isoformat()
    )


def match(client, run, emb, appearance=None, body=BODY):
    """POST /match with a torso and a body unless told otherwise."""
    payload = {"runId": run, "embedding": emb, "quality": 90.0}
    if appearance is not None:
        payload["appearance"] = appearance
    if body is not None:
        payload["body"] = body
    res = client.post("/match", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


def sightings(client, run, emb, reads):
    """One identity seen once per read; returns its key."""
    key = None
    for r in reads:
        key = match(client, run, emb, r)["personKey"]
    return key


def review(client, run="r", **kw):
    """Ask one run's gallery which identity pairs a human should look at."""
    res = client.post("/review/duplicates", json={"runId": run, **kw})
    assert res.status_code == 200, res.text
    return res.json()


def in_queue(report, a, b) -> bool:
    """Whether the ranked queue still asks about the pair."""
    return any({p["a"], p["b"]} == {a, b} for p in report["pairs"])


# ------------------------------------------------------------- the arithmetic


def test_self_agreement_is_the_median_pair_and_one_read_is_none():
    """The two statistics the rule reads, on known histograms."""
    red, blue = np.asarray(RED), np.asarray(BLUE)
    assert self_agreement([red]) is None, "one read agrees with nothing"
    assert self_agreement([red, red, red]) == pytest.approx(1.0)
    # 2 red + 2 blue: pairs 1, 1, and four at 0.10 -> median 0.10
    assert self_agreement([red, red, blue, blue]) == pytest.approx(0.10)
    assert best_cross([red], [blue]) == pytest.approx(0.10)
    assert best_cross([red, blue], [blue]) == pytest.approx(1.0), "BEST, not mean"
    assert best_cross([], [blue]) is None


def test_spread_keeps_the_ends_and_caps_evenly():
    """The cap keeps the first and last read, so the span survives it."""
    reads = list(range(100))
    kept = spread(reads, 24)
    assert len(kept) == 24 and kept[0] == 0 and kept[-1] == 99
    assert spread(reads[:5], 24) == reads[:5]


# ------------------------------------------------------------- the rule


def test_two_steady_garments_that_clash_are_set_aside(client, ticking, tmp_path):
    """Hub in red three times over 5 s, spoke in blue three times: set aside."""
    kh = sightings(client, "r", hub(), [RED] * 3)
    ks = sightings(client, "r", spoke(1), [BLUE] * 3)
    got = review(client)
    assert not in_queue(got, kh, ks)
    assert got["excluded"]["clothes"] == 1
    assert got["returned"] == 0 and got["dropped"] == 0


def test_a_single_agreeing_read_keeps_the_pair(client, ticking):
    """Every spoke read is half red: its best against the hub is 0.55."""
    kh = sightings(client, "r", hub(), [RED] * 3)
    ks = sightings(client, "r", spoke(1), [RED_BLUE] * 3)
    got = review(client)
    assert in_queue(got, kh, ks) and got["excluded"]["clothes"] == 0


def test_an_identity_whose_own_reads_disagree_has_no_clothing(client, ticking):
    """Red, blue, red, blue: a merged pair of people, or a band on a pillar."""
    kh = sightings(client, "r", hub(), [RED, BLUE, RED, BLUE])
    ks = sightings(client, "r", spoke(1), [torso({12: 1.0})] * 3)
    got = review(client)
    assert in_queue(got, kh, ks), "self-agreement 0.10 < 0.6: no testimony"


def test_under_three_reads_never_sets_aside(client, ticking):
    """Two reads are not enough testimony, however they clash."""
    kh = sightings(client, "r", hub(), [RED] * 2)
    ks = sightings(client, "r", spoke(1), [BLUE] * 3)
    assert in_queue(review(client), kh, ks)


def test_three_reads_inside_two_seconds_are_one_moment(client, monkeypatch):
    """Three consecutive frames are one pose under one light, read three times."""
    t0 = datetime(2026, 9, 24, 21, 0, 0, tzinfo=UTC)
    counter = itertools.count()
    monkeypatch.setattr(
        store, "_now",
        lambda: (t0 + timedelta(milliseconds=250 * next(counter))).isoformat(),
    )
    kh = sightings(client, "r", hub(), [RED] * 3)
    ks = sightings(client, "r", spoke(1), [BLUE] * 3)
    assert in_queue(review(client), kh, ks)


def test_an_unmeasured_side_never_excludes(client, ticking):
    """No torso on one side is one opinion, not a disagreement."""
    kh = sightings(client, "r", hub(), [RED] * 3)
    ks = sightings(client, "r", spoke(1), [None] * 3)
    got = review(client)
    assert in_queue(got, kh, ks) and got["excluded"]["clothes"] == 0


def test_v2_torsos_never_count(client, ticking):
    """A 48-float row is the chin-down partition: not clothing evidence here."""
    v2_red = [0.0] * 48
    v2_red[0] = 1.0
    v2_blue = [0.0] * 48
    v2_blue[24] = 1.0
    kh = sightings(client, "r", hub(), [v2_red] * 3)
    ks = sightings(client, "r", spoke(1), [v2_blue] * 3)
    got = review(client)
    assert in_queue(got, kh, ks) and got["excluded"]["clothes"] == 0


def test_setting_aside_writes_nothing_and_the_operator_can_still_merge(
    client, ticking, tmp_path
):
    """Set aside removes a question, never answers one."""
    kh = sightings(client, "r", hub(), [RED] * 3)
    ks = sightings(client, "r", spoke(1), [BLUE] * 3)
    assert review(client)["excluded"]["clothes"] == 1
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.reading():
        assert s.cannot_link(kh, ks) is False, "set aside is not a recorded fact"
    merged = client.post("/merge", json={"runId": "r", "keep": kh, "drop": ks}).json()
    assert merged["merged"] is True


def test_clash_zero_is_off_and_the_evidence_is_still_shown(client, ticking, monkeypatch):
    """Off: the pair is back in the queue, nothing set aside, why.clothes intact."""
    kh = sightings(client, "r", hub(), [RED] * 3)
    ks = sightings(client, "r", spoke(1), [BLUE] * 3)
    on = review(client)
    assert on["excluded"]["clothes"] == 1 and len(on["setAside"]) == 1

    monkeypatch.setenv("HECO_REVIEW_CLOTHES_CLASH", "0")
    off = review(client)
    assert off["excluded"] == excluded()
    assert off["setAside"] == []
    assert in_queue(off, kh, ks)
    (pair,) = off["pairs"]
    assert pair["clothes"] == pytest.approx(0.10), "the ranking clothes number is unchanged"
    assert pair["why"]["clothes"]["cross"] == pytest.approx(0.10)
    assert on["setAside"][0]["why"] == pair["why"], "the evidence does not depend on the knob"


# ------------------------------------------------------------- the well-seen tier
#
# Run e5bae3 (2026-09-25): a white shirt against a blue one agreed at 0.374 in
# its best pair of reads, a black kurta against a light check at 0.496 — both
# identities seen 17-37 times, both asked, while 27 well-seen identities
# scored their own early half against their late half at 0.887 and up.

#: 0.35 in bin 0 (red) and 0.55 in bin 24 (blue): agrees with RED at exactly
#: 0.45 — over the 0.35 clash, under the well-seen 0.55.
MOSTLY_BLUE = torso({0: 7.0, 24: 11.0})
#: 0.5 red and 0.4 blue: agrees with RED at 0.60 — over the well-seen clash.
MOSTLY_RED = torso({0: 5.0, 24: 4.0})


def test_the_tier_fixtures_agree_with_red_where_the_tests_say():
    """The fixtures' arithmetic, pinned so the tests below mean what they say."""
    assert best_cross([np.asarray(RED)], [np.asarray(MOSTLY_BLUE)]) == pytest.approx(0.45)
    assert best_cross([np.asarray(RED)], [np.asarray(MOSTLY_RED)]) == pytest.approx(0.60)


def test_a_well_seen_pair_that_half_agrees_is_set_aside(client, ticking):
    """Eight reads each over eight seconds, best cross 0.45: set aside."""
    kh = sightings(client, "r", hub(), [RED] * 8)
    ks = sightings(client, "r", spoke(1), [MOSTLY_BLUE] * 8)
    got = review(client)
    assert not in_queue(got, kh, ks)
    assert got["excluded"]["clothes"] == 1
    (row,) = got["setAside"]
    assert row["reasons"] == ["clothes"]
    assert row["why"]["clothes"]["cross"] == pytest.approx(0.45)


def test_one_thinly_seen_side_keeps_the_charitable_clash(client, ticking):
    """Seven reads on one side: the 0.35 rule alone speaks, and 0.45 is asked."""
    kh = sightings(client, "r", hub(), [RED] * 8)
    ks = sightings(client, "r", spoke(1), [MOSTLY_BLUE] * 7)
    got = review(client)
    assert in_queue(got, kh, ks) and got["excluded"]["clothes"] == 0


def test_a_well_seen_pair_over_the_clash_is_asked(client, ticking):
    """Mostly red against red, 0.60: a garment that may be the same, asked."""
    kh = sightings(client, "r", hub(), [RED] * 8)
    ks = sightings(client, "r", spoke(1), [MOSTLY_RED] * 8)
    got = review(client)
    assert in_queue(got, kh, ks) and got["excluded"]["clothes"] == 0


def test_well_seen_but_self_disagreeing_has_no_clothing(client, ticking):
    """Eight reads that disagree with each other are no testimony either."""
    kh = sightings(client, "r", hub(), [RED, BLUE] * 4)
    ks = sightings(client, "r", spoke(1), [torso({12: 1.0})] * 8)
    assert in_queue(review(client), kh, ks)


@pytest.mark.parametrize("knob", ["HECO_REVIEW_CLOTHES_WELL_SEEN_N", "HECO_REVIEW_CLOTHES_CLASH"])
def test_the_tier_turns_off(client, ticking, monkeypatch, knob):
    """WELL_SEEN_N=0 turns the tier off; CLOTHES_CLASH=0 turns clothing off, tier and all."""
    kh = sightings(client, "r", hub(), [RED] * 8)
    ks = sightings(client, "r", spoke(1), [MOSTLY_BLUE] * 8)
    assert not in_queue(review(client), kh, ks)
    monkeypatch.setenv(knob, "0")
    got = review(client)
    assert in_queue(got, kh, ks) and got["excluded"]["clothes"] == 0


# ------------------------------------------------------------- setAside


def test_a_set_aside_pair_comes_back_with_its_evidence_and_reason(client, ticking):
    """The machine's exclusion stays inspectable: evidence, reason, face score."""
    kh = sightings(client, "r", hub(), [RED] * 3)
    ks = sightings(client, "r", spoke(1), [BLUE] * 4)
    got = review(client)
    (row,) = got["setAside"]
    assert {row["a"], row["b"]} == {kh, ks}
    assert row["reasons"] == ["clothes"]
    assert row["cosine"] == pytest.approx(0.30, abs=1e-5)
    assert row["clothes"] == pytest.approx(0.10), "best template torso, as on queue rows"
    wc = row["why"]["clothes"]
    hub_side, spoke_side = ("A", "B") if row["a"] == kh else ("B", "A")
    assert wc["self" + hub_side] == pytest.approx(1.0)
    assert wc["self" + spoke_side] == pytest.approx(1.0)
    assert wc["cross"] == pytest.approx(0.10)
    assert (wc["n" + hub_side], wc["n" + spoke_side]) == (3, 4)


def test_queue_rows_carry_why_clothes_and_null_is_not_zero(client, ticking):
    """One side unmeasured: its self is null, the cross is null, its n is 0."""
    kh = sightings(client, "r", hub(), [RED] * 3)
    sightings(client, "r", spoke(1), [None] * 2)
    (row,) = review(client)["pairs"]
    wc = row["why"]["clothes"]
    hub_side, spoke_side = ("A", "B") if row["a"] == kh else ("B", "A")
    assert wc["self" + hub_side] == pytest.approx(1.0)
    assert wc["self" + spoke_side] is None
    assert wc["cross"] is None
    assert (wc["n" + hub_side], wc["n" + spoke_side]) == (3, 0)


def test_side_signals_land_in_set_aside_with_every_reason_that_spoke(client, ticking):
    """A child-aged hub in red against an adult in blue: age AND clothes.

    Counted once, under the first reason; listed with both.
    """
    kh = None
    for _ in range(3):
        res = client.post("/match", json={
            "runId": "r", "embedding": hub(), "quality": 90.0, "appearance": RED,
            "body": BODY, "attributes": {"gender": "M", "genderP": 0.6, "age": 8.0},
        }).json()
        kh = res["personKey"]
    ks = None
    for _ in range(3):
        res = client.post("/match", json={
            "runId": "r", "embedding": spoke(1), "quality": 90.0, "appearance": BLUE,
            "body": BODY, "attributes": {"gender": "M", "genderP": 0.6, "age": 40.0},
        }).json()
        ks = res["personKey"]
    got = review(client)
    assert got["excluded"] == excluded(age=1)
    (row,) = got["setAside"]
    assert {row["a"], row["b"]} == {kh, ks}
    assert row["reasons"] == ["age", "clothes"]


def test_set_aside_is_ranked_like_the_queue_and_capped_at_limit(client, ticking):
    """Three set-aside pairs, limit 2: the two strongest faces come back."""
    kh = sightings(client, "r", hub(), [RED] * 3)
    keys = {}
    for i, cos in ((1, 0.30), (2, 0.33), (3, 0.25)):
        keys[cos] = sightings(client, "r", spoke(i, cos), [BLUE] * 3)
    got = review(client, limit=2)
    assert got["excluded"]["clothes"] == 3
    assert [round(r["cosine"], 2) for r in got["setAside"]] == [0.33, 0.30]
    assert all(kh in (r["a"], r["b"]) for r in got["setAside"])


def test_an_operator_can_merge_a_pair_straight_out_of_set_aside(client, ticking):
    """Mergeable means the ordinary /merge, on the keys setAside names."""
    sightings(client, "r", hub(), [RED] * 3)
    sightings(client, "r", spoke(1), [BLUE] * 3)
    (row,) = review(client)["setAside"]
    before = client.post("/match", json={"runId": "r", "embedding": hub()}).json()["galleryN"]
    res = client.post("/merge", json={"runId": "r", "keep": row["a"], "drop": row["b"]}).json()
    assert res == {"merged": True, "galleryN": before - 1}


def test_health_reports_the_clothing_policy_and_min_n_is_at_least_two(client, monkeypatch):
    """The policy that produced a queue is readable next to it."""
    body = client.get("/health").json()
    assert body["reviewClothesClash"] == pytest.approx(config.DEFAULT_REVIEW_CLOTHES_CLASH)
    assert body["reviewClothesMinN"] == 3
    assert body["reviewClothesSelfMin"] == pytest.approx(0.6)
    assert body["reviewClothesWellSeenN"] == 8
    assert body["reviewClothesWellSeenClash"] == pytest.approx(0.55)
    for name in (
        "HECO_REVIEW_CLOTHES_CLASH", "HECO_REVIEW_CLOTHES_MIN_N", "HECO_REVIEW_CLOTHES_SELF_MIN",
    ):
        monkeypatch.setenv(name, "")
    assert config.review_clothes_clash() == pytest.approx(0.35)
    assert config.review_clothes_min_n() == 3
    monkeypatch.setenv("HECO_REVIEW_CLOTHES_MIN_N", "1")
    assert config.review_clothes_min_n() == 2, "one read has no self-agreement"


# ------------------------------------------------------------- the body log


def test_every_sighting_logs_its_torso_on_its_body_row(client, tmp_path):
    """Per sighting, not per template; no body, no row; no torso, no reading."""
    kh = match(client, "r", hub(), RED)["personKey"]
    match(client, "r", hub(), BLUE)
    match(client, "r", hub(), None)                 # unmeasured torso: no reading
    match(client, "r", hub(), RED, body=None)       # no body: no row at all
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.reading():
        ev = s.sighting_evidence()
        assert [e.key for e in ev] == [kh, kh]
        assert [int(np.argmax(e.appearance[:39])) for e in ev] == [0, 24]
        assert len(s.body_sightings()) == 3


def test_a_retracted_body_row_takes_its_torso_with_it(client, tmp_path):
    """The same-frame guard's bodyId retraction removes the torso too."""
    kh = match(client, "r", hub(), RED)["personKey"]
    body_id = match(client, "r", hub(), BLUE)["bodyId"]
    res = client.post("/template/forget", json={"runId": "r", "bodyId": body_id}).json()
    assert res["bodyForgotten"] is True
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.reading():
        assert [int(np.argmax(e.appearance[:39])) for e in s.sighting_evidence()] == [0]
        assert s.keys() == [kh]


def test_a_body_table_without_the_appearance_column_migrates_and_reads_nothing(tmp_path):
    """Tonight's galleries: body_sightings with face_w and no appearance."""
    path = tmp_path / "gallery-legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE body_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, h REAL NOT NULL,
            w REAL NOT NULL, y_bottom REAL NOT NULL, frame_h INTEGER NOT NULL,
            created_at TEXT NOT NULL, face_w REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO body_sightings (key, h, w, y_bottom, frame_h, created_at, face_w)"
        " VALUES ('p00001', 1200, 400, 1500, 2160, '2026-09-24T09:00:00+00:00', 110)"
    )
    conn.commit()
    conn.close()
    with store.VectorStore(path) as s:
        cols = {row[1] for row in s.conn.execute("PRAGMA table_info(body_sightings)")}
        assert "appearance" in cols
        assert s.sighting_evidence() == []
        assert len(s.body_sightings()) == 1
        s.add_body_sighting("p00001", 1200.0, 400.0, 1500.0, FRAME_H, 110.0, appearance=RED)
        (row,) = s.sighting_evidence()
        assert row.appearance.size == 64


def test_torso_reads_drop_v2_and_time_the_span():
    """v2 rows are skipped and the span is first-to-last write time."""
    rows = [
        store.SightingEvidence("p1", "2026-09-24T21:00:00+00:00", np.asarray(RED, np.float32)),
        store.SightingEvidence("p1", "2026-09-24T21:00:03+00:00", np.asarray(RED, np.float32)),
        store.SightingEvidence("p1", "2026-09-24T21:00:01+00:00", np.zeros(48, np.float32)),
        store.SightingEvidence("p2", "2026-09-24T21:00:00+00:00", np.zeros(48, np.float32)),
    ]
    reads = gallery.torso_reads(rows)
    assert set(reads) == {"p1"}, "an identity with only v2 rows is absent"
    assert reads["p1"].n == 2 and reads["p1"].span_s == pytest.approx(3.0)
    assert reads["p1"].agreement == pytest.approx(1.0)


def test_templates_are_the_fallback_only_for_an_identity_without_body_log_torsos(
    client, ticking, tmp_path
):
    """A pre-0.13.0 gallery (run c84098) holds its torsos on templates alone,
    and they still testify; an identity WITH body-log reads never mixes its
    templates in — each template's torso is also on its sighting's body row.
    """
    kh = sightings(client, "r", hub(), [RED] * 3)        # body log: 3 red reads
    ks = match(client, "r", spoke(1), None, body=None)["personKey"]
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.transaction():
        for _ in range(3):                                # templates only, as c84098
            s.add(ks, spoke(1), quality=90.0, appearance=BLUE)
        s.add(kh, hub(), quality=90.0, appearance=BLUE)   # never read: kh has a body log
    got = review(client)
    (row,) = got["setAside"]
    assert {row["a"], row["b"]} == {kh, ks} and row["reasons"] == ["clothes"]
    wc = row["why"]["clothes"]
    hub_side, spoke_side = ("A", "B") if row["a"] == kh else ("B", "A")
    assert (wc["n" + hub_side], wc["n" + spoke_side]) == (3, 3)
    assert wc["self" + hub_side] == pytest.approx(1.0), "kh's stray BLUE template is not a read"
