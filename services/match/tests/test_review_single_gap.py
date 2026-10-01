"""Two set-aside rules calibrated on the Sharon clip (2026-09-30):

* the CLOTHES GAP — two well-seen sides whose typical-against-typical clothing
  comparison sits well under each side's self-agreement (config
  DEFAULT_REVIEW_CLOTHES_GAP says the measurement);
* the SINGLE GOOD VIEW — a once-seen side against an established one, on two
  or more hard clashes at once (sex, beard, turban-vs-bare).

Galleries are written directly so each test controls exactly how many reads,
templates and body rows each side holds.
"""

import itertools
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from app import config, gallery, store
from app.gallery import clothes_gap_of, identity_reads, single_view_apart
from app.headwear import headwear_tallies
from fastapi.testclient import TestClient

from tests.test_review_head_beard import DARK, NONE, hub, spoke
from tests.test_review_headwear import BARE, TURBAN

DIM = 128
BODY = {"h": 1200.0, "w": 400.0, "yBottom": 1500.0, "frameH": 2160}


def torso(bins: dict[int, float]) -> np.ndarray:
    """A 64-float v3 torso descriptor with mass in ``bins`` (sums to 1)."""
    v = np.zeros(64)
    total = float(sum(bins.values()))
    for b, w in bins.items():
        v[b] = w / total
    return v


#: Two garments that share most of their colour mass but not all: their
#: best cross reading is high (0.75) while each is self-consistent (1.0).
CREAM = torso({3: 0.75, 40: 0.25})
STRIPED = torso({3: 0.75, 41: 0.25})
#: Something else entirely.
NAVY = torso({20: 1.0})


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with gallery data redirected to a temp directory."""
    from app import main
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    store.close_all_stores()
    with TestClient(main.app) as c:
        yield c
    store.close_all_stores()


@pytest.fixture()
def ticking(monkeypatch):
    """A store clock that moves one second per write."""
    t0 = datetime(2026, 9, 30, 1, 0, 0, tzinfo=UTC)
    counter = itertools.count()
    monkeypatch.setattr(store, "_now", lambda: (t0 + timedelta(seconds=next(counter))).isoformat())


def identity(st, key, emb, *, templates=1, torsos=(), beard=None, headwear=None,
             gender=None, gender_p=None):
    """Write one identity: ``templates`` face templates carrying ``gender``,
    and one body row per torso read (each with the beard / headwear given)."""
    attrs = {"gender": gender, "genderP": gender_p, "age": 30.0} if gender else None
    for _ in range(templates):
        st.add(key, emb, quality=100.0, attributes=attrs)
    for t in torsos:
        row = st.add_body_sighting(
            key, BODY["h"], BODY["w"], BODY["yBottom"], BODY["frameH"], 100.0,
            appearance=t, beard=None if beard is None else np.array(beard, dtype=np.float32),
        )
        if headwear is not None:
            st.set_body_headwear(row, np.array(headwear, dtype=np.float32), "test-model")


def review(client, monkeypatch, run="r", **env):
    """The run's review queue, with the named config knobs set for this call."""
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    res = client.post("/review/duplicates", json={"runId": run})
    assert res.status_code == 200, res.text
    return res.json()


def the_pair(report):
    """The one in-band pair, wherever it landed."""
    rows = report["pairs"] + report["setAside"]
    assert len(rows) == 1, rows
    return rows[0]


# ------------------------------------------------------------- the gap


def test_the_gap_is_typical_against_typical_and_needs_both_sides_well_seen():
    """A pair whose best cross is high but whose typical cross is low has a
    large gap; under min_n reads on either side there is no gap at all."""
    a = identity_reads([(float(i), CREAM) for i in range(10)])
    b = identity_reads([(float(i), STRIPED) for i in range(10)])
    gap = clothes_gap_of(a, b, 8)
    assert gap == pytest.approx(1.0 - 0.75, abs=1e-6), "self 1.0, median cross 0.75"
    few = identity_reads([(0.0, STRIPED)] * 3)
    assert clothes_gap_of(a, few, 8) is None, "3 reads: not well seen"
    assert clothes_gap_of(None, b, 8) is None
    far = identity_reads([(float(i), NAVY) for i in range(10)])
    assert clothes_gap_of(a, far, 8) == pytest.approx(1.0, abs=1e-6)


def test_a_well_seen_pair_far_apart_on_the_gap_is_set_aside_on_clothes(
    client, tmp_path, ticking, monkeypatch,
):
    """Two guests of 10 reads each, cross 0.75 (over every clash bar) and gap
    0.25: asked at gap 0.32, set aside at 0.2 — and the gap is in ``why``."""
    st = store.open_store(gallery.db_path(tmp_path, "r"))
    with st.transaction():
        identity(st, "p00001", hub(), torsos=[CREAM] * 10)
        identity(st, "p00002", spoke(1, 0.30), torsos=[STRIPED] * 10)
    strict = review(client, monkeypatch, HECO_REVIEW_CLOTHES_GAP=0.32)
    row = the_pair(strict)
    assert row["why"]["clothes"]["gap"] == pytest.approx(0.25, abs=1e-6)
    assert strict["pairs"], "0.25 is under a 0.32 bar: still asked"
    loose = review(client, monkeypatch, HECO_REVIEW_CLOTHES_GAP=0.2)
    assert loose["setAside"] and loose["setAside"][0]["reasons"] == ["clothes"]
    assert loose["excluded"]["clothes"] == 1
    assert review(client, monkeypatch, HECO_REVIEW_CLOTHES_GAP=0)["pairs"], "0 turns it off"


# --------------------------------------------------------- single view


def _tallies(rows):
    return headwear_tallies(rows, 0.8, 0.5, 0.0)


def test_single_view_needs_one_once_seen_side_an_established_other_and_two_clashes():
    """The rule's shape, on the pure function."""
    from app.gallery import beard_reads, identity_gender
    from app.store import SightingEvidence

    def ev(key, beard, hw, n):
        return [SightingEvidence(key, f"2026-09-30T01:00:{i:02d}Z", None, None,
                                 np.array(beard, dtype=np.float32), None,
                                 np.array(hw, dtype=np.float32)) for i in range(n)]

    rows = ev("man", DARK, TURBAN, 12) + ev("woman", NONE, BARE, 1)
    beards = beard_reads(rows)
    coverings = _tallies(rows)
    singles = {"woman": ("F", 0.996, 36.0)}
    genders = {"man": identity_gender([("M", 0.9, 30.0)] * 6), "woman": ("F", 0.67)}
    kw = dict(min_clashes=2, est_n=8, sex_p=0.95, gender_min_p=0.8, pale=False, headwear_min_n=2)
    r = single_view_apart("man", "woman", singles, genders, beards["man"], beards["woman"],
                          coverings, 12, 1, **kw)
    assert r == {"single": "woman", "clashes": ["sex", "beard", "turban"]}
    # The established side needs its rows.
    assert single_view_apart("man", "woman", singles, genders, beards["man"], beards["woman"],
                             coverings, 5, 1, **kw) is None
    # Both once-seen, or neither: no side to trust.
    assert single_view_apart("man", "woman", {}, genders, beards["man"], beards["woman"],
                             coverings, 12, 1, **kw) is None
    # An unsure single view (sex 0.7) does not clash on sex.
    r2 = single_view_apart("man", "woman", {"woman": ("F", 0.7, 36.0)}, genders, beards["man"],
                           beards["woman"], coverings, 12, 1, **kw)
    assert r2["clashes"] == ["beard", "turban"]
    # Off at 0.
    assert single_view_apart("man", "woman", singles, genders, beards["man"], beards["woman"],
                             coverings, 12, 1, **{**kw, "min_clashes": 0}) is None


def test_a_once_seen_guest_two_hard_clashes_from_an_established_one_is_set_aside(
    client, tmp_path, ticking, monkeypatch,
):
    """p00004 (bearded, turbaned, male over many reads) against p00023 (one
    view: female 0.996, no beard, bare head): set aside as 'single' with the
    clashes named; with the woman's sex unsure and no beard read, one clash
    is not enough and the pair is asked."""
    st = store.open_store(gallery.db_path(tmp_path, "r"))
    with st.transaction():
        identity(st, "p00004", hub(), templates=6, torsos=[NAVY] * 12, beard=DARK,
                 headwear=TURBAN, gender="M", gender_p=0.9)
        identity(st, "p00023", spoke(1, 0.30), templates=1, torsos=[CREAM], beard=NONE,
                 headwear=BARE, gender="F", gender_p=0.996)
    rep = review(client, monkeypatch, HECO_REVIEW_SINGLE_MIN_CLASHES=2, HECO_REVIEW_HEADWEAR=1)
    assert rep["setAside"] and rep["setAside"][0]["reasons"] == ["single"]
    assert rep["setAside"][0]["why"]["single"] == {
        "single": "p00023", "clashes": ["sex", "beard", "turban"],
    }
    assert rep["excluded"]["single"] == 1
    assert review(client, monkeypatch, HECO_REVIEW_SINGLE_MIN_CLASHES=0)["pairs"], "0 turns it off"
    # Four clashes needed: still asked.
    assert review(client, monkeypatch, HECO_REVIEW_SINGLE_MIN_CLASHES=4)["pairs"]


def test_the_defaults_are_the_calibrated_ones():
    """The bars the service ships with are the measured ones."""
    assert config.DEFAULT_REVIEW_CLOTHES_GAP == 0.32
    assert config.DEFAULT_REVIEW_CLOTHES_GAP_MIN_N == 8
    assert config.DEFAULT_REVIEW_SINGLE_MIN_CLASHES == 2
    assert config.DEFAULT_REVIEW_SINGLE_EST_N == 8
    assert config.DEFAULT_REVIEW_SINGLE_SEX_P == 0.95
    assert math.isclose(config.review_clothes_gap(), 0.32)
