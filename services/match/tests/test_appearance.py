"""The torso-appearance tie-breaker (v1): storage, similarity, the enrol veto.

Why every test here checks TWO things at once.  Clothing is constant within
one event, so a torso descriptor is real evidence — but the bench measured
its limit: the closest impostor pair on this camera, two DIFFERENT men at
cosine 0.377 (above the 0.363 face threshold), were BOTH IN LIGHT SHIRTS.
Appearance may therefore only VETO a write (refuse a template enrolment),
never make or unmake a match — so alongside each veto assertion these tests
pin that the verdict (isNew / personKey / cosine) is exactly what the face
alone would have produced.

Vector helpers follow tests/test_templates.py: pose() places views at exact
angles on a great circle so cosines are exact, because the interesting
behaviour lives within a few hundredths of the thresholds.  Descriptors are
built by hist(), which places unit mass in named bins so intersections are
exact too (1.0 same shirt, 0.0 different shirt, 0.5 half-shared).
"""

import math
import sqlite3

import numpy as np
import pytest
from app import config, gallery, main, staff, store
from app.appearance import best_intersection, intersection
from fastapi.testclient import TestClient

DIM = 128
THRESHOLD = config.DEFAULT_THRESHOLD  # 0.363 — never moved by these tests
#: A cosine that clears EVERY M1 enrolment gate (threshold + confidence with
#: margin to spare, below the 0.90 near-duplicate ceiling, no rival), so the
#: only thing that can refuse the template is the appearance veto under test.
ENROLLABLE = THRESHOLD + config.DEFAULT_TEMPLATE_CONFIDENCE + 0.02


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise to unit length (the store does this too; be explicit here)."""
    return v / np.linalg.norm(v)


def _json(v: np.ndarray) -> list[float]:
    """A vector as the JSON list the /match endpoint takes."""
    return [float(x) for x in v]


def pose(degrees: float) -> np.ndarray:
    """One view at a stated angle on a fixed great circle (exact cosines)."""
    e1, e2 = np.zeros(DIM), np.zeros(DIM)
    e1[0], e2[1] = 1.0, 1.0
    return np.cos(np.radians(degrees)) * e1 + np.sin(np.radians(degrees)) * e2


def at(cosine: float) -> np.ndarray:
    """A probe at an EXACT cosine from pose(0) — the geometry of a boundary test."""
    return pose(math.degrees(math.acos(cosine)))


def hist(*bins: int) -> list[float]:
    """A 48-bin torso histogram with uniform mass over the named H×S bins.

    hist(0) vs hist(0) intersects at exactly 1.0 (same shirt), hist(0) vs
    hist(1) at 0.0 (different shirt), hist(0, 1) vs hist(1, 2) at 0.5 — so
    the clash boundary can be tested exactly rather than approximately.
    """
    h = np.zeros(48, dtype=np.float32)
    for b in bins:
        h[b] += 1.0
    return (h / h.sum()).astype(float).tolist()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with gallery + staff data redirected to a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    store.close_all_stores()
    with TestClient(main.app) as c:
        yield c
    store.close_all_stores()


def _match(client, run_id: str, v: np.ndarray, quality: float = 70.0,
           appearance: list[float] | None = None, site_id: str | None = None) -> dict:
    """POST /match, optionally carrying a torso descriptor and a siteId."""
    body = {"runId": run_id, "embedding": _json(v), "quality": quality}
    if appearance is not None:
        body["appearance"] = appearance
    if site_id is not None:
        body["siteId"] = site_id
    res = client.post("/match", json=body)
    assert res.status_code == 200, res.text
    return res.json()


# ------------------------------------------------------- the pure similarity


def test_intersection_identical_is_one():
    """The same shirt in two frames: full mass overlap, similarity 1.0."""
    assert intersection(hist(0, 5, 12), hist(0, 5, 12)) == pytest.approx(1.0, abs=1e-6)


def test_intersection_disjoint_is_zero():
    """Mass in entirely different H×S bins: nothing shared, similarity 0.0."""
    assert intersection(hist(0), hist(1)) == pytest.approx(0.0, abs=1e-6)


def test_intersection_partial_overlap_is_exact():
    """Half-shared mass scores exactly 0.5 — the boundary the knob sits on."""
    assert intersection(hist(0, 1), hist(1, 2)) == pytest.approx(0.5, abs=1e-6)


def test_intersection_rejects_length_mismatch():
    """A wire bug must raise, not quietly score 0.0 and become a veto."""
    with pytest.raises(ValueError):
        intersection(hist(0), [0.5, 0.5])
    with pytest.raises(ValueError):
        intersection([], [])


def test_best_intersection_absent_is_none_not_zero():
    """Absent is not zero: no query or no stored descriptors means None."""
    stored = [np.asarray(hist(0), dtype=np.float32)]
    assert best_intersection(None, stored) is None
    assert best_intersection(hist(0), []) is None
    # And BEST is charitable: agreement with any one stored descriptor wins.
    several = [np.asarray(hist(1), dtype=np.float32), np.asarray(hist(0), dtype=np.float32)]
    assert best_intersection(hist(0), several) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- storage


def test_appearance_round_trip_skips_nulls_and_stays_out_of_the_scan(tmp_path):
    """Descriptors survive a reopen; NULL rows are skipped; the scan is face-only."""
    path = tmp_path / "s.db"
    with store.VectorStore(path) as s:
        s.add("p00001", pose(0.0), quality=70.0, appearance=hist(0))
        s.add("p00001", pose(35.0), quality=70.0)  # no descriptor -> NULL row
        s.add("p00002", pose(90.0), quality=70.0, appearance=hist(1))
    with store.VectorStore(path) as s:
        got = s.appearances_for("p00001")
        assert len(got) == 1, "the NULL row must be skipped, not decoded as zeros"
        assert np.allclose(got[0], np.asarray(hist(0), dtype=np.float32))
        # The cosine scan answers from faces alone: descriptors on p00002 do
        # not move a probe that is geometrically p00001's.
        hit = s.search(pose(0.0))
        assert hit.key == "p00001"
        assert hit.cosine == pytest.approx(1.0, abs=1e-6)


#: The vectors schema exactly as it shipped BEFORE the appearance column —
#: what every gallery and staff file on disk looked like on 2026-08-05.
_OLD_SCHEMA = """
CREATE TABLE vectors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL,
    vec        BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    quality    REAL,
    sub_canon  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_vectors_key ON vectors(key);
CREATE TABLE cannot_link (a TEXT NOT NULL, b TEXT NOT NULL, PRIMARY KEY (a, b));
CREATE TABLE manual (key TEXT PRIMARY KEY, note TEXT, created_at TEXT NOT NULL);
CREATE TABLE meta (k TEXT PRIMARY KEY, v INTEGER NOT NULL);
"""


def test_an_old_gallery_file_is_altered_in_and_keeps_working(tmp_path):
    """Opening a pre-appearance file ADDs the column; its old rows mean 'absent'.

    Galleries now outlive their runs (24 h retention) and staff stores persist
    for years, so files created before this feature WILL be reopened by the
    new code.  The migration must be invisible: old rows read as
    descriptor-absent (never as zero histograms, which would veto every
    enrolment against them), and new templates written into the old file carry
    descriptors normally.
    """
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    vec = np.zeros(DIM, dtype=np.float32)
    vec[0] = 1.0
    conn.execute(
        "INSERT INTO vectors (key, vec, dim, quality, sub_canon, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("p00001", vec.tobytes(), DIM, 70.0, 0, "2026-08-05T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    with store.VectorStore(path) as s:
        cols = {row[1] for row in s.conn.execute("PRAGMA table_info(vectors)")}
        assert "appearance" in cols, "opening an older file must ALTER the column in"
        hit = s.search(pose(0.0))
        assert hit.key == "p00001" and hit.cosine == pytest.approx(1.0, abs=1e-6)
        assert s.appearances_for("p00001") == [], "pre-migration rows are absent, not zero"
        s.add("p00001", pose(35.0), quality=70.0, appearance=hist(3))
    with store.VectorStore(path) as s:
        got = s.appearances_for("p00001")
        assert len(got) == 1
        assert np.allclose(got[0], np.asarray(hist(3), dtype=np.float32))


def test_merge_carries_appearance_rows_with_their_vectors(tmp_path):
    """A merge re-points rows wholesale, so descriptors ride with their templates."""
    with store.VectorStore(tmp_path / "s.db") as s:
        s.add("p00001", pose(0.0), quality=70.0, appearance=hist(0))
        s.add("p00002", pose(90.0), quality=70.0, appearance=hist(1))
        assert s.merge("p00001", "p00002") is True
        assert len(s.appearances_for("p00001")) == 2
        assert s.appearances_for("p00002") == []


# ------------------------------------------------------- the enrolment veto


def test_enrol_veto_fires_on_clash_and_the_verdict_is_untouched(client, tmp_path):
    """A clashing torso refuses the template; isNew/personKey/cosine stand.

    The face match here (0.433) clears every M1 gate — without appearance
    this exact sighting IS enrolled (pinned below in the agreement test) —
    so the refusal can only be the veto.  The verdict must read exactly as
    the face alone decided: same key, isNew false, cosine 0.433.
    """
    first = _match(client, "run-veto", pose(0.0), appearance=hist(0))
    assert first["isNew"] is True

    out = _match(client, "run-veto", at(ENROLLABLE), appearance=hist(1))
    assert out["isNew"] is False, "the verdict belongs to the face alone"
    assert out["personKey"] == first["personKey"]
    assert out["cosine"] == pytest.approx(ENROLLABLE, abs=1e-5)
    assert out["appearanceSim"] == pytest.approx(0.0, abs=1e-6)
    assert out["appearanceVetoed"] is True
    assert out["templateAdded"] is False
    assert out["templateN"] == 1, "nothing stored"

    s = store.open_store(gallery.db_path(tmp_path, "run-veto"))
    with s.reading():
        assert s.count_for(first["personKey"]) == 1
        assert len(s.appearances_for(first["personKey"])) == 1, (
            "the vetoed sighting's descriptor must not be stored either"
        )


def test_enrol_proceeds_on_agreement_and_stores_the_descriptor(client, tmp_path):
    """Matching clothes: the template is kept, descriptor and all."""
    first = _match(client, "run-agree", pose(0.0), appearance=hist(0))
    out = _match(client, "run-agree", at(ENROLLABLE), appearance=hist(0))
    assert out["appearanceSim"] == pytest.approx(1.0, abs=1e-6)
    assert out["appearanceVetoed"] is False
    assert out["templateAdded"] is True
    assert out["templateN"] == 2

    s = store.open_store(gallery.db_path(tmp_path, "run-agree"))
    with s.reading():
        assert len(s.appearances_for(first["personKey"])) == 2


def test_exactly_at_the_clash_threshold_is_not_a_clash(client):
    """'Below = clash': intersection exactly 0.50 against a 0.50 knob enrols."""
    _match(client, "run-edge", pose(0.0), appearance=hist(0, 1))
    out = _match(client, "run-edge", at(ENROLLABLE), appearance=hist(1, 2))
    assert out["appearanceSim"] == pytest.approx(0.5, abs=1e-6)
    assert out["appearanceVetoed"] is False
    assert out["templateAdded"] is True


def test_absent_descriptors_never_veto(client):
    """Absence on EITHER side disables the veto — absent is not zero.

    Old runs, faces without a containing person box, and sub-24 px crops all
    arrive descriptor-less; punishing them would veto enrolments exactly
    where evidence is thinnest, and under-counting is already the dominant
    failure mode.
    """
    # Probe carries no descriptor; identity has one stored.
    _match(client, "run-abs1", pose(0.0), appearance=hist(0))
    out = _match(client, "run-abs1", at(ENROLLABLE))
    assert out["appearanceSim"] is None
    assert out["appearanceVetoed"] is False
    assert out["templateAdded"] is True

    # Identity has none stored (pre-feature rows); probe carries one.
    _match(client, "run-abs2", pose(0.0))
    out = _match(client, "run-abs2", at(ENROLLABLE), appearance=hist(1))
    assert out["appearanceSim"] is None
    assert out["appearanceVetoed"] is False
    assert out["templateAdded"] is True


def test_knob_zero_disables_the_veto_but_keeps_the_visibility(client, monkeypatch):
    """HECO_MATCH_APPEARANCE_CLASH=0 is the off switch; appearanceSim still reports."""
    monkeypatch.setenv("HECO_MATCH_APPEARANCE_CLASH", "0")
    _match(client, "run-off", pose(0.0), appearance=hist(0))
    out = _match(client, "run-off", at(ENROLLABLE), appearance=hist(1))
    assert out["appearanceSim"] == pytest.approx(0.0, abs=1e-6), "visibility is unconditional"
    assert out["appearanceVetoed"] is False
    assert out["templateAdded"] is True


# ------------------------------------------- the verdict is never appearance's


def test_clothing_never_rescues_a_face_miss(client):
    """The two-white-shirts rule: a sub-threshold face stays NEW despite a
    perfectly matching torso.

    The measured 0.377 impostor pair — two different men, both in light
    shirts — is exactly the pair a rescue would merge.  So a probe at cosine
    threshold−0.01 with intersection 1.0 against the stored descriptor must
    still mint a new person, with appearanceSim null (a mint has nothing
    stored to compare against).
    """
    first = _match(client, "run-rescue", pose(0.0), appearance=hist(0))
    below = _match(client, "run-rescue", at(THRESHOLD - 0.01), appearance=hist(0))
    assert below["isNew"] is True, "clothing must never rescue a face miss"
    assert below["personKey"] != first["personKey"]
    assert below["cosine"] == pytest.approx(THRESHOLD - 0.01, abs=1e-5)
    assert below["appearanceSim"] is None
    assert below["appearanceVetoed"] is False


def test_verdicts_match_a_descriptorless_run_frame_for_frame(client):
    """The same faces with and without (clashing) descriptors verdict identically.

    The walk deliberately contains no enrolling sighting — a mint, a
    dead-band re-sighting (matched, never learned from), and a sub-threshold
    second mint — so both runs hold identical galleries throughout and the
    comparison is exact.  (Where enrolment DOES differ, later cosines
    legitimately differ too; the per-call verdict invariance of that case is
    pinned in test_enrol_veto_fires_on_clash_and_the_verdict_is_untouched.)
    """
    walk = [pose(0.0), at(THRESHOLD + 0.001), at(THRESHOLD - 0.01)]
    plain = [_match(client, "run-plain", v) for v in walk]
    dressed = [
        _match(client, "run-dressed", v, appearance=hist(i)) for i, v in enumerate(walk)
    ]
    for p, d in zip(plain, dressed, strict=True):
        assert p["isNew"] == d["isNew"]
        assert p["personKey"] == d["personKey"]
        assert p["cosine"] == pytest.approx(d["cosine"], abs=1e-6)
    assert dressed[1]["appearanceVetoed"] is False, (
        "the dead band is _should_enrol refusing, not the appearance veto"
    )


# ----------------------------------------------------------- the wire surface


def test_wrong_length_appearance_is_a_readable_422(client):
    """47 or 2 floats is a caller bug, named as such before it can veto anything."""
    for bad in ([0.5, 0.5], [1.0 / 47] * 47):
        res = client.post(
            "/match",
            json={"runId": "run-bad", "embedding": _json(pose(0.0)), "appearance": bad},
        )
        assert res.status_code == 422, res.text
        assert "48" in res.text, "the error must name the contract"


def test_appearance_sim_is_null_for_staff_hits_and_first_mints(client, tmp_path):
    """Staff flows never touch appearance; a mint has nothing to compare against."""
    minted = _match(client, "run-null", pose(0.0), appearance=hist(0))
    assert minted["isNew"] is True
    assert minted["appearanceSim"] is None
    assert minted["appearanceVetoed"] is False

    res = client.post(
        "/staff/enrol",
        json={
            "siteId": "site-app",
            "staffId": "st-1",
            "samples": [{"embedding": _json(pose(0.0)), "quality": 80.0}],
        },
    )
    assert res.status_code == 200, res.text
    hit = _match(client, "run-null2", pose(0.0), appearance=hist(0), site_id="site-app")
    assert hit["isStaff"] is True and hit["staffId"] == "st-1"
    assert hit["appearanceSim"] is None
    assert hit["appearanceVetoed"] is False

    s = store.open_store(staff.db_path(tmp_path, "site-app"))
    with s.reading():
        assert s.appearances_for("st-1") == [], (
            "the staff store must never gain appearance rows"
        )


def test_health_reports_the_clash_knob_and_empty_means_unset(client, monkeypatch):
    """The bench must read back which veto policy produced a count."""
    body = client.get("/health").json()
    assert body["appearanceClash"] == pytest.approx(config.DEFAULT_APPEARANCE_CLASH)

    monkeypatch.setenv("HECO_MATCH_APPEARANCE_CLASH", "")
    assert config.appearance_clash() == config.DEFAULT_APPEARANCE_CLASH, (
        "compose renders ${VAR-} as empty string: empty means unset"
    )
    monkeypatch.setenv("HECO_MATCH_APPEARANCE_CLASH", "0.7")
    assert config.appearance_clash() == 0.7
