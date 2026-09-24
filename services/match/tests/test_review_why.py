"""The review queue's side evidence (v4, 2026-09-24): sex, age, stature.

THE MEASUREMENT.  Run f0bfc5 — a Punjab wedding hall from an overview camera,
74 guests counted — put 500 pairs in the review queue.  Its top five, read by
eye: #1 a black beard against a white one, #3 a man in a blue shirt against an
elderly woman in glasses, #5 a girl in a yellow top against a woman in a green
dress.  Every one a question the pipeline held the evidence to not ask.

So the gallery now records, per template, what the embed service's attribute
head read (sex + its probability, age) and, per sighting, the person box; the
review queue reports both sides' readings in a per-pair ``why`` and SETS ASIDE
pairs the evidence says cannot be one person, counting them in ``excluded``.

What these tests pin, and why each matters:

* an exclusion needs its evidence on BOTH sides — one confident sex reading
  and one unsure (or absent) one is one opinion, not a disagreement;
* absent is not zero, everywhere: a legacy gallery with none of the new
  columns answers null in every ``why`` field and zero in every ``excluded``;
* the stature fit is replay.py's algorithm (the scratchpad reference that
  reproduced the run's h = 0.602 * y_bottom + 300 px line);
* exclusion happens after the band test and BEFORE the cap, writes nothing
  (no cannot_link row — that stays a human's or co-presence's word), and
  every knob's zero is its off switch.

Geometry: a HUB identity at e0 and SPOKES at cosine 0.30 from it along
distinct orthogonal axes.  Every (hub, spoke) pair sits in the review band;
every (spoke, spoke) pair sits at 0.09, below the 0.15 floor.  One run can
therefore hold several in-band pairs whose attributes differ, without any
two spokes ever being asked about each other.
"""

import math
import sqlite3

import numpy as np
import pytest
from app import config, gallery, main, store
from app.gallery import identity_age, identity_gender, stature_fit, stature_ratios
from fastapi.testclient import TestClient

DIM = 128
FRAME_H = 2160
#: Run f0bfc5's measured perspective line — used to synthesise standing boxes
#: whose "true" stature ratio is known exactly.
SLOPE, INTERCEPT = 0.602, 300.2


def _e(i: int) -> np.ndarray:
    """The i-th standard basis vector in 128-d (an orthonormal anchor)."""
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def _json(v: np.ndarray) -> list[float]:
    """A vector as the JSON list the /match endpoint takes."""
    return [float(x) for x in v]


def hub() -> np.ndarray:
    """The identity every in-band pair is measured against."""
    return _e(0)


def spoke(i: int, cosine: float = 0.30) -> np.ndarray:
    """A probe at an EXACT cosine from the hub along its own orthogonal axis."""
    return cosine * _e(0) + math.sqrt(1.0 - cosine * cosine) * _e(i)


def excluded(**counts) -> dict:
    """The reply's ``excluded`` block: every signal at zero but those named."""
    return {"gender": 0, "age": 0, "stature": 0, "clothes": 0, "head": 0, "beard": 0, **counts}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with gallery data redirected to a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    store.close_all_stores()
    with TestClient(main.app) as c:
        yield c
    store.close_all_stores()


def attrs(gender: str, p: float, age: float) -> dict:
    """One face's attribute reading as the wire carries it."""
    return {"gender": gender, "genderP": p, "age": age}


def standing_box(y_bottom: float, ratio: float = 1.0) -> dict:
    """A standing person box on f0bfc5's line, ``ratio`` times an adult's height."""
    h = ratio * (SLOPE * y_bottom + INTERCEPT)
    return {"h": h, "w": h / 3.0, "yBottom": y_bottom, "frameH": FRAME_H}


def match(client, run, v, attributes=None, body=None, feat_norm=None, exclude=None):
    """POST /match with whichever of the v4 side fields the test carries."""
    payload = {"runId": run, "embedding": _json(v), "quality": 90.0}
    if attributes is not None:
        payload["attributes"] = attributes
    if body is not None:
        payload["body"] = body
    if feat_norm is not None:
        payload["featNorm"] = feat_norm
    if exclude:
        payload["excludeKeys"] = exclude
    res = client.post("/match", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


def review(client, run="r", **kw):
    """Ask one run's gallery which identity pairs a human should look at."""
    res = client.post("/review/duplicates", json={"runId": run, **kw})
    assert res.status_code == 200, res.text
    return res.json()


def pairs_of(report) -> dict:
    """The queue keyed by unordered pair, for order-free lookups."""
    return {frozenset((p["a"], p["b"])): p for p in report["pairs"]}


def opened(tmp_path, run):
    """The cached store behind a run, for reading what the API wrote."""
    return store.open_store(gallery.db_path(tmp_path, run))


def settle(tmp_path, run, key, vec, gender, p, age, n=3):
    """Give ``key`` ``n`` more templates of the SAME view, each reading (gender, p, age).

    An identity's sex is the template mean shrunk by two pseudo-votes, so one
    template is never confident (one M 1.0 reads 0.67); a settled identity
    needs several agreeing views.  The vector is repeated so the pair's face
    cosine — the band test — does not move.
    """
    s = opened(tmp_path, run)
    with s.transaction():
        for _ in range(n):
            s.add(key, vec, quality=70.0, attributes=attrs(gender, p, age))


#: Four unanimous 0.97 reads: (4 * 0.97 + 1) / 6 = 0.8133, over the 0.8 bar.
SURE_P = 0.97
SURE_N = 3  # ...three added to the one the /match call enrolled
SURE_IDENTITY_P = (4 * SURE_P + 1.0) / 6.0


@pytest.fixture()
def ticking(monkeypatch):
    """A store clock that moves one second per write.

    The stature reader counts an identity's standing boxes by distinct write
    SECONDS, so sixty /match calls inside one real second are one moment.
    Tests that want sixty moments tick the clock.
    """
    import itertools
    from datetime import UTC, datetime, timedelta

    t0 = datetime(2026, 9, 24, 21, 0, 0, tzinfo=UTC)
    counter = itertools.count()
    monkeypatch.setattr(
        store, "_now", lambda: (t0 + timedelta(seconds=next(counter))).isoformat()
    )


# ------------------------------------------------------------ the store (M1)

#: The vectors schema exactly as galleries on disk looked on 2026-09-23 —
#: the appearance column present (48-d blobs), none of the attribute columns,
#: no body_sightings table.  Run f0bfc5's own gallery has this shape.
_V2_SCHEMA = """
CREATE TABLE vectors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL,
    vec        BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    quality    REAL,
    sub_canon  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    appearance BLOB
);
CREATE INDEX idx_vectors_key ON vectors(key);
CREATE TABLE cannot_link (a TEXT NOT NULL, b TEXT NOT NULL, PRIMARY KEY (a, b));
CREATE TABLE manual (key TEXT PRIMARY KEY, note TEXT, created_at TEXT NOT NULL);
CREATE TABLE meta (k TEXT PRIMARY KEY, v INTEGER NOT NULL);
CREATE TABLE store_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


def _write_v2_gallery(path, keys_and_vecs):
    """A gallery file as the previous release wrote it: 48-d torsos, no attributes."""
    conn = sqlite3.connect(path)
    conn.executescript(_V2_SCHEMA)
    conn.execute(
        "INSERT INTO store_meta (k, v) VALUES ('embedderId', ?)", (store.EMBEDDER_ID,)
    )
    torso = (np.ones(48, dtype=np.float32) / 48).tobytes()
    for key, vec in keys_and_vecs:
        conn.execute(
            "INSERT INTO vectors (key, vec, dim, quality, sub_canon, created_at, appearance)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, np.asarray(vec, dtype=np.float32).tobytes(), DIM, 70.0, 0,
             "2026-09-23T00:00:00+00:00", torso),
        )
    conn.commit()
    conn.close()


def test_a_v2_gallery_opens_migrates_and_reads_absent_everywhere(tmp_path):
    """The previous release's file: no new columns, 48-d blobs, no body table.

    Galleries outlive their runs by 24 h, so the day this ships every store on
    disk has this shape.  Opening one must ALTER the columns in, keep the
    48-d descriptors readable, and answer 'not measured' for everything new.
    """
    path = tmp_path / "gallery-v2.db"
    _write_v2_gallery(path, [("p00001", hub()), ("p00002", spoke(1))])

    with store.VectorStore(path) as s:
        cols = {row[1] for row in s.conn.execute("PRAGMA table_info(vectors)")}
        assert {"gender", "gender_p", "age", "feat_norm"} <= cols
        tables = {r[0] for r in s.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "body_sightings" in tables
        assert s.attributes_for("p00001") == [(None, None, None)]
        assert s.feat_norms_for("p00001") == []
        assert s.body_sightings() == []
        assert [a.size for a in s.appearances_for("p00001")] == [48]
        assert s.search(hub()).key == "p00001"
        # ...and a v3 template written into the old file coexists with the v2 rows.
        s.add("p00001", spoke(3), quality=70.0, appearance=[1.0 / 64] * 64,
              attributes=attrs("M", 0.9, 40.0), feat_norm=22.5)
    with store.VectorStore(path) as s:
        assert sorted(a.size for a in s.appearances_for("p00001")) == [48, 64]
        assert s.attributes_for("p00001") == [(None, None, None), ("M", 0.9, 40.0)]
        assert s.feat_norms_for("p00001") == [22.5]


def test_attributes_ride_with_templates_and_bodies_log_every_call(client, tmp_path):
    """A template stores its face's reading; the body log grows on EVERY guest call.

    The second hub sighting matches at 1.0 — above the near-duplicate ceiling,
    so no template is written — yet its body must still be logged: the stature
    fit wants every standing box in the run, not the five a cap keeps.
    """
    first = match(client, "r", hub(), attrs("M", 0.95, 41.0), standing_box(1500.0), 21.0)
    again = match(client, "r", hub(), attrs("M", 0.6, 45.0), standing_box(1600.0), 19.0)
    assert again["personKey"] == first["personKey"] and again["templateAdded"] is False

    s = opened(tmp_path, "r")
    with s.reading():
        assert s.attributes_for(first["personKey"]) == [("M", 0.95, 41.0)]
        assert s.feat_norms_for(first["personKey"]) == [21.0]
        rows = s.body_sightings()
    assert [(r.key, r.y_bottom, r.frame_h, r.face_w) for r in rows] == [
        (first["personKey"], 1500.0, FRAME_H, 90.0), (first["personKey"], 1600.0, FRAME_H, 90.0)
    ], "the face width (the call's quality) rides with the box"
    assert all(r.created_at for r in rows)
    assert (first["bodyId"], again["bodyId"]) == (1, 2), "the body rowid rides the reply"


def test_merge_rekeys_body_sightings_and_keeps_attribute_columns(tmp_path):
    """The survivor inherits the dropped key's boxes and readings — all of them.

    A sighting log naming a key that no longer exists would leave the merged
    guest's stature read off half its data.
    """
    with store.VectorStore(tmp_path / "s.db") as s:
        s.add("p00001", hub(), attributes=attrs("F", 0.9, 30.0), feat_norm=20.0)
        s.add("p00002", spoke(1), attributes=attrs("F", 0.8, 34.0), feat_norm=18.0)
        s.add_body_sighting("p00001", 1200.0, 400.0, 1500.0, FRAME_H)
        s.add_body_sighting("p00002", 1210.0, 400.0, 1510.0, FRAME_H)
        assert s.merge("p00001", "p00002") is True
        assert s.attributes_for("p00001") == [("F", 0.9, 30.0), ("F", 0.8, 34.0)]
        assert s.feat_norms_for("p00001") == [20.0, 18.0]
        assert [r[0] for r in s.body_sightings()] == ["p00001", "p00001"]
        assert s.body_sightings_count("p00002") == 0


def test_remove_takes_the_body_log_with_the_person(tmp_path):
    """mark-staff lifts a person out; their boxes must not keep feeding the fit."""
    with store.VectorStore(tmp_path / "s.db") as s:
        s.add("p00001", hub())
        s.add_body_sighting("p00001", 1200.0, 400.0, 1500.0, FRAME_H)
        s.remove("p00001")
        assert s.body_sightings() == []


def test_a_null_or_partial_body_is_silently_not_logged(client, tmp_path):
    """No box, or a box with no height, is 'not measured' — never a row, never a 500."""
    r = match(client, "r", hub(), body=None)
    res = client.post("/match", json={
        "runId": "r", "embedding": _json(hub()),
        "body": {"h": 0, "w": 100, "yBottom": 1, "frameH": FRAME_H},
    })
    assert res.status_code == 422, "a non-positive height is a wire bug, named at the boundary"
    s = opened(tmp_path, "r")
    with s.reading():
        assert s.body_sightings() == [] and s.count_for(r["personKey"]) == 1


# ---------------------------------------------------- per-identity aggregates


def test_identity_gender_is_the_shrunk_template_mean_and_unsure_views_pull_it_down():
    """A frontal 0.97 M and a head-down 0.55 F read M at 0.605 — under the 0.8 bar.

    The sure view is not outvoted; the unsure one still costs enough
    confidence that the review bar will not trust the identity's sex.  Two
    pseudo-votes of 0.5 shrink every mean: (0.97 + 0.45 + 1) / 4.
    """
    assert identity_gender([("M", 0.97, 40.0), ("F", 0.55, 38.0)]) == (
        "M", pytest.approx(0.605)
    )
    assert identity_gender([("F", 0.9, 30.0), ("F", 0.8, 31.0)]) == ("F", pytest.approx(0.675))
    assert identity_gender([("M", 0.5, 30.0)]) == ("M", pytest.approx(0.5))
    assert identity_gender([("M", 0.4, 30.0)]) == ("F", pytest.approx(1.0 - 1.4 / 3.0)), (
        "genderP is the probability OF THE REPORTED sex: 0.4 male is 0.6 female"
    )


def test_one_confident_template_is_not_a_confident_identity():
    """THE p00048 CASE: a girl with eleven M >= 0.8 reads and one F 0.92.

    A duplicate is minted on the view that failed to match — the view the
    attribute head flips on — so the identity's confidence must grow with
    agreeing views, not arrive whole with the first.  One M 1.0 reads 0.67,
    three unanimous 0.80 (the bar), five 0.857; and one dissenting confident
    view among five (four M 0.97, one F 0.92) drops the identity to 0.727,
    under the bar, so the pair is asked about rather than set aside.
    """
    assert identity_gender([("M", 1.0, 30.0)]) == ("M", pytest.approx(2.0 / 3.0))
    assert identity_gender([("M", 1.0, 30.0)] * 3) == ("M", pytest.approx(0.8))
    assert identity_gender([("M", 1.0, 30.0)] * 5) == ("M", pytest.approx(6.0 / 7.0))
    mixed = [("M", 0.97, 30.0)] * 4 + [("F", 0.92, 30.0)]
    g, p = identity_gender(mixed)
    assert g == "M" and p == pytest.approx((4 * 0.97 + 0.08 + 1.0) / 7.0) and p < 0.8
    assert identity_gender([("M", SURE_P, 30.0)] * 4)[1] == pytest.approx(SURE_IDENTITY_P)
    assert SURE_IDENTITY_P >= config.DEFAULT_REVIEW_GENDER_MIN_P


def test_identity_gender_and_age_are_none_when_nothing_was_read():
    """Absent is not a default sex and not age 0."""
    assert identity_gender([]) == (None, None)
    assert identity_gender([(None, None, None), (None, None, None)]) == (None, None)
    assert identity_gender([("M", None, 30.0)]) == (None, None), (
        "a sex without a probability is not a vote"
    )
    assert identity_age([]) is None
    assert identity_age([(None, None, None)]) is None


def test_identity_age_is_the_median_so_one_wild_view_cannot_age_a_child():
    """One 31-year reading among two under-ten ones leaves the child a child."""
    assert identity_age([("F", 0.9, 9.0), ("F", 0.9, 10.0), ("F", 0.6, 31.0)]) == 10.0
    assert identity_age([(None, None, 8.0), (None, None, 12.0)]) == 10.0


# ------------------------------------------------------------ the stature fit


def _boxes(key, n, ratio, y0=800.0, y1=2000.0, aspect=3.0):
    """``n`` standing boxes for ``key`` spread down f0bfc5's perspective line."""
    out = []
    for i in range(n):
        yb = y0 + (y1 - y0) * i / max(1, n - 1)
        h = ratio * (SLOPE * yb + INTERCEPT)
        out.append((key, h, h / aspect, yb, FRAME_H))
    return out


def test_stature_fit_recovers_the_line_and_the_trim_discards_seated_and_junk():
    """replay.py's algorithm: least squares, then three passes trimming 0.6..1.5.

    Exact adults alone give the exact line.  Adding seated bodies (0.4 of the
    predicted height — the aspect test passes when a chair hides the legs)
    and tracker junk (2.0) must give the SAME line after trimming, because
    every trimmed point is gone from the refit entirely.
    """
    adults = _boxes("a", 200, 1.0)
    a, b = stature_fit(gallery.standing_sightings(adults))
    assert (a, b) == (pytest.approx(SLOPE, abs=1e-9), pytest.approx(INTERCEPT, abs=1e-6))

    polluted = adults + _boxes("seated", 30, 0.4) + _boxes("junk", 20, 2.0)
    a2, b2 = stature_fit(gallery.standing_sightings(polluted))
    assert (a2, b2) == (pytest.approx(SLOPE, abs=1e-9), pytest.approx(INTERCEPT, abs=1e-6))


def test_stature_ratios_place_a_child_under_the_adults_and_need_min_n():
    """f0bfc5's p00009 read 0.73 against adults at 1.03-1.04; here 0.72 vs 1.0.

    The child stays in the fit (0.72 is inside the trim band, as in replay.py),
    so the line bends a hair towards them — hence the tolerance.  An identity
    with fewer standing boxes than ``min_n`` is absent, not 0.
    """
    sightings = (
        _boxes("adult1", 60, 1.0) + _boxes("adult2", 40, 1.0)
        + _boxes("child", 10, 0.72) + _boxes("glimpsed", 7, 1.0)
    )
    ratios = stature_ratios(sightings, min_n=8)
    assert ratios["adult1"] == pytest.approx(1.0, abs=0.03)
    assert ratios["adult2"] == pytest.approx(1.0, abs=0.03)
    assert ratios["child"] == pytest.approx(0.72, abs=0.03)
    assert "glimpsed" not in ratios, "seven boxes is under the bar: not measured"
    assert "glimpsed" in stature_ratios(sightings, min_n=7)


def test_standing_filter_is_replay_py_geometry():
    """Squat boxes, boxes cut by the frame bottom, boxes touching the top: out."""
    ok = ("k", 900.0, 300.0, 1500.0, FRAME_H)
    squat = ("k", 500.0, 300.0, 1500.0, FRAME_H)            # h/w < 2: seated/bending
    cut = ("k", 900.0, 300.0, FRAME_H - 30.0, FRAME_H)      # feet under the frame edge
    top = ("k", 1497.0, 300.0, 1500.0, FRAME_H)             # top edge at y=3: cropped
    assert gallery.standing_sightings([ok, squat, cut, top]) == [("k", 1500.0, 900.0, None)]


def test_a_head_and_shoulders_box_is_not_a_standing_body():
    """THE p00052 CASE: 205x470 px boxes on a 112 px face, aspect 2.3.

    An occlusion (a table, a pillar) hands the detector a waist-up box whose
    aspect passes the standing test; fifteen of them in a row read her at
    0.59 of adult height against 1.05 from her full boxes.  A standing body
    is at least six face widths tall (adults 10-12, the smallest child 5.7
    at minimum, 9.9 at p10); the waist-up boxes read 2.9-4.3 and are out.
    A row from before the column (face_w None) is not filtered.
    """
    B = store.BodySighting
    waist_up = B("p00052", 470.0, 205.0, 870.0, FRAME_H, 112.0, "2026-09-24T21:10:05+00:00")
    full = B("p00052", 1232.0, 400.0, 1400.0, FRAME_H, 112.0, "2026-09-24T21:11:40+00:00")
    legacy = B("p00052", 470.0, 205.0, 870.0, FRAME_H, None, None)
    just_under = B("k", 671.0, 220.0, 1400.0, FRAME_H, 112.0, "2026-09-24T21:11:41+00:00")
    just_over = B("k", 672.0, 220.0, 1400.0, FRAME_H, 112.0, "2026-09-24T21:11:42+00:00")
    assert gallery.standing_sightings([waist_up, full, legacy, just_under, just_over]) == [
        ("p00052", 1400.0, 1232.0, "2026-09-24T21:11:40"),
        ("p00052", 870.0, 470.0, None),
        ("k", 1400.0, 672.0, "2026-09-24T21:11:42"),
    ]
    assert gallery._STANDING_MIN_FACE_WIDTHS == 6.0


def test_min_n_boxes_must_span_two_moments():
    """Twelve boxes from ONE second are one measurement; from two seconds, twelve.

    The eight-box floor gave no independence when eight boxes could be eight
    consecutive frames of one occlusion (15 frames is one second at 15 fps).
    The bar is two distinct write seconds, not eight: on f0bfc5 the 53
    identities with eight standing boxes spanned a median of four seconds
    and only three spanned eight.  Untimed rows (a pure-function caller)
    count one each.
    """
    adults = _boxes("adult", 200, 1.0)  # enough that 24 child boxes barely bend the line
    same_second = [
        store.BodySighting("burst", h, w, yb, fh, None, f"2026-09-24T21:00:00.{i:06d}+00:00")
        for i, (_, h, w, yb, fh) in enumerate(_boxes("burst", 12, 0.7))
    ]
    spread = [
        store.BodySighting("spread", h, w, yb, fh, None, f"2026-09-24T21:00:{i:02d}+00:00")
        for i, (_, h, w, yb, fh) in enumerate(_boxes("spread", 12, 0.7))
    ]
    two_seconds = [
        store.BodySighting("two", h, w, yb, fh, None, f"2026-09-24T21:01:0{i % 2}.{i:06d}+00:00")
        for i, (_, h, w, yb, fh) in enumerate(_boxes("two", 12, 0.7))
    ]
    ratios = stature_ratios(adults + same_second + spread + two_seconds, min_n=8)
    assert "burst" not in ratios, "twelve boxes, one moment: not measured"
    assert ratios["spread"] == pytest.approx(0.7, abs=0.06)
    assert ratios["two"] == pytest.approx(0.7, abs=0.06), "two seconds is the bar"
    assert gallery._STATURE_MIN_MOMENTS == 2
    assert ratios["adult"] == pytest.approx(1.0, abs=0.06), "untimed rows count one each"


def test_too_few_standing_boxes_means_no_fit_and_no_stature():
    """Under 50 standing boxes there is no perspective line, so nothing is measured."""
    assert stature_fit(gallery.standing_sightings(_boxes("a", 49, 1.0))) is None
    assert stature_ratios(_boxes("a", 49, 1.0), min_n=1) == {}


# ---------------------------------------------------- the review queue (M2)


def test_gender_exclusion_needs_both_sides_confident_and_a_null_never_excludes(client, tmp_path):
    """Three spokes against a settled-male hub: a settled F is set aside, a single
    F@0.6 read and an unmeasured one are still asked — readings on the row.

    "Settled" is four agreeing 0.97 views (0.813 after the shrink); a lone
    0.95 view would read 0.65 and never be confident on its own.
    """
    kh = match(client, "r", hub(), attrs("M", SURE_P, 41.0))["personKey"]
    settle(tmp_path, "r", kh, hub(), "M", SURE_P, 41.0)
    k_sure = match(client, "r", spoke(1), attrs("F", SURE_P, 43.0))["personKey"]
    settle(tmp_path, "r", k_sure, spoke(1), "F", SURE_P, 43.0)
    k_unsure = match(client, "r", spoke(2), attrs("F", 0.6, 39.0))["personKey"]
    k_none = match(client, "r", spoke(3))["personKey"]

    got = review(client)
    by = pairs_of(got)
    assert frozenset((kh, k_sure)) not in by, "both confident, different: set aside"
    assert got["excluded"] == excluded(gender=1)
    assert got["considered"] == 6, "still examined — the count beyond the queue is not hidden"

    unsure = by[frozenset((kh, k_unsure))]["why"]
    a_is_hub = unsure["gender"]["a"] == "M" and by[frozenset((kh, k_unsure))]["a"] == kh
    assert a_is_hub
    assert unsure["gender"] == {
        "a": "M", "b": "F", "pA": pytest.approx(SURE_IDENTITY_P),
        "pB": pytest.approx(1.0 - 1.4 / 3.0),
    }
    assert unsure["age"] == {"a": 41.0, "b": 39.0}
    assert unsure["stature"]["a"] is None and unsure["stature"]["b"] is None
    assert unsure["stature"]["adultM"] == pytest.approx(1.75)

    none = by[frozenset((kh, k_none))]["why"]
    assert none["gender"] == {
        "a": "M", "b": None, "pA": pytest.approx(SURE_IDENTITY_P), "pB": None,
    }
    assert none["age"] == {"a": 41.0, "b": None}


def test_a_lone_confident_read_on_either_side_still_asks(client, tmp_path):
    """A settled man against ONE F 0.95 view: asked, not set aside.

    The lone view is exactly the head-down or turned sighting a duplicate is
    minted on; it reads F at 0.65 after the shrink and cannot vote the pair
    away.  The row still says what was read.
    """
    kh = match(client, "r", hub(), attrs("M", SURE_P, 41.0))["personKey"]
    settle(tmp_path, "r", kh, hub(), "M", SURE_P, 41.0)
    ks = match(client, "r", spoke(1), attrs("F", 0.95, 43.0))["personKey"]
    got = review(client)
    assert got["excluded"] == excluded()
    (p,) = got["pairs"]
    assert {p["a"], p["b"]} == {kh, ks}
    lone = p["why"]["gender"]["pB"] if p["b"] == ks else p["why"]["gender"]["pA"]
    assert lone == pytest.approx(1.0 - 1.05 / 3.0), "one F 0.95 view: F at 0.65"


def test_age_exclusion_is_child_against_adult_and_the_dead_zone_still_asks(client):
    """8 vs 35 is set aside; 15 vs 35 straddles the 12..20 gap and is asked."""
    kh = match(client, "r", hub(), attrs("F", 0.7, 35.0))["personKey"]
    k_child = match(client, "r", spoke(1), attrs("F", 0.7, 8.0))["personKey"]
    k_teen = match(client, "r", spoke(2), attrs("F", 0.7, 15.0))["personKey"]

    got = review(client)
    by = pairs_of(got)
    assert frozenset((kh, k_child)) not in by
    assert frozenset((kh, k_teen)) in by
    assert got["excluded"] == excluded(age=1)


def test_stature_exclusion_sets_aside_a_child_sized_body_against_an_adult(client, ticking):
    """Two identities in the face band; one walks the hall at 0.7 of adult height."""
    kh = match(client, "r", hub())["personKey"]
    ks = match(client, "r", spoke(1))["personKey"]
    for i in range(60):                       # the adult population: the fit
        match(client, "r", hub(), body=standing_box(800.0 + 20.0 * i, 1.0))
    for i in range(12):                       # the child: 12 standing boxes at 0.7
        match(client, "r", spoke(1), body=standing_box(900.0 + 80.0 * i, 0.7))

    got = review(client)
    assert got["pairs"] == []
    assert got["excluded"] == excluded(stature=1)

    # Turn the signal off: the pair is back, and its `why` carries the ratios
    # AND the metres a human reads (ratio × the 1.75 m anchor).
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HECO_REVIEW_STATURE_GAP", "0")
        got = review(client)
    (p,) = got["pairs"]
    st = p["why"]["stature"]
    ratio_hub, ratio_spoke = (st["a"], st["b"]) if p["a"] == kh else (st["b"], st["a"])
    # The child is 12 of the 72 boxes and sits inside the trim band, so the
    # line bends towards them (adults read ~1.05, the child ~0.74) — the same
    # behaviour replay.py showed on f0bfc5, where adults read 1.03-1.08.  The
    # GAP is what the exclusion tests, and it is untouched by that lean.
    assert ratio_hub == pytest.approx(1.0, abs=0.08)
    assert ratio_spoke == pytest.approx(0.7, abs=0.08)
    assert ratio_hub - ratio_spoke >= 0.2
    assert st["aM"] == pytest.approx(st["a"] * 1.75) and st["bM"] == pytest.approx(st["b"] * 1.75)
    assert got["excluded"]["stature"] == 0
    assert {p["a"], p["b"]} == {kh, ks}


def test_stature_is_null_under_min_n_and_null_never_excludes(client, ticking):
    """Seven standing boxes is not a measurement; a null side cannot be set aside."""
    kh = match(client, "r", hub())["personKey"]
    match(client, "r", spoke(1))
    for i in range(60):
        match(client, "r", hub(), body=standing_box(800.0 + 20.0 * i, 1.0))
    for i in range(7):
        match(client, "r", spoke(1), body=standing_box(900.0 + 80.0 * i, 0.5))

    got = review(client)
    (p,) = got["pairs"]
    st = p["why"]["stature"]
    hub_side, spoke_side = ("a", "b") if p["a"] == kh else ("b", "a")
    assert st[hub_side] == pytest.approx(1.0, abs=0.05)
    assert st[spoke_side] is None, "under HECO_REVIEW_STATURE_MIN_N: not measured"
    assert got["excluded"] == excluded()


def test_a_legacy_gallery_answers_null_everywhere_and_excludes_nothing(client, tmp_path):
    """A run from the previous release: every `why` field null, every count zero.

    Regression insurance for the day this ships: the retained galleries on
    disk hold no readings, and the queue must look exactly as it did — plus
    nulls — not empty, and not 500.
    """
    path = gallery.db_path(tmp_path, "legacy")
    _write_v2_gallery(path, [("p00001", hub()), ("p00002", spoke(1)), ("p00003", spoke(2))])

    got = review(client, run="legacy")
    assert got["excluded"] == excluded()
    assert len(got["pairs"]) == 2, "both in-band pairs are still asked about"
    for p in got["pairs"]:
        assert p["why"] == {
            "gender": {"a": None, "b": None, "pA": None, "pB": None},
            "age": {"a": None, "b": None},
            "stature": {
                "a": None, "b": None, "adultM": pytest.approx(1.75), "aM": None, "bM": None,
            },
            # v2 torsos are not clothing EVIDENCE (nA/nB count v3 reads), even
            # though the 48-d pair still ranks by its own intersection below.
            "clothes": {"selfA": None, "selfB": None, "cross": None, "nA": 0, "nB": 0},
            "head": {
                "a": None, "b": None, "sim": None, "selfA": None, "selfB": None,
                "nA": 0, "nB": 0, "wearA": None, "wearB": None,
            },
            "beard": {"a": None, "b": None, "nA": 0, "nB": 0},
            "light": {"a": None, "b": None, "shift": None, "held": []},
        }
        assert p["clothes"] == pytest.approx(1.0), "the 48-d torsos still compare"
    assert got["setAside"] == []


def test_a_v2_torso_against_a_v3_torso_is_unmeasured_not_zero(client):
    """Mixed descriptor generations in one queue read as clothes: null."""
    match(client, "r", hub(), body=None)  # no torso on the hub
    kh = match(client, "r", hub())["personKey"]
    client.post("/match", json={
        "runId": "r", "embedding": _json(hub()), "appearance": [1.0 / 48] * 48,
        "excludeKeys": [kh],
    })
    client.post("/match", json={
        "runId": "r", "embedding": _json(spoke(1)), "appearance": [1.0 / 64] * 64,
    })
    got = review(client)
    clothes = {frozenset((p["a"], p["b"])): p["clothes"] for p in got["pairs"]}
    assert clothes and all(c is None for c in clothes.values()), clothes


def test_exclusion_happens_after_the_band_and_before_the_cap_and_writes_nothing(client, tmp_path):
    """Three in-band pairs, one excluded, limit 1: returned 1, dropped 1, excluded 1.

    The excluded pair costs no slot and is not 'dropped' — dropped means
    'worth a look, no room this call', and a pair the evidence settled is
    neither.  And nothing is written: cannot_link is a human's or
    co-presence's word, so a later operator merge of the pair is still
    possible if the attribute model was wrong.
    """
    kh = match(client, "r", hub(), attrs("M", SURE_P, 40.0))["personKey"]
    settle(tmp_path, "r", kh, hub(), "M", SURE_P, 40.0)
    kx = match(client, "r", spoke(1), attrs("F", SURE_P, 40.0))["personKey"]   # set aside
    settle(tmp_path, "r", kx, spoke(1), "F", SURE_P, 40.0)
    match(client, "r", spoke(2), attrs("M", 0.95, 40.0))
    match(client, "r", spoke(3), attrs("M", 0.95, 40.0))

    got = review(client, limit=1)
    assert (got["returned"], got["dropped"], got["excluded"]["gender"]) == (1, 1, 1)
    assert got["considered"] == 6

    s = opened(tmp_path, "r")
    with s.reading():
        assert s.cannot_link(kh, kx) is False, "set aside is not a recorded fact"
    merged = client.post("/merge", json={"runId": "r", "keep": kh, "drop": kx}).json()
    assert merged["merged"] is True, "the operator's word still outranks the attribute model"


def test_every_exclusion_knob_at_zero_is_its_off_switch(client, monkeypatch, tmp_path):
    """Gender speaks first, then age; zero each in turn and the pair comes back."""
    kh = match(client, "r", hub(), attrs("M", SURE_P, 8.0))["personKey"]
    settle(tmp_path, "r", kh, hub(), "M", SURE_P, 8.0)
    ks = match(client, "r", spoke(1), attrs("F", SURE_P, 40.0))["personKey"]
    settle(tmp_path, "r", ks, spoke(1), "F", SURE_P, 40.0)
    assert review(client)["excluded"]["gender"] == 1, "gender speaks first"

    monkeypatch.setenv("HECO_REVIEW_GENDER_MIN_P", "0")
    got = review(client)
    assert got["excluded"] == excluded(age=1), "then age"

    monkeypatch.setenv("HECO_REVIEW_AGE_CHILD_MAX", "0")
    got = review(client)
    assert got["excluded"] == excluded()
    assert frozenset((kh, ks)) in pairs_of(got), "nothing left to set it aside"


def test_health_reports_the_review_policy_and_empty_means_unset(client, monkeypatch):
    """A queue of 12 under one policy and 500 under another must be tellable apart."""
    body = client.get("/health").json()
    assert body["reviewFloor"] == pytest.approx(0.15)
    assert body["reviewGenderMinP"] == pytest.approx(config.DEFAULT_REVIEW_GENDER_MIN_P)
    assert body["reviewAgeChildMax"] == pytest.approx(12.0)
    assert body["reviewAgeAdultMin"] == pytest.approx(20.0)
    assert body["reviewStatureGap"] == pytest.approx(0.2)
    assert body["reviewStatureMinN"] == 8
    assert body["adultM"] == pytest.approx(1.75), "the North Indian adult average, 5'9\""

    for name in (
        "HECO_REVIEW_GENDER_MIN_P", "HECO_REVIEW_AGE_CHILD_MAX", "HECO_REVIEW_AGE_ADULT_MIN",
        "HECO_REVIEW_STATURE_GAP", "HECO_REVIEW_STATURE_MIN_N", "HECO_STATURE_ADULT_M",
    ):
        monkeypatch.setenv(name, "")
    assert config.review_gender_min_p() == pytest.approx(0.8)
    assert config.review_stature_min_n() == 8
    assert config.adult_height_m() == pytest.approx(1.75)


def test_the_anchor_knob_is_the_one_compose_and_demo_up_name(client, monkeypatch):
    """HECO_STATURE_ADULT_M reaches /health and every why.stature.

    docker-compose.yml passes HECO_STATURE_ADULT_M into the container and
    demo-up.sh reads it back as THE knob; the first cut read a differently
    named variable here, so `status` printed 1.80 while every row stayed at
    1.75 — a knob that looks set and changes nothing.
    """
    monkeypatch.setenv("HECO_STATURE_ADULT_M", "1.80")
    assert client.get("/health").json()["adultM"] == pytest.approx(1.80)
    match(client, "r", hub())
    match(client, "r", spoke(1))
    (p,) = review(client)["pairs"]
    assert p["why"]["stature"]["adultM"] == pytest.approx(1.80)


def test_body_id_rides_the_reply_and_forget_retracts_the_row(client, tmp_path):
    """The same-frame guard's undo for the body log, beside the template's.

    A sighting's box is logged under the key /match resolved BEFORE the
    runner can see it was a different body; the re-ask then logs it again
    under the right key.  /template/forget takes the row back by ``bodyId``
    — with or without a ``templateId``, because a hit that enrolled nothing
    still logged a box — and names neither as the caller's bug.
    """
    first = match(client, "r", hub(), body=standing_box(1500.0))
    assert first["bodyId"] == 1
    none = match(client, "r", hub())
    assert none["bodyId"] is None, "no body on the call, no row, no id"

    got = client.post("/template/forget", json={"runId": "r", "bodyId": 1}).json()
    assert got == {"ok": True, "forgotten": False, "bodyForgotten": True, "galleryN": 1}
    s = opened(tmp_path, "r")
    with s.reading():
        assert s.body_sightings() == []
    again = client.post("/template/forget", json={"runId": "r", "bodyId": 1}).json()
    assert again["bodyForgotten"] is False, "already gone is ordinary, not an error"
    assert client.post("/template/forget", json={"runId": "r"}).status_code == 422


def test_a_body_table_without_face_w_migrates_in_place(tmp_path):
    """The evening's column is added to a morning's file; old rows read None."""
    path = tmp_path / "morning.db"
    conn = sqlite3.connect(path)
    conn.executescript(_V2_SCHEMA + """
CREATE TABLE body_sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, h REAL NOT NULL,
    w REAL NOT NULL, y_bottom REAL NOT NULL, frame_h INTEGER NOT NULL,
    created_at TEXT NOT NULL
);""")
    conn.execute(
        "INSERT INTO store_meta (k, v) VALUES ('embedderId', ?)", (store.EMBEDDER_ID,)
    )
    conn.execute(
        "INSERT INTO body_sightings (key, h, w, y_bottom, frame_h, created_at)"
        " VALUES ('p00001', 1200.0, 400.0, 1500.0, ?, '2026-09-24T09:00:00+00:00')",
        (FRAME_H,),
    )
    conn.commit()
    conn.close()
    with store.VectorStore(path) as s:
        (row,) = s.body_sightings()
        assert row.face_w is None and row.created_at == "2026-09-24T09:00:00+00:00"
        s.add_body_sighting("p00001", 1200.0, 400.0, 1500.0, FRAME_H, 110.0)
        assert [r.face_w for r in s.body_sightings()] == [None, 110.0]


def test_attribute_wire_shape_is_validated_at_the_boundary(client):
    """A sex outside M/F or a probability outside 0..1 is a 422, not a stored lie."""
    bad = [
        {"gender": "X", "genderP": 0.9, "age": 30},
        {"gender": "M", "genderP": 1.5, "age": 30},
        {"gender": "M", "genderP": 0.9},
    ]
    for a in bad:
        res = client.post(
            "/match", json={"runId": "r", "embedding": _json(hub()), "attributes": a}
        )
        assert res.status_code == 422, res.text
    res = client.post("/match", json={
        "runId": "r", "embedding": _json(hub()),
        "attributes": None, "featNorm": None, "body": None,
    })
    assert res.status_code == 200, "every side field is optional and null is not measured"


# ------------------------------------------------- the clothing set-aside

#: Two v3 torsos with no bin in common: intersection 0.  Real reads on the
#: Sharon re-run scored 0.10-0.25 for the operator's different-people pairs
#: and never under 0.52 for a person against themselves.
def _torso(bins) -> list[float]:
    """A v3 (64-float) descriptor with its mass spread evenly over ``bins``."""
    v = np.zeros(64)
    v[list(bins)] = 1.0
    return [float(x) for x in v / v.sum()]


RED, BLUE, GREEN = _torso(range(0, 3)), _torso(range(20, 23)), _torso(range(10, 13))


def clothe(tmp_path, run, key, vec, torsos):
    """Add one template per torso read to ``key`` (same face, so the pair's
    band test does not move)."""
    s = opened(tmp_path, run)
    with s.transaction():
        for t in torsos:
            s.add(key, vec, quality=70.0, appearance=t)


def _pair(client, tmp_path, a_torsos, b_torsos):
    """Hub + one spoke in the review band, each wearing the given reads."""
    kh = match(client, "r", hub())["personKey"]
    ks = match(client, "r", spoke(1))["personKey"]
    clothe(tmp_path, "r", kh, hub(), a_torsos)
    clothe(tmp_path, "r", ks, spoke(1), b_torsos)
    return kh, ks


def test_a_clear_clothing_clash_sets_the_pair_aside_and_keeps_it_visible(client, tmp_path, ticking):
    """Both sides: three reads, over three seconds, agreeing with themselves;
    against each other: nothing.  Set aside — counted, listed with its reason,
    written nowhere, and still mergeable."""
    kh, ks = _pair(client, tmp_path, [RED] * 3, [BLUE] * 3)
    got = review(client)
    assert got["excluded"]["clothes"] == 1
    assert frozenset((kh, ks)) not in pairs_of(got)
    [row] = got["setAside"]
    assert (frozenset((row["a"], row["b"])), row["reasons"]) == (frozenset((kh, ks)), ["clothes"])
    # approx: the reads are float32 thirds and the agreement is summed in
    # float64 (1.0000000298), the arithmetic the body-log rule uses.
    assert row["why"]["clothes"] == {
        "selfA": pytest.approx(1.0), "selfB": pytest.approx(1.0), "cross": pytest.approx(0.0),
        "nA": 3, "nB": 3,
    }
    s = opened(tmp_path, "r")
    with s.reading():
        assert s.cannot_link(kh, ks) is False, "set aside is not a recorded fact"
    merged = client.post("/merge", json={"runId": "r", "keep": kh, "drop": ks}).json()
    assert merged["merged"] is True


def test_matching_clothes_still_ask(client, tmp_path, ticking):
    """Same clothes on both: exactly the pair a human should look at."""
    kh, ks = _pair(client, tmp_path, [RED] * 3, [RED] * 3)
    got = review(client)
    assert got["excluded"]["clothes"] == 0
    assert pairs_of(got)[frozenset((kh, ks))]["why"]["clothes"]["cross"] == pytest.approx(1.0)


def test_too_few_reads_on_either_side_still_asks(client, tmp_path, ticking):
    """Two reads is one opinion held twice, not a person's clothing."""
    kh, ks = _pair(client, tmp_path, [RED] * 3, [BLUE] * 2)
    assert frozenset((kh, ks)) in pairs_of(review(client))


def test_a_person_whose_own_reads_scatter_is_not_evidence(client, tmp_path, ticking):
    """Blue, green, blue: median self-agreement 0 — lighting, a shawl, a
    crowded torso.  Such a person's clothing cannot rule anyone out."""
    kh, ks = _pair(client, tmp_path, [RED] * 3, [BLUE, GREEN, BLUE])
    got = review(client)
    assert got["excluded"]["clothes"] == 0
    assert pairs_of(got)[frozenset((kh, ks))]["why"]["clothes"]["selfB"] == pytest.approx(0.0)


def test_reads_from_one_moment_are_one_read(client, tmp_path, monkeypatch):
    """Three reads written in the same second agree trivially: still asks."""
    monkeypatch.setattr(store, "_now", lambda: "2026-09-24T21:00:00+00:00")
    kh, ks = _pair(client, tmp_path, [RED] * 3, [BLUE] * 3)
    assert frozenset((kh, ks)) in pairs_of(review(client))


def test_v2_torsos_are_never_clothing_evidence(client, tmp_path, ticking):
    """A 48-float read is a different histogram: counted as zero v3 reads."""
    v2 = [1.0 / 48] * 48
    kh, ks = _pair(client, tmp_path, [v2] * 3, [BLUE] * 3)
    row = pairs_of(review(client))[frozenset((kh, ks))]
    assert (row["why"]["clothes"]["nA"], row["why"]["clothes"]["cross"]) == (0, None)


def test_clothes_clash_zero_is_the_off_switch(client, tmp_path, ticking, monkeypatch):
    """HECO_REVIEW_CLOTHES_CLASH=0: clothing ranks again and removes nobody."""
    kh, ks = _pair(client, tmp_path, [RED] * 3, [BLUE] * 3)
    monkeypatch.setenv("HECO_REVIEW_CLOTHES_CLASH", "0")
    got = review(client)
    assert (got["excluded"]["clothes"], got["setAside"]) == (0, [])
    assert frozenset((kh, ks)) in pairs_of(got)


def test_health_reports_the_clothing_policy(client, monkeypatch):
    """The queue's policy must be readable beside the count it produced."""
    body = client.get("/health").json()
    assert body["reviewClothesClash"] == pytest.approx(0.35)
    assert body["reviewClothesMinN"] == 3
    assert body["reviewClothesSelfMin"] == pytest.approx(0.6)
    monkeypatch.setenv("HECO_REVIEW_CLOTHES_CLASH", "")
    assert config.review_clothes_clash() == pytest.approx(0.35), "empty means unset"
