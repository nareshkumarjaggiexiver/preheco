"""The review queue's head and beard evidence (match 0.14.0, 2026-09-24 night).

Run f0bfc5's review pair #1 was a maroon turban and a black beard against a
peach turban and a white beard.  The runner now sends, per kept face, a head
descriptor (40 floats: 24 soft hue bins, 3 brightness bins, 13 reserved) and
a beard reading ([skin, dark, grey, white]); the match service logs both on
the sighting's body row and the review may set a pair aside on either:

* head — each identity reads ONE head (three reads over two seconds whose
  median pairwise intersection is at least 0.6) and the best cross reading
  is under HECO_REVIEW_HEAD_CLASH;
* beard — each identity has HECO_REVIEW_BEARD_MIN_N reads over two seconds,
  two thirds of them name one class, and the two classes cannot be one
  face: none against any beard, dark against white.

Both sides needed, null never excludes, a legacy body table opens with the
columns ALTERed in and reads nothing, and the reasons land in setAside.
Geometry as in test_review_clothes: a hub on e0, spokes at cosine 0.30.
"""

import itertools
import math
import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from app import config, gallery, main, store
from app.appearance import beard_class, beard_read_class, beards_differ, head_label
from fastapi.testclient import TestClient

DIM = 128
BODY = {"h": 1200.0, "w": 400.0, "yBottom": 1500.0, "frameH": 2160}


def _e(i: int) -> np.ndarray:
    """The i-th standard basis vector."""
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


def head(bins: dict[int, float]) -> list[float]:
    """A 40-float head descriptor with mass in ``bins``."""
    v = np.zeros(40)
    total = float(sum(bins.values()))
    for b, w in bins.items():
        v[b] = w / total
    return [float(x) for x in v]


RED_HEAD = head({23: 1.0, 0: 1.0})
ORANGE_HEAD = head({1: 0.8, 2: 0.2})
BLACK_HAIR = head({24: 1.0})
DARK = [0.10, 0.85, 0.03, 0.02]
NONE = [0.92, 0.04, 0.02, 0.02]
WHITE = [0.20, 0.05, 0.15, 0.60]
GREY = [0.20, 0.05, 0.60, 0.15]
UNSURE = [0.50, 0.35, 0.10, 0.05]


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


@pytest.fixture()
def ticking(monkeypatch):
    """A store clock that moves one second per write."""
    t0 = datetime(2026, 9, 24, 21, 0, 0, tzinfo=UTC)
    counter = itertools.count()
    monkeypatch.setattr(
        store, "_now", lambda: (t0 + timedelta(seconds=next(counter))).isoformat()
    )


def match(client, run, emb, head_=None, beard=None, body=BODY):
    """POST /match carrying whichever readings the test gives it."""
    payload = {"runId": run, "embedding": emb, "quality": 90.0}
    if head_ is not None:
        payload["head"] = head_
    if beard is not None:
        payload["beard"] = beard
    if body is not None:
        payload["body"] = body
    res = client.post("/match", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


def seen(client, emb, heads=(), beards=(), run="r"):
    """One identity sighted once per reading; its key."""
    key = None
    for h, b in itertools.zip_longest(heads, beards):
        key = match(client, run, emb, h, b)["personKey"]
    return key


def review(client, run="r", **kw):
    """The run's review queue."""
    res = client.post("/review/duplicates", json={"runId": run, **kw})
    assert res.status_code == 200, res.text
    return res.json()


def only_pair(report) -> dict:
    """The single pair of a two-identity run, queued or set aside."""
    rows = report["pairs"] + report["setAside"]
    assert len(rows) == 1, rows
    return rows[0]


def side(row, key) -> tuple[str, str]:
    """('A', 'B') when ``key`` is the row's a, else ('B', 'A')."""
    return ("A", "B") if row["a"] == key else ("B", "A")


# ------------------------------------------------------------- the wire


def test_match_takes_head_and_beard_and_refuses_the_wrong_shape(client):
    """40 and 4 floats or absent; anything else names the contract."""
    assert match(client, "r", hub(), RED_HEAD, DARK)["isNew"] is True
    for bad in ({"head": [0.1] * 39}, {"beard": [0.5, 0.5]}, {"beard": [1.5, 0, 0, 0]}):
        res = client.post("/match", json={"runId": "r", "embedding": hub(), **bad})
        assert res.status_code == 422, bad
        assert ("head" in res.text) or ("beard" in res.text)


def test_the_body_row_logs_head_and_beard_and_a_legacy_table_migrates(client, tmp_path):
    """Per sighting, beside the torso; an old body table reads None for both."""
    kh = match(client, "r", hub(), RED_HEAD, DARK)["personKey"]
    match(client, "r", hub(), None, NONE)
    s = store.open_store(gallery.db_path(tmp_path, "r"))
    with s.reading():
        ev = s.sighting_evidence()
    assert [e.key for e in ev] == [kh, kh]
    assert ev[0].head.size == 40 and ev[0].beard.tolist() == pytest.approx(DARK)
    assert ev[1].head is None and ev[1].appearance is None

    path = tmp_path / "gallery-legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE body_sightings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL,"
        " h REAL NOT NULL, w REAL NOT NULL, y_bottom REAL NOT NULL, frame_h INTEGER NOT NULL,"
        " created_at TEXT NOT NULL, face_w REAL, appearance BLOB);"
    )
    conn.execute(
        "INSERT INTO body_sightings (key, h, w, y_bottom, frame_h, created_at, appearance)"
        " VALUES ('p00001', 1200, 400, 1500, 2160, '2026-09-24T21:00:00+00:00', ?)",
        (np.full(64, 1 / 64, np.float32).tobytes(),),
    )
    conn.commit()
    conn.close()
    with store.VectorStore(path) as legacy:
        cols = {row[1] for row in legacy.conn.execute("PRAGMA table_info(body_sightings)")}
        assert {"head", "beard"} <= cols
        (row,) = legacy.sighting_evidence()
        assert row.appearance.size == 64 and row.head is None and row.beard is None


# ------------------------------------------------------------- the head


def test_two_steady_heads_that_clash_are_set_aside(client, ticking):
    """A red turban against an orange one: set aside, reason 'head'."""
    kh = seen(client, hub(), heads=[RED_HEAD] * 3)
    seen(client, spoke(1), heads=[ORANGE_HEAD] * 3)
    got = review(client)
    assert got["excluded"] == excluded(head=1) and got["pairs"] == []
    (row,) = got["setAside"]
    assert row["reasons"] == ["head"]
    wh = row["why"]["head"]
    me, other = side(row, kh)
    assert (wh["a"], wh["b"]) == (("red", "orange") if me == "A" else ("orange", "red"))
    assert wh["sim"] == pytest.approx(0.0)
    assert wh["self" + me] == pytest.approx(1.0) and wh["n" + other] == 3


def test_a_head_that_matches_in_any_read_keeps_the_pair(client, ticking):
    """One of the spoke's reads is red too: best cross 1.0."""
    seen(client, hub(), heads=[RED_HEAD] * 3)
    seen(client, spoke(1), heads=[ORANGE_HEAD, ORANGE_HEAD, ORANGE_HEAD, RED_HEAD])
    got = review(client)
    assert got["excluded"]["head"] == 0 and len(got["pairs"]) == 1


def test_head_needs_both_sides_three_reads_and_its_own_agreement(client, ticking):
    """Two reads, a disagreeing side, or no reads at all: never a reason."""
    seen(client, hub(), heads=[RED_HEAD] * 3)
    seen(client, spoke(1), heads=[ORANGE_HEAD] * 2)                       # two reads
    seen(client, spoke(2), heads=[ORANGE_HEAD, BLACK_HAIR, ORANGE_HEAD, BLACK_HAIR])
    seen(client, spoke(3), heads=[None] * 3)                              # unmeasured
    got = review(client)
    assert got["excluded"]["head"] == 0 and len(got["pairs"]) == 3
    unmeasured = [p for p in got["pairs"] if p["why"]["head"]["sim"] is None]
    assert len(unmeasured) == 1


def test_a_covered_head_is_never_set_against_a_bare_one(client, ticking):
    """A turban (or a dupatta over the hair) against black hair: still asked.

    The same guest's head is covered and uncovered within one wedding — a
    dupatta for the ceremony, a rumal for the Gurdwara — so headwear against
    hair is what one person minted twice can look like.  Cross 0.0, kept.
    """
    kh = seen(client, hub(), heads=[RED_HEAD] * 3)
    seen(client, spoke(1), heads=[BLACK_HAIR] * 3)
    got = review(client)
    assert got["excluded"]["head"] == 0 and len(got["pairs"]) == 1
    (row,) = got["pairs"]
    me, _other = side(row, kh)
    wh = row["why"]["head"]
    assert wh["sim"] == pytest.approx(0.0)
    assert (wh["a"], wh["b"]) == (("red", "black") if me == "A" else ("black", "red"))


def worn(bins: dict[int, float], wear: float) -> list[float]:
    """A head descriptor as a 2026-09-25 runner sends it: histogram, then the
    headwear share (skin left out) in slot 27 and its flag in slot 28."""
    v = head(bins)
    v[27], v[28] = wear, 1.0
    return v


def test_a_bald_scalp_is_bare_however_chromatic_its_skin(client, ticking):
    """f0bfc5's p00062, balding: his head histogram read 0.91 chromatic (his
    skin's orange) and the rule took him for headwear, set against real
    turbans at 0.21-0.26 — a balding guest in a safa for the baraat, set
    aside against himself bare-headed. Skin left out, his share is 0.03:
    bare, and a covered head is never set against a bare one."""
    kh = seen(client, hub(), heads=[worn({1: 0.9, 24: 0.1}, 0.03)] * 3)
    seen(client, spoke(1), heads=[worn({11: 0.85, 24: 0.15}, 0.8)] * 3)  # blue turban
    got = review(client)
    assert got["excluded"]["head"] == 0 and len(got["pairs"]) == 1
    wh = got["pairs"][0]["why"]["head"]
    me, other = side(got["pairs"][0], kh)
    assert wh["wear" + me] == pytest.approx(0.03) and wh["wear" + other] == pytest.approx(0.8)
    assert wh["sim"] < config.DEFAULT_REVIEW_HEAD_CLASH, "the colours still disagree"


def test_two_turbans_measured_skin_free_are_still_set_aside(client, ticking):
    """Blue against maroon, both well over the skin-free floor: set aside, as
    before; the flag and the share never count as histogram agreement."""
    seen(client, hub(), heads=[worn({11: 0.9, 24: 0.1}, 0.8)] * 3)
    seen(client, spoke(1), heads=[worn({22: 0.6, 23: 0.3, 24: 0.1}, 0.7)] * 3)
    got = review(client)
    assert got["excluded"]["head"] == 1
    (row,) = got["setAside"]
    assert row["why"]["head"]["sim"] == pytest.approx(0.1), "bins 0..26 only"


def test_head_clash_zero_is_off(client, ticking, monkeypatch):
    """Off: the pair is back in the queue and why.head is still shown."""
    seen(client, hub(), heads=[RED_HEAD] * 3)
    seen(client, spoke(1), heads=[ORANGE_HEAD] * 3)
    monkeypatch.setenv("HECO_REVIEW_HEAD_CLASH", "0")
    got = review(client)
    assert got["excluded"] == excluded() and got["setAside"] == []
    assert got["pairs"][0]["why"]["head"]["sim"] == pytest.approx(0.0)


# ------------------------------------------------------------- the beard


@pytest.mark.parametrize(
    ("a", "b", "apart"),
    [
        (DARK, NONE, True),      # a full beard against a clean chin
        (WHITE, NONE, False),    # a warm light reads a white beard as none: off
        (GREY, NONE, False),     # ...by default (HECO_REVIEW_BEARD_PALE)
        (DARK, WHITE, True),     # pair #1: black beard against white
        (DARK, GREY, False),     # salt-and-pepper sits between: never set against
        (WHITE, GREY, False),
        (DARK, DARK, False),
        (DARK, UNSURE, False),   # a moustache, stubble: names no class
    ],
)
def test_beard_classes_one_face_cannot_show_both(client, ticking, a, b, apart):
    """The beard rule on each pairing of classes."""
    seen(client, hub(), beards=[a] * 3)
    seen(client, spoke(1), beards=[b] * 3)
    got = review(client)
    assert got["excluded"]["beard"] == (1 if apart else 0)
    row = only_pair(got)
    assert (row.get("reasons") == ["beard"]) is apart


def test_beard_needs_min_n_reads_over_two_seconds_on_both_sides(client, monkeypatch):
    """Three reads inside one second are one moment; min_n 0 is off."""
    t0 = datetime(2026, 9, 24, 21, 0, 0, tzinfo=UTC)
    counter = itertools.count()
    monkeypatch.setattr(
        store, "_now",
        lambda: (t0 + timedelta(milliseconds=300 * next(counter))).isoformat(),
    )
    seen(client, hub(), beards=[DARK] * 3)
    seen(client, spoke(1), beards=[NONE] * 3)
    got = review(client)
    assert got["excluded"]["beard"] == 0
    row = only_pair(got)
    assert {row["why"]["beard"]["a"], row["why"]["beard"]["b"]} == {"dark", "none"}, (
        "the classes are still shown"
    )


@pytest.mark.parametrize("pale", [WHITE, GREY])
def test_none_against_a_pale_beard_only_with_its_switch(client, ticking, monkeypatch, pale):
    """HECO_REVIEW_BEARD_PALE=1 brings back none-vs-grey/white.

    Off by default: the pale test compares the chin's saturation with the
    cheek's, and an 8% warm light read f0bfc5's white-bearded elder as
    "none" — his genuine duplicate set aside.  The classes are still shown.
    """
    seen(client, hub(), beards=[pale] * 3)
    seen(client, spoke(1), beards=[NONE] * 3)
    got = review(client)
    assert got["excluded"]["beard"] == 0
    assert {only_pair(got)["why"]["beard"]["a"], only_pair(got)["why"]["beard"]["b"]} == {
        "none", "white" if pale is WHITE else "grey"}
    monkeypatch.setenv("HECO_REVIEW_BEARD_PALE", "1")
    assert review(client)["excluded"]["beard"] == 1
    assert client.get("/health").json()["reviewBeardPale"] is True


def test_beard_min_n_zero_is_off(client, ticking, monkeypatch):
    """HECO_REVIEW_BEARD_MIN_N=0: nothing set aside on beards."""
    seen(client, hub(), beards=[DARK] * 3)
    seen(client, spoke(1), beards=[NONE] * 3)
    monkeypatch.setenv("HECO_REVIEW_BEARD_MIN_N", "0")
    assert review(client)["excluded"] == excluded()


def test_an_unsure_majority_names_no_class():
    """Two thirds of the reads, counting the unsure ones against it."""
    v = [np.asarray(x) for x in (DARK, DARK, UNSURE)]
    assert beard_class(v) == "dark"
    assert beard_class([np.asarray(x) for x in (DARK, UNSURE, UNSURE)]) is None
    assert beard_read_class(np.asarray(UNSURE)) is None
    assert beards_differ(None, "dark") is False, "absent is not a class"


def test_head_labels_follow_the_dominant_family():
    """Display only: maroon is red, peach is orange, black is black."""
    assert head_label([np.asarray(RED_HEAD)]) == "red"
    assert head_label([np.asarray(ORANGE_HEAD)]) == "orange"
    peach = head({0: 0.83, 1: 0.17})     # f0bfc5's peach turban, H ~5
    maroon = head({22: 0.6, 23: 0.4})    # ...and its maroon one, H ~168
    assert (head_label([np.asarray(maroon)]), head_label([np.asarray(peach)])) == ("red", "orange")
    assert head_label([np.asarray(BLACK_HAIR)]) == "black"
    assert head_label([np.asarray(head({21: 1.0}))]) == "pink"
    assert head_label([]) is None


# ------------------------------------------------------------- together


def test_every_signal_that_spoke_is_listed_and_the_pair_counted_once(client, ticking):
    """Clothes, head and beard all against the pair: one count, three reasons."""
    red = [0.0] * 64
    red[0], red[39], red[49] = 0.9, 0.07, 0.03
    blue = [0.0] * 64
    blue[24], blue[39], blue[49] = 0.9, 0.07, 0.03
    for _ in range(3):
        client.post("/match", json={
            "runId": "r", "embedding": hub(), "appearance": red, "head": RED_HEAD,
            "beard": DARK, "body": BODY,
        })
        client.post("/match", json={
            "runId": "r", "embedding": spoke(1), "appearance": blue, "head": ORANGE_HEAD,
            "beard": WHITE, "body": BODY,
        })
    got = review(client)
    assert got["excluded"] == excluded(clothes=1)
    (row,) = got["setAside"]
    assert row["reasons"] == ["clothes", "head", "beard"]


def test_health_reports_head_and_beard_policy_and_empty_means_unset(client, monkeypatch):
    """The policy that produced a queue is readable beside it."""
    body = client.get("/health").json()
    assert body["reviewHeadClash"] == pytest.approx(config.DEFAULT_REVIEW_HEAD_CLASH)
    assert body["reviewBeardMinN"] == config.DEFAULT_REVIEW_BEARD_MIN_N == 3
    monkeypatch.setenv("HECO_REVIEW_HEAD_CLASH", "")
    monkeypatch.setenv("HECO_REVIEW_BEARD_MIN_N", "")
    assert config.review_head_clash() == pytest.approx(config.DEFAULT_REVIEW_HEAD_CLASH)
    assert config.review_beard_min_n() == 3
