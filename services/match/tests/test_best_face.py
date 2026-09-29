"""The best-face options (match 0.20.0): the probe, the stricter bar, and the
best-face template ("anchor") that follows a guest's clearest face.

Geometry is built exactly: a probe at a KNOWN cosine to a stored face, so every
threshold decision below is the measurement it names, not an approximation.
"""

import numpy as np
import pytest
from app import config, main, store
from fastapi.testclient import TestClient

DIM = 128
THRESHOLD = config.DEFAULT_THRESHOLD  # 0.363


def _unit(v):
    """Normalise to unit length."""
    return v / np.linalg.norm(v)


def _json(v):
    """A vector as the JSON list the API takes."""
    return [float(x) for x in v]


def _base(i):
    """The i-th unit axis: mutually orthogonal faces of different people."""
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def _at(a, cos, axis):
    """A unit vector at exactly ``cos`` to unit ``a``, tilted along ``axis``."""
    return _unit(cos * a + np.sqrt(1.0 - cos * cos) * _base(axis))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with the gallery redirected to a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    store.close_all_stores()
    with TestClient(main.app) as c:
        yield c
    store.close_all_stores()


def _match(client, v, quality=120.0, **extra):
    """POST /match on run r1 and return the reply."""
    body = {"runId": "r1", "embedding": _json(v), "quality": quality, **extra}
    res = client.post("/match", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def _rows(tmp_path):
    """(key, anchor) for every stored template of run r1."""
    store.close_all_stores()
    import sqlite3
    db = sqlite3.connect(str(tmp_path / "gallery-r1.db"))
    try:
        return db.execute("SELECT key, COALESCE(anchor, 0) FROM vectors ORDER BY id").fetchall()
    finally:
        db.close()


def test_a_probe_that_matches_nobody_writes_nothing(client, tmp_path):
    """mint:false on an unknown face: personKey null, deferred, gallery untouched."""
    r = _match(client, _base(0), mint=False)
    assert r["personKey"] is None and r["deferred"] is True and r["isNew"] is False
    assert r["galleryN"] == 0
    assert _rows(tmp_path) == []


def test_a_probe_that_matches_a_known_guest_is_an_ordinary_match(client):
    """The probe changes nothing for a face the gallery already knows."""
    a = _base(0)
    minted = _match(client, a)
    assert minted["isNew"] is True
    again = _match(client, _at(a, 0.8, 1), mint=False)
    assert again["personKey"] == minted["personKey"] and again["deferred"] is False
    assert again["galleryN"] == 1


def test_the_stricter_bar_only_ever_raises_the_threshold(client):
    """minCosine 0.45 turns a 0.40 match away; a minCosine under the threshold
    cannot let a 0.30 face in."""
    a = _base(0)
    _match(client, a)
    probe = _at(a, 0.40, 1)
    assert _match(client, probe, mint=False, minCosine=0.45)["personKey"] is None
    assert _match(client, probe, mint=False)["personKey"] is not None, "0.40 clears 0.363"
    assert _match(client, _at(a, 0.30, 2), mint=False, minCosine=0.1)["personKey"] is None


def test_a_mint_can_found_its_guest_on_a_best_face_template(client, tmp_path):
    """bestFaceAnchor flags the founding view; without it nothing is flagged."""
    _match(client, _base(0), bestFaceAnchor=True)
    _match(client, _base(1))
    assert _rows(tmp_path) == [("p00001", 1), ("p00002", 0)]


def test_a_better_face_replaces_the_best_face_template(client, tmp_path):
    """/template/anchor swaps the anchor: one best face per guest, others kept."""
    a = _base(0)
    key = _match(client, a, bestFaceAnchor=True)["personKey"]
    # A second, different view enrolled the ordinary way (an extra angle).
    _match(client, _at(a, 0.75, 3))
    better = _at(a, 0.9, 4)
    res = client.post("/template/anchor", json={
        "runId": "r1", "personKey": key, "embedding": _json(better), "quality": 160.0,
    })
    assert res.status_code == 200, res.text
    rows = _rows(tmp_path)
    assert sum(anchor for _, anchor in rows) == 1, "exactly one best face"
    assert all(k == key for k, _ in rows)
    assert res.json()["templateN"] == len(rows)
    # The founding row (id 1) is gone; the new best face is the anchor.
    assert rows[-1] == (key, 1)


def test_an_anchor_never_creates_a_guest(client):
    """A key the gallery does not hold is a 404: the count cannot move here."""
    _match(client, _base(0))
    res = client.post("/template/anchor", json={
        "runId": "r1", "personKey": "p00099", "embedding": _json(_base(5)),
    })
    assert res.status_code == 404


def test_pruning_never_evicts_the_best_face(tmp_path):
    """prune_redundant drops the most redundant views, never the anchor — even
    when the anchor is the most redundant of them all."""
    s = store.VectorStore(tmp_path / "g.db")
    a = _base(0)
    with s.transaction():
        s.add("p00001", a)
        s.set_anchor("p00001", _at(a, 0.99, 1))  # nearly a twin of the founding view
        s.add("p00001", _at(a, 0.5, 2))
        s.add("p00001", _at(a, 0.4, 3))
        s.prune_redundant("p00001", 2)
        assert s.count_for("p00001") == 2
        assert s.anchor_count("p00001") == 1
        s.prune_to_cap("p00001", 1)
        assert s.anchor_count("p00001") == 1, "the quality cap keeps the anchor too"


def test_anchor_only_compares_a_guest_on_the_best_face_alone(tmp_path):
    """A probe close to an extra view but under the threshold to the best face
    matches the guest ordinarily, and nobody under anchor_only."""
    from app import gallery

    def probe_key(v, **kw):
        return gallery.match(
            tmp_path, "r1", _json(v), 120.0, threshold=THRESHOLD, canon_px=80.0, mint=False, **kw
        ).person_key

    a = _base(0)
    key = gallery.match(
        tmp_path, "r1", _json(a), 120.0, threshold=THRESHOLD, canon_px=80.0, best_face_anchor=True
    ).person_key
    side = _at(a, 0.4, 1)                 # another angle of the same guest
    s = store.open_store(gallery.db_path(tmp_path, "r1"))
    with s.transaction():
        s.add(key, side)                  # held as an extra template
    probe = _unit(0.85 * side + np.sqrt(1 - 0.85 ** 2) * _base(2))
    assert abs(float(probe @ a) - 0.34) < 1e-6, "0.34 to the best face: under 0.363"
    assert probe_key(probe) == key, "the side view carries it ordinarily"
    assert probe_key(probe, anchor_only=True) is None, "the best face alone does not"
    # A guest with no anchor is still compared on every template.
    b = _base(10)
    other = gallery.match(
        tmp_path, "r1", _json(b), 120.0, threshold=THRESHOLD, canon_px=80.0
    ).person_key
    assert probe_key(_at(b, 0.8, 11), anchor_only=True) == other
