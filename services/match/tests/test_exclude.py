"""POST /exclude — the operator's *not-a-guest* removal (match 0.17.0).

THE ASK (2026-09-25): "there should be an option to remove the guest from the
list". The key keeps its templates, so the same face sighted again MATCHES it
and is not counted as a new guest, and the review queue stops asking about it.
"""

import math

import numpy as np
import pytest
from app import main, store
from fastapi.testclient import TestClient

DIM = 128


def unit(i: int, cosine: float = 1.0) -> list[float]:
    """A probe at ``cosine`` from axis 0 along axis ``i`` (i=0: axis 0 itself)."""
    v = np.zeros(DIM)
    if i == 0:
        v[0] = 1.0
    else:
        v[0] = cosine
        v[i] = math.sqrt(1.0 - cosine * cosine)
    return [float(x) for x in v]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with gallery data in a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    store.close_all_stores()
    with TestClient(main.app) as c:
        yield c
    store.close_all_stores()


def match(client, emb):
    """One /match of ``emb`` in run r; the reply."""
    res = client.post("/match", json={"runId": "r", "embedding": emb, "quality": 90.0})
    assert res.status_code == 200, res.text
    return res.json()


def test_an_excluded_guest_is_matched_again_not_recounted(client):
    """Removed, then seen again: the sighting matches the removed key, no new guest."""
    first = match(client, unit(0))
    assert first["isNew"] is True
    key = first["personKey"]
    res = client.post("/exclude", json={"runId": "r", "personKey": key}).json()
    assert res == {"excluded": True, "personKey": key}
    again = match(client, unit(0))
    assert again["personKey"] == key and again["isNew"] is False


def test_excluding_twice_or_an_unknown_key_says_false(client):
    """The caller lowers its count only on True: never twice, never for nobody."""
    key = match(client, unit(0))["personKey"]
    def excluded(k):
        return client.post("/exclude", json={"runId": "r", "personKey": k}).json()["excluded"]

    assert excluded(key) is True
    assert excluded(key) is False, "never twice"
    assert excluded("p09999") is False, "never for nobody"


def test_an_excluded_key_leaves_the_review_queue(client):
    """A pair in the review band disappears once either side is removed."""
    a = match(client, unit(0))["personKey"]
    b = match(client, unit(1, cosine=0.30))["personKey"]
    q = client.post("/review/duplicates", json={"runId": "r"}).json()
    assert any({p["a"], p["b"]} == {a, b} for p in q["pairs"])
    client.post("/exclude", json={"runId": "r", "personKey": b})
    q = client.post("/review/duplicates", json={"runId": "r"}).json()
    assert not any(b in (p["a"], p["b"]) for p in q["pairs"] + q.get("setAside", []))


def test_a_bad_run_id_is_refused(client):
    """The same run-id rule as every gallery call."""
    res = client.post("/exclude", json={"runId": "../x", "personKey": "p00001"})
    assert res.status_code == 422
