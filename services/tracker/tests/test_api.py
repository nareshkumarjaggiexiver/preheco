"""API tests: per-run state isolation, reset semantics, wire shapes."""

import pytest
from app.main import _last_used, _runs, app
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """In-process client with a clean run table per test."""
    _runs.clear()
    _last_used.clear()
    with TestClient(app) as c:
        yield c
    _runs.clear()
    _last_used.clear()


def _track(client, run_id: str, boxes):
    """POST one frame of detections for a run."""
    payload = {
        "runId": run_id,
        "tMs": 0,
        "boxes": [{"x": x, "y": y, "w": 20, "h": 40, "conf": 0.9} for x, y in boxes],
    }
    res = client.post("/track", json=payload)
    assert res.status_code == 200, res.text
    return res.json()["tracks"]


def test_health(client):
    """Health names the algorithm honestly — this is not a DNN."""
    body = client.get("/health").json()
    assert body["ok"] is True and body["model"] == "sort-lite-iou-velocity"


def test_track_auto_creates_run_and_confirms(client):
    """First /track on a new runId works; warm-up reports immediately."""
    tracks = _track(client, "r1", [(10, 50)])
    assert len(tracks) == 1
    t = tracks[0]
    assert set(t) == {"id", "box", "ageFrames", "hits"}
    assert t["ageFrames"] == 1 and t["hits"] == 1
    assert t["box"]["w"] == 20 and t["box"]["conf"] == 0.9


def test_runs_are_isolated(client):
    """Two runs have independent trackers and independent id spaces."""
    for i in range(4):
        a = _track(client, "run-a", [(10 + i * 8, 50)])
    for i in range(4):
        b = _track(client, "run-b", [(300 - i * 8, 90)])
    assert a[0]["id"] == 1
    assert b[0]["id"] == 1  # same id 1, different run — no cross-talk
    assert a[0]["box"]["x"] != b[0]["box"]["x"]


def test_reset_clears_state(client):
    """After /reset the run starts fresh: ids and ages restart."""
    for i in range(5):
        before = _track(client, "r1", [(10 + i * 8, 50)])
    assert before[0]["ageFrames"] == 5
    res = client.post("/reset", json={"runId": "r1"})
    assert res.status_code == 200 and res.json()["ok"] is True
    after = _track(client, "r1", [(10, 50)])
    assert after[0]["ageFrames"] == 1 and after[0]["hits"] == 1


def test_env_tuning_applies_on_reset(client, monkeypatch):
    """/reset rebuilds the tracker from current env (min_hits here)."""
    monkeypatch.setenv("TRACKER_MIN_HITS", "1")
    client.post("/reset", json={"runId": "r1"})
    for i in range(5):
        _track(client, "r1", [(10 + i * 8, 50)])
    # min_hits=1: a brand-new second box reports immediately even past warm-up
    tracks = _track(client, "r1", [(58, 50), (400, 200)])
    assert len(tracks) == 2


def test_empty_boxes_ok(client):
    """A frame with no detections is valid and returns no tracks."""
    assert _track(client, "r1", []) == []


def test_track_rejects_malformed_box(client):
    """Schema validation rejects negative sizes with 422."""
    res = client.post(
        "/track",
        json={"runId": "r", "tMs": 0, "boxes": [{"x": 0, "y": 0, "w": -5, "h": 4}]},
    )
    assert res.status_code == 422


# ---------------------------------------------- regressions (2026-08-05 review)


def test_release_forgets_a_finished_runs_state(client):
    """A finished run's tracker must not stay resident forever.

    Regression: ``_runs`` only ever gained entries. Every run gets a fresh
    uuid-suffixed id, so /reset never replaced an old one and a season of
    events left one dead SortLite per run in memory until a process restart.
    """
    _track(client, "r1", [(10, 50)])
    assert client.get("/health").json()["runs"] == 1

    res = client.post("/release", json={"runId": "r1"})
    assert res.status_code == 200 and res.json()["released"] is True
    assert _runs == {} and _last_used == {}
    assert client.get("/health").json()["runs"] == 0

    # Idempotent: releasing again (or an unknown run) is still success.
    assert client.post("/release", json={"runId": "r1"}).json()["released"] is False


def test_idle_runs_are_evicted_on_the_next_track(client, monkeypatch):
    """A run that died without releasing is swept by the TTL, not kept for ever."""
    monkeypatch.setenv("TRACKER_RUN_TTL_S", "0.05")
    _track(client, "dead-run", [(10, 50)])
    assert "dead-run" in _runs

    import time
    time.sleep(0.06)
    _track(client, "live-run", [(10, 50)])  # any traffic sweeps the stale ones

    assert "dead-run" not in _runs, "an idle run's tracker must be evicted"
    assert "live-run" in _runs, "the run being served is never evicted"


def test_ttl_never_evicts_the_run_being_served(client, monkeypatch):
    """A long, quiet run must not evict itself mid-event."""
    monkeypatch.setenv("TRACKER_RUN_TTL_S", "0.01")
    for i in range(4):
        tracks = _track(client, "slow", [(10 + i * 8, 50)])
    assert tracks[0]["ageFrames"] == 4, "its own state must survive every tick"
