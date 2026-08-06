"""Staff-whitelist and operator-correction tests, over the HTTP surface.

Synthetic clustered-gaussian identities as elsewhere; FastAPI's TestClient
drives the ASGI app in-process (no network). Covers: enrol -> staff-first
match, staff excluded from the guest gallery, and the merge/split/mark-staff
corrections the runner applies from the feedback loop.
"""

import os
import time

import numpy as np
import pytest
from app import config, gallery, main, staff, store
from fastapi.testclient import TestClient

RNG = np.random.default_rng(11)


def _unit(v: np.ndarray) -> list[float]:
    return (v / np.linalg.norm(v)).astype(float).tolist()


def centroid() -> np.ndarray:
    """A synthetic identity: a random point on the 128-d unit sphere."""
    return RNG.normal(size=128)


def sighting(c: np.ndarray, noise: float = 0.05) -> list[float]:
    """One face capture: the centroid plus small gaussian noise, unit-length."""
    return _unit(c + RNG.normal(scale=noise, size=128))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with the data dir redirected to a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    with TestClient(main.app) as c:
        yield c


def _match(client, run_id, embedding, quality=85.0, site_id=None):
    body = {"runId": run_id, "embedding": embedding, "quality": quality}
    if site_id is not None:
        body["siteId"] = site_id
    res = client.post("/match", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_enrol_then_staff_first_match(client):
    """An enrolled walk-through is recognised as staff before the gallery."""
    c = centroid()
    samples = [{"embedding": sighting(c), "quality": 80.0} for _ in range(5)]
    res = client.post(
        "/staff/enrol", json={"siteId": "site-1", "staffId": "st-1", "samples": samples}
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"staffId": "st-1", "sampleCount": 5}

    out = _match(client, "run-a", sighting(c), site_id="site-1")
    assert out["isStaff"] is True
    assert out["staffId"] == "st-1"
    assert out["galleryN"] == 0  # staff never enters the guest gallery


def test_staff_excluded_from_guest_count(client):
    """A staff face and a guest face: only the guest grows galleryN."""
    staff_c, guest_c = centroid(), centroid()
    samples = [{"embedding": sighting(staff_c), "quality": 80.0} for _ in range(3)]
    client.post("/staff/enrol", json={"siteId": "s", "staffId": "st", "samples": samples})

    a = _match(client, "run-b", sighting(staff_c), site_id="s")
    assert a["isStaff"] is True and a["galleryN"] == 0
    b = _match(client, "run-b", sighting(guest_c), site_id="s")
    assert b["isStaff"] is False and b["isNew"] is True and b["galleryN"] == 1


def test_no_site_id_skips_staff_check(client):
    """Without siteId every face is a guest (staff store never consulted)."""
    c = centroid()
    client.post(
        "/staff/enrol",
        json={"siteId": "s", "staffId": "st", "samples": [{"embedding": sighting(c)}]},
    )
    out = _match(client, "run-c", sighting(c))  # no siteId
    assert out["isStaff"] is False and out["isNew"] is True


def test_re_enrol_supersedes(client):
    """Re-enrolling a member replaces prior samples — count stays exact."""
    c = centroid()
    first = client.post(
        "/staff/enrol",
        json={
            "siteId": "s",
            "staffId": "st",
            "samples": [{"embedding": sighting(c)} for _ in range(5)],
        },
    ).json()
    assert first["sampleCount"] == 5
    again = client.post(
        "/staff/enrol",
        json={
            "siteId": "s",
            "staffId": "st",
            "samples": [{"embedding": sighting(c)} for _ in range(3)],
        },
    ).json()
    assert again["sampleCount"] == 3  # replaced, not 8


def test_duplicate_merge_endpoint(client):
    """Two keys for one person merge to one; galleryN falls by one."""
    c = centroid()
    a = _match(client, "run-d", sighting(c, noise=0.3))
    # Force a second key by using a well-separated sighting, then merge them.
    b = _match(client, "run-d", sighting(centroid()))
    assert a["personKey"] != b["personKey"]
    res = client.post(
        "/merge", json={"runId": "run-d", "keep": a["personKey"], "drop": b["personKey"]}
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"merged": True, "galleryN": 1}


def test_false_match_split_then_merge_refused(client):
    """A split records a do-not-merge pair the merge endpoint then refuses."""
    a = _match(client, "run-e", sighting(centroid()))
    b = _match(client, "run-e", sighting(centroid()))
    split = client.post(
        "/split", json={"runId": "run-e", "a": a["personKey"], "b": b["personKey"]}
    )
    assert split.status_code == 200 and split.json()["galleryN"] == 2
    merged = client.post(
        "/merge", json={"runId": "run-e", "keep": a["personKey"], "drop": b["personKey"]}
    ).json()
    assert merged == {"merged": False, "galleryN": 2}


def test_mark_staff_moves_guest_to_store(client):
    """Mark-staff lifts a guest out of the gallery and into the staff store."""
    c = centroid()
    g = _match(client, "run-f", sighting(c), site_id="site-x")
    assert g["galleryN"] == 1
    res = client.post(
        "/mark-staff",
        json={"runId": "run-f", "personKey": g["personKey"], "siteId": "site-x", "staffId": "st-9"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["moved"] == 1 and body["galleryN"] == 0 and body["staffKey"] == "st-9"
    # The moved person is now recognised as staff.
    again = _match(client, "run-f", sighting(c), site_id="site-x")
    assert again["isStaff"] is True and again["staffId"] == "st-9"


def test_mark_staff_anonymous_when_no_staff_id(client):
    """Without a roster id, mark-staff mints an anonymous staff key."""
    c = centroid()
    g = _match(client, "run-g", sighting(c), site_id="site-y")
    res = client.post(
        "/mark-staff", json={"runId": "run-g", "personKey": g["personKey"], "siteId": "site-y"}
    ).json()
    assert res["moved"] == 1 and res["staffKey"].startswith("anon")


def test_enrol_rejects_bad_site_id(client):
    """Path-traversal siteIds never reach the filesystem."""
    res = client.post(
        "/staff/enrol",
        json={"siteId": "../evil", "staffId": "x", "samples": [{"embedding": [0.1] * 128}]},
    )
    assert res.status_code == 422


def test_staff_check_module_missing_store(tmp_path):
    """staff.check on a site with no enrolments is a clean 'not staff'."""
    assert staff.check(tmp_path, "unseen-site", [0.1] * 128, config.DEFAULT_THRESHOLD) is None


def test_purge_erases_only_the_named_member(tmp_path):
    """The erasure half of consent: purge removes one member's templates,
    leaves the colleague's, and is idempotent."""
    import numpy as np
    from app import staff

    a = np.zeros(128, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(128, dtype=np.float32)
    b[1] = 1.0
    staff.enrol(tmp_path, "site1", "ravi", [{"embedding": a.tolist(), "quality": 90}])
    staff.enrol(tmp_path, "site1", "sunita", [{"embedding": b.tolist(), "quality": 88}])

    removed = staff.purge(tmp_path, "site1", ["ravi"])
    assert removed == {"ravi": 1}

    # ravi no longer matches; sunita still does
    assert staff.check(tmp_path, "site1", a, threshold=0.9) is None
    hit = staff.check(tmp_path, "site1", b, threshold=0.9)
    assert hit is not None and hit.key == "sunita"

    # idempotent: purging again (or an unknown id) reports 0 and succeeds
    assert staff.purge(tmp_path, "site1", ["ravi", "ghost"]) == {"ravi": 0, "ghost": 0}


def test_purge_with_no_store_file_is_trivially_done(tmp_path):
    """A site that never enrolled anyone has trivially honoured every erasure."""
    from app import staff
    assert staff.purge(tmp_path, "empty-site", ["x"]) == {"x": 0}


# ---------------------------------------------- regressions (2026-08-05 review)


def test_mark_staff_without_site_id_is_refused_and_keeps_the_templates(client):
    """Never destroy a person's templates when there is nowhere to move them.

    Regression: /mark-staff removed the guest's templates UNCONDITIONALLY but
    only absorbed them into a staff store ``if templates and siteId`` — with no
    siteId they were simply deleted, while moved>0 was still returned. The
    runner then decremented unique and resolved the correction 'applied', and
    the person, now unknown to every store, was counted as a brand-new guest
    at their next crossing. The correction silently undid itself.
    """
    c = centroid()
    g = _match(client, "run-z", sighting(c))
    assert g["galleryN"] == 1

    res = client.post("/mark-staff", json={"runId": "run-z", "personKey": g["personKey"]})
    assert res.status_code == 400
    assert "siteId" in res.json()["detail"]

    # The person is still there, still the same key: nothing was destroyed.
    again = _match(client, "run-z", sighting(c))
    assert again["isNew"] is False and again["personKey"] == g["personKey"]
    assert again["galleryN"] == 1


def test_missed_correction_adds_an_attested_person_to_the_count(client):
    """The one lever that moves the count UP, and it is marked as human-made.

    Regression: 'missed' feedback was acknowledged and changed nothing, so an
    operator watching an uncounted guest walk through had no way to fix the
    number they were about to deliver.
    """
    c = centroid()
    first = _match(client, "run-mm", sighting(c))
    assert first["galleryN"] == 1

    res = client.post("/count/manual", json={"runId": "run-mm", "note": "by the bar"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["manual"] is True and body["galleryN"] == 2
    assert body["personKey"].startswith("m"), "human-added people stay distinguishable"

    # An attested person owns no template, so they can never absorb a sighting.
    again = _match(client, "run-mm", sighting(c))
    assert again["personKey"] == first["personKey"] and again["galleryN"] == 2


def test_manual_additions_survive_a_reopen_and_never_match(client, tmp_path):
    """The manual row is durable and stays out of the cosine scan."""
    from app import gallery
    key, n = gallery.add_manual(tmp_path, "run-p", "wheelchair user")
    assert (key.startswith("m"), n) == (True, 1)
    from app.store import close_all_stores
    close_all_stores()  # force a reload from disk
    assert gallery.count(tmp_path, "run-p") == 1


def test_gallery_sweep_deletes_orphaned_run_files_only(client, tmp_path):
    """Gallery files have a lifecycle; staff stores deliberately do not.

    Regression: gallery.reset only deleted the CURRENT run's file, and a run id
    is freshly minted per run, so no prior file was ever removed — data/ grew
    one gallery per run indefinitely, each holding real guests' face
    embeddings. That is a retention liability, not just disk.
    """
    import os
    import time

    from app import gallery

    _match(client, "old-run", sighting(centroid()))
    _match(client, "new-run", sighting(centroid()))
    client.post("/staff/enrol", json={
        "siteId": "site-keep", "staffId": "st", "samples": [{"embedding": sighting(centroid())}],
    })
    old = gallery.db_path(tmp_path, "old-run")
    stale = time.time() - 7200
    os.utime(old, (stale, stale))

    res = client.post("/gallery/sweep", json={"maxAgeS": 3600})
    assert res.status_code == 200, res.text
    assert res.json()["swept"] == ["old-run"]
    assert not old.exists()
    assert gallery.db_path(tmp_path, "new-run").exists(), "a live run must be untouched"
    assert staff.db_path(tmp_path, "site-keep").exists(), "staff stores persist by design"


def test_the_retention_sweep_runs_itself(tmp_path, monkeypatch):
    """The timed sweep is real code on the real clock, not an endpoint nobody calls.

    Galleries now OUTLIVE their runs (the runner stopped deleting at settle —
    the gallery is the evidence behind an invoice figure, and deleting it at
    settle meant a disputed count could be shown nothing the next morning).
    That is only safe because retention moved HERE: a background task started
    by the service lifespan ages every gallery out after
    HECO_GALLERY_RETENTION_S, settled and crashed runs alike.  This test pins
    the moving parts an integration test can reach without waiting an hour:
    the env knobs parse (empty string = unset, the compose ${VAR-} rendering),
    the startup pass sweeps an already-stale file, and staff stores survive.
    """
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))

    # Env parsing: empty means unset, values parse, absent means default.
    monkeypatch.setenv("HECO_GALLERY_RETENTION_S", "")
    assert main.retention_s() == main.DEFAULT_SWEEP_MAX_AGE_S
    monkeypatch.setenv("HECO_GALLERY_RETENTION_S", "7200")
    assert main.retention_s() == 7200.0
    monkeypatch.delenv("HECO_GALLERY_RETENTION_S")
    assert main.retention_s() == main.DEFAULT_SWEEP_MAX_AGE_S

    # A gallery already older than the retention window when the service
    # starts, plus a staff store that must never be touched.
    stale_gallery = gallery.db_path(tmp_path, "died-yesterday")
    store.open_store(stale_gallery)
    store.close_all_stores()
    staff_store = staff.db_path(tmp_path, "site-keep")
    store.open_store(staff_store)
    store.close_all_stores()
    long_ago = time.time() - main.DEFAULT_SWEEP_MAX_AGE_S - 60
    os.utime(stale_gallery, (long_ago, long_ago))

    # Entering the TestClient context runs the lifespan, which starts the
    # sweeper; its first pass is immediate.  Poll briefly rather than sleep a
    # fixed amount — the pass is fast but runs on the event loop's schedule.
    with TestClient(main.app):
        deadline = time.time() + 5.0
        while stale_gallery.exists() and time.time() < deadline:
            time.sleep(0.05)
    assert not stale_gallery.exists(), "the startup pass must sweep a stale gallery"
    assert staff_store.exists(), "staff stores persist by design"
