"""M1 — multiple templates per guest, proved on tonight's corridor geometry.

Why this file exists.  Before M1 a guest was represented forever by the FIRST
view of them and every later sighting was compared against that one vector.
The corridor bench (ground truth: ONE man walking) produced THREE gallery
identities whose views were mutually **0.296 / 0.307 / 0.347** against a 0.363
threshold — every pair a near miss, the closest by 0.016.  These tests rebuild
that exact geometry from its Gram matrix and show what the template policy does
to it, with the threshold left at its documented 0.363 throughout.

The synthetic vectors here are NOT gaussian blobs like ``test_match.py``'s;
they are constructed so their cosines are exact, because the whole question is
what happens within a few hundredths of the threshold.
"""

import numpy as np
import pytest
from app import config, gallery, main, staff, store
from fastapi.testclient import TestClient

DIM = 128
THRESHOLD = config.DEFAULT_THRESHOLD  # 0.363 — never moved by these tests

#: Tonight's measured pairwise cosines between the three gallery identities the
#: single man produced: A-B 0.347 (the closest, missing the threshold by 0.016),
#: B-C 0.307, A-C 0.296 (the widest pose separation, so the path runs A→B→C).
TONIGHT_AB, TONIGHT_BC, TONIGHT_AC = 0.347, 0.307, 0.296


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise to unit length (the store does this too; be explicit here)."""
    return v / np.linalg.norm(v)


def _json(v: np.ndarray) -> list[float]:
    """A vector as the JSON list the /match endpoint takes."""
    return [float(x) for x in v]


def tonight_views() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three unit vectors with EXACTLY tonight's measured pairwise cosines.

    Built by Cholesky of the Gram matrix, so the geometry is the measurement
    rather than an approximation of it.  The matrix is positive-definite
    (eigenvalues 0.65 / 0.71 / 1.63), i.e. tonight's numbers are realisable —
    which is itself worth asserting, since an unrealisable Gram would mean the
    reported cosines could not all have come from one measurement.
    """
    g = np.array(
        [
            [1.0, TONIGHT_AB, TONIGHT_AC],
            [TONIGHT_AB, 1.0, TONIGHT_BC],
            [TONIGHT_AC, TONIGHT_BC, 1.0],
        ]
    )
    assert (np.linalg.eigvalsh(g) > 0).all(), "tonight's cosines must be realisable"
    v = np.zeros((3, DIM))
    v[:, :3] = np.linalg.cholesky(g)
    return v[0], v[1], v[2]


def bridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A view halfway between two poses — the frame captured mid-turn.

    A corridor crossing is not three snapshots; it is a continuous pose sweep,
    and the intermediate frames are what let a profile view be chained back to
    a frontal one.  The normalised bisector of two views 70 deg apart sits at
    cosine ~0.82 from BOTH, comfortably inside the threshold either way.
    """
    return _unit(x + y)


def arc(steps: int, degrees: float) -> list[np.ndarray]:
    """A pose sweep: ``steps`` unit vectors ``degrees`` apart on a great circle.

    Models a guest turning steadily past the camera.  At 35 deg per step,
    adjacent frames sit at cosine 0.819 (a solid match) and frames two apart at
    0.342 (a miss) — so the sweep only holds together if the gallery keeps the
    intermediate views.
    """
    e1, e2 = np.zeros(DIM), np.zeros(DIM)
    e1[0], e2[1] = 1.0, 1.0
    return [
        np.cos(np.radians(degrees * i)) * e1 + np.sin(np.radians(degrees * i)) * e2
        for i in range(steps)
    ]


def orthogonal_arc(steps: int, degrees: float, cross: float) -> list[np.ndarray]:
    """A second person's sweep, tilted to sit ``cross`` from the first's views.

    Same internal geometry as :func:`arc` (cosines are preserved by the tilt),
    but every cross-person cosine is at most ``cross``.  With ``cross=0.30``
    this is a genuine near-miss impostor pair — 0.063 below the threshold,
    comparable to the 0.016 by which tonight's same-person views missed it —
    which is the pair a multi-template gallery is most likely to merge by
    mistake.
    """
    e3, e4 = np.zeros(DIM), np.zeros(DIM)
    e3[2], e4[3] = 1.0, 1.0
    own = [
        np.cos(np.radians(degrees * i)) * e3 + np.sin(np.radians(degrees * i)) * e4
        for i in range(steps)
    ]
    mine = arc(steps, degrees)
    tilt = np.sqrt(1.0 - cross * cross)
    return [_unit(cross * a + tilt * o) for a, o in zip(mine, own, strict=True)]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with gallery + staff data redirected to a temp directory."""
    monkeypatch.setenv("HECO_MATCH_DATA_DIR", str(tmp_path))
    store.close_all_stores()
    with TestClient(main.app) as c:
        yield c
    store.close_all_stores()


def _match(client, run_id: str, v: np.ndarray, quality: float = 70.0) -> dict:
    """POST /match with a 70 px face — the POC close-zone default."""
    res = client.post(
        "/match", json={"runId": run_id, "embedding": _json(v), "quality": quality}
    )
    assert res.status_code == 200, res.text
    return res.json()


def _walk(client, run_id: str, views, qualities=None) -> list[dict]:
    """Feed a sequence of sightings in order and return every verdict."""
    if qualities is None:
        qualities = [70.0] * len(views)
    return [_match(client, run_id, v, q) for v, q in zip(views, qualities, strict=True)]


def _templates(tmp_path, run_id: str, key: str) -> list[float | None]:
    """Read one identity's stored template qualities straight from its gallery."""
    s = store.open_store(gallery.db_path(tmp_path, run_id))
    with s.reading():
        return s.qualities_for(key)


# ------------------------------------------------------------------ mechanics


def test_search_is_per_identity_best(tmp_path):
    """VectorStore.search already returns an identity's BEST template score.

    The M1 brief said to verify this before rebuilding it.  ``search`` is an
    argmax over ROWS, and max-over-rows equals max-over-(per-identity maxima),
    so the moment several rows share a key the returned cosine IS that
    identity's best template.  Nothing to rebuild; this test pins it.
    """
    views = arc(4, 35.0)
    with store.VectorStore(tmp_path / "s.db") as s:
        for v in views:
            s.add("p00001", v, quality=70.0)
        s.add("p00002", orthogonal_arc(1, 35.0, 0.30)[0], quality=70.0)
        identities = s.keys()  # VectorStore.keys(), not a mapping — do not inline
        for probe in views:
            hit = s.search(probe)
            per_identity = {
                k: max(float(probe @ x) for x in s.vectors_for(k)) for k in identities
            }
            best_key = max(per_identity, key=per_identity.get)
            assert hit.key == best_key
            assert hit.cosine == pytest.approx(per_identity[best_key], abs=1e-6)


def test_runner_up_ignores_the_named_key(tmp_path):
    """runner_up reports the nearest RIVAL identity, never the one asked about."""
    with store.VectorStore(tmp_path / "s.db") as s:
        for v in arc(3, 35.0):
            s.add("p00001", v, quality=70.0)
        assert s.runner_up(arc(1, 35.0)[0], "p00001") is None  # no rival exists
        s.add("p00002", orthogonal_arc(1, 35.0, 0.30)[0], quality=70.0)
        rival = s.runner_up(arc(1, 35.0)[0], "p00001")
        assert rival is not None and rival.key == "p00002"
        assert rival.cosine == pytest.approx(0.30, abs=1e-5)


# ------------------------------------------------- (a) match ANY stored view


def test_probe_matches_any_template_of_an_identity(client, tmp_path):
    """A probe near ANY held view matches that identity — not just the first.

    The sweep leaves the guest holding several poses.  A probe taken at the far
    end of the sweep is BELOW the threshold against the founding template, so a
    match can only have come from a later one: that is the whole M1 mechanism,
    asserted rather than assumed.
    """
    views = arc(4, 35.0)
    verdicts = _walk(client, "run-any", views)
    key = verdicts[0]["personKey"]
    assert {v["personKey"] for v in verdicts} == {key}
    assert verdicts[-1]["templateN"] == 4

    far = _unit(views[-1] + 0.05 * views[-2])  # a fresh capture near the last pose
    assert float(far @ views[0]) < THRESHOLD, "must not be matchable via view 0"
    out = _match(client, "run-any", far)
    assert out["isNew"] is False
    assert out["personKey"] == key
    assert out["galleryN"] == 1


# ------------------------------------------------------- (b) the cap and eviction


def test_prune_keeps_the_best_qualities_and_drops_the_worst(tmp_path):
    """prune_to_cap evicts by lowest capture quality, never the best view."""
    with store.VectorStore(tmp_path / "s.db") as s:
        for v, q in zip(arc(6, 35.0), [90.0, 60.0, 95.0, 88.0, 92.0, 85.0], strict=True):
            s.add("p00001", v, quality=q)
        assert s.count_for("p00001") == 6
        assert s.prune_to_cap("p00001", 5) == 1
        assert s.qualities_for("p00001") == [95.0, 92.0, 90.0, 88.0, 85.0]
        assert s.count_for("p00001") == 5  # in-memory index agrees with SQLite


def test_prune_evicts_quality_less_templates_first(tmp_path):
    """A template with no recorded quality is the first to go, then the lowest."""
    with store.VectorStore(tmp_path / "s.db") as s:
        for v, q in zip(arc(4, 35.0), [90.0, None, 55.0, 80.0], strict=True):
            s.add("p00001", v, quality=q)
        assert s.prune_to_cap("p00001", 2) == 2
        assert s.qualities_for("p00001") == [90.0, 80.0]


def test_gallery_respects_the_cap_and_keeps_the_best_views(client, tmp_path, monkeypatch):
    """Over a long sweep the guest holds 5 views — the best-quality 5."""
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    qualities = [90.0, 60.0, 95.0, 88.0, 92.0, 85.0]
    verdicts = _walk(client, "run-cap", arc(6, 35.0), qualities)
    key = verdicts[0]["personKey"]
    assert {v["personKey"] for v in verdicts} == {key}, "one sweep, one guest"
    assert verdicts[-1]["galleryN"] == 1
    assert verdicts[-1]["templateN"] == 5
    held = _templates(tmp_path, "run-cap", key)
    assert held == [95.0, 92.0, 90.0, 88.0, 85.0]
    assert 60.0 not in held, "the worst view was evicted"
    assert max(held) == 95.0, "the best view was never evicted"


def test_a_worse_view_does_not_displace_a_better_one_at_cap(client, tmp_path, monkeypatch):
    """At the cap, a low-quality sighting is refused rather than churned in."""
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "3")
    views = arc(5, 35.0)
    _walk(client, "run-worse", views[:3], [90.0, 91.0, 92.0])
    out = _match(client, "run-worse", views[3], quality=40.0)
    assert out["templateAdded"] is False
    assert _templates(tmp_path, "run-worse", out["personKey"]) == [92.0, 91.0, 90.0]


def test_eviction_survives_a_reopen_and_leaves_the_index_honest(tmp_path):
    """Evicted templates are gone from SQLite AND from the resident scan matrix.

    ``prune_to_cap`` deletes particular rows and writes that through to the
    in-memory index rather than dropping it.  The failure mode if the
    bookkeeping is wrong is silent and expensive: RAM would keep answering with
    a vector the database no longer has, so a guest would keep matching a view
    that was evicted — and a reopen (or any rollback) would change the count.
    """
    path = tmp_path / "s.db"
    views = arc(6, 35.0)
    with store.VectorStore(path) as s:
        for v, q in zip(views, [90.0, 60.0, 95.0, 88.0, 92.0, 85.0], strict=True):
            s.add("p00001", v, quality=q)
        s.prune_to_cap("p00001", 5)
        live = s.search(views[1])  # the evicted pose
        assert live.cosine < 0.99, "the evicted view must no longer answer as itself"
        assert len(s.vectors_for("p00001")) == 5

    with store.VectorStore(path) as s:
        assert s.count_for("p00001") == 5
        assert s.qualities_for("p00001") == [95.0, 92.0, 90.0, 88.0, 85.0]
        assert s.search(views[1]).cosine == pytest.approx(live.cosine, abs=1e-6)


def test_corrections_still_work_on_a_multi_template_identity(tmp_path):
    """merge / split / remove keep working once identities hold several views.

    These are the operator's only levers over the invoice figure, and M1 changed
    the row bookkeeping underneath them.
    """
    path = tmp_path / "s.db"
    with store.VectorStore(path) as s:
        for v in arc(3, 35.0):
            s.add("p00001", v, quality=70.0)
        for v in orthogonal_arc(3, 35.0, 0.30):
            s.add("p00002", v, quality=70.0)
        s.prune_to_cap("p00001", 2)
        assert s.distinct_count() == 2

        # 2 + 3 = 5 templates on the survivor, ABOVE the cap of 2 used above.
        # That is correct at this layer: VectorStore.merge is a pure op and does
        # not hold policy.  gallery.merge applies the cap — see
        # test_a_merge_is_pruned_back_to_the_cap, which is where that is pinned.
        assert s.merge("p00001", "p00002") is True
        assert s.count_for("p00001") == 5 and s.distinct_count() == 1

        removed = s.remove("p00001")
        assert len(removed) == 5 and s.distinct_count() == 0
        assert s.search(arc(1, 35.0)[0]) is None


# --------------------------------------------- (c) tonight's corridor measurement


def test_tonight_corridor_walk_before_and_after(client, monkeypatch):
    """THE headline: one man's crossing, 3 identities before M1, 1 after.

    The walk is A → B → C with the mid-turn frames the camera actually sees.
    Its three anchor views carry tonight's exact measured cosines, so cap=1
    reproduces the measurement (three identities for one man) and cap=5 — the
    same vectors, the same 0.363 threshold — resolves it to one guest.
    """
    a, b, c = tonight_views()
    walk = [a, bridge(a, b), b, bridge(b, c), c]

    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "1")
    before = _walk(client, "run-before", walk)
    assert before[-1]["galleryN"] == 3, "pre-M1: one man counted three times"
    assert sum(v["isNew"] for v in before) == 3

    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    after = _walk(client, "run-after", walk)
    assert after[-1]["galleryN"] == 1, "post-M1: one man, one guest"
    assert sum(v["isNew"] for v in after) == 1
    assert len({v["personKey"] for v in after}) == 1
    assert after[-1]["templateN"] == 5

    # Every anchor pair is still a mutual near miss — nothing was merged by
    # relaxing what counts as a match.
    for x, y, expected in ((a, b, TONIGHT_AB), (b, c, TONIGHT_BC), (a, c, TONIGHT_AC)):
        assert float(x @ y) == pytest.approx(expected, abs=1e-6)
        assert float(x @ y) < THRESHOLD


def test_three_anchor_views_alone_still_do_not_collapse(client, monkeypatch):
    """DISPROOF: multi-template alone cannot bridge a gap nothing spans.

    Fed ONLY the three anchor views — no mid-turn frames — the gallery still
    reports three people, and it is right to.  The first view is stored; the
    second misses it by 0.056; there is no third party to compare against.  No
    template policy can join two vectors at cosine 0.307 while the threshold is
    0.363 — anything that did would BE a lower threshold, which is M2 and is
    blocked on impostor data.

    So M1's win on tonight's run is conditional on the crossing having supplied
    intermediate poses that passed the quality gate.  This test is the honest
    boundary of the claim, kept executable so nobody later "fixes" it by
    quietly relaxing the threshold.
    """
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    a, b, c = tonight_views()
    verdicts = _walk(client, "run-anchors", [a, b, c])
    assert verdicts[-1]["galleryN"] == 3
    assert [v["isNew"] for v in verdicts] == [True, True, True]
    assert verdicts[1]["cosine"] == pytest.approx(TONIGHT_AB, abs=1e-6)
    assert verdicts[2]["cosine"] == pytest.approx(TONIGHT_BC, abs=1e-6)


# -------------------------------------------------- (d) different people stay apart


def test_two_near_miss_people_do_not_merge(client, monkeypatch):
    """Two people 0.30 apart keep separate keys even at 5 templates each.

    The risk multi-template introduces is over-MERGING: more templates mean
    more chances for a rival's view to land inside somebody's accept region.
    Both people sweep the full pose range, interleaved as they would be walking
    together, and every cross-person cosine tops out at 0.30 — a 0.063 miss,
    tighter than most real impostor pairs.  Two guests in, two guests out.
    """
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    mine, theirs = arc(6, 35.0), orthogonal_arc(6, 35.0, 0.30)
    cross = max(float(x @ y) for x in mine for y in theirs)
    assert cross == pytest.approx(0.30, abs=1e-6) and cross < THRESHOLD

    keys = {"mine": set(), "theirs": set()}
    for x, y in zip(mine, theirs, strict=True):
        keys["mine"].add(_match(client, "run-two", x)["personKey"])
        keys["theirs"].add(_match(client, "run-two", y)["personKey"])
    assert len(keys["mine"]) == 1 and len(keys["theirs"]) == 1
    assert keys["mine"].isdisjoint(keys["theirs"])
    assert _match(client, "run-two", mine[0])["galleryN"] == 2


def test_ambiguous_sighting_is_not_enrolled(client, monkeypatch):
    """A view sitting between two people is refused as a template (no bridges).

    Storing it would attach a point that half-belongs to somebody else, and the
    next probe near it would collapse two paying guests into one — an invisible
    under-count, the failure mode nobody can audit.
    """
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    monkeypatch.setenv("HECO_MATCH_TEMPLATE_MARGIN", "0.05")
    e1, e2 = np.zeros(DIM), np.zeros(DIM)
    e1[0], e2[1] = 1.0, 1.0
    first = _match(client, "run-amb", e1)
    second = _match(client, "run-amb", e2)
    assert first["personKey"] != second["personKey"]

    midway = _unit(e1 + e2)  # 0.7071 from BOTH — maximally ambiguous
    out = _match(client, "run-amb", midway)
    assert out["isNew"] is False, "it does match somebody"
    assert out["templateAdded"] is False, "but it must not become a template"
    assert out["templateN"] == 1


# ---------------------------------------------------------- the enrolment gates


def test_near_duplicate_sightings_do_not_consume_the_cap(client, monkeypatch):
    """Consecutive frames of a standing guest add nothing and are not stored.

    Without this rule five near-identical frames would fill the cap in the
    first second of a crossing and the guest would end up NARROWER than under
    the old single-template gallery.
    """
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    rng = np.random.default_rng(7)
    base = _unit(rng.normal(size=DIM))
    first = _match(client, "run-dup", base)
    for _ in range(8):
        jitter = _unit(base + rng.normal(scale=0.02, size=DIM))
        assert float(jitter @ base) > config.DEFAULT_TEMPLATE_MAX_COSINE
        out = _match(client, "run-dup", jitter)
        assert out["personKey"] == first["personKey"]
        assert out["templateAdded"] is False
        assert out["templateN"] == 1


def test_a_barely_matching_sighting_is_not_enrolled(client, monkeypatch):
    """A match AT the threshold counts as a re-sighting but never as a template.

    The drift brake.  A bare-minimum match is the least certain evidence in the
    system; promoting it to a template would let an identity annex the region
    around a point we are not sure of, and the error would compound with every
    template built on it.
    """
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    e1, e2 = np.zeros(DIM), np.zeros(DIM)
    e1[0], e2[1] = 1.0, 1.0

    def at(cos: float) -> np.ndarray:
        return cos * e1 + np.sqrt(1.0 - cos * cos) * e2

    _match(client, "run-conf", e1)
    barely = _match(client, "run-conf", at(THRESHOLD + 0.001))
    assert barely["isNew"] is False, "it IS a re-sighting — the threshold did not move"
    assert barely["templateAdded"] is False, "but it is too uncertain to learn from"

    # Comfortably past the confidence floor rather than exactly on it: the store
    # keeps float32 vectors, so a cosine constructed at the boundary lands a few
    # 1e-8 either side of it and the assertion would be about rounding.
    clear = THRESHOLD + config.DEFAULT_TEMPLATE_CONFIDENCE + 0.02
    confident = _match(client, "run-conf", at(clear))
    assert confident["isNew"] is False and confident["templateAdded"] is True


def test_cap_of_one_is_exactly_the_old_behaviour(client, monkeypatch):
    """TEMPLATES_PER_PERSON=1 restores the pre-M1 gallery — the off switch."""
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "1")
    verdicts = _walk(client, "run-off", arc(4, 35.0))
    assert all(v["templateAdded"] is False for v in verdicts)
    assert all(v["templateN"] == 1 for v in verdicts)
    # 35 deg steps: every second frame is a 0.342 miss against its predecessor,
    # so the sweep fragments exactly as the corridor bench did.
    assert verdicts[-1]["galleryN"] == 2


def test_health_reports_the_template_policy(client):
    """The bench must be able to read back which policy produced a count."""
    body = client.get("/health").json()
    assert body["threshold"] == pytest.approx(config.DEFAULT_THRESHOLD)
    assert body["templatesPerPerson"] == config.DEFAULT_TEMPLATES_PER_PERSON
    assert body["templateConfidence"] == pytest.approx(config.DEFAULT_TEMPLATE_CONFIDENCE)
    assert body["templateMaxCosine"] == pytest.approx(config.DEFAULT_TEMPLATE_MAX_COSINE)


# ------------------------------------------------------------- staff untouched


def test_staff_store_never_grows_from_crossings(client, tmp_path, monkeypatch):
    """Requirement 5: the staff store behaves exactly as it did before M1.

    Staff templates come from the operator-supervised walk-through only.  If an
    unsupervised crossing could enrol into the staff store, a mis-tagged guest
    would permanently poison a roster member — and staff persist across every
    event at the site, so the damage would outlive the run.
    """
    monkeypatch.setenv("HECO_MATCH_TEMPLATES_PER_PERSON", "5")
    views = arc(4, 35.0)
    res = client.post(
        "/staff/enrol",
        json={
            "siteId": "site-1",
            "staffId": "s-anita",
            "samples": [{"embedding": _json(v), "quality": 70.0} for v in views[:2]],
        },
    )
    assert res.status_code == 200 and res.json()["sampleCount"] == 2

    # Cross the camera repeatedly at the poses she enrolled from: the staff hit
    # must keep recognising her from exactly the two enrolled views and no more.
    for v in (views[0], views[1], _unit(views[1] + 0.1 * views[0])):
        out = client.post(
            "/match",
            json={"runId": "run-staff", "embedding": _json(v), "quality": 70.0,
                  "siteId": "site-1"},
        ).json()
        assert out["isStaff"] is True
        assert out["staffId"] == "s-anita"
        assert out["templateAdded"] is False
        assert out["galleryN"] == 0, "staff never enter the guest count"

    s = store.open_store(staff.db_path(tmp_path, "site-1"))
    with s.reading():
        assert s.count_for("s-anita") == 2, "crossings must not enrol into staff"

    again = client.post(
        "/staff/enrol",
        json={
            "siteId": "site-1",
            "staffId": "s-anita",
            "samples": [{"embedding": _json(v), "quality": 70.0} for v in views[:3]],
        },
    )
    assert again.json()["sampleCount"] == 3, "re-enrolment still supersedes"


# ------------------------------------------------ the cap applies to merges too


def test_a_merge_is_pruned_back_to_the_cap(tmp_path):
    """A merge cannot smuggle an identity past HECO_MATCH_TEMPLATES_PER_PERSON.

    The survivor inherits BOTH identities' views, so merge is the one path that
    can carry a key over its cap.  Left unpruned it compounds: the verifier
    measured six single-template people merged one after another leaving six
    views against a cap of five, and an identity whose accept region grows
    without bound is what causes the NEXT silent merge — an under-count, which
    is invisible in an invoice figure in a way an over-count never is.
    """
    run = "run-merge-cap"
    db = gallery.db_path(tmp_path, run)
    s = store.open_store(db)
    with s.transaction():
        for v in arc(4, 35.0):
            s.add("p00001", v, quality=90.0)
        for v in orthogonal_arc(4, 35.0, 0.30):
            s.add("p00002", v, quality=60.0)
    store.close_all_stores()

    merged, n = gallery.merge(tmp_path, run, "p00001", "p00002", templates_per_person=5)
    assert merged is True and n == 1

    s = store.open_store(db)
    with s.reading():
        assert s.count_for("p00001") == 5, "8 inherited views must be pruned to the cap"


def test_a_merge_keeps_the_views_that_caused_the_split(tmp_path):
    """Eviction after a merge must be by redundancy, NOT by capture quality.

    This is the whole reason prune_redundant exists.  An operator merges two
    identities precisely because the pipeline could not join them — so the
    merged-in views are the distinctive ones, and they are frequently the WORSE
    photographs (a face at the edge of the lens, half-turned).  Prune on quality
    and every one of them is evicted, the identity is left exactly as narrow as
    it was before, the same pose mints a new key at the next crossing, and the
    operator merges the same guest again tomorrow.

    Here p00002's views are deliberately the low-quality ones.  A quality-based
    prune keeps five of p00001's six and drops all of p00002's; the redundancy
    rule must keep p00002's views alive, because they are what the merge was
    asserting.
    """
    run = "run-merge-keeps"
    db = gallery.db_path(tmp_path, run)
    good = arc(6, 12.0)  # six near-duplicates: adjacent cosine 0.978
    distinct = orthogonal_arc(2, 35.0, 0.30)  # the far-apart, badly-shot views
    s = store.open_store(db)
    with s.transaction():
        for v in good:
            s.add("p00001", v, quality=95.0)
        for v in distinct:
            s.add("p00002", v, quality=55.0)  # worst quality in the gallery
    store.close_all_stores()

    gallery.merge(tmp_path, run, "p00001", "p00002", templates_per_person=5)

    s = store.open_store(db)
    with s.reading():
        kept = s.vectors_for("p00001")
        assert len(kept) == 5
        survivors = [
            max(float(np.dot(_unit(k), _unit(d))) for k in kept) for d in distinct
        ]
    assert all(c > 0.999 for c in survivors), (
        "the merged-in views were evicted — a quality prune would do exactly "
        f"this, and it puts the operator on a re-merge treadmill (got {survivors})"
    )


def test_a_merge_without_the_cap_is_exactly_the_old_behaviour(tmp_path):
    """templates_per_person 0 or 1 prunes nothing — pre-M1 semantics, unchanged.

    A caller that has not opted into multi-template must get what it always got,
    so the cap can be rolled back in the field by one env var without a merge
    quietly behaving differently from the rest of the service.
    """
    run = "run-merge-uncapped"
    db = gallery.db_path(tmp_path, run)
    s = store.open_store(db)
    with s.transaction():
        for v in arc(3, 35.0):
            s.add("p00001", v, quality=90.0)
        for v in orthogonal_arc(3, 35.0, 0.30):
            s.add("p00002", v, quality=90.0)
    store.close_all_stores()

    merged, n = gallery.merge(tmp_path, run, "p00001", "p00002")
    assert merged is True and n == 1

    s = store.open_store(db)
    with s.reading():
        assert s.count_for("p00001") == 6, "no cap passed, so nothing may be evicted"


def test_prune_redundant_evicts_the_twin_and_keeps_the_anchor(tmp_path):
    """The eviction rule itself: drop the least distinctive, keep the founding view.

    Three views — two of them near-identical, one far away.  The near-duplicate
    pair carries the redundancy, so one of them must go and the distant view must
    survive.  Ties break towards keeping the OLDEST row, so the anchor the
    identity was minted on is the last thing to be evicted.
    """
    path = tmp_path / "s.db"
    with store.VectorStore(path) as s:
        anchor, twin = arc(2, 2.0)  # cosine 0.999 — the redundant pair
        far = arc(3, 60.0)[2]  # 120 deg from the anchor
        s.add("p00001", anchor, quality=70.0)
        s.add("p00001", twin, quality=70.0)
        s.add("p00001", far, quality=70.0)

        assert s.prune_redundant("p00001", 2) == 1
        kept = s.vectors_for("p00001")
        assert len(kept) == 2
        assert max(float(np.dot(_unit(k), _unit(far))) for k in kept) > 0.999, (
            "the distinctive view was evicted — that is the quality rule's bug"
        )
        assert max(float(np.dot(_unit(k), _unit(anchor))) for k in kept) > 0.999, (
            "the founding view must outlive its twin"
        )


def test_prune_redundant_is_a_no_op_within_the_cap(tmp_path):
    """Nothing is evicted when the identity already fits — no churn, no surprise."""
    path = tmp_path / "s.db"
    with store.VectorStore(path) as s:
        for v in arc(3, 35.0):
            s.add("p00001", v, quality=70.0)
        assert s.prune_redundant("p00001", 5) == 0
        assert s.count_for("p00001") == 3
        with pytest.raises(ValueError):
            s.prune_redundant("p00001", 0)
