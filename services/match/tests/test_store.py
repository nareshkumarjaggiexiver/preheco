"""Unit tests for VectorStore: add/search/threshold, keys, merge/split/remove.

Synthetic identities are clustered gaussians on the unit sphere (the same
geometry SFace promises, without weights): a random unit centroid per person,
each sighting the centroid plus small noise, re-normalised. No model, no
network — just the store and numpy.
"""

import numpy as np
import pytest
from app.store import VectorStore, as_unit

RNG = np.random.default_rng(7)


def centroid() -> np.ndarray:
    """A synthetic identity: a random point on the 128-d unit sphere."""
    return RNG.normal(size=128)


def sighting(c: np.ndarray, noise: float = 0.05) -> list[float]:
    """One face capture: the centroid plus small gaussian noise, unit-length."""
    return as_unit(c + RNG.normal(scale=noise, size=128)).tolist()


def test_as_unit_rejects_degenerate():
    """Too-short and zero vectors have no direction to compare."""
    with pytest.raises(ValueError):
        as_unit([1.0, 2.0])
    with pytest.raises(ValueError):
        as_unit([0.0] * 128)


def test_empty_search_is_none(tmp_path):
    """A fresh store has nothing to match against."""
    with VectorStore(tmp_path / "s.db") as store:
        assert store.search([0.1] * 128) is None
        assert store.distinct_count() == 0


def test_add_and_nearest(tmp_path):
    """Two clusters: a query near one lands on that key, high cosine."""
    a, b = centroid(), centroid()
    with VectorStore(tmp_path / "s.db") as store:
        ka = store.add_auto(sighting(a))
        kb = store.add_auto(sighting(b))
        assert ka != kb
        assert store.distinct_count() == 2
        near_a = store.search(sighting(a))
        assert near_a.key == ka
        assert near_a.cosine > 0.9


def test_threshold_boundary(tmp_path):
    """A query at an exact cosine sits either side of a chosen threshold."""
    e1 = np.zeros(128)
    e1[0] = 1.0
    e2 = np.zeros(128)
    e2[1] = 1.0

    def at_cosine(c: float) -> list[float]:
        import math

        return (c * e1 + math.sqrt(1.0 - c * c) * e2).tolist()

    with VectorStore(tmp_path / "s.db") as store:
        store.add("p1", e1.tolist())
        assert store.search(at_cosine(0.4)).cosine == pytest.approx(0.4, abs=1e-5)
        assert store.search(at_cosine(0.3)).cosine == pytest.approx(0.3, abs=1e-5)


def test_mint_key_is_monotonic_across_removal(tmp_path):
    """Keys never repeat even after a removal shrinks the distinct count."""
    with VectorStore(tmp_path / "s.db") as store:
        k1 = store.add_auto(sighting(centroid()))
        k2 = store.add_auto(sighting(centroid()))
        assert (k1, k2) == ("p00001", "p00002")
        store.remove(k2)
        k3 = store.add_auto(sighting(centroid()))
        assert k3 == "p00003"  # not a reused p00002


def test_merge_folds_and_decrements(tmp_path):
    """Merge re-points one key's templates onto another; count falls by one."""
    c = centroid()
    with VectorStore(tmp_path / "s.db") as store:
        keep = store.add_auto(sighting(c))
        drop = store.add_auto(sighting(c))
        assert store.distinct_count() == 2
        assert store.merge(keep, drop) is True
        assert store.distinct_count() == 1
        assert store.count_for(keep) == 2  # both templates now under keep
        assert store.count_for(drop) == 0


def test_merge_refuses_unknown_and_self(tmp_path):
    """A merge with an unknown key, or a key onto itself, changes nothing."""
    with VectorStore(tmp_path / "s.db") as store:
        k = store.add_auto(sighting(centroid()))
        assert store.merge(k, "nope") is False
        assert store.merge(k, k) is False
        assert store.distinct_count() == 1


def test_split_blocks_a_later_merge(tmp_path):
    """A false-match split records a constraint merge() then honours."""
    c = centroid()
    with VectorStore(tmp_path / "s.db") as store:
        a = store.add_auto(sighting(c))
        b = store.add_auto(sighting(c))
        store.split(a, b)
        assert store.cannot_link(a, b) is True
        assert store.cannot_link(b, a) is True  # order-independent
        assert store.merge(a, b) is False  # constraint refuses the merge
        assert store.distinct_count() == 2


def test_remove_returns_templates_for_rehoming(tmp_path):
    """Remove hands back the raw rows so mark-staff can re-home them."""
    with VectorStore(tmp_path / "s.db") as store:
        k = store.add_auto(sighting(centroid()), quality=71.0, sub_canon=True)
        store.add(k, sighting(centroid()), quality=88.0)
        rows = store.remove(k)
        assert len(rows) == 2
        blob, quality, sub_canon = rows[0]
        assert isinstance(blob, bytes) and len(blob) == 128 * 4  # float32[128]
        assert quality == 71.0 and sub_canon == 1
        assert store.distinct_count() == 0


def test_persistence_across_reopen(tmp_path):
    """A store reopened from disk still answers with what was committed."""
    path = tmp_path / "s.db"
    c = centroid()
    with VectorStore(path) as store:
        k = store.add_auto(sighting(c))
    with VectorStore(path) as store:
        assert store.distinct_count() == 1
        assert store.search(sighting(c)).key == k


# ---------------------------------------------- regressions (2026-08-05 review)


@pytest.fixture(autouse=True)
def _no_store_leak():
    """Never let one test's cached connections reach the next."""
    yield
    from app.store import close_all_stores

    close_all_stores()


def test_open_store_is_one_connection_per_file(tmp_path):
    """The whole point of the cache: open once, not once per /match.

    Regression: every /match constructed a fresh VectorStore — sqlite3.connect
    + PRAGMA + the schema script + a SELECT of every row + a numpy rebuild from
    BLOBs, two or three times per face, dozens of times a second.
    """
    from app.store import close_store, open_store

    path = tmp_path / "s.db"
    first = open_store(path)
    assert open_store(path) is first
    close_store(path)
    assert open_store(path) is not first


def test_write_through_index_never_diverges_from_disk(tmp_path):
    """Whatever the in-memory scan matrix says, a reload from SQLite agrees."""
    from app.store import close_all_stores, open_store

    path = tmp_path / "s.db"
    store = open_store(path)
    cents = [centroid() for _ in range(6)]
    with store.transaction():
        keys = [store.add_auto(sighting(c)) for c in cents]
    with store.transaction():
        store.merge(keys[0], keys[1])  # rename in the index
        store.remove(keys[2])  # drop rows from the index
        store.add(keys[3], sighting(cents[3]))  # append to the index
        store.add_manual("operator saw one more")  # counted, never matched

    query = sighting(cents[4])  # ONE query vector, asked of both views
    live = (store.distinct_count(), store.keys(), store.manual_keys())
    hit_live = store.search(query)

    close_all_stores()  # force everything to come back off disk
    fresh = open_store(path)
    assert (fresh.distinct_count(), fresh.keys(), fresh.manual_keys()) == live
    hit_fresh = fresh.search(query)
    assert hit_fresh.key == hit_live.key
    assert hit_fresh.cosine == pytest.approx(hit_live.cosine)


def test_rolled_back_write_leaves_no_phantom_row_in_memory(tmp_path):
    """A failed transaction must not leave the index claiming rows SQLite lost."""
    from app.store import open_store

    store = open_store(tmp_path / "s.db")
    with store.transaction():
        store.add_auto(sighting(centroid()))
    assert store.distinct_count() == 1

    with pytest.raises(RuntimeError), store.transaction():
        store.add_auto(sighting(centroid()))
        raise RuntimeError("something failed after the write")

    assert store.distinct_count() == 1, "the rolled-back person must be gone from RAM too"
    assert len(store.keys()) == 1


def test_manual_entries_count_but_never_match(tmp_path):
    """An operator-attested person exists in the count and nowhere else."""
    from app.store import open_store

    store = open_store(tmp_path / "s.db")
    c = centroid()
    with store.transaction():
        real = store.add_auto(sighting(c))
        manual = store.add_manual("child behind the pillar")
    assert manual.startswith("m")
    assert store.distinct_count() == 2
    assert store.search(sighting(c)).key == real, "no vector, so never the nearest"


def test_concurrent_matches_do_not_corrupt_the_shared_store(tmp_path):
    """The cached store is shared by the request threadpool; it must hold up."""
    import threading

    from app import gallery

    cents = [centroid() for _ in range(24)]
    errors: list[BaseException] = []

    def worker(lo: int, hi: int) -> None:
        try:
            for i in range(lo, hi):
                gallery.match(
                    tmp_path, "run-t", sighting(cents[i]), 85.0,
                    threshold=0.9, canon_px=80.0,
                )
        except BaseException as e:  # noqa: BLE001 — surface it in the assertion
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i * 6, i * 6 + 6)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert gallery.count(tmp_path, "run-t") == 24


def test_reset_closes_the_store_before_deleting_the_file(tmp_path):
    """A gallery reset must really empty the gallery, not just unlink the file.

    An open connection to an unlinked inode keeps answering from a database
    nobody can see, so a run would silently resume a previous run's people.
    """
    from app import gallery

    c = centroid()
    kw = {"threshold": 0.9, "canon_px": 80.0}
    assert gallery.match(tmp_path, "r", sighting(c), 85.0, **kw).is_new is True
    assert gallery.match(tmp_path, "r", sighting(c), 85.0, **kw).is_new is False

    gallery.reset(tmp_path, "r")
    assert not gallery.db_path(tmp_path, "r").exists()
    assert gallery.count(tmp_path, "r") == 0

    after = gallery.match(tmp_path, "r", sighting(c), 85.0, **kw)
    assert after.is_new is True and after.gallery_n == 1
