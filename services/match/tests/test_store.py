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
