"""The calibration sweep, proven on synthetic embedding spaces with known truth.

Two well-separated clusters must yield a pack whose threshold sits between
the distributions; an unseparable space must be REFUSED, not thresholded.
"""

import numpy as np

from eval.sweep import (
    collect_observations,
    kept_faces_with_embeddings,
    pair_cosines,
    propose_pack,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _cluster(center, n, spread, rng):
    """n unit vectors near a unit center — one person's re-sightings."""
    return [_unit(center + rng.normal(0, spread, size=center.shape)) for _ in range(n)]


def _labels(frames):
    return {"format": "heco-labels/1", "set": {"id": "s1"}, "frames": frames}


def test_the_join_follows_seq_iou_and_kept_order():
    """The same three joins the scorecard verified on real footage: seq picks
    the picture, IoU picks the face, kept-order binds the embedding — and a
    gated face contributes nothing even when an embedding count would fit."""
    golden = [{
        "seq": 7,
        "embeddings": [[1.0, 0.0], [0.0, 1.0]],
        "faces": {"faces": [
            {"gate": "kept", "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
            {"gate": "frontality", "box": {"x": 50, "y": 0, "w": 10, "h": 10}},
            {"gate": "kept", "box": {"x": 100, "y": 0, "w": 10, "h": 10}},
        ]},
    }]
    pairs = kept_faces_with_embeddings(golden[0])
    assert [p[0]["box"]["x"] for p in pairs] == [0, 100], "gated face skipped, order held"

    labels = _labels([{
        "seq": 7,
        "boxes": [
            {"kind": "face", "identity": "A", "x": 1, "y": 1, "w": 10, "h": 10},
            {"kind": "face", "identity": "B", "x": 99, "y": 0, "w": 10, "h": 10},
            {"kind": "face", "x": 200, "y": 0, "w": 10, "h": 10},  # anonymous — no side
        ],
    }])
    obs = collect_observations(labels, golden)
    assert set(obs) == {"A", "B"}
    assert np.allclose(obs["A"][0], [1.0, 0.0])
    assert np.allclose(obs["B"][0], [0.0, 1.0])


def test_separated_clusters_yield_a_threshold_between_the_distributions():
    rng = np.random.default_rng(7)
    dim = 32
    centers = [_unit(rng.normal(size=dim)) for _ in range(6)]
    obs = {f"id{i}": _cluster(c, 8, 0.05, rng) for i, c in enumerate(centers)}

    genuine, impostor = pair_cosines(obs)
    assert genuine.min() > impostor.max(), "the synthetic truth: fully separated"

    pack = propose_pack(obs, "synthetic-32", calibrated_at="2026-08-14")
    assert pack["refusal" if pack["proposal"] is None else "proposal"] is not None
    p = pack["proposal"]
    assert impostor.max() < p["threshold"] <= genuine.min() + 1e-6, (
        f"threshold {p['threshold']} must sit in the gap "
        f"({impostor.max():.3f}, {genuine.min():.3f}]"
    )
    assert p["atProposal"]["missRate"] == 0.0
    assert p["atProposal"]["falseMatchRate"] == 0.0
    assert p["healMinCosine"] > impostor.max(), "heal floor clears the worst impostor"


def test_an_unseparable_space_is_refused_not_thresholded():
    """The doc-06 doctrine as code: overlap beyond repair is an acquisition
    problem, and a sweep that silently picked the least-bad threshold would
    launder it into a calibration."""
    rng = np.random.default_rng(3)
    dim = 32
    base = _unit(rng.normal(size=dim))
    # Everyone looks the same: identity clusters wider than their separation.
    obs = {f"id{i}": _cluster(base, 8, 0.4, rng) for i in range(6)}
    pack = propose_pack(obs, "mush-32", calibrated_at="2026-08-14")
    assert pack["proposal"] is None
    assert "acquisition problem" in pack["refusal"]


def test_too_few_pairs_is_an_answer_about_labelling_not_a_guess():
    obs = {"A": [_unit([1, 0])], "B": [_unit([0, 1])]}
    pack = propose_pack(obs, "sparse", calibrated_at="2026-08-14")
    assert pack["proposal"] is None
    assert "label more" in pack["refusal"]
