"""Contested torso reads (match 0.21.0): a read taken with another person's
box across its band is stored flagged, and the review keeps to clean reads
while the guest has any."""

import numpy as np
from app.gallery import torso_reads
from app.store import SightingEvidence, VectorStore


def _t(bins):
    v = np.zeros(64, dtype=np.float32)
    for b, w in bins.items():
        v[b] = w
    return v / v.sum()


def test_the_flag_rides_the_body_row_and_comes_back_on_the_evidence(tmp_path):
    """add_body_sighting(contested=True) is read back as SightingEvidence.contested."""
    s = VectorStore(tmp_path / "g.db")
    with s.transaction():
        s.add("p00001", np.eye(128, dtype=np.float32)[0])
        s.add_body_sighting("p00001", 1200.0, 400.0, 1500.0, 2160, 100.0, appearance=_t({1: 1.0}))
        s.add_body_sighting(
            "p00001", 1200.0, 400.0, 1500.0, 2160, 100.0, appearance=_t({2: 1.0}), contested=True,
        )
    with s.reading():
        ev = s.sighting_evidence()
    assert [r.contested for r in ev] == [False, True]
    # A row from before the column exists reads as clean.
    assert SightingEvidence("p", "2026-10-01T00:00:00Z", None).contested is False


def test_the_review_prefers_clean_reads_and_falls_back_to_contested_ones():
    """Clean reads define the guest's clothes; contested ones only when there is nothing else."""
    clean = [SightingEvidence("a", f"2026-10-01T00:00:{i:02d}Z", _t({1: 1.0})) for i in range(3)]
    mixed = [
        SightingEvidence("a", f"2026-10-01T00:01:{i:02d}Z", _t({9: 1.0}), contested=True)
        for i in range(5)
    ]
    reads = torso_reads(clean + mixed, {})["a"]
    assert reads.n == 3, "the five contested reads do not count while clean ones exist"
    assert all(v[1] == 1.0 for v in reads.vectors)
    only_mixed = torso_reads(mixed, {})["a"]
    assert only_mixed.n == 5, "with no clean read, the contested ones still speak"
