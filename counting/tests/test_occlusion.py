"""A face behind a nearer person's head is not counted (HECO_QUALITY_MAX_HEAD_OVERLAP).

THE CASE (2026-09-25, run 125001 of the D02 recording, p00005): a man's face
half hidden behind the grey head of a guest standing in front of him passed
the strict gate. The geometry below is that frame's, scaled from the 1280x720
keyframe to 4K.
"""

import pytest
from heco_counting.association import nearer_head_overlap
from heco_counting.gate import GateThresholds, gate_face, gate_faces


def box(x, y, w, h):
    """A detector box."""
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}


GREY = box(1449, 225, 951, 1890)      # the nearer guest, back to camera, arms out
BEHIND = box(1950, 285, 270, 765)     # p00005's visible body, feet hidden
P00005 = {"box": box(1950, 270, 120, 180)}


def test_run_125001_p00005_is_mostly_behind_the_nearer_head():
    """The reported half face: nearly all of it inside the grey head."""
    covered = nearer_head_overlap(P00005, [GREY, BEHIND])
    assert covered == pytest.approx(0.97, abs=0.02)


def test_two_guests_side_by_side_are_never_ordered():
    """Same floor line: neither is nearer, so neither covers the other."""
    a, b = box(1000, 300, 300, 1800), box(1250, 300, 300, 1800)
    face = {"box": box(1180, 340, 110, 150)}   # A's face, right beside B's head
    assert nearer_head_overlap(face, [a, b]) == 0.0


def test_a_face_beside_a_nearer_shoulder_is_not_covered():
    """Over the shoulder of someone nearer, clear of their head: counted."""
    nearer = box(1000, 400, 400, 1700)
    farther = box(1350, 250, 300, 1250)
    face = {"box": box(1400, 300, 100, 150)}
    assert nearer_head_overlap(face, [nearer, farther]) == 0.0


def test_a_face_in_no_person_box_cannot_be_ordered():
    """No own body, no depth: None, which the gate treats as unmeasured."""
    assert nearer_head_overlap({"box": box(10, 10, 50, 60)}, [GREY]) is None


def test_the_gate_rejects_at_the_floor_and_reports_it():
    """Armed at 0.3: the half face goes, a sliver stays, absent is unmeasured."""
    t = GateThresholds(max_head_overlap=0.3)
    assert "occluded" in t.armed
    face = {"box": box(1950, 270, 120, 180), "headOverlap": 0.97}
    assert gate_face(face, t).reason == "occluded"
    assert gate_face({"box": box(0, 0, 120, 180), "headOverlap": 0.1}, t).kept
    unknown = gate_face({"box": box(0, 0, 120, 180)}, t)
    assert unknown.kept and "occluded" in unknown.unmeasured
    out = gate_faces([dict(face), {"box": box(0, 0, 120, 180), "headOverlap": 0.0}], t)
    assert out.gated_by == {"occluded": 1} and len(out.kept) == 1


def test_off_by_default():
    """The default thresholds never look at headOverlap."""
    t = GateThresholds()
    assert "occluded" not in t.armed
    assert gate_face({"box": box(0, 0, 120, 180), "headOverlap": 1.0}, t).kept
