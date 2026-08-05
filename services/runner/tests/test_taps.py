"""Unit tests for the pure debug-tap payload builders (no I/O, no OpenCV)."""

from app import taps


def test_quality_band_thresholds():
    """gated below the embed floor, sub-canon below the production canon."""
    assert taps.quality_band(40.0, 56.0, 80.0) == "gated"
    assert taps.quality_band(64.0, 56.0, 80.0) == "sub-canon"
    assert taps.quality_band(90.0, 56.0, 80.0) == "canon"
    assert taps.quality_band(56.0, 56.0, 80.0) == "sub-canon"  # floor is inclusive
    assert taps.quality_band(80.0, 56.0, 80.0) == "canon"  # canon is inclusive


def test_person_payload_counts_and_rounds():
    """Exact count plus rounded boxes with confidence carried through."""
    boxes = [{"x": 10.04, "y": 20.06, "w": 40.44, "h": 100.5, "conf": 0.912}]
    p = taps.person_payload(boxes)
    assert p["count"] == 1
    assert p["boxes"][0] == {"x": 10.0, "y": 20.1, "w": 40.4, "h": 100.5, "conf": 0.912}


def test_track_payload_carries_ids_and_ages():
    """Track ids with their ages are what a debugger needs."""
    tracks = [{"id": 3, "box": {"x": 1, "y": 2, "w": 3, "h": 4}, "ageFrames": 12, "hits": 9}]
    t = taps.track_payload(tracks)
    assert t["count"] == 1
    assert t["tracks"][0]["id"] == 3 and t["tracks"][0]["ageFrames"] == 12


def test_face_payload_kept_vs_gated():
    """Faces at/above the floor count as kept; the quality band is tagged."""
    faces = [
        {"box": {"x": 0, "y": 0, "w": 90.0, "h": 110}},  # canon
        {"box": {"x": 0, "y": 0, "w": 64.0, "h": 80}},  # sub-canon
        {"box": {"x": 0, "y": 0, "w": 40.0, "h": 50}},  # gated
    ]
    f = taps.face_payload(faces, min_px=56.0, canon_px=80.0)
    assert (f["count"], f["kept"], f["gated"]) == (3, 2, 1)
    assert [row["quality"] for row in f["faces"]] == ["canon", "sub-canon", "gated"]
    assert [row["widthPx"] for row in f["faces"]] == [90.0, 64.0, 40.0]


def test_face_payload_reads_the_gate_verdict_rather_than_recomputing_it():
    """The reporting path must not be a second implementation of the gate.

    While the gate was width-only in two places the two agreed by luck; the
    moment either grew a signal they would disagree about the same frame, and
    the console's number is the one an operator argues an invoice from.
    """
    faces = [
        {"box": {"x": 0, "y": 0, "w": 90.0, "h": 110}, "gateReason": None,
         "iedPx": 44.0, "frontality": 0.9},
        # Comfortably wide, and still rejected — only the verdict knows why.
        {"box": {"x": 0, "y": 0, "w": 120.0, "h": 150}, "gateReason": "frontality",
         "frontality": 0.05},
        {"box": {"x": 0, "y": 0, "w": 40.0, "h": 50}, "gateReason": "width"},
    ]
    f = taps.face_payload(faces, min_px=56.0, canon_px=80.0)
    assert (f["count"], f["kept"], f["gated"]) == (3, 1, 2)
    assert f["gatedBy"] == {"frontality": 1, "width": 1}
    assert [row["gate"] for row in f["faces"]] == ["kept", "frontality", "width"]
    # The evidence travels with the verdict, or the operator cannot check it.
    assert f["faces"][0]["iedPx"] == 44.0
    assert f["faces"][1]["frontality"] == 0.05


def test_face_payload_falls_back_to_width_for_an_ungated_caller():
    """An unstamped face list keeps the historical width-only reading."""
    faces = [{"box": {"x": 0, "y": 0, "w": 64.0, "h": 80}},
             {"box": {"x": 0, "y": 0, "w": 40.0, "h": 50}}]
    f = taps.face_payload(faces, min_px=56.0, canon_px=80.0)
    assert (f["kept"], f["gated"]) == (1, 1)
    assert f["gatedBy"] == {"width": 1}


def test_match_payload_counters_and_staff_flag():
    """Verdicts carry personKey/cosine/staff; live counters ride alongside."""
    verdicts = [
        {"personKey": "p00001", "cosine": 0.42, "isNew": True, "isStaff": False},
        {"personKey": "st-1", "cosine": 0.55, "isNew": False, "isStaff": True},
    ]
    m = taps.match_payload(verdicts, unique=1, staff_crossings=1)
    assert m["unique"] == 1 and m["staffCrossings"] == 1 and m["matched"] == 2
    assert m["matches"][1]["isStaff"] is True


def test_build_payloads_covers_the_five_stages():
    """One frame snapshot yields a payload per visual stage."""
    last = {
        "t_ms": 500,
        "seq": 5,
        "boxes": [{"x": 1, "y": 1, "w": 40, "h": 100, "conf": 0.9}],
        "tracks": [
            {"id": 1, "box": {"x": 1, "y": 1, "w": 40, "h": 100}, "ageFrames": 3, "hits": 2}
        ],
        "faces": [{"box": {"x": 2, "y": 2, "w": 85, "h": 110}}],
        "verdicts": [{"personKey": "p00001", "cosine": 0.4, "isNew": True, "isStaff": False}],
    }
    out = taps.build_payloads(last, 56.0, 80.0, unique=1, staff_crossings=0)
    assert set(out) == {"ingest", "person-detect", "track", "face-detect", "match"}
    assert out["ingest"] == {"tMs": 500, "seq": 5}
    assert out["match"]["unique"] == 1


def test_row_cap_and_budget():
    """Row lists are capped, but counts stay exact and the payload fits 32 KB."""
    boxes = [{"x": i, "y": 0, "w": 40, "h": 100, "conf": 0.5} for i in range(200)]
    p = taps.person_payload(boxes)
    assert p["count"] == 200  # exact
    assert len(p["boxes"]) == taps.ROW_CAP  # capped
    assert taps.within_budget(p) is True


def test_face_payload_zone_excluded_faces_are_visible_but_uncounted():
    """A zone-eaten face rides the tap flagged, outside kept/gated entirely.

    The operator who drew the polygon audits it from this payload: hiding the
    face would make the zone unfalsifiable, and counting it as gated would
    blame the quality gate for a placement decision.
    """
    faces = [
        {"box": {"x": 0, "y": 0, "w": 90.0, "h": 110}, "gateReason": None},
        {"box": {"x": 100, "y": 0, "w": 60.0, "h": 75},
         "excludedByZone": True, "excludedZone": "frosted partition"},
    ]
    f = taps.face_payload(faces, min_px=56.0, canon_px=80.0)
    assert (f["count"], f["kept"], f["gated"], f["excludedByZone"]) == (2, 1, 0, 1)
    row = f["faces"][1]
    assert row["excludedByZone"] is True
    assert row["zone"] == "frosted partition"
    assert row["gate"] == "zone"
