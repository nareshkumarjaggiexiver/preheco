"""The operator's drawing: what it eats, what it only labels, and when it refuses."""

from heco_counting.zones import (
    apply_detection_zones,
    apply_face_zones,
    mark_person_zones,
)


class Obs:
    """A recording CountingObserver."""

    def __init__(self):
        self.counts, self.metrics, self.events = {}, [], []

    def bump(self, key, by=1):
        """Add to a named counter."""
        self.counts[key] = self.counts.get(key, 0) + by

    def observe(self, stage, metric, value):
        """Record one metric sample."""
        self.metrics.append((stage, metric, value))

    def event(self, text):
        """Record one event line."""
        self.events.append(text)


def face(cx, cy, w=40.0):
    """A face whose box CENTRE sits at (cx, cy)."""
    return {"box": {"x": cx - w / 2, "y": cy - w / 2, "w": w, "h": w}}


def frame(w=1000, h=1000):
    """A frame carrying its pixel dimensions."""
    return {"w": w, "h": h}


#: The left half of the picture.
LEFT_HALF = [{"label": "office", "points": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}]


def test_a_face_inside_a_zone_is_dropped_and_stamped():
    """Dropped from the countable list, but KEPT in the caller's list stamped.

    The operator has to be able to see what their polygon is eating, in the
    tap and on the annotated frame — an invisible filter on an invoice figure
    is not correctable.
    """
    faces = [face(200, 500), face(800, 500)]
    obs = Obs()
    kept = apply_face_zones(faces, frame(), LEFT_HALF, obs)
    assert len(kept) == 1 and kept[0] is faces[1]
    assert faces[0]["excludedByZone"] is True
    assert faces[0]["excludedZone"] == "office"
    assert obs.counts["excludedByZone"] == 1


def test_the_rule_is_the_CENTRE_not_overlap():
    """A guest walking past a partition counts; the face behind it does not.

    Overlap would eat everyone who passes near the edge of a polygon, which
    at a doorway is everyone.
    """
    straddling = face(505, 500, w=200)   # centre right of the line, box crosses it
    kept = apply_face_zones([straddling], frame(), LEFT_HALF, Obs())
    assert kept == [straddling], "a box may overlap a zone; only its centre decides"


def test_no_frame_dimensions_means_NO_exclusion_and_a_counter():
    """Absent is not zero, and it must never be silently treated as inside.

    Zones are normalized and need a pixel size to become polygons. A frame
    without one leaves every face untested — and dropping untested faces would
    remove guests from an invoice with nothing to point at.
    """
    faces = [face(200, 500)]
    obs = Obs()
    kept = apply_face_zones(faces, {"w": None, "h": None}, LEFT_HALF, obs)
    assert kept == faces, "nothing may be dropped when nothing could be tested"
    assert obs.counts["zoneUnmeasured"] == 1
    assert "excludedByZone" not in faces[0]


def test_overlapping_zones_report_the_FIRST_label():
    """Which label reaches the operator is decided by their own zone order."""
    two = [
        {"label": "first", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        {"label": "second", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    ]
    faces = [face(500, 500)]
    apply_face_zones(faces, frame(), two, Obs())
    assert faces[0]["excludedZone"] == "first"


def test_no_zones_is_a_fast_path_that_touches_nothing():
    """A run with no zones must be indistinguishable from one before zones."""
    faces = [face(200, 500)]
    obs = Obs()
    assert apply_face_zones(faces, frame(), [], obs) == faces
    assert obs.counts == {}
    assert "excludedByZone" not in faces[0]


# ------------------------------------------------- detections-mode (persons)

DETECT_LEFT = [{"label": "tv", "mode": "detections",
                "points": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}]


def test_a_detections_zone_keeps_a_body_out_of_the_TRACKER():
    """Fires before tracking so a wall TV never forms a track at all."""
    boxes = [{"x": 100, "y": 400, "w": 100, "h": 200},
             {"x": 800, "y": 400, "w": 100, "h": 200}]
    obs = Obs()
    trackable = apply_detection_zones(boxes, frame(), DETECT_LEFT, obs)
    assert len(trackable) == 1 and trackable[0] is boxes[1]
    assert obs.counts["personsZoned"] == 1


def test_a_faces_mode_zone_does_not_filter_person_boxes():
    """The two modes are separate on purpose; only opt-in zones cut bodies."""
    boxes = [{"x": 100, "y": 400, "w": 100, "h": 200}]
    assert apply_detection_zones(boxes, frame(), LEFT_HALF, Obs()) == boxes


# ------------------------------------------------------------- label-only

def test_mark_person_zones_LABELS_and_never_filters():
    """Deleting a body would cut the track and re-mint the guest beyond it.

    This is the one zone rule that must not remove anything: it exists so the
    console can say "persons 2 · 1 in zone", and nothing more.
    """
    boxes = [{"x": 100, "y": 400, "w": 100, "h": 200},
             {"x": 800, "y": 400, "w": 100, "h": 200}]
    mark_person_zones(boxes, frame(), LEFT_HALF, Obs())
    assert boxes[0].get("inZone") is True
    assert "inZone" not in boxes[1]
    assert len(boxes) == 2, "labelling must never shorten the list"
