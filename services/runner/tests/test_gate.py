"""Unit tests for the composite quality gate (point #6).

The gate is the pipeline's only irreversible discard — a face it rejects is
never embedded, never matched and therefore never counted — so its boundaries
and, above all, its DEFAULTS are pinned here rather than inferred from a loop
test. Pure functions, no I/O.
"""

from app.config import Settings
from app.gate import GateThresholds, gate_face, gate_faces


def face(w=85.0, **signals):
    """A detected face with the given box width and optional signals."""
    return {"box": {"x": 0, "y": 0, "w": w, "h": w * 1.3}, "conf": 0.9, **signals}


# ------------------------------------------------------- defaults do nothing


def test_the_shipped_default_is_exactly_the_old_width_only_gate():
    """THE load-bearing test of this change.

    Every new floor defaults to unarmed, because we have no measured value for
    any of them: the eval harness that would price a floor does not exist yet,
    and a guessed floor costs guests off an invoice. Until an operator opts in,
    the gate must be byte-for-byte the width gate it has always been.
    """
    t = GateThresholds.from_settings(Settings())
    assert t.min_px == 56.0
    assert t.armed == (), "no signal beyond width may gate by default"

    # A face that would fail every other axis still passes on width alone.
    awful = face(w=56.0, iedPx=4.0, frontality=0.01, sharpness=0.1)
    assert gate_face(awful, t).kept is True
    assert gate_face(face(w=55.9), t).reason == "width"


def test_settings_carry_the_env_overrides_through_to_the_gate():
    """The floors are tunable per camera without a code change."""
    t = GateThresholds.from_settings(
        Settings(quality_min_ied_px=30.0, quality_min_frontality=0.5,
                 quality_min_sharpness=25.0)
    )
    assert t.armed == ("ied", "frontality", "sharpness")


# ------------------------------------------------------------- each signal


def test_ied_floor_rejects_a_wide_but_distant_face():
    """Width is the wrong size measure: 56 px of box is only ~24 px of IED.

    A face can clear the box-width floor while carrying far too little of the
    geometry recognition actually consumes.
    """
    t = GateThresholds(min_px=56.0, min_ied_px=30.0)
    assert gate_face(face(w=90.0, iedPx=29.9), t).reason == "ied"
    assert gate_face(face(w=90.0, iedPx=30.0), t).kept is True  # inclusive floor


def test_frontality_floor_rejects_a_wide_face_turned_side_on():
    """Size says nothing about pose, and a profile has little identity left."""
    t = GateThresholds(min_px=56.0, min_frontality=0.4)
    assert gate_face(face(w=120.0, frontality=0.2), t).reason == "frontality"
    assert gate_face(face(w=120.0, frontality=0.4), t).kept is True


def test_sharpness_floor_rejects_a_wide_frontal_but_smeared_face():
    """The failure no other signal here can see: motion blur at any size."""
    t = GateThresholds(min_px=56.0, min_sharpness=20.0)
    assert gate_face(face(w=120.0, frontality=1.0, sharpness=3.0), t).reason == "sharpness"
    assert gate_face(face(w=120.0, frontality=1.0, sharpness=20.0), t).kept is True


def test_width_is_reported_first_when_several_floors_fail():
    """One face, one label: the console shows the first failing signal.

    Order is width, ied, frontality, sharpness — cheapest and oldest first.
    """
    t = GateThresholds(min_px=56.0, min_ied_px=30.0, min_frontality=0.5)
    assert gate_face(face(w=10.0, iedPx=1.0, frontality=0.0), t).reason == "width"
    assert gate_face(face(w=90.0, iedPx=1.0, frontality=0.0), t).reason == "ied"
    assert gate_face(face(w=90.0, iedPx=99.0, frontality=0.0), t).reason == "frontality"


# --------------------------------------------- an unknown signal is not a bad one


def test_a_missing_signal_never_rejects_a_face():
    """Absence means UNKNOWN, and unknown must not cost a guest.

    Under-counting is this pipeline's dominant failure mode and the count is an
    invoice figure, so a face the faces service could not measure (no
    landmarks, a degenerate crop) has to pass an armed floor rather than be
    dropped by it. The alternative is a silent third way to lose people, worst
    exactly where detection is already hardest.
    """
    t = GateThresholds(min_px=56.0, min_ied_px=30.0, min_sharpness=20.0)
    v = gate_face(face(w=85.0), t)  # no iedPx, no sharpness
    assert v.kept is True
    assert v.unmeasured == ("ied", "sharpness"), "but it must be reported"


def test_a_malformed_signal_is_treated_as_unmeasured_not_as_zero():
    """A stage that drifts must not become a mass rejection.

    Reading a non-numeric field as 0.0 would fail every armed floor at once and
    silently zero a night's count; crashing would kill a live run mid-event.
    """
    t = GateThresholds(min_px=56.0, min_ied_px=30.0)
    v = gate_face(face(w=85.0, iedPx=None), t)
    assert v.kept is True and v.unmeasured == ("ied",)
    assert gate_face(face(w=85.0, iedPx="wide"), t).kept is True


# ------------------------------------------------------------ frame-level


def test_gate_faces_stamps_every_face_with_its_reason():
    """The reason has to survive to the taps and the overlay, so it rides on
    the face itself — three consumers downstream re-order or truncate the
    list, and a parallel array would be wrong in all of them."""
    t = GateThresholds(min_px=56.0, min_frontality=0.5)
    faces = [face(w=90.0, frontality=0.9), face(w=40.0), face(w=90.0, frontality=0.1)]
    out = gate_faces(faces, t)

    assert len(out.kept) == 1
    assert [f["gateReason"] for f in faces] == [None, "width", "frontality"]
    assert out.gated_by == {"width": 1, "frontality": 1}


def test_gate_faces_counts_faces_that_slipped_past_an_armed_floor():
    """An operator who arms an IED floor needs to know how much of it runs.

    If most faces carry no landmarks, an armed IED floor is gating almost
    nothing — and the count would look reassuringly unchanged while the gate
    the operator thinks they configured is not happening.
    """
    t = GateThresholds(min_px=56.0, min_ied_px=30.0)
    faces = [face(w=90.0, iedPx=40.0), face(w=90.0), face(w=90.0)]
    out = gate_faces(faces, t)
    assert len(out.kept) == 3
    assert out.unmeasured == 2
    assert faces[1]["gateUnmeasured"] == ["ied"]
    assert "gateUnmeasured" not in faces[0]


# ---------------------------------------- landmarks, eye span, and re-verify
# Harvested from the sibling face-detection pipeline (docs/planning/13-…):
# the two signals it gates on that we did not have, and the per-track
# re-verification saving that is the cheapest idea in that file.

from app.gate import reverify_filter  # noqa: E402


def test_the_new_floors_are_unarmed_by_default_too():
    """Same rule as every other floor: shipped off. A guessed floor costs
    guests off an invoice, and these two were measured on somebody else's
    camera, mount and lighting."""
    t = GateThresholds.from_settings(Settings())
    assert t.min_eye_span == 0.0
    assert t.require_landmarks is False
    assert t.armed == ()
    # A face with implausible landmarks is KEPT while the switch is off.
    assert gate_face(face(landmarksPlausible=False), t).kept


def test_armed_landmarks_reject_a_detection_that_is_not_a_face():
    """Armed, the switch turns an implausible layout into a refusal."""
    t = GateThresholds(require_landmarks=True)
    assert gate_face(face(landmarksPlausible=False), t).reason == "landmarks"
    assert gate_face(face(landmarksPlausible=True), t).kept
    assert t.armed == ("landmarks",)


def test_landmarks_are_reported_before_the_quality_floors():
    """A detection on a shirt must read as NOT-A-FACE, not as a badly-posed
    face: the two send an operator looking in different places."""
    t = GateThresholds(require_landmarks=True, min_frontality=0.55)
    v = gate_face(face(landmarksPlausible=False, frontality=0.1), t)
    assert v.reason == "landmarks"


def test_unmeasured_landmarks_keep_the_face_and_say_so():
    """Absence is UNKNOWN, never BAD — and an operator who armed the switch
    needs to know how many faces are slipping past it unmeasured."""
    t = GateThresholds(require_landmarks=True)
    v = gate_face(face(), t)
    assert v.kept
    assert v.unmeasured == ("landmarks",)


def test_armed_eye_span_rejects_a_profile_view():
    """The pose axis iedPx cannot see: a profile face two feet from the lens
    has a large IED in pixels and almost no eye span relative to its box."""
    t = GateThresholds(min_eye_span=0.30)
    assert gate_face(face(eyeSpanRatio=0.17), t).reason == "eyespan"
    assert gate_face(face(eyeSpanRatio=0.42), t).kept
    assert gate_face(face(iedPx=90.0), t).unmeasured == ("eyespan",)


def test_the_measured_frontality_floor_splits_the_observed_bands():
    """The sibling pipeline measured frontal crops at a nose-offset/eye-
    distance of 0.05-0.43 and the off-angle crops behind phantom identities
    at 0.48-1.02. Their yaw ratio is our frontality inverted, so their 0.45
    ceiling is our 0.55 floor — this pins that translation."""
    t = GateThresholds(min_frontality=0.55)
    assert gate_face(face(frontality=1 - 0.43), t).kept  # worst frontal: 0.57
    assert gate_face(face(frontality=1 - 0.48), t).reason == "frontality"  # best off-angle: 0.52


def _box(x, y, w=40, h=90):
    return {"x": x, "y": y, "w": w, "h": h}


def test_reverify_filter_is_a_no_op_with_nothing_settled():
    """Nothing identified yet means nothing to skip — the common early case."""
    regions = [_box(0, 0), _box(100, 0)]
    kept, skipped = reverify_filter(regions, [])
    assert kept == regions and skipped == 0


def test_reverify_filter_drops_only_the_settled_person():
    """The saving must come out of re-confirming a settled answer, never out
    of searching someone who could still change the count."""
    identified, stranger = _box(0, 0), _box(300, 0)
    kept, skipped = reverify_filter([identified, stranger], [_box(2, 3)])
    assert kept == [stranger]
    assert skipped == 1


def test_reverify_filter_removes_the_raw_detection_too_not_just_the_track_box():
    """THE load-bearing property. The search list is the deduped union of raw
    detection boxes and track boxes, so a settled track's own detection
    describes the same person. Filtering by track box alone would remove
    nothing — the person would still be searched via their detection."""
    track_box = _box(10, 10)
    raw_detection = _box(12, 11)  # same person, detector's own box
    kept, skipped = reverify_filter([raw_detection], [track_box])
    assert kept == [] and skipped == 1


def test_reverify_filter_keeps_a_neighbour_standing_close():
    """Two people shoulder to shoulder are two searches. The overlap rule is
    the same one that merged the two lists, so the boundary is consistent."""
    settled = _box(0, 0, w=40, h=90)
    neighbour = _box(30, 0, w=40, h=90)  # IoU ~0.14, well under the threshold
    kept, _ = reverify_filter([neighbour], [settled])
    assert kept == [neighbour]
