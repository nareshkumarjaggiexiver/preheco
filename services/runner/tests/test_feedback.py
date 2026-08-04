"""Unit tests for the pure feedback-item interpreter."""

from app.feedback import plan_action


def test_duplicate_maps_to_merge():
    """A duplicate correction merges the second key into the first."""
    a = plan_action({"id": 1, "kind": "duplicate", "payload": {"personKeys": ["p1", "p2"]}})
    assert a.kind == "merge" and a.key_a == "p1" and a.key_b == "p2"


def test_false_match_maps_to_split():
    """A false-match correction records a do-not-merge pair."""
    a = plan_action({"id": 2, "kind": "false-match", "payload": {"personKeys": ["p1", "p3"]}})
    assert a.kind == "split" and (a.key_a, a.key_b) == ("p1", "p3")


def test_mark_staff_with_and_without_roster_id():
    """mark-staff carries the personKey, and the staffId when the operator set one."""
    named = plan_action(
        {"id": 3, "kind": "mark-staff", "payload": {"personKey": "p9", "staffId": "st-2"}}
    )
    assert named.kind == "mark-staff" and named.person_key == "p9" and named.staff_id == "st-2"
    anon = plan_action({"id": 4, "kind": "mark-staff", "payload": {"personKey": "p9"}})
    assert anon.kind == "mark-staff" and anon.staff_id is None


def test_missed_becomes_a_manual_count_and_note_stays_audit_only():
    """'missed' is the operator's +1 lever; 'note' is still audit-only.

    Regression: 'missed' mapped to 'ack' and changed nothing, so every
    implemented lever moved the count DOWN and an operator watching an
    uncounted guest walk through had no way to correct it.
    """
    missed = plan_action({"id": 5, "kind": "missed", "payload": {"note": "by the bar"}})
    assert missed.kind == "count-missed" and missed.note == "by the bar"
    assert plan_action({"id": 6, "kind": "note", "payload": {}}).kind == "ack"


def test_unknown_kind_is_rejected_not_claimed_applied():
    """An unrecognised kind is rejected (with a reason), never 'applied'.

    Regression: it mapped to 'ack', which the loop reported as APPLIED — the
    operator was told their correction had landed when this runner did not
    even understand it.  Rejecting still closes the item, so it does not
    re-poll forever.
    """
    a = plan_action({"id": 7, "kind": "whatever"})
    assert a.kind == "invalid"
    assert "whatever" in a.reason


def test_mark_staff_needs_a_run_with_a_site():
    """mark-staff on a siteless run is invalid, not a template-destroying no-op.

    Regression: with no siteId the match service removed the guest's templates
    and absorbed them nowhere, yet reported moved>0 — so the runner decremented
    unique, resolved the item 'applied', and the person was re-counted as a new
    guest at their next crossing.
    """
    item = {"id": 13, "kind": "mark-staff", "payload": {"personKey": "p9"}}
    refused = plan_action(item, has_site_id=False)
    assert refused.kind == "invalid" and "siteId" in refused.reason
    assert plan_action(item, has_site_id=True).kind == "mark-staff"


def test_malformed_pair_is_invalid():
    """A merge/split without a valid two-key pair is rejected, not attempted."""
    one_key = {"id": 8, "kind": "duplicate", "payload": {"personKeys": ["only"]}}
    same_key = {"id": 9, "kind": "duplicate", "payload": {"personKeys": ["x", "x"]}}
    no_pair = {"id": 10, "kind": "false-match", "payload": {}}
    no_person = {"id": 11, "kind": "mark-staff", "payload": {}}
    assert plan_action(one_key).kind == "invalid"
    assert plan_action(same_key).kind == "invalid"
    assert plan_action(no_pair).kind == "invalid"
    assert plan_action(no_person).kind == "invalid"


def test_payload_as_json_string_is_parsed():
    """The planner may hand the payload back as a JSON string; parse it."""
    a = plan_action({"id": 12, "kind": "duplicate", "payload": '{"personKeys": ["p1", "p2"]}'})
    assert a.kind == "merge" and a.key_a == "p1" and a.key_b == "p2"
