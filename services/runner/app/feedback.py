"""Interpret operator feedback items into gallery actions (pure).

The runner polls the planner's feedback endpoint and, for each open item, must
decide what to do to the live gallery.  This module is the *pure* half of that:
it maps one feedback item (as the planner returns it) to an :class:`Action`
describing the correction, with no HTTP and no gallery access — the loop owns
those side effects.  Keeping the mapping pure makes every branch of the
CONTRACTS.md v1 feedback table unit-testable from a plain dict.

The kinds (planner ``run_feedback.kind``, mirrored in api.js ``FEEDBACK_KINDS``):

    duplicate   {personKeys:[a,b]}   -> merge b into a   (unique -1 on success)
    false-match {personKeys:[a,b]}   -> split (do-not-merge; unique unchanged)
    mark-staff  {personKey, staffId?} -> move to staff store (unique -1)
    missed | note                     -> acknowledge only (audit trail)
"""

import json
from dataclasses import dataclass


@dataclass
class Action:
    """A planned correction for one feedback item.

    ``kind`` is one of ``merge`` | ``split`` | ``mark-staff`` | ``ack`` |
    ``invalid``.  ``ack`` items are filed (acknowledged) with no gallery
    change; ``invalid`` items have a malformed payload and are rejected without
    a gallery call.  The key fields carry only what the chosen kind needs.
    """

    feedback_id: object
    kind: str
    key_a: str | None = None
    key_b: str | None = None
    person_key: str | None = None
    staff_id: str | None = None


def _payload(item: dict) -> dict:
    """Return the item payload as a dict (parsing a JSON string if needed)."""
    p = item.get("payload")
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except (json.JSONDecodeError, TypeError):
            return {}
    return p if isinstance(p, dict) else {}


def _two_keys(payload: dict) -> tuple[str, str] | None:
    """Extract a valid two-key pair from ``personKeys``, else None."""
    keys = payload.get("personKeys")
    if isinstance(keys, list) and len(keys) >= 2 and keys[0] and keys[1] and keys[0] != keys[1]:
        return str(keys[0]), str(keys[1])
    return None


def plan_action(item: dict) -> Action:
    """Map one feedback item to the correction the runner should apply.

    A malformed correction (a duplicate/false-match without a valid key pair,
    a mark-staff without a personKey) becomes an ``invalid`` action so the loop
    rejects it cleanly rather than issuing a bad gallery call.
    """
    fid = item.get("id")
    kind = item.get("kind")
    payload = _payload(item)

    if kind == "duplicate":
        pair = _two_keys(payload)
        if pair is None:
            return Action(fid, "invalid")
        return Action(fid, "merge", key_a=pair[0], key_b=pair[1])

    if kind == "false-match":
        pair = _two_keys(payload)
        if pair is None:
            return Action(fid, "invalid")
        return Action(fid, "split", key_a=pair[0], key_b=pair[1])

    if kind == "mark-staff":
        person_key = payload.get("personKey")
        if not person_key:
            return Action(fid, "invalid")
        staff_id = payload.get("staffId")
        return Action(fid, "mark-staff", person_key=str(person_key),
                      staff_id=str(staff_id) if staff_id else None)

    if kind in ("missed", "note"):
        return Action(fid, "ack")

    # Unknown kind: acknowledge so it does not re-poll forever, but touch nothing.
    return Action(fid, "ack")
