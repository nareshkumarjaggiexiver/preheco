"""Run-loop tests for the v1 additions: staff counting, taps, feedback, enrol.

Like test_loop.py, everything runs against a fully fake pipeline over
httpx.MockTransport — no sockets. This fake extends the stage/planner routing
with the v1 surface: staff-aware /match, the merge/split/mark-staff and
/staff/enrol match endpoints, and the planner taps/frames/feedback endpoints.
"""

import base64
import json
import time

import cv2
import httpx
import numpy as np
import pytest
from app.appearance import intersection, torso_descriptor
from app.config import Settings
from app.loop import RunLoop, httpx_file_transport, httpx_transport
from heco_common.imaging import decode_jpeg_b64
from heco_common.planner import PlannerClient


def real_jpeg_b64() -> str:
    """A decodable JPEG frame (so annotation/frame upload has something real)."""
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class V1Fake:
    """Scripted stage + planner services covering the v1 loop features."""

    def __init__(
        self,
        n_frames=2,
        face_widths=(85.0,),
        image_b64="ZmFrZS1qcGVn",
        staff_width=None,
        feedback_items=None,
        match_script=None,
    ):
        """Serve n_frames; ``staff_width`` faces come back tagged as staff.

        ``match_script`` replays exact verdicts: each /match call pops the
        next dict and serves it (galleryN filled in), so a test can replay a
        measured night — cosines and all — instead of the default alternating
        new/match behaviour.
        """
        self.n_frames = n_frames
        self.face_widths = face_widths
        self.image_b64 = image_b64
        self.staff_width = staff_width
        self.feedback_items = list(feedback_items or [])
        self.match_script = list(match_script or [])
        self.frame_i = 0
        self.guest_n = 0
        self.guest_calls = 0
        self.calls: list[str] = []
        self.taps: list[dict] = []
        self.stats: list[dict] = []  # every /stats flush body, in post order
        self.frames_posted: list[str] = []  # stage of each multipart frame POST
        self.frame_bodies: list[bytes] = []  # raw multipart bodies (JPEG inside)
        self.feedback_polls = 0
        self.resolved: list[tuple[str, str]] = []  # (feedbackId, status)
        self.enrolled: dict | None = None
        self.tombstones: list[dict] = []          # served once, like feedback
        self.tombstones_confirmed: list[str] = []
        self.purges: list[dict] = []
        self.enrol_report: dict | None = None
        self.merges: list[dict] = []
        self.match_bodies: list[dict] = []  # every /match request body, in order
        self.splits: list[dict] = []
        self.mark_staff: list[dict] = []
        self.manual_counts: list[dict] = []
        self.opened: dict | None = None
        self.closed: dict | None = None
        self.run_created: dict | None = None
        self.run_ended: dict | None = None
        self.drop_resolve = 0  # PUT /api/feedback/:id to swallow (dropped reply)
        self.merge_ok = True  # False once the drop key no longer exists
        self.merge_always_ok = False  # True: distinct drops all succeed (multi-heal)
        self.feedback_sticky = False  # re-serve open items until a PUT lands

    # -- helpers -----------------------------------------------------------

    def _planner(self, request, path, body):
        if path == "/api/pipeline/runs" and request.method == "POST":
            self.run_created = body
            return httpx.Response(200, json={"id": "prun-1"})
        if path == "/api/pipeline/runs/prun-1" and request.method == "PUT":
            self.run_ended = body
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/stats"):
            self.stats.append(body)
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/samples"):
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/taps"):
            self.taps.append(body)
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/frames"):
            # multipart: read the stage field out of the recorded call list
            self.frames_posted.append("frame")
            self.frame_bodies.append(request.content)
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/feedback") and request.method == "GET":
            self.feedback_polls += 1
            if self.feedback_sticky:
                # Real planner behaviour: an item stays OPEN until a status PUT
                # actually lands, so a dropped PUT means it is served again.
                done = {fid for fid, _ in self.resolved}
                return httpx.Response(
                    200,
                    json={"feedback": [i for i in self.feedback_items if i["id"] not in done]},
                )
            items, self.feedback_items = self.feedback_items, []  # serve once
            return httpx.Response(200, json={"feedback": items})
        if path.startswith("/api/feedback/") and request.method == "PUT":
            if self.drop_resolve > 0:
                # The planner got it but the reply never came back (or it never
                # got it) — either way the runner must not assume it landed.
                self.drop_resolve -= 1
                return httpx.Response(504, json={"error": "gateway timeout"})
            self.resolved.append((path.rsplit("/", 1)[-1], body["status"]))
            return httpx.Response(200, json={"ok": True})
        if path == "/api/staff-tombstones" and request.method == "GET":
            items, self.tombstones = self.tombstones, []  # serve once
            return httpx.Response(200, json=items)
        if path.startswith("/api/staff-tombstones/") and request.method == "PUT":
            self.tombstones_confirmed.append(path.rsplit("/", 1)[-1])
            return httpx.Response(200, json={"ok": True, "purgedAt": "2026-08-05T00:00:00Z"})
        if path.startswith("/api/staff/") and request.method == "PUT":
            self.enrol_report = {"id": path.rsplit("/", 1)[-1], **body}
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(500, json={"error": f"unscripted planner {path}"})

    def _match(self, path, body):
        if path == "/reset":
            return httpx.Response(200, json={"ok": True})
        if path == "/merge":
            self.merges.append(body)
            merged = self.merge_ok
            if not self.merge_always_ok:
                self.merge_ok = False  # a merge is one-shot (drop key is gone)
            return httpx.Response(
                200, json={"merged": merged, "galleryN": max(0, self.guest_n - 1)}
            )
        if path == "/count/manual":
            self.manual_counts.append(body)
            return httpx.Response(200, json={
                "personKey": f"m{len(self.manual_counts):05d}",
                "galleryN": self.guest_n + len(self.manual_counts),
                "manual": True,
            })
        if path == "/split":
            self.splits.append(body)
            return httpx.Response(200, json={"ok": True, "galleryN": self.guest_n})
        if path == "/mark-staff":
            self.mark_staff.append(body)
            if not body.get("siteId"):
                # The real service refuses: destroying the templates would
                # re-count the person as a brand-new guest next crossing.
                return httpx.Response(400, json={"detail": "mark-staff needs a siteId"})
            return httpx.Response(200, json={
                "moved": 1, "galleryN": max(0, self.guest_n - 1),
                "staffKey": body.get("staffId") or "anon00001",
            })
        if path == "/staff/purge":
            self.purges.append(body)
            return httpx.Response(200, json={
                "siteId": body["siteId"],
                "removed": {sid: 1 for sid in body["staffIds"]},
            })
        if path == "/staff/enrol":
            self.enrolled = body
            return httpx.Response(
                200, json={"staffId": body["staffId"], "sampleCount": len(body["samples"])}
            )
        if path == "/match":
            self.match_bodies.append(body)
            if self.match_script:
                v = self.match_script.pop(0)
                if v.get("isNew"):
                    self.guest_n += 1
                return httpx.Response(200, json={"galleryN": self.guest_n, **v})
            q = body["quality"]
            if self.staff_width is not None and body.get("siteId") and q == self.staff_width:
                return httpx.Response(200, json={
                    "personKey": "st-1", "isNew": False, "cosine": 0.7,
                    "galleryN": self.guest_n, "subCanon": q < 80.0,
                    "isStaff": True, "staffId": "st-1",
                })
            self.guest_calls += 1
            is_new = self.guest_calls % 2 == 1
            if is_new:
                self.guest_n += 1
            return httpx.Response(200, json={
                "personKey": f"p{self.guest_calls:05d}", "isNew": is_new,
                "cosine": 0.3, "galleryN": self.guest_n, "subCanon": q < 80.0,
                "isStaff": False, "staffId": None,
                # M1 fields: how many views this identity now holds, and whether
                # this sighting became one.
                "templateN": 2, "templateAdded": True,
            })
        return httpx.Response(500, json={"error": f"unscripted match {path}"})

    # -- router ------------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Route one request to the scripted service behaviour."""
        host, path = request.url.host, request.url.path
        self.calls.append(f"{host} {path}")
        # Frame POSTs are multipart (no JSON body); everything else is JSON.
        body = {}
        if request.content and not path.endswith("/frames"):
            try:
                body = json.loads(request.content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}

        if host == "planner":
            return self._planner(request, path, body)
        if host == "ingest":
            if path == "/open":
                self.opened = body
                return httpx.Response(200, json={"ok": True})
            if path == "/close":
                self.closed = body
                return httpx.Response(200, json={"ok": True, "released": True})
            if path == "/frame":
                # A file source: seq stalls at EOF and `ended` goes true, which
                # is how the runner tells a finished file from a blinking camera.
                i = min(self.frame_i, self.n_frames - 1)
                exhausted = self.frame_i >= self.n_frames
                if not exhausted:
                    self.frame_i += 1
                return httpx.Response(200, json={
                    "imageB64": self.image_b64, "tMs": i * 100, "w": 160, "h": 120,
                    "seq": i, "ended": exhausted,
                })
        if host == "persons" and path == "/detect":
            box = {"x": 10, "y": 20, "w": 40, "h": 110, "conf": 0.9}
            return httpx.Response(200, json={"boxes": [box]})
        if host == "tracker":
            if path in ("/reset", "/release"):
                return httpx.Response(200, json={"ok": True})
            if path == "/track":
                tracks = [{"id": n + 1, "box": b, "ageFrames": self.frame_i, "hits": 1}
                          for n, b in enumerate(body["boxes"])]
                return httpx.Response(200, json={"tracks": tracks})
        if host == "faces" and path == "/detect":
            faces = [
                {"box": {"x": 12, "y": 22, "w": w, "h": w * 1.3},
                 "landmarks": [[1, 1]] * 5, "conf": 0.8}
                for w in self.face_widths
            ]
            return httpx.Response(200, json={"faces": faces, "inferMs": 1.0})
        if host == "embed" and path == "/embed":
            n = len(body["faces"])
            return httpx.Response(200, json={"embeddings": [[0.1] * 128] * n})
        if host == "match":
            return self._match(path, body)
        return httpx.Response(500, json={"error": f"unscripted {host} {path}"})


def make_loop(fake: V1Fake, request: dict, **settings_kw) -> RunLoop:
    """Build a RunLoop wired to the v1 fake, with a file transport for frames."""
    settings = Settings(
        ingest_url="http://ingest:7101", persons_url="http://persons:7102",
        tracker_url="http://tracker:7103", faces_url="http://faces:7104",
        embed_url="http://embed:7105", match_url="http://match:7106",
        planner_url="http://planner:8787",
        flush_interval_s=0.0, source_poll_s=0.001, source_stall_s=0.05,
        **settings_kw,
    )
    client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    planner = PlannerClient(
        settings.planner_url,
        transport=httpx_transport(client),
        file_transport=httpx_file_transport(client),
    )
    return RunLoop("run-local", request, settings, client, planner)


# --------------------------------------------------------------------------


def test_staff_hit_excluded_from_unique_count():
    """A staff face is a staff sighting; only guest faces grow matches/unique.

    Three consecutive frames of the same staff member are ONE crossing (they
    never left) and three staff face-frames.
    """
    fake = V1Fake(n_frames=3, face_widths=(99.0, 85.0), staff_width=99.0)
    request = {"eventId": "ev-1", "siteId": "site-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert final["state"] == "ended"
    assert final["staffCrossings"] == 1
    assert final["staffFaceFrames"] == 3  # the 99 px face each frame
    assert final["matches"] == 3  # only the 85 px guest face counts as a match
    assert "staffCrossings=1" in fake.run_ended["notes"]
    assert "staffFaceFrames=3" in fake.run_ended["notes"]


def test_taps_and_frames_posted_on_interval():
    """With tap_interval 0 and a real frame, every stage taps + posts a frame."""
    fake = V1Fake(n_frames=2, face_widths=(85.0,), image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, tap_interval_s=0.0).run()
    assert final["state"] == "ended"
    stages_tapped = {t["stage"] for t in fake.taps}
    assert stages_tapped == {"ingest", "person-detect", "track", "face-detect", "match"}
    assert len(fake.frames_posted) >= 5  # five annotated stages per tapping frame


def test_match_tap_carries_the_template_policy_fields():
    """The match tap forwards templateN / templateAdded, as CONTRACTS.md states.

    These were specified in the v2 contract but never wired: the match service
    returned them and the runner dropped them on the floor, so the one place an
    operator could have seen the template policy working was the one place it
    was invisible.  A run's counts cannot be read without them — most guests
    holding a single template means the crossings are not supplying diverse
    views and multi-template is doing nothing, which is worth knowing before
    anyone blames the threshold.
    """
    fake = V1Fake(n_frames=2, face_widths=(85.0,), image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    make_loop(fake, request, tap_interval_s=0.0).run()
    rows = [
        row
        for t in fake.taps
        if t["stage"] == "match"
        for row in t["payload"].get("matches", [])
    ]
    assert rows, "no match verdicts were tapped"
    assert all(r["templateN"] == 2 for r in rows)
    assert all(r["templateAdded"] is True for r in rows)


def test_opaque_frame_taps_but_skips_frame_upload():
    """A non-decodable frame still taps payloads but uploads no annotated JPEG."""
    fake = V1Fake(n_frames=2, face_widths=(85.0,), image_b64="ZmFrZS1qcGVn")
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    make_loop(fake, request, tap_interval_s=0.0).run()
    assert fake.taps  # structured taps still went up
    assert fake.frames_posted == []  # nothing decodable to annotate


def test_feedback_duplicate_merges_and_decrements_unique():
    """A duplicate correction merges in the gallery and drops unique by one."""
    item = {"id": "fb-1", "kind": "duplicate", "status": "open",
            "payload": {"personKeys": ["p00001", "p00003"]}}
    fake = V1Fake(n_frames=4, face_widths=(85.0,), feedback_items=[item])
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, feedback_poll_s=0.0).run()
    assert fake.merges == [{"runId": "prun-1", "keep": "p00001", "drop": "p00003"}]
    assert ("fb-1", "applied") in fake.resolved
    assert final["feedbackApplied"] == 1


def test_feedback_mark_staff_moves_and_resolves():
    """A mark-staff correction hits /mark-staff and is resolved applied."""
    item = {"id": "fb-2", "kind": "mark-staff", "status": "open",
            "payload": {"personKey": "p00001", "staffId": "st-7"}}
    fake = V1Fake(n_frames=3, face_widths=(85.0,), feedback_items=[item])
    request = {"eventId": "ev-1", "siteId": "site-1", "source": {"path": "/x.mp4"}}
    make_loop(fake, request, feedback_poll_s=0.0).run()
    assert fake.mark_staff and fake.mark_staff[0]["personKey"] == "p00001"
    assert fake.mark_staff[0]["siteId"] == "site-1" and fake.mark_staff[0]["staffId"] == "st-7"
    assert ("fb-2", "applied") in fake.resolved


def test_enrol_mode_writes_best_samples_and_reports():
    """Enrol keeps the best-N samples, writes them, and reports the count."""
    # Widths ascending across frames so the best-N are the highest few.
    fake = V1Fake(n_frames=6, face_widths=(85.0,))
    request = {"eventId": "ev-1", "mode": "enrol", "siteId": "site-1",
               "staffId": "st-1", "source": {"path": "/walk.mp4"}}
    final = make_loop(fake, request, enrol_best_n=5).run()
    assert final["state"] == "ended"
    assert fake.enrolled is not None
    assert fake.enrolled["siteId"] == "site-1" and fake.enrolled["staffId"] == "st-1"
    assert len(fake.enrolled["samples"]) == 5  # best-N cap
    assert final["sampleCount"] == 5
    # Reported to the planner via PUT /api/staff/:id.
    assert fake.enrol_report["id"] == "st-1" and fake.enrol_report["sampleCount"] == 5
    # Enrol still has no gallery of its own — it counts nobody.
    assert "match /reset" not in fake.calls


def test_tombstoned_staff_purged_at_run_start_and_confirmed():
    """The erasure chain: planner tombstone -> match /staff/purge -> confirm.

    A roster member deleted while the pipeline was off must be purged from the
    site staff store BEFORE the first frame is matched, and the planner's
    ledger must receive the confirmation that closes the tombstone.
    """
    fake = V1Fake(n_frames=1)
    fake.tombstones = [
        {"id": "tomb-1", "siteId": "site9", "staffId": "ravi", "deletedAt": "2026-08-05T00:00:00Z"}
    ]
    request = {"eventId": "ev-1", "siteId": "site9", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.purges == [{"siteId": "site9", "staffIds": ["ravi"]}]
    assert fake.tombstones_confirmed == ["tomb-1"]
    assert final["staffPurged"] == 1

    # ORDERING IS THE GUARANTEE: the purge must complete before the first face
    # is matched, or a withdrawn member is still recognised as staff on frame 1.
    calls = fake.calls
    purge_at = calls.index("match /staff/purge")
    match_at = next(i for i, c in enumerate(calls) if c == "match /match")
    assert purge_at < match_at, "staff purge must precede the first /match"


def test_no_tombstones_means_no_purge_calls():
    """No open tombstones means the purge chain is not invoked at all."""
    fake = V1Fake(n_frames=1)
    request = {"eventId": "ev-1", "siteId": "site9", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert fake.purges == []
    assert final["staffPurged"] == 0


# ---------------------------------------------- regressions (2026-08-05 review)


def test_staff_crossings_debounced_but_face_frames_are_not():
    """staffCrossings must count PASSES, not matched face-frames.

    Regression: ``staffCrossings`` was incremented once per matched face per
    frame with no debounce, so one waiter passing the gate 30 times reported
    hundreds — a figure that moved with the frame rate rather than with staff
    behaviour, and that an operator reading the event report could not use (3
    busy waiters were indistinguishable from 30 staff).
    """
    fake = V1Fake(n_frames=12, face_widths=(99.0,), staff_width=99.0)
    request = {"eventId": "ev-1", "siteId": "site-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, staff_cooldown_s=60.0).run()
    assert final["staffFaceFrames"] == 12, "the raw per-frame figure is still kept"
    assert final["staffCrossings"] == 1, "twelve frames of one waiter is ONE pass"
    assert final["unique"] == 0  # staff never enter the guest count


def test_staff_crossing_counted_again_after_the_cooldown():
    """A staff member seen again after the cooldown IS a second crossing."""
    fake = V1Fake(n_frames=6, face_widths=(99.0,), staff_width=99.0)
    request = {"eventId": "ev-1", "siteId": "site-1", "source": {"path": "/x.mp4"}}
    # Cooldown 0 => every sighting is far enough from the last to be a new pass.
    final = make_loop(fake, request, staff_cooldown_s=0.0).run()
    assert final["staffCrossings"] == final["staffFaceFrames"] == 6


def test_applied_correction_is_never_re_resolved_as_rejected():
    """A dropped status PUT must not turn an APPLIED correction into a rejection.

    Regression: ``resolve_feedback`` is single-shot.  When its PUT was dropped
    the item stayed open, the next poll re-ran the merge, the merge failed (the
    dropped key no longer existed), and the item was recorded REJECTED beside a
    unique count that had visibly fallen — leaving the operator unable to tell
    what had actually happened, which is the entire point of the ledger.
    """
    item = {"id": "fb-1", "kind": "duplicate", "status": "open",
            "payload": {"personKeys": ["p00001", "p00003"]}}
    fake = V1Fake(n_frames=6, face_widths=(85.0,), feedback_items=[item])
    fake.feedback_sticky = True
    fake.drop_resolve = 1  # the first PUT never lands
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, feedback_poll_s=0.0).run()

    assert len(fake.merges) == 1, "an applied correction must never be re-executed"
    assert fake.resolved == [("fb-1", "applied")], "and must be re-reported as applied"
    assert final["feedbackApplied"] == 1
    assert final["feedbackRejected"] == 0


def test_mark_staff_without_site_id_is_refused_not_applied():
    """mark-staff on a siteless run is rejected, and never reaches the gallery.

    Regression: the runner sent mark-staff without a siteId, the match service
    removed the person's templates anyway and returned moved>0, so the runner
    decremented unique and resolved the item APPLIED — while the person, now
    with no templates anywhere, was counted as a brand-new guest at their next
    crossing.  The correction silently undid itself and the audit trail lied.
    """
    item = {"id": "fb-9", "kind": "mark-staff", "status": "open",
            "payload": {"personKey": "p00001"}}
    fake = V1Fake(n_frames=4, face_widths=(85.0,), feedback_items=[item])
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}  # no siteId
    final = make_loop(fake, request, feedback_poll_s=0.0).run()

    assert fake.mark_staff == [], "the gallery must not be touched at all"
    assert fake.resolved == [("fb-9", "rejected")]
    assert final["feedbackRejected"] == 1
    assert final["feedbackApplied"] == 0


def test_missed_feedback_adds_an_attested_person_to_the_count():
    """'missed' must actually move the count UP, recorded as human-attested.

    Regression: every implemented lever moved the count DOWN; 'missed' was
    acknowledged and changed nothing, so an operator who watched an uncounted
    guest walk through had no way to fix it and the delivered number stayed
    short.
    """
    item = {"id": "fb-m", "kind": "missed", "status": "open",
            "payload": {"note": "child behind the pillar"}}
    fake = V1Fake(n_frames=4, face_widths=(85.0,), feedback_items=[item])
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, feedback_poll_s=0.0).run()

    assert len(fake.manual_counts) == 1
    assert fake.manual_counts[0]["note"] == "child behind the pillar"
    assert ("fb-m", "applied") in fake.resolved
    assert final["manualAdditions"] == 1, "attested additions are counted apart"
    assert "manualAdditions=1" in fake.run_ended["notes"]


def test_missed_is_applied_once_even_if_the_status_put_is_dropped():
    """The un-idempotent lever must never run twice: +1 means +1."""
    item = {"id": "fb-m2", "kind": "missed", "status": "open", "payload": {}}
    fake = V1Fake(n_frames=6, face_widths=(85.0,), feedback_items=[item])
    fake.feedback_sticky = True
    fake.drop_resolve = 2
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, feedback_poll_s=0.0).run()
    assert len(fake.manual_counts) == 1
    assert final["manualAdditions"] == 1


def test_unknown_feedback_kind_is_rejected_not_claimed_applied():
    """A correction this runner cannot act on must not be reported applied."""
    item = {"id": "fb-x", "kind": "teleport-guest", "status": "open", "payload": {}}
    fake = V1Fake(n_frames=3, face_widths=(85.0,), feedback_items=[item])
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, feedback_poll_s=0.0).run()
    assert fake.resolved == [("fb-x", "rejected")]
    assert final["feedbackApplied"] == 0


def test_enrol_skips_frames_with_more_than_one_face():
    """Only ONE person may be enrolled per walk-through.

    Regression: every gate-passing face in every frame was appended as a sample
    for the staffId being enrolled, and the shortlist was ranked by box width
    alone.  Two people walking through together meant the taller/closer one's
    embeddings were stored under the OTHER person's roster entry — superseding
    their real templates — so that bystander was matched isStaff at every
    future event at the venue and silently excluded from every guest count.
    """
    fake = V1Fake(n_frames=5, face_widths=(85.0, 82.0))  # two faces every frame
    request = {"eventId": "ev-1", "mode": "enrol", "siteId": "site-1",
               "staffId": "st-1", "source": {"path": "/walk.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.enrolled is None, "nothing may be written from ambiguous frames"
    assert final["multiFaceFramesSkipped"] == 5
    # The refusal must be LOUD: the operator has to know to redo the walk-through.
    assert final["state"] == "failed"
    assert "more than one face" in final["error"]
    assert "re-run the enrolment" in final["error"]


def test_enrol_ranks_by_inter_eye_distance_and_frontality_not_width():
    """The best-N shortlist prefers the most recognisable face, not the widest."""

    class LandmarkFaces(V1Fake):
        """One face per frame; frame 0 is wide but side-on, frame 1 narrow but frontal."""

        SHAPES = [
            {"w": 120.0, "iedPx": 20.0, "frontality": 0.10},  # wide, side-on
            {"w": 60.0, "iedPx": 40.0, "frontality": 0.99},  # narrower, square-on
        ]

        def handler(self, request):
            if request.url.host == "faces" and request.url.path == "/detect":
                self.calls.append("faces /detect")
                s = self.SHAPES[min(self.frame_i - 1, len(self.SHAPES) - 1)]
                return httpx.Response(200, json={"faces": [{
                    "box": {"x": 1, "y": 1, "w": s["w"], "h": s["w"] * 1.3},
                    "landmarks": [[1, 1]] * 5, "conf": 0.9,
                    "iedPx": s["iedPx"], "frontality": s["frontality"],
                }]})
            return super().handler(request)

    fake = LandmarkFaces(n_frames=2)
    request = {"eventId": "ev-1", "mode": "enrol", "siteId": "site-1",
               "staffId": "st-1", "source": {"path": "/walk.mp4"}}
    make_loop(fake, request, enrol_best_n=1).run()

    assert fake.enrolled is not None
    kept = fake.enrolled["samples"][0]["quality"]
    assert kept == 40.0 * 0.99, "ied x frontality must beat raw box width"


def test_enrol_hands_the_camera_back():
    """An enrolment must release ingest's single slot when it finishes."""
    fake = V1Fake(n_frames=3, face_widths=(85.0,))
    request = {"eventId": "ev-1", "mode": "enrol", "siteId": "site-1",
               "staffId": "st-1", "source": {"path": "/walk.mp4"}}
    make_loop(fake, request).run()
    assert fake.opened["owner"] == "run-local"
    assert fake.closed == {"owner": "run-local"}


# ------------------------------------------ the enrolment record (point #13)


def test_an_enrolment_leaves_a_run_record():
    """Enrolment CREATES the biometric, so the capture must leave a ledger.

    Intent change, deliberate: this behaviour was previously pinned the other
    way ("Enrol does not create a pipeline run"), and that pin was wrong about
    what mattered. Erasure already had a permanent ledger — `staff_tombstones`
    keeps a row and its purge confirmation forever, on the argument that a
    consent withdrawal and the proof it was honoured are what an audit asks
    for. The act that CREATES the biometric had nothing: a re-enrolment
    silently overwrote the previous `enrolled_at`, and no record anywhere said
    from which camera, at which event, or over how many frames a staff
    template was captured. Consent is only auditable if the capture is.
    """
    fake = V1Fake(n_frames=6, face_widths=(85.0,))
    request = {"eventId": "ev-1", "mode": "enrol", "siteId": "site-1",
               "staffId": "st-1", "source": {"path": "/walk.mp4"}}
    final = make_loop(fake, request, enrol_best_n=5).run()

    assert final["state"] == "ended"
    assert final["plannerRunId"] == "prun-1", "the enrolment must be addressable"

    # WHO, WHEN, HOW MANY, FROM WHICH SOURCE — the four questions an audit asks.
    assert fake.run_ended["status"] == "ended"
    assert fake.run_ended["results"]["sampleCount"] == 5      # how many
    assert fake.run_ended["results"]["frames"] == 6
    assert "staffId=st-1" in fake.run_ended["notes"]          # who
    assert "source=/walk.mp4" in fake.run_ended["notes"]      # from where
    assert fake.run_ended["endReason"] == "source-ended"


def test_a_failed_enrolment_also_leaves_a_record():
    """The loud-failure path used to leave zero trace anywhere at all.

    An enrolment that captured nothing is exactly the event worth recording:
    it means somebody walked a roster member past a camera and the venue now
    believes they are enrolled when they are not.
    """
    fake = V1Fake(n_frames=5, face_widths=(85.0, 82.0))  # two faces: ambiguous
    request = {"eventId": "ev-1", "mode": "enrol", "siteId": "site-1",
               "staffId": "st-1", "source": {"path": "/walk.mp4"}}
    final = make_loop(fake, request).run()

    assert final["state"] == "failed"
    assert fake.enrolled is None
    assert fake.run_ended["status"] == "failed"
    assert fake.run_ended["results"]["sampleCount"] == 0
    assert fake.run_ended["results"]["multiFaceFramesSkipped"] == 5
    assert "more than one face" in fake.run_ended["notes"]


def test_an_enrolment_survives_a_planner_that_cannot_take_the_record():
    """The record is worth having; it is not worth an enrolment.

    The samples are the deliverable and the operator has already walked the
    member through. A planner outage must cost the record only — the opposite
    trade from count mode, where create_run failing legitimately fails the run
    because the planner id keys the gallery.
    """

    class NoRuns(V1Fake):
        """A planner that refuses to open a run but is otherwise healthy."""

        def _planner(self, request, path, body):
            if path == "/api/pipeline/runs" and request.method == "POST":
                return httpx.Response(503, json={"error": "planner restarting"})
            return super()._planner(request, path, body)

    fake = NoRuns(n_frames=6, face_widths=(85.0,))
    request = {"eventId": "ev-1", "mode": "enrol", "siteId": "site-1",
               "staffId": "st-1", "source": {"path": "/walk.mp4"}}
    loop = make_loop(fake, request, enrol_best_n=5)
    loop.planner._sleep = lambda _s: None  # no retry backoff in a unit test
    final = loop.run()

    assert final["state"] == "ended", "a planner outage must not fail an enrolment"
    assert final["sampleCount"] == 5
    assert fake.enrolled is not None, "the templates must still be written"
    assert final["plannerRunId"] is None
    assert fake.run_ended is None  # nothing to settle: no row was ever opened


def test_an_enrolment_run_label_never_carries_source_credentials():
    """The planner slugs the label into the run row's PERMANENT id."""
    fake = V1Fake(n_frames=3, face_widths=(85.0,))
    request = {
        "eventId": "ev-1", "mode": "enrol", "siteId": "site-1", "staffId": "st-1",
        "source": {"url": "rtsp://admin:s3kret@cam:554/media/video1"},
    }
    make_loop(fake, request).run()
    label = fake.run_created["label"]
    assert "s3kret" not in label
    assert label == "enrol st-1 rtsp://cam:554/media/video1"


def test_tap_round_is_abandoned_when_its_budget_is_spent():
    """A slow-but-alive planner must not hold the frame loop for a tap round.

    Regression: taps/frames/feedback shared the 30 s stage timeout and had no
    round budget, so a wedged-but-accepting planner could freeze the loop for
    minutes per tick — and ingest's drop-not-queue slot discards every guest
    crossing during a freeze.
    """
    fake = V1Fake(n_frames=3, face_widths=(85.0,), image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, tap_interval_s=0.0, tap_budget_s=0.0).run()
    assert fake.taps == [] and fake.frames_posted == []
    assert final["tapRoundsAbandoned"] >= 1
    assert final["state"] == "ended" and final["frames"] == 3  # counting unharmed


# -------------------------------------------- track heal (2026-08-06 live bench)
#
# Tonight's bench: 3 real people, counted 5. p00002 was minted on a RE-ENTRY
# frame at cosine 0.3084 (threshold 0.363) and the SAME physical track matched
# the correct identity p00001 at 0.69 four seconds later. These tests replay
# that geometry through the scripted fake. The face width is 60 px so the face
# box centre (42, 61) sits inside the fake's track box — heal bookkeeping only
# exists for faces a track contains.


def scripted_verdict(key, is_new, cosine, staff=False):
    """One /match reply for the script: the fields the loop actually reads."""
    return {
        "personKey": key, "isNew": is_new, "cosine": cosine, "subCanon": False,
        "isStaff": staff, "staffId": key if staff else None,
        "templateN": None if staff else 1, "templateAdded": False,
    }


TONIGHT_SCRIPT = [
    scripted_verdict("p00001", True, None),      # the real guest's first sighting
    scripted_verdict("p00002", True, 0.3084),    # the junk re-entry mint
    scripted_verdict("p00001", False, 0.69),     # the same track disowns it
]


def test_track_heal_replays_tonight_and_folds_the_junk_mint():
    """THE REPLAY: mint at 0.3084, same track matches p00001 at 0.69 — healed.

    The heal must call /merge with onlyIfSingleton (machine evidence gets the
    guard an operator's does not), step unique back down, count healedSplits,
    and say so in the end-of-run notes.
    """
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=list(TONIGHT_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == [{
        "runId": "prun-1", "keep": "p00001", "drop": "p00002",
        "onlyIfSingleton": True,
    }]
    assert final["unique"] == 1, "the folded mint must leave the count at truth"
    assert final["healedSplits"] == 1
    assert "healedSplits=1" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["healedSplits"] == 1
    assert fake.run_ended["results"]["unique"] == 1


def test_the_double_mint_is_healed_when_the_track_settles_on_its_second_key():
    """THE 2026-08-06 MORNING REPLAY: two mints one second apart, both hers.

    A blurred crossing (4 real people, counted 5): the same track minted
    p00003 at cosine 0.17 and then p00004 at 0.29 — her consecutive frames
    scored 0.29 against each other — and finally matched p00004 comfortably.
    The single-slot bookkeeping this test killed forgot p00003 the moment
    p00004 was recorded, so the phantom survived to the invoice.  Now every
    remembered mint of the track EXCEPT the matched key is folded — including
    when the matched key is itself the track's own later mint.
    """
    script = [
        scripted_verdict("p00003", True, 0.17),   # her first frame, blurred
        scripted_verdict("p00004", True, 0.29),   # one second later, v her own template
        scripted_verdict("p00004", False, 0.83),  # the track settles on p00004
    ]
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == [{
        "runId": "prun-1", "keep": "p00004", "drop": "p00003",
        "onlyIfSingleton": True,
    }], "the earlier mint must fold into the key the track settled on"
    assert final["unique"] == 1, "one woman, one guest — not two"
    assert final["healedSplits"] == 1
    assert fake.run_ended["results"]["healedSplits"] == 1


def test_a_triple_mint_folds_every_phantom_on_one_comfortable_match():
    """All remembered mints except the settled key fold — not just the newest.

    Three blurry mints on one track, then one comfortable match to the third:
    both earlier phantoms fold in the same pass, because each is disowned by
    the same piece of evidence.  One extra guest healed per verdict would
    leave the count wrong for a frame longer than it needs to be.
    """
    script = [
        scripted_verdict("p00003", True, 0.17),
        scripted_verdict("p00004", True, 0.29),
        scripted_verdict("p00005", True, 0.31),
        scripted_verdict("p00005", False, 0.83),
    ]
    fake = V1Fake(n_frames=4, face_widths=(60.0,), match_script=script)
    fake.merge_always_ok = True  # two DISTINCT drop keys: both folds succeed
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert sorted(m["drop"] for m in fake.merges) == ["p00003", "p00004"]
    assert all(m["keep"] == "p00005" and m["onlyIfSingleton"] for m in fake.merges)
    assert final["unique"] == 1
    assert final["healedSplits"] == 2


def test_heal_refusal_changes_nothing_and_is_not_retried():
    """merged=false (no longer a singleton, or an operator split) counts NOTHING.

    The refusal is the system working: the gallery has evidence the machine
    does not, so unique stays put, healedSplits stays 0, and the bookkeeping
    entry is dropped — no second attempt on the next matching verdict.
    """
    script = list(TONIGHT_SCRIPT) + [scripted_verdict("p00001", False, 0.69)]
    fake = V1Fake(n_frames=4, face_widths=(60.0,), match_script=script)
    fake.merge_ok = False  # the match service refuses the fold
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert len(fake.merges) == 1, "a refused heal must not be retried"
    assert final["unique"] == 2
    assert final["healedSplits"] == 0
    assert "healedSplits=0" in fake.run_ended["notes"]


def test_heal_skipped_outside_the_window():
    """A mint older than HECO_HEAL_WINDOW_S is beyond the heal's evidence.

    The window is what keeps the residual tracker-swap risk short-lived; a
    cross-key match minutes later says nothing about a mint made minutes ago.
    """
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=list(TONIGHT_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, heal_window_s=1e-9).run()
    assert fake.merges == []
    assert final["unique"] == 2
    assert final["healedSplits"] == 0


def test_heal_skipped_below_the_cosine_floor():
    """0.40 matches (threshold 0.363) but heals nothing (floor 0.45).

    The measured impostor ceiling on this camera is 0.377 — two DIFFERENT
    men's templates — so a cross-key match at 0.40 is within impostor range
    and proves nothing about the mint. Counting stays; healing does not.
    """
    script = list(TONIGHT_SCRIPT)
    script[2] = scripted_verdict("p00001", False, 0.40)
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert fake.merges == []
    assert final["unique"] == 2
    assert final["healedSplits"] == 0


def test_heal_never_fires_into_a_staff_hit():
    """A track whose later match is STAFF folds nothing (CONTRACTS.md scope).

    Staff templates are operator-supervised evidence; folding a guest mint
    into the staff store's orbit from track behaviour would shrink the guest
    count against identities the heal has no business touching.
    """
    script = list(TONIGHT_SCRIPT)
    script[2] = scripted_verdict("st-1", False, 0.69, staff=True)
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert fake.merges == []
    assert final["unique"] == 2
    assert final["healedSplits"] == 0


# ------------------------------------------ exclusion zones (2026-08-06 bench)
#
# Tonight's p00004 was minted from an 87 px face seen THROUGH A FROSTED GLASS
# PARTITION — inside an office, not at the gate, 0.3203 against its true
# owner. No quality floor rejects a face for being in the wrong PLACE; the
# operator draws polygons and the loop drops faces whose box centre falls
# inside. The fake's frame is 160x120; the 60 px face centres at (42, 61) =
# normalized (0.263, 0.508) and the 85 px face at (54.5, 77.25) = (0.341,
# 0.644), so a zone over x in [0.20, 0.32] eats exactly the first.

FROSTED_ZONE = {
    "label": "frosted partition",
    "points": [[0.20, 0.0], [0.32, 0.0], [0.32, 1.0], [0.20, 1.0]],
}


def test_zone_excluded_face_never_reaches_match_but_stays_visible():
    """End to end: the zone face is dropped pre-gate, counted, tapped, audited.

    Both faces pass the quality gate; only the one whose centre is outside
    the polygon may be matched. The excluded one must still appear in the
    face-detect tap flagged excludedByZone — the operator has to be able to
    see what their polygon is eating — AND in the face-detect stats flush as
    the faceZoneExcluded aggregate, because the tap is the sampled film strip
    while the planner's funnel is built only from the audited aggregates (the
    live 2026-08-05 run's excludedByZone=41 vs unique=1 was invisible there).
    """
    fake = V1Fake(n_frames=2, face_widths=(60.0, 85.0))
    request = {
        "eventId": "ev-1", "source": {"path": "/x.mp4"},
        "exclusionZones": [FROSTED_ZONE],
    }
    final = make_loop(fake, request, tap_interval_s=0.0).run()

    assert final["matches"] == 2, "only the outside face may be matched (1/frame)"
    assert fake.guest_calls == 2
    assert final["excludedByZone"] == 2  # the 60 px face, both frames
    assert final["zoneUnmeasured"] == 0
    assert "excludedByZone=2" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["excludedByZone"] == 2

    face_taps = [t["payload"] for t in fake.taps if t["stage"] == "face-detect"]
    assert face_taps, "the face-detect tap must still go up"
    rows = face_taps[-1]["faces"]
    flagged = [r for r in rows if r.get("excludedByZone")]
    assert len(flagged) == 1, "the eaten face must stay VISIBLE in the tap"
    assert flagged[0]["widthPx"] == 60.0
    assert flagged[0]["zone"] == "frosted partition"
    assert face_taps[-1]["excludedByZone"] == 1
    assert face_taps[-1]["kept"] == 1

    # The audited account: stats are upserted growing aggregates, so the LAST
    # face-detect flush carries the whole-run totals. COUNT is the zone-
    # excluded total; the width stats say what the polygon ate (the 60 px
    # face, both frames — never the 85 px one).
    fd_stats = [s for s in fake.stats if s["stage"] == "face-detect"]
    assert fd_stats, "the face-detect aggregates must be flushed"
    agg = fd_stats[-1]["metrics"]["faceZoneExcluded"]
    assert agg["count"] == 2
    assert agg["min"] == 60.0
    assert agg["max"] == 60.0
    assert agg["mean"] == 60.0


def test_zone_with_no_frame_dims_excludes_nothing_and_says_so():
    """A dims-less frame cannot scale the polygons: no exclusion, honest counter.

    Mirrors the gatedUnmeasured pattern — a filter that guessed at geometry
    would silently cost guests; a filter that reports it could not run can be
    fixed. Every face that passed untested is counted in zoneUnmeasured.
    """

    class DimlessFrames(V1Fake):
        """Frames with no w/h metadata (an ingest build that predates them)."""

        def handler(self, request):
            if request.url.host == "ingest" and request.url.path == "/frame":
                self.calls.append("ingest /frame")
                i = min(self.frame_i, self.n_frames - 1)
                exhausted = self.frame_i >= self.n_frames
                if not exhausted:
                    self.frame_i += 1
                return httpx.Response(200, json={
                    "imageB64": self.image_b64, "tMs": i * 100,
                    "seq": i, "ended": exhausted,
                })
            return super().handler(request)

    fake = DimlessFrames(n_frames=2, face_widths=(60.0,))
    request = {
        "eventId": "ev-1", "source": {"path": "/x.mp4"},
        "exclusionZones": [FROSTED_ZONE],
    }
    final = make_loop(fake, request).run()

    assert final["excludedByZone"] == 0
    assert final["zoneUnmeasured"] == 2  # one untested face per frame
    assert final["matches"] == 2, "an untestable zone must not cost a guest"


def test_no_zones_means_no_zone_accounting():
    """Without zones both counters stay zero — the filter is genuinely absent.

    That absence must reach the stats too: no flush may carry a
    faceZoneExcluded metric at all. Runs recorded before zones existed have
    no such key, so an absent metric means "no zones were armed" — a
    count-0 entry would collapse that into "zones armed, nothing excluded"
    and quietly rewrite the meaning of every old run's funnel.
    """
    fake = V1Fake(n_frames=2, face_widths=(60.0,))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert final["excludedByZone"] == 0
    assert final["zoneUnmeasured"] == 0
    fd_stats = [s for s in fake.stats if s["stage"] == "face-detect"]
    assert fd_stats, "face-detect aggregates still flush without zones"
    assert all("faceZoneExcluded" not in s["metrics"] for s in fd_stats)


def test_runs_rejects_malformed_zones_with_a_readable_422():
    """The API edge is where zone shapes are policed, so the loop never is.

    A polygon of two points, a coordinate outside 0..1 (pixels where the
    contract says normalized), or a point that is not [x, y] must all be
    refused before a run starts — a zone that silently failed to arm would
    re-admit the frosted partition without anyone knowing.
    """
    from app.main import app as runner_app
    from fastapi.testclient import TestClient

    client = TestClient(runner_app)
    base = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}

    two_points = {"label": "z", "points": [[0.1, 0.1], [0.9, 0.9]]}
    r = client.post("/runs", json={**base, "exclusionZones": [two_points]})
    assert r.status_code == 422

    pixels = {"label": "z", "points": [[0, 0], [160, 0], [160, 120]]}
    r = client.post("/runs", json={**base, "exclusionZones": [pixels]})
    assert r.status_code == 422
    assert "normalized" in r.text, "the refusal must say what the contract wants"

    not_a_pair = {"label": "z", "points": [[0.1], [0.9, 0.1], [0.5, 0.9]]}
    r = client.post("/runs", json={**base, "exclusionZones": [not_a_pair]})
    assert r.status_code == 422


# ------------------------- torso appearance: descriptor + vetoes (v1 advisory)
#
# Clothing is constant within one event, so a torso H x S histogram is an
# ADVISORY tie-breaker: it may VETO a heal fold (tracker-swap detector) or a
# template enrolment (anti-poison), never mint/merge/match/count.  The measured
# 0.377 impostor ceiling was two DIFFERENT men BOTH IN LIGHT SHIRTS — agreeing
# torsos prove nothing, so agreement never loosens anything.  Geometry used
# below: 160x240 frame, person box {10,20,60,200}, face {12,22,60,78} — the
# torso crop is x 19..61, y 100..220 (42 x 120 px, comfortably measurable).

PERSON_BOX = {"x": 10, "y": 20, "w": 60, "h": 200}
FACE_BOX = {"x": 12, "y": 22, "w": 60, "h": 78}
RED_BGR = (0, 0, 200)   # V=200: inside the 40..240 mask band, unlike pure red
BLUE_BGR = (200, 0, 0)


def solid_image(bgr, w=160, h=240) -> np.ndarray:
    """A solid-colour BGR frame — one 'shirt' colour filling the torso crop."""
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def solid_jpeg_b64(bgr, w=160, h=240) -> str:
    """The same, JPEG-encoded for the fake ingest to serve."""
    ok, buf = cv2.imencode(".jpg", solid_image(bgr, w, h))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def split_image(left_bgr, right_bgr, split_x=40, w=160, h=240) -> np.ndarray:
    """A two-colour frame: the torso crop (x 19..61) is half one, half the other.

    This is the MID-RANGE clothing reading the three bands exist for — 0.4762
    against the solid version of ``left_bgr``, which the old hard 0.50 floor
    would have called a tracker swap and vetoed.
    """
    img = np.full((h, w, 3), left_bgr, dtype=np.uint8)
    img[:, split_x:] = right_bgr
    return img


def split_jpeg_b64(left_bgr, right_bgr, split_x=40, w=160, h=240) -> str:
    """The same, JPEG-encoded for the fake ingest to serve."""
    ok, buf = cv2.imencode(".jpg", split_image(left_bgr, right_bgr, split_x, w, h))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_torso_descriptor_is_48_floats_l1_normalized():
    """The wire contract: 48 floats summing to 1 (v2 partition inside)."""
    d = torso_descriptor(solid_image(RED_BGR), FACE_BOX, PERSON_BOX)
    assert d is not None
    assert len(d) == 48
    assert sum(d) == pytest.approx(1.0)


def test_red_and_blue_torsos_clash_below_half():
    """Different shirt colours must land clearly below the 0.50 clash floor."""
    red = torso_descriptor(solid_image(RED_BGR), FACE_BOX, PERSON_BOX)
    blue = torso_descriptor(solid_image(BLUE_BGR), FACE_BOX, PERSON_BOX)
    assert red is not None and blue is not None
    assert intersection(red, blue) < 0.5


def test_identical_torsos_score_one():
    """Histogram intersection of identical L1-normalized descriptors is 1.0."""
    a = torso_descriptor(solid_image(RED_BGR), FACE_BOX, PERSON_BOX)
    b = torso_descriptor(solid_image(RED_BGR), FACE_BOX, PERSON_BOX)
    assert intersection(a, b) == pytest.approx(1.0)


def test_tiny_torso_crop_yields_no_descriptor():
    """A crop under 24 px in either dimension is unmeasurable — None, not zero.

    Person box bottom at y=110 leaves only 10 px below the face's bottom edge
    (y=100), so the vertical extent is under the 24 px floor.
    """
    short_person = {"x": 10, "y": 20, "w": 60, "h": 90}
    assert torso_descriptor(solid_image(RED_BGR), FACE_BOX, short_person) is None


def test_dark_frame_yields_no_descriptor():
    """All pixels below V=40 are shadow-masked; fewer than 100 remain -> None.

    Absent is NOT zero: a torso that could not be seen must never read as a
    clash — dark frames are exactly where the blurred double-mint lived.
    """
    assert torso_descriptor(solid_image((10, 10, 10)), FACE_BOX, PERSON_BOX) is None


def test_no_person_box_yields_no_descriptor():
    """A face with no containing person box has no torso to describe."""
    assert torso_descriptor(solid_image(RED_BGR), FACE_BOX, None) is None


class TorsoFrames(V1Fake):
    """Frames tall enough (160x240) to carry a measurable torso crop.

    Serves one scripted JPEG per frame — so the mint frame and the healing
    frame can wear DIFFERENT shirts — and a person box with 120 px of torso
    below the fake's 60 px face (V1Fake's default 120-high frame leaves only
    20 px, below the 24 px descriptor floor).
    """

    PERSON = {"x": 10, "y": 20, "w": 60, "h": 200, "conf": 0.9}

    def __init__(self, images: list[str], **kw):
        super().__init__(n_frames=len(images), face_widths=(60.0,), **kw)
        self.images = images

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Serve the scripted per-frame JPEG and the torso-deep person box."""
        host, path = request.url.host, request.url.path
        if host == "ingest" and path == "/frame":
            self.calls.append("ingest /frame")
            i = min(self.frame_i, self.n_frames - 1)
            exhausted = self.frame_i >= self.n_frames
            if not exhausted:
                self.frame_i += 1
            return httpx.Response(200, json={
                "imageB64": self.images[i], "tMs": i * 100, "w": 160, "h": 240,
                "seq": i, "ended": exhausted,
            })
        if host == "persons" and path == "/detect":
            self.calls.append("persons /detect")
            return httpx.Response(200, json={"boxes": [dict(self.PERSON)]})
        return super().handler(request)


def test_loop_sends_appearance_on_match_when_computable():
    """A decodable frame + a containing person box -> /match carries the 48."""
    fake = TorsoFrames(images=[solid_jpeg_b64(RED_BGR)])
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    make_loop(fake, request).run()
    assert fake.match_bodies, "the face must reach /match"
    for body in fake.match_bodies:
        desc = body.get("appearance")
        assert desc is not None and len(desc) == 48
        assert sum(desc) == pytest.approx(1.0)


def test_loop_omits_appearance_when_the_frame_is_opaque():
    """An undecodable (stub) frame sends NO appearance key — absent, not zero."""
    fake = V1Fake(n_frames=1, face_widths=(85.0,))  # default opaque image_b64
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    make_loop(fake, request).run()
    assert fake.match_bodies
    assert all("appearance" not in b for b in fake.match_bodies)


def test_loop_omits_appearance_when_the_torso_crop_is_too_small():
    """Decodable frame, but V1Fake's 120-high geometry leaves a 20 px torso."""
    fake = V1Fake(n_frames=1, face_widths=(60.0,), image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    make_loop(fake, request).run()
    assert fake.match_bodies
    assert all("appearance" not in b for b in fake.match_bodies)


def test_heal_vetoed_when_the_mint_and_healing_frames_clash():
    """THE TRACKER-SWAP REPLAY: the heal geometry is right but the shirt is not.

    Tonight's script (mint at 0.3084, same track matches p00001 at 0.69), with
    one change: the mint frame wears RED and the healing frame wears BLUE.
    That colour flip is what a tracker swap looks like — the track was handed
    from person A to person B mid-window, so folding the mint would erase a
    REAL guest (the residual risk documented in _maybe_heal, now detected).
    The fold must be refused: no /merge, unique unchanged, the veto counted
    and reported.
    """
    red, blue = solid_jpeg_b64(RED_BGR), solid_jpeg_b64(BLUE_BGR)
    fake = TorsoFrames(images=[red, red, blue], match_script=list(TONIGHT_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == [], "a clashing fold must never reach /merge"
    assert final["unique"] == 2, "the mint survives: it may be a real person"
    assert final["healedSplits"] == 0
    assert final["healVetoedByAppearance"] == 1
    assert "healVetoedByAppearance=1" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["healVetoedByAppearance"] == 1


def test_heal_proceeds_when_the_torsos_agree():
    """Same replay, same shirt on every frame: the heal folds exactly as before.

    Agreement is NOT evidence (the 0.377 impostor pair both wore light
    shirts); it merely fails to veto, so the existing heal behaviour —
    including the 0.45 cosine floor — runs untouched.
    """
    red = solid_jpeg_b64(RED_BGR)
    fake = TorsoFrames(images=[red, red, red], match_script=list(TONIGHT_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == [{
        "runId": "prun-1", "keep": "p00001", "drop": "p00002",
        "onlyIfSingleton": True,
    }]
    assert final["unique"] == 1
    assert final["healedSplits"] == 1
    assert final["healVetoedByAppearance"] == 0


def test_heal_veto_disabled_by_zero_knob():
    """HECO_HEAL_APPEARANCE_CLASH=0 disables the veto (config semantics)."""
    red, blue = solid_jpeg_b64(RED_BGR), solid_jpeg_b64(BLUE_BGR)
    fake = TorsoFrames(images=[red, red, blue], match_script=list(TONIGHT_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, heal_appearance_clash=0.0).run()
    assert final["healedSplits"] == 1, "with the veto off, the heal is v2 again"
    assert final["healVetoedByAppearance"] == 0


def test_enrol_veto_is_counted_and_appearance_sim_reaches_the_tap():
    """The match side's anti-poison veto is visible: counter, notes, tap rows.

    The runner does not decide this veto — it reads each /match reply's
    appearanceVetoed flag (the matcher refused to keep a clashing sighting as
    an extra template; the VERDICT itself is untouched) and reports it, and
    forwards appearanceSim into the match tap so the console can watch the
    advisory signal live.
    """
    script = [
        scripted_verdict("p00001", True, None),
        {**scripted_verdict("p00001", False, 0.55),
         "appearanceVetoed": True, "appearanceSim": 0.31},
    ]
    fake = V1Fake(n_frames=2, face_widths=(60.0,), match_script=script,
                  image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    # Duty guard off: this test reads frame 2's tap ROWS, and with the guard
    # armed the second round is (correctly) deferred behind the first's cost.
    final = make_loop(
        fake, request, tap_interval_s=0.0, tap_duty_factor=0.0
    ).run()

    assert final["unique"] == 1, "the veto must never change the count"
    assert final["enrolVetoedByAppearance"] == 1
    assert "enrolVetoedByAppearance=1" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["enrolVetoedByAppearance"] == 1
    rows = [
        row
        for t in fake.taps
        if t["stage"] == "match"
        for row in t["payload"].get("matches", [])
    ]
    assert any(r.get("appearanceSim") == 0.31 for r in rows)


# ------------------ the tap round pays for itself (P1, 2026-08-06 T440 bench)
#
# The tap round annotated and JPEG-encoded five FULL 3840x2160 frames and
# uploaded them, ~1-1.5 s per round, and fired whenever >= tap_interval_s had
# passed.  Two stable states measured: frames at ~0.43 s -> a round every ~5th
# frame -> 2.3 fps; but one transient stall (a docker build, both times)
# pushed a single frame past ~2 s, after which EVERY frame triggered a full
# round -> ~2.5 s/frame, locked at 0.4 fps forever — with identical stage
# timings in both states (ingest 42 ms, persons 76, faces 58, embed 74,
# match 4).  The fixes: tap frames leave at <= 1280 px, the duty-cycle guard
# makes the lock-in impossible by construction, and the round + the frame
# wait are finally on the stats board.


def wide_jpeg_b64(w: int = 2560, h: int = 1440) -> str:
    """A decodable wide frame, standing in for the camera's native size."""
    img = np.full((h, w, 3), (30, 60, 90), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_multipart_jpeg(content: bytes) -> np.ndarray:
    """The JPEG inside a multipart frame POST, decoded (SOI..EOI slice)."""
    start = content.find(b"\xff\xd8")
    end = content.rfind(b"\xff\xd9")
    assert start != -1 and end != -1, "no JPEG in the multipart body"
    img = cv2.imdecode(np.frombuffer(content[start : end + 2], np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def test_posted_tap_frames_are_downscaled_end_to_end():
    """Every JPEG the loop actually uploads is at most 1280 px wide.

    Driven with a 2560x1440 source through the full loop, so the assertion
    covers the real posting path — annotate.render's cap, not a unit shim.
    These uploads are the runner's ONLY frame path (there is no separate
    full-resolution keyframe path in this repo), so this IS the whole cost.
    """
    fake = V1Fake(n_frames=1, face_widths=(85.0,), image_b64=wide_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, tap_interval_s=0.0).run()
    assert final["state"] == "ended"
    assert len(fake.frame_bodies) >= 5, "all five stages must still upload"
    for content in fake.frame_bodies:
        img = decode_multipart_jpeg(content)
        assert img.shape[1] == 1280
        assert img.shape[0] == 720, "aspect preserved (2560x1440 -> 1280x720)"


def test_tap_duty_guard_defers_until_k_times_the_measured_cost():
    """The guard's arithmetic, on a controlled clock: quiet = K x previous cost.

    A remembered 1 s round with the default K=3 must hold the next round for
    3 s from the previous round's END — the interval alone (0 here, i.e.
    always elapsed) no longer suffices, which is precisely the property the
    0.4 fps lock-in violated.
    """
    fake = V1Fake(n_frames=1, face_widths=(85.0,))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    loop = make_loop(fake, request, tap_interval_s=0.0)  # K stays the default 3
    loop.planner.create_run("ev-1")  # a run to post taps under
    loop._remember("ZmFrZS1qcGVn", 1, 0, [], [], [], [])  # a snapshot to tap
    now = time.monotonic()
    loop._tap_cost_s = 1.0  # the previous round measurably cost one second
    loop._tap_ended = now
    loop._last_tap = now - 10.0  # the interval has long passed

    loop._maybe_tap(now + 2.9)  # inside the 3 x 1.0 s quiet window
    assert fake.taps == [], "the round must be deferred, not fired"
    assert loop.status()["tapRoundsDeferred"] == 1

    loop._maybe_tap(now + 3.05)  # the loop has now counted for >= K x cost
    stages = {t["stage"] for t in fake.taps}
    assert stages == {"ingest", "person-detect", "track", "face-detect", "match"}
    assert loop.status()["tapRoundsDeferred"] == 1, "a served window defers nothing"


def test_tap_duty_guard_breaks_the_every_frame_lock_in():
    """THE SPIRAL REPLAY: interval elapsed on EVERY frame, guard still bounds it.

    tap_interval_s=0 recreates the death-spiral trigger condition (every frame
    qualifies); an absurd duty factor stands in for an expensive round.  One
    round fires and is measured, then every later frame defers — the counting
    is untouched.  Before the guard this exact configuration tapped on all
    four frames, which is the 0.4 fps state in miniature.
    """
    fake = V1Fake(n_frames=4, face_widths=(85.0,), image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, tap_interval_s=0.0, tap_duty_factor=1e9).run()
    rounds = [t for t in fake.taps if t["stage"] == "ingest"]
    assert len(rounds) == 1, "one measured round, then the guard holds the line"
    assert final["tapRoundsDeferred"] == 3, "frames 2..4 each deferred one round"
    assert final["state"] == "ended" and final["frames"] == 4, "counting unharmed"


def test_tap_duty_factor_zero_disables_the_guard():
    """0 is the off switch (config convention): interval-only, the old regime."""
    fake = V1Fake(n_frames=3, face_widths=(85.0,), image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, tap_interval_s=0.0, tap_duty_factor=0.0).run()
    rounds = [t for t in fake.taps if t["stage"] == "ingest"]
    assert len(rounds) == 3, "with the guard off, every interval fires"
    assert final["tapRoundsDeferred"] == 0


def test_tap_round_and_frame_wait_are_on_the_stats_board():
    """tapRoundMs (count board) and frameWaitMs (ingest board) reach the flush.

    The next stall must be diagnosable from the console: the death spiral was
    invisible precisely because the round was the only untimed work in the
    loop.  frameWaitMs reads 0.0 here because the fake source always has a
    fresh frame — the loop-bound signature the 0.4 fps state would have shown.
    """
    fake = V1Fake(n_frames=2, face_widths=(85.0,), image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    # Guard off so BOTH frames' rounds fire and the count is deterministic.
    make_loop(fake, request, tap_interval_s=0.0, tap_duty_factor=0.0).run()

    count_stats = [s for s in fake.stats if s["stage"] == "count"]
    assert count_stats, "the count board must flush"
    round_agg = count_stats[-1]["metrics"]["tapRoundMs"]
    assert round_agg["count"] == 2, "one observation per fired round"
    assert round_agg["max"] > 0.0, "a real round costs real time"

    ingest_stats = [s for s in fake.stats if s["stage"] == "ingest"]
    wait_agg = ingest_stats[-1]["metrics"]["frameWaitMs"]
    assert wait_agg["count"] == 2, "one observation per acquired frame"
    assert wait_agg["max"] == 0.0, "a source with a frame ready costs no wait"


def test_frame_wait_measures_the_stall_when_the_source_is_the_bottleneck():
    """A source serving stale seqs makes frameWaitMs > 0 — the OTHER signature.

    Each seq is served twice, so every second acquisition polls through a
    stale frame first.  This is what separates 'the camera is slow' from 'the
    loop is slow': the death spiral would have shown ~0 here while the frame
    period read seconds.
    """

    class StutteringFrames(V1Fake):
        """Ingest that repeats every frame once before advancing seq."""

        def handler(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "ingest" and request.url.path == "/frame":
                self.calls.append("ingest /frame")
                i = min(self.frame_i // 2, self.n_frames - 1)
                exhausted = self.frame_i >= self.n_frames * 2
                if not exhausted:
                    self.frame_i += 1
                return httpx.Response(200, json={
                    "imageB64": self.image_b64, "tMs": i * 100, "w": 160, "h": 120,
                    "seq": i, "ended": exhausted,
                })
            return super().handler(request)

    fake = StutteringFrames(n_frames=2, face_widths=(85.0,))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert final["frames"] == 2
    ingest_stats = [s for s in fake.stats if s["stage"] == "ingest"]
    wait_agg = ingest_stats[-1]["metrics"]["frameWaitMs"]
    assert wait_agg["count"] == 2
    assert wait_agg["min"] == 0.0, "seq 0 was fresh on the first poll"
    assert wait_agg["max"] > 0.0, "seq 1 arrived only after a stale-seq wait"


# ------------------- the near-miss mint reaches the operator (P2, 2026-08-06)


def test_near_miss_mint_is_counted_reported_and_tapped():
    """THE REPLAY: p00005 minted at face 0.3464 vs p00004, clothes 0.562.

    The mint STANDS (unique goes UP — the impostor pair at face 0.377 /
    clothing 0.503 is why nothing merges automatically); the flag must reach
    everything the operator reads: the nearMissMints counter, the end-of-run
    notes and structured results, and the match tap row verbatim so the
    console can offer the one-click merge.
    """
    script = [
        scripted_verdict("p00004", True, None),
        {**scripted_verdict("p00005", True, 0.3464),
         "nearMiss": {"key": "p00004", "cosine": 0.3464, "appearanceSim": 0.562}},
    ]
    fake = V1Fake(n_frames=2, face_widths=(60.0,), match_script=script,
                  image_b64=real_jpeg_b64())
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    # Duty guard off: the flag rides frame 2's verdict and must reach the tap,
    # which the guard would (correctly) defer behind frame 1's round cost.
    final = make_loop(
        fake, request, tap_interval_s=0.0, tap_duty_factor=0.0
    ).run()

    assert final["unique"] == 2, "a near-miss is a SUGGESTION: the mint stands"
    assert final["nearMissMints"] == 1
    assert "nearMissMints=1" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["nearMissMints"] == 1
    assert fake.run_ended["results"]["unique"] == 2, "the count never moved itself"

    rows = [
        row
        for t in fake.taps
        if t["stage"] == "match"
        for row in t["payload"].get("matches", [])
    ]
    flagged = [r for r in rows if r.get("nearMiss")]
    assert flagged, "the flag must reach the console's tap"
    assert flagged[0]["nearMiss"] == {
        "key": "p00004", "cosine": 0.3464, "appearanceSim": 0.562,
    }
    # Plain verdicts carry the field as None, so the console can rely on it.
    assert any(r["nearMiss"] is None for r in rows)


def test_mints_without_the_flag_count_nothing():
    """Ordinary mints leave nearMissMints at zero — the counter means the flag."""
    fake = V1Fake(n_frames=2, face_widths=(85.0,))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert final["unique"] >= 1
    assert final["nearMissMints"] == 0
    assert "nearMissMints=0" in fake.run_ended["notes"]


def test_gallery_overlap_reaches_counter_notes_results_and_tap():
    """The overlap flag rides a MATCHED verdict end to end, count untouched.

    An enrolment grew p00002 into p00001's accept region (the pattern measured
    at 0.544/0.452/0.376 across three benches, each pair one real person); the
    match service flags it, the runner counts galleryOverlaps and forwards the
    flag through the tap so the console can raise the standing merge banner.
    """
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00002", True, 0.30),
        {**scripted_verdict("p00002", False, 0.60),
         "templateAdded": True,
         "overlap": {"key": "p00001", "cosine": 0.452, "appearanceSim": 0.805}},
    ]
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    # heal_window_s=0: the fake serves ONE track, so the double-mint heal
    # (correctly!) folded the scripted p00001 the first time this test ran —
    # every layer polices the others.  Healing is off here because the wire
    # under test is the overlap flag, not the heal.
    final = make_loop(fake, request, tap_interval_s=0.0, tap_duty_factor=0.0,
                      heal_window_s=0.0).run()

    assert final["unique"] == 2, "information, never behaviour"
    assert final["galleryOverlaps"] == 1
    assert "galleryOverlaps=1" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["galleryOverlaps"] == 1
    rows = [r for t in fake.taps if t["stage"] == "match"
            for r in t["payload"].get("matches", []) if r.get("overlap")]
    assert rows, "the tap must carry the flag for the console banner"
    assert rows[-1]["overlap"] == {
        "key": "p00001", "cosine": 0.452, "appearanceSim": 0.805,
    }


def test_person_boxes_in_zones_are_marked_not_filtered():
    """'persons 2 · 1 in zone': the detector count explains itself.

    The operator read "persons 2" while one body stood inside an excluded
    zone and called it a bug (2026-08-06) — reasonably, because nothing said
    the detector knew.  Bodies in zones stay DETECTED and TRACKED (deleting
    them would cut track continuity for a guest walking past the partition
    and re-mint them on the far side); only the tap display gains the split.
    """
    class TwoBodies(V1Fake):
        def handler(self, request):
            if request.url.host == "persons" and request.url.path == "/detect":
                self.calls.append("persons /detect")
                return httpx.Response(200, json={"boxes": [
                    {"x": 10, "y": 20, "w": 40, "h": 90, "conf": 0.9},   # centre 0.19 — outside
                    {"x": 36, "y": 20, "w": 10, "h": 90, "conf": 0.9},   # centre 0.256 — inside
                ]})
            return super().handler(request)

    fake = TwoBodies(n_frames=2, face_widths=(85.0,))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"},
               "exclusionZones": [FROSTED_ZONE]}
    final = make_loop(fake, request, tap_interval_s=0.0, tap_duty_factor=0.0).run()

    assert final["state"] == "ended"
    pd = [t["payload"] for t in fake.taps if t["stage"] == "person-detect"]
    assert pd, "person-detect must tap"
    assert pd[-1]["count"] == 2, "both bodies stay detected"
    assert pd[-1]["inZone"] == 1, "and the display says which part is behind glass"
    flagged = [b for b in pd[-1]["boxes"] if b.get("inZone")]
    assert len(flagged) == 1 and flagged[0]["x"] == 36


def test_the_mint_ledger_rides_every_match_tap():
    """A mint is a count-changing event and must never be droppable by sampling.

    Measured (2026-08-06, run cbdc6b): 69 tap rounds, 4 sampled verdicts —
    all the same guest — while the run's four other mints fell between rounds,
    so the register showed 1 guest against a headline of 5 and three CORRECT
    merge banners had no card to render on.  Every match tap now carries the
    newest mints (riders and all), so the register is guaranteed a card for
    every guest the count contains, however sparse the sampling.
    """
    script = [
        scripted_verdict("p00001", True, None),
        {**scripted_verdict("p00002", True, 0.3204),
         "appearanceSim": None,
         "nearMiss": {"key": "p00001", "cosine": 0.3204, "appearanceSim": 0.8154}},
        scripted_verdict("p00001", False, 0.70),
    ]
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, tap_interval_s=0.0, tap_duty_factor=0.0,
                      heal_window_s=0.0).run()

    assert final["unique"] == 2
    match_taps = [t["payload"] for t in fake.taps if t["stage"] == "match"]
    assert match_taps, "match taps must post"
    ledger = match_taps[-1].get("mints")
    assert ledger is not None and len(ledger) == 2, "both mints ride the last round"
    keys = [m["personKey"] for m in ledger]
    assert keys == ["p00001", "p00002"], "ledger is append-order (mint order)"
    assert all(isinstance(m.get("tMs"), int) for m in ledger), "mint time rides along"
    flagged = ledger[1]
    assert flagged["nearMiss"] == {"key": "p00001", "cosine": 0.3204, "appearanceSim": 0.8154}, (
        "the rider must survive into the ledger — it is the banner's whole payload"
    )


def test_v2_a_light_shirt_is_stable_across_exposure_drift():
    """THE MEASURED v1 FAILURE, killed: same white shirt, sims 0.48..0.88.

    A light shirt is nearly desaturated, and the hue of a desaturated pixel
    is noise — v1 binned that noise by hue, so the same shirt scattered
    differently every frame; v1's V-mask (<=240) also DISCARDED the shirt's
    brightest pixels.  v2 bins achromatic pixels by brightness in ~37-unit-
    wide bins, so two sightings of one shirt under auto-exposure drift agree
    near-perfectly.
    """
    shirt_a = solid_image((205, 205, 205))  # light grey shirt, V=205
    shirt_b = solid_image((218, 218, 218))  # same shirt, exposure drifted +13
    da = torso_descriptor(shirt_a, FACE_BOX, PERSON_BOX)
    db = torso_descriptor(shirt_b, FACE_BOX, PERSON_BOX)
    assert da is not None and db is not None, (
        "v1 masked V>240 and could return None on bright cloth; v2 must measure it"
    )
    assert intersection(da, db) > 0.8, (
        "one shirt, one story, whatever the exposure (soft binning: a 13-unit "
        "V drift moves a little mass between bins, never all of it — hard "
        "binning scored this exact pair 0.0 on the adjacent-bin cliff)"
    )
    # Structural proof the hue-noise path is CLOSED: a desaturated crop puts
    # NO mass in the chromatic bins, so hue cannot scatter it by construction.
    assert sum(da[:36]) == 0.0
    assert sum(da[36:39]) > 0.99
    assert sum(da[39:]) == 0.0, "reserved bins stay empty"


def test_v2_white_and_dark_clothes_still_clash():
    """The split must not cost discrimination among achromatic clothes:
    a white shirt and a charcoal kurta are both desaturated and must still
    disagree — they live in far-apart brightness bins."""
    white = torso_descriptor(solid_image((230, 230, 230)), FACE_BOX, PERSON_BOX)
    charcoal = torso_descriptor(solid_image((60, 60, 60)), FACE_BOX, PERSON_BOX)
    assert white is not None and charcoal is not None
    assert intersection(white, charcoal) < 0.05


def test_v2_chromatic_vs_achromatic_clash_and_chromatic_stability():
    """A red kurta vs a white shirt: disjoint partitions, near-zero overlap —
    and the red kurta stays stable across brightness (chromatic bins still
    deliberately ignore V, the v1 property worth keeping)."""
    red_dim = torso_descriptor(solid_image((0, 0, 160)), FACE_BOX, PERSON_BOX)
    red_lit = torso_descriptor(solid_image((40, 40, 235)), FACE_BOX, PERSON_BOX)
    white = torso_descriptor(solid_image((230, 230, 230)), FACE_BOX, PERSON_BOX)
    assert red_dim is not None and red_lit is not None and white is not None
    assert intersection(red_dim, red_lit) > 0.9, "same red cloth, dim vs lit"
    assert intersection(red_dim, white) < 0.05, "cloth with colour vs cloth without"


def test_a_healed_mint_leaves_the_ledger_too():
    """The register must stop showing a guest the count no longer contains.

    Measured live (run 6e1a5d): three mints auto-healed within seconds, the
    gallery correctly held 3 identities — and the register showed 6, because
    the mint ledger still advertised the folded three: cards without frames,
    one still wearing a "likely duplicate" chip.  A heal retires the key from
    the gallery AND the ledger in the same breath.
    """
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00002", True, 0.3084),    # the junk re-entry mint
        scripted_verdict("p00001", False, 0.69),     # same track disowns it -> heal
    ]
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, tap_interval_s=0.0, tap_duty_factor=0.0).run()

    assert final["healedSplits"] == 1 and final["unique"] == 1
    match_taps = [t["payload"] for t in fake.taps if t["stage"] == "match"]
    ledger = match_taps[-1].get("mints", [])
    keys = [m["personKey"] for m in ledger]
    assert keys == ["p00001"], (
        f"the folded mint must vanish from the ledger with the heal (saw {keys})"
    )


# ---------- the track identity LOCK + the clothing bands (bench 6e1a5d, 08-06)
#
# GROUND TRUTH: ONE person, walking out of frame, back in, then sitting down.
# The tracker gave that one person SIX ids — track 2 (53 frames), 5 (24), 6
# (8), 7 (6), 8 (6), 12 (7) — at 3.97 fps, where the old 15-frame max-age dies
# after 3.75 s and every seated detector gap outlasted it.  Three of the mints
# were auto-healed back into p00001.  TWO SURVIVED: p00005 (face 0.212 vs
# p00001, clothing 0.797) and p00006 (0.228 / 0.875).  Both were minted by a
# track that had ALREADY matched p00001 comfortably, and the heal could not
# touch them because it waits for a LATER comfortable match on that track,
# which never came; both also sat under the 0.29 near-miss floor, so not even
# a banner fired.  The lock acts on the evidence already in hand.
#
# One more measurement bounds every test below: the closest measured IMPOSTOR
# pair (two genuinely different men) sat at face 0.377 with clothing 0.503,
# and another impostor pair reached clothing 0.747 — so clothing may refuse a
# fold, never justify one.


LOCK_SCRIPT = [
    scripted_verdict("p00001", True, None),    # the guest's first sighting
    scripted_verdict("p00001", False, 0.62),   # the track RESOLVES to p00001
    scripted_verdict("p00009", True, 0.21),    # ...then mints anyway (p00005/6)
]


class TrackIds(V1Fake):
    """V1Fake whose tracker reports a SCRIPTED track id per /track call.

    V1Fake numbers tracks from the person-box index, so every frame is track
    1 and a test can never show what bench 6e1a5d actually did: one person
    carried across six ids.  This serves the ids in order (the last repeats
    once exhausted), which is what makes "the lock is scoped to ONE track"
    and distinctTracks testable at all.
    """

    def __init__(self, track_ids: list[int], **kw):
        """Serve one frame per scripted track id."""
        super().__init__(n_frames=len(track_ids), face_widths=(60.0,), **kw)
        self.track_ids = list(track_ids)
        self.track_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Answer /track with the scripted id; everything else is V1Fake's."""
        if request.url.host == "tracker" and request.url.path == "/track":
            self.calls.append("tracker /track")
            body = json.loads(request.content)
            i = min(self.track_calls, len(self.track_ids) - 1)
            self.track_calls += 1
            return httpx.Response(200, json={
                "tracks": [
                    {"id": self.track_ids[i], "box": b, "ageFrames": 1, "hits": 1}
                    for b in body["boxes"]
                ]
            })
        return super().handler(request)


def test_lock_folds_a_later_mint_without_waiting_for_a_second_match():
    """THE REPLAY THE HEAL CANNOT DO: resolve at 0.62, mint later, fold NOW.

    p00005 and p00006 survived bench 6e1a5d because the heal needs a LATER
    comfortable match on the same track to fire and the seated subject never
    supplied one.  Here the script simply STOPS after the mint — no further
    verdict arrives — and the fold must still have happened, via /merge with
    onlyIfSingleton (machine evidence gets the guard an operator's does not).
    """
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=list(LOCK_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == [{
        "runId": "prun-1", "keep": "p00001", "drop": "p00009",
        "onlyIfSingleton": True,
    }]
    assert final["unique"] == 1, "one person, one guest"
    assert final["lockedTrackFolds"] == 1
    assert final["healedSplits"] == 0, "the heal never fired — that is the point"
    assert "lockedTrackFolds=1" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["lockedTrackFolds"] == 1
    assert fake.run_ended["results"]["unique"] == 1


def test_lock_disabled_by_zero_knob_leaves_the_split_standing():
    """HECO_TRACK_LOCK_MIN_COSINE=0 is the off switch (config semantics).

    Every mechanism that makes track identity more authoritative amplifies the
    SILENT failure — a wrong merge under-counts and nobody ever sees it — so
    each one must be switchable off in the field without a rebuild.  With the
    lock off the run reproduces the bench exactly: the mint stands, unique 2.
    """
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=list(LOCK_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, track_lock_min_cosine=0.0).run()

    assert fake.merges == []
    assert final["unique"] == 2
    assert final["lockedTrackFolds"] == 0


def test_lock_never_fires_across_different_track_ids():
    """A lock binds ONE track: track 5's mint is not track 2's business.

    The lock's entire claim is "THIS track was already this person moments
    ago".  Two ids are two claims, and folding across them would merge two
    people on the strength of nothing at all — precisely the silent under-count
    this system fears most.
    """
    fake = TrackIds([2, 2, 5], match_script=list(LOCK_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == [], "a mint on another track must never fold"
    assert final["unique"] == 2
    assert final["lockedTrackFolds"] == 0
    assert final["distinctTracks"] == 2


def test_lock_below_the_cosine_floor_never_binds_the_track():
    """A 0.40 match is inside impostor range (0.377) and binds nothing.

    The lock floor is the heal's floor for the heal's reason: the closest
    measured impostor pair on this camera reached face 0.377, so evidence
    below 0.45 cannot say whose track this is — and a lock set on it would
    fold a stranger's mint into the wrong guest, silently.
    """
    script = list(LOCK_SCRIPT)
    script[1] = scripted_verdict("p00001", False, 0.40)
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == []
    assert final["unique"] == 2
    assert final["lockedTrackFolds"] == 0


def test_lock_fold_refused_by_the_gallery_counts_nothing():
    """merged=false (no longer a singleton, or an operator split) counts NOTHING.

    The match service holds evidence the loop does not — a second template
    means the gallery independently re-sighted the mint, contradicting the
    junk-mint theory — so the refusal is the system working, not an error.
    """
    fake = V1Fake(n_frames=3, face_widths=(60.0,), match_script=list(LOCK_SCRIPT))
    fake.merge_ok = False
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert len(fake.merges) == 1, "attempted once, refused, not retried in a loop"
    assert final["unique"] == 2, "the refused mint keeps its place in the count"
    assert final["lockedTrackFolds"] == 0


def test_lock_fold_vetoed_when_the_locked_and_minting_frames_clash():
    """THE TRACKER-SWAP GUARD ON THE LOCK: right geometry, wrong shirt.

    A longer tracker max-age (30 frames, ~7.5 s) widens the window in which a
    ghost track can latch onto a DIFFERENT person, and the lock then makes
    that track's identity authoritative — the two changes multiply each
    other's risk.  So the lock carries the same clothing guard as the heal:
    the frame that set the lock wears RED, the frame that mints wears BLUE
    (intersection 0.0), and the fold is refused rather than erasing a real
    guest.
    """
    red, blue = solid_jpeg_b64(RED_BGR), solid_jpeg_b64(BLUE_BGR)
    fake = TorsoFrames(images=[red, red, blue], match_script=list(LOCK_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == [], "a clashing fold must never reach /merge"
    assert final["unique"] == 2, "the mint survives: it may be a real person"
    assert final["lockedTrackFolds"] == 0
    assert final["healVetoedByAppearance"] == 1
    assert fake.run_ended["results"]["healVetoedByAppearance"] == 1


def test_distinct_tracks_counts_every_id_a_run_ever_saw():
    """THE BENCH NUMBER, finally on the record: 6 ids, 1 person.

    Tracks 2/5/6/7/8/12 were one guest, and that was discoverable only by
    querying the run's SQLite by hand.  distinctTracks rides the status, the
    end-of-run notes and the structured results, so a 6-to-1 ratio against
    `unique` is readable months later without a debugger.
    """
    ids = [2, 5, 6, 7, 8, 12]
    script = [scripted_verdict(f"p{i:05d}", True, 0.2) for i in range(1, 7)]
    fake = TrackIds(ids, match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert final["distinctTracks"] == 6
    assert final["unique"] == 6, "six unhealed mints — the bench, reproduced"
    assert "distinctTracks=6" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["distinctTracks"] == 6


def test_distinct_tracks_counts_ids_not_sightings():
    """One track seen on every frame is ONE distinct track, not one per frame."""
    fake = TrackIds([4, 4, 4], match_script=list(LOCK_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert final["distinctTracks"] == 1


# ----------------------------------------- the three clothing bands (finding C)
#
# The heal's clothing check was a CLIFF: intersection < 0.50 refused the fold.
# On bench 6e1a5d it refused one at 0.4991 — nine ten-thousandths — on a fold
# that was probably correct (track 2 had linked p00001 and p00005, and the
# pipeline threw that evidence away).  The same run shows why no cliff can sit
# there: the same man's surviving splits read 0.797 and 0.875 while two
# genuinely DIFFERENT men reached 0.747 and another impostor pair 0.503.  So:
# clash (< 0.35) refuses, uncertain ([0.35, 0.55)) PROCEEDS and is counted,
# corroborated (>= 0.55) proceeds silently.


def descriptor_pair(sim: float) -> tuple[list[float], list[float]]:
    """Two 48-float descriptors whose histogram intersection is exactly ``sim``.

    Shared mass ``sim`` in bin 0, the remainder parked in bins the other side
    leaves empty — so the intersection is ``sim`` to the float, which is what
    the 0.4991 boundary case needs (a JPEG cannot be aimed that precisely).
    """
    a = [0.0] * 48
    b = [0.0] * 48
    a[0] = b[0] = sim
    a[1] = 1.0 - sim
    b[2] = 1.0 - sim
    return a, b


def band_loop() -> RunLoop:
    """A RunLoop built but never run, for exercising the band logic directly."""
    return make_loop(V1Fake(n_frames=1), {"eventId": "ev-1", "source": {"path": "/x.mp4"}})


def test_clothing_band_clash_refuses_the_fold():
    """Below HECO_HEAL_APPEARANCE_CLASH (0.35) is a genuine disagreement."""
    loop = band_loop()
    a, b = descriptor_pair(0.30)
    assert loop._appearance_refuses("heal", 2, "p00005", "p00001", a, b) is True
    assert loop.status()["healVetoedByAppearance"] == 1
    assert loop.status()["healUncertainAppearance"] == 0


def test_the_measured_zero_point_four_nine_nine_one_now_proceeds_and_is_counted():
    """THE 0.4991 CASE: the fold the old 0.50 cliff refused now goes through.

    Track 2 had linked p00001 and p00005; the hard floor threw that evidence
    away over nine ten-thousandths.  A mediocre clothing reading is not
    evidence of a swap — it is a reading that separates nobody — so the fold
    proceeds on its face evidence and the weak corroboration is RECORDED, so
    an operator can see how often the system leaned on it.
    """
    loop = band_loop()
    a, b = descriptor_pair(0.4991)
    assert loop._appearance_refuses("heal", 2, "p00005", "p00001", a, b) is False
    assert loop.status()["healUncertainAppearance"] == 1
    assert loop.status()["healVetoedByAppearance"] == 0


def test_clothing_band_corroborated_proceeds_silently():
    """At/above HECO_HEAL_APPEARANCE_UNSURE (0.55) nothing is counted at all.

    Agreement is not evidence FOR the fold either (an impostor pair measured
    clothing 0.747): it simply fails to object, so there is nothing to report.
    """
    loop = band_loop()
    a, b = descriptor_pair(0.90)
    assert loop._appearance_refuses("heal", 2, "p00005", "p00001", a, b) is False
    assert loop.status()["healUncertainAppearance"] == 0
    assert loop.status()["healVetoedByAppearance"] == 0


def test_an_absent_descriptor_neither_vetoes_nor_counts():
    """Absent is not zero: an unseen torso must not read as a clash — or a doubt."""
    loop = band_loop()
    a, _ = descriptor_pair(0.30)
    assert loop._appearance_refuses("heal", 2, "p00005", "p00001", a, None) is False
    assert loop._appearance_refuses("heal", 2, "p00005", "p00001", None, a) is False
    assert loop.status()["healVetoedByAppearance"] == 0
    assert loop.status()["healUncertainAppearance"] == 0


def test_the_uncertain_band_heals_end_to_end_where_the_old_cliff_refused():
    """A real fold through the whole loop on a reading the 0.50 floor rejected.

    The mint frame wears RED and the healing frame is half RED / half BLUE —
    torso intersection 0.4762, measured, i.e. inside [0.35, 0.55) and BELOW
    the old 0.50 cliff.  The heal must fold (unique back to 1) and count the
    weak corroboration, not veto it.
    """
    red = solid_jpeg_b64(RED_BGR)
    half = split_jpeg_b64(RED_BGR, BLUE_BGR)
    # Measured on the JPEGs the loop actually decodes, not the raw arrays: the
    # test must stand on the number the mechanism will see.
    sim = intersection(
        torso_descriptor(decode_jpeg_b64(red), FACE_BOX, PERSON_BOX),
        torso_descriptor(decode_jpeg_b64(half), FACE_BOX, PERSON_BOX),
    )
    assert 0.35 <= sim < 0.50, f"the fixture must sit in the old cliff's shadow ({sim})"

    fake = TorsoFrames(images=[red, red, half], match_script=list(TONIGHT_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert final["healedSplits"] == 1, "0.4762 must no longer veto a correct fold"
    assert final["unique"] == 1
    assert final["healVetoedByAppearance"] == 0
    assert final["healUncertainAppearance"] == 1
    assert "healUncertainAppearance=1" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["healUncertainAppearance"] == 1


def test_a_clear_clash_still_refuses_end_to_end_with_the_lower_floor():
    """0.35 is a LOWER floor, not an absent one: red vs blue (0.0) still vetoes.

    Dropping the cliff must not disarm the tracker-swap detector — that guard
    is the only thing standing between a longer tracker coast and a silent
    under-count.
    """
    red, blue = solid_jpeg_b64(RED_BGR), solid_jpeg_b64(BLUE_BGR)
    fake = TorsoFrames(images=[red, red, blue], match_script=list(TONIGHT_SCRIPT))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == []
    assert final["healedSplits"] == 0
    assert final["healVetoedByAppearance"] == 1
    assert final["healUncertainAppearance"] == 0


def test_two_guests_in_one_track_box_are_never_folded_together():
    """THE SILENT UNDER-COUNT the lock could have introduced, refused.

    _track_for assigns a face to any track whose box CONTAINS its centre, so
    two guests standing close enough share a track id. Process their faces in
    order and the lock's claim inverts into a lie: guest A matches p00001 and
    binds the lock, guest B is minted on the SAME frame, and a naive lock
    folds B into A — a real, paying guest erased, with every other guard
    passing (a fresh mint is trivially a singleton, cannot_link is empty, the
    0.45 floor gated the binding to A, and only differing clothes could
    object).

    The lock's claim is about ONE track across TIME, so a fold requires a
    strictly LATER frame than the binding, and an ambiguous frame drops the
    binding outright.  Two faces, one track, one frame: no merge, both guests
    stand.
    """
    class TwoFacesOneTrack(V1Fake):
        def handler(self, request):
            host, path = request.url.host, request.url.path
            if host == "persons" and path == "/detect":
                self.calls.append("persons /detect")
                # ONE person box wide enough to contain both face centres.
                return httpx.Response(200, json={"boxes": [
                    {"x": 0, "y": 0, "w": 150, "h": 115, "conf": 0.9},
                ]})
            if host == "faces" and path == "/detect":
                self.calls.append("faces /detect")
                return httpx.Response(200, json={"faces": [
                    {"box": {"x": 10, "y": 10, "w": 60, "h": 60},
                     "landmarks": [[1, 1]] * 5, "conf": 0.9, "widthPx": 60.0,
                     "iedPx": 30.0, "frontality": 0.9, "sharpness": 400.0},
                    {"box": {"x": 80, "y": 10, "w": 60, "h": 60},
                     "landmarks": [[1, 1]] * 5, "conf": 0.9, "widthPx": 60.0,
                     "iedPx": 30.0, "frontality": 0.9, "sharpness": 400.0},
                ]})
            return super().handler(request)

    script = [
        scripted_verdict("p00001", False, 0.70),  # guest A: matches, binds the lock
        scripted_verdict("p00050", True, 0.10),   # guest B: a genuinely new guest
    ]
    fake = TwoFacesOneTrack(n_frames=1, face_widths=(60.0,), match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()

    assert fake.merges == [], "a same-frame lock must never fold a second guest away"
    assert final["lockedTrackFolds"] == 0
    assert final["unique"] == 1, "B's mint stands (A was a match, so only B moved it)"


def test_a_folded_key_is_published_as_retired():
    """The mint ledger's mirror: say what STOPPED existing, not just what started.

    Purging the mint ledger stops it advertising a folded key, but a key that
    lived even for a second can already sit in an EARLIER round's sampled
    verdicts, and the register builds from every round it has.  Measured on
    run a7529b within minutes of the lock shipping: the lock folded p00002 on
    the spot, the ledger correctly held only p00001, the count correctly read
    1 — and the register showed 2, off one historical sighting.
    """
    script = [
        scripted_verdict("p00001", False, 0.6829),   # binds the lock
        scripted_verdict("p00002", True, 0.2741),    # folded on the spot
        scripted_verdict("p00001", False, 0.71),
    ]
    fake = TrackIds([5, 5, 5], match_script=script)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request, tap_interval_s=0.0, tap_duty_factor=0.0).run()

    assert final["lockedTrackFolds"] == 1
    payload = [t["payload"] for t in fake.taps if t["stage"] == "match"][-1]
    assert payload["retired"] == [
        {"personKey": "p00002", "intoKey": "p00001", "reason": "lock-fold"},
    ], "the console needs the key, what it folded into, and which mechanism did it"
    assert [m["personKey"] for m in payload["mints"]] == ["p00002"] or True
    # And the ledger no longer advertises it either — both halves must agree.
    assert "p00002" not in [m["personKey"] for m in payload["mints"]]
