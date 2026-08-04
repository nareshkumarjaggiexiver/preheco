"""Run-loop tests for the v1 additions: staff counting, taps, feedback, enrol.

Like test_loop.py, everything runs against a fully fake pipeline over
httpx.MockTransport — no sockets. This fake extends the stage/planner routing
with the v1 surface: staff-aware /match, the merge/split/mark-staff and
/staff/enrol match endpoints, and the planner taps/frames/feedback endpoints.
"""

import base64
import json

import cv2
import httpx
import numpy as np
from app.config import Settings
from app.loop import RunLoop, httpx_file_transport, httpx_transport
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
    ):
        """Serve n_frames; ``staff_width`` faces come back tagged as staff."""
        self.n_frames = n_frames
        self.face_widths = face_widths
        self.image_b64 = image_b64
        self.staff_width = staff_width
        self.feedback_items = list(feedback_items or [])
        self.frame_i = 0
        self.guest_n = 0
        self.guest_calls = 0
        self.calls: list[str] = []
        self.taps: list[dict] = []
        self.frames_posted: list[str] = []  # stage of each multipart frame POST
        self.feedback_polls = 0
        self.resolved: list[tuple[str, str]] = []  # (feedbackId, status)
        self.enrolled: dict | None = None
        self.tombstones: list[dict] = []          # served once, like feedback
        self.tombstones_confirmed: list[str] = []
        self.purges: list[dict] = []
        self.enrol_report: dict | None = None
        self.merges: list[dict] = []
        self.splits: list[dict] = []
        self.mark_staff: list[dict] = []
        self.manual_counts: list[dict] = []
        self.opened: dict | None = None
        self.closed: dict | None = None
        self.run_ended: dict | None = None
        self.drop_resolve = 0  # PUT /api/feedback/:id to swallow (dropped reply)
        self.merge_ok = True  # False once the drop key no longer exists
        self.feedback_sticky = False  # re-serve open items until a PUT lands

    # -- helpers -----------------------------------------------------------

    def _planner(self, request, path, body):
        if path == "/api/pipeline/runs" and request.method == "POST":
            return httpx.Response(200, json={"id": "prun-1"})
        if path == "/api/pipeline/runs/prun-1" and request.method == "PUT":
            self.run_ended = body
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/stats") or path.endswith("/samples"):
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/taps"):
            self.taps.append(body)
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/frames"):
            # multipart: read the stage field out of the recorded call list
            self.frames_posted.append("frame")
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
            merged, self.merge_ok = self.merge_ok, False  # a merge is one-shot
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
                i = min(self.frame_i, self.n_frames - 1)
                if self.frame_i < self.n_frames:
                    self.frame_i += 1
                return httpx.Response(200, json={
                    "imageB64": self.image_b64, "tMs": i * 100, "w": 160, "h": 120, "seq": i,
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
    # Enrol does not create a pipeline run or reset the gallery.
    assert "match /reset" not in fake.calls and "planner /api/pipeline/runs" not in fake.calls


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
