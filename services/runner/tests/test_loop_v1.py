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
        self.enrol_report: dict | None = None
        self.merges: list[dict] = []
        self.splits: list[dict] = []
        self.mark_staff: list[dict] = []
        self.run_ended: dict | None = None

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
            items, self.feedback_items = self.feedback_items, []  # serve once
            return httpx.Response(200, json={"feedback": items})
        if path.startswith("/api/feedback/") and request.method == "PUT":
            self.resolved.append((path.rsplit("/", 1)[-1], body["status"]))
            return httpx.Response(200, json={"ok": True})
        if path.startswith("/api/staff/") and request.method == "PUT":
            self.enrol_report = {"id": path.rsplit("/", 1)[-1], **body}
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(500, json={"error": f"unscripted planner {path}"})

    def _match(self, path, body):
        if path == "/reset":
            return httpx.Response(200, json={"ok": True})
        if path == "/merge":
            self.merges.append(body)
            return httpx.Response(200, json={"merged": True, "galleryN": max(0, self.guest_n - 1)})
        if path == "/split":
            self.splits.append(body)
            return httpx.Response(200, json={"ok": True, "galleryN": self.guest_n})
        if path == "/mark-staff":
            self.mark_staff.append(body)
            return httpx.Response(200, json={
                "moved": 1, "galleryN": max(0, self.guest_n - 1),
                "staffKey": body.get("staffId") or "anon00001",
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
                return httpx.Response(200, json={"ok": True})
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
            if path == "/reset":
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
    """A staff face is a staffCrossing; only guest faces grow matches/unique."""
    fake = V1Fake(n_frames=3, face_widths=(99.0, 85.0), staff_width=99.0)
    request = {"eventId": "ev-1", "siteId": "site-1", "source": {"path": "/x.mp4"}}
    final = make_loop(fake, request).run()
    assert final["state"] == "ended"
    assert final["staffCrossings"] == 3  # the 99 px face each frame
    assert final["matches"] == 3  # only the 85 px guest face counts as a match
    assert "staffCrossings=3" in fake.run_ended["notes"]


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
