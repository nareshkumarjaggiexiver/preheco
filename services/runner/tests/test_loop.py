"""Run-loop tests against a fully fake pipeline (httpx.MockTransport).

No sockets, no network: the MockTransport routes by hostname (ingest, persons,
tracker, faces, embed, match, planner) exactly as compose DNS would, and a
FakePipeline records every call so the tests can verify orchestration order,
aggregation maths, batch flushing, and end-of-source handling.

The fake ingest mimics the real service's semantics: it serves the LATEST
frame with an increasing ``seq`` and, once the source is exhausted, keeps
re-serving the last frame with a stalled seq — the loop must detect the stall
and end the run (there is no explicit EOF flag on the real service).
"""

import json

import httpx
import pytest
from app.config import Settings
from app.loop import RunLoop, httpx_transport
from heco_common.planner import PlannerClient

B64 = "ZmFrZS1qcGVn"  # the runner treats frames as opaque base64


class FakePipeline:
    """Scripted stand-ins for all six stage services plus the planner."""

    def __init__(self, n_frames=3, face_widths=(85.0, 64.0, 40.0), boxes_per_frame=1):
        """Serve n_frames frames; each has faces of the given widths."""
        self.n_frames = n_frames
        self.face_widths = face_widths
        self.boxes_per_frame = boxes_per_frame
        self.frame_i = 0
        self.match_calls = 0
        self.calls: list[str] = []  # "host path" in arrival order
        self.stats: list[dict] = []
        self.sample_batches: list[list] = []
        self.run_created: dict | None = None
        self.run_ended: dict | None = None
        self.embed_face_counts: list[int] = []
        self.match_qualities: list[float] = []
        self.match_reset: dict | None = None
        self.tracker_reset: dict | None = None
        self.opened: dict | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Route one request to the scripted service behaviour."""
        host, path = request.url.host, request.url.path
        self.calls.append(f"{host} {path}")
        body = json.loads(request.content) if request.content else {}

        if host == "planner":
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
                self.sample_batches.append(body["samples"])
                return httpx.Response(200, json={"ok": True})

        if host == "ingest":
            if path == "/open":
                self.opened = body
                return httpx.Response(200, json={"ok": True})
            if path == "/frame":
                # Real-ingest semantics: latest frame only; seq stalls at EOF.
                i = min(self.frame_i, self.n_frames - 1)
                if self.frame_i < self.n_frames:
                    self.frame_i += 1
                return httpx.Response(
                    200,
                    json={"imageB64": B64, "tMs": i * 100, "w": 320, "h": 240, "seq": i},
                )

        if host == "persons" and path == "/detect":
            i = self.frame_i - 1  # frame currently in flight
            boxes = [
                {"x": 10, "y": 20, "w": 40, "h": 100.0 + 10 * i, "conf": 0.9}
                for _ in range(self.boxes_per_frame)
            ]
            return httpx.Response(200, json={"boxes": boxes})

        if host == "tracker":
            if path == "/reset":
                self.tracker_reset = body
                return httpx.Response(200, json={"ok": True, "runId": body["runId"]})
            if path == "/track":
                if "runId" not in body or not isinstance(body.get("tMs"), int):
                    return httpx.Response(422, json={"detail": "runId/tMs required"})
                tracks = [
                    {"id": n + 1, "box": b, "ageFrames": self.frame_i, "hits": 1}
                    for n, b in enumerate(body["boxes"])
                ]
                return httpx.Response(200, json={"tracks": tracks})

        if host == "faces" and path == "/detect":
            assert body.get("within"), "faces must be searched within tracked boxes"
            faces = [
                {
                    "box": {"x": 12, "y": 22, "w": w, "h": w * 1.3},
                    "landmarks": [[1, 1]] * 5,
                    "conf": 0.8,
                    "widthPx": w,
                }
                for w in self.face_widths
            ]
            return httpx.Response(200, json={"faces": faces, "inferMs": 1.0})

        if host == "embed" and path == "/embed":
            n = len(body["faces"])
            self.embed_face_counts.append(n)
            return httpx.Response(200, json={"embeddings": [[0.1] * 128] * n, "alignMs": 0.5})

        if host == "match":
            if path == "/reset":
                self.match_reset = body
                return httpx.Response(200, json={"ok": True})
            if path == "/match":
                self.match_calls += 1
                self.match_qualities.append(body["quality"])
                is_new = self.match_calls % 2 == 1  # every other face is new
                return httpx.Response(
                    200,
                    json={
                        "personKey": f"p{self.match_calls:05d}",
                        "isNew": is_new,
                        "cosine": 0.2 + 0.1 * (self.match_calls % 3),
                        "galleryN": (self.match_calls + 1) // 2,
                        "subCanon": body["quality"] < 80.0,
                    },
                )

        return httpx.Response(500, json={"error": f"unscripted {host} {path}"})


def make_loop(fake: FakePipeline, **settings_kw) -> RunLoop:
    """Build a RunLoop wired to the fake pipeline over MockTransport."""
    settings = Settings(
        ingest_url="http://ingest:7101",
        persons_url="http://persons:7102",
        tracker_url="http://tracker:7103",
        faces_url="http://faces:7104",
        embed_url="http://embed:7105",
        match_url="http://match:7106",
        planner_url="http://planner:8787",
        flush_interval_s=0.0,  # flush every frame — deterministic in tests
        source_poll_s=0.001,
        source_stall_s=0.05,  # stalled-seq EOF detection, test-fast
        **settings_kw,
    )
    client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    planner = PlannerClient(settings.planner_url, transport=httpx_transport(client))
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4", "loop": False}}
    return RunLoop("run-local", request, settings, client, planner)


def test_orchestration_order():
    """Setup order, then per-frame stage order, then teardown order."""
    fake = FakePipeline(n_frames=2)
    loop = make_loop(fake)
    final = loop.run()
    assert final["state"] == "ended"

    non_planner = [c for c in fake.calls if not c.startswith("planner")]
    # Setup: gallery + tracker reset, then source open — after the planner
    # run exists (its id keys all downstream per-run state).
    assert fake.calls[0] == "planner /api/pipeline/runs"
    assert non_planner[0] == "match /reset"
    assert non_planner[1] == "tracker /reset"
    assert non_planner[2] == "ingest /open"
    # Per-frame order (2 kept faces -> 2 match calls per frame).
    per_frame = [
        "ingest /frame",
        "persons /detect",
        "tracker /track",
        "faces /detect",
        "embed /embed",
        "match /match",
        "match /match",
    ]
    assert non_planner[3 : 3 + 7] == per_frame
    assert non_planner[10 : 10 + 7] == per_frame
    # Teardown: stalled-seq polls on ingest, then the planner PUT last.
    assert non_planner[-1] == "ingest /frame"
    assert fake.calls[-1] == "planner /api/pipeline/runs/prun-1"
    assert fake.match_reset == {"runId": "prun-1"}
    assert fake.tracker_reset == {"runId": "prun-1"}
    assert fake.opened == {"path": "/x.mp4", "loop": False}


def test_quality_gate_and_sub_canon_share():
    """40 px face never reaches embed; 64 px is matched but sub-canon."""
    fake = FakePipeline(n_frames=2, face_widths=(85.0, 64.0, 40.0))
    final = make_loop(fake).run()
    assert fake.embed_face_counts == [2, 2]  # the 40 px face was gated out
    assert sorted(set(fake.match_qualities)) == [64.0, 85.0]
    # Per frame: matches for 85 (canon) and 64 (sub-canon) -> share 0.5.
    assert final["matches"] == 4
    assert final["subCanonShare"] == pytest.approx(0.5)
    assert final["unique"] == 2  # fake match alternates isNew


def test_stat_aggregation_maths():
    """Box heights 100, 110, 120 across frames aggregate to mean 110."""
    fake = FakePipeline(n_frames=3)
    make_loop(fake).run()
    pd = [s for s in fake.stats if s["stage"] == "person-detect"][-1]
    m = pd["metrics"]["personBoxHPx"]
    assert m == {"count": 3, "min": 100.0, "mean": 110.0, "max": 120.0}
    assert pd["frames"] == 3
    assert pd["fps"] >= 0.0
    # Every contract stage reported stats at least once.
    seen = {s["stage"] for s in fake.stats}
    assert seen == {
        "ingest", "person-detect", "track", "face-detect",
        "quality", "embed", "match", "count",
    }
    # matchCosine flows through to the match stage aggregates.
    mc = [s for s in fake.stats if s["stage"] == "match"][-1]["metrics"]["matchCosine"]
    assert mc["count"] == 6  # 3 frames x 2 kept faces


def test_batch_flushing_respects_cap():
    """More raw rows than the cap: batches stay <= cap, overflow is counted."""
    fake = FakePipeline(n_frames=1, boxes_per_frame=9, face_widths=(85.0,))
    # 9 person rows + 1 face row + 1 embed row + 1 match row = 12 candidates
    loop = make_loop(fake, sample_batch_max=5)
    final = loop.run()
    assert fake.sample_batches, "samples must be posted"
    assert all(len(b) <= 5 for b in fake.sample_batches)
    assert final["samplesDropped"] > 0
    total_rows = sum(len(b) for b in fake.sample_batches)
    assert total_rows + final["samplesDropped"] == 12


def test_end_of_source_ends_run_with_notes():
    """Seq stall -> final flush -> PUT ended with the run summary."""
    fake = FakePipeline(n_frames=3)
    final = make_loop(fake).run()
    assert final["state"] == "ended"
    assert final["frames"] == 3  # the stalled repeats are never re-processed
    assert fake.run_ended is not None
    assert fake.run_ended["status"] == "ended"
    assert "unique=3" in fake.run_ended["notes"]  # 6 matches, alternating isNew
    assert "frames=3" in fake.run_ended["notes"]
    assert "subCanonShare=0.50" in fake.run_ended["notes"]
    # The PUT is the very last planner interaction (after the final flush).
    planner_calls = [c for c in fake.calls if c.startswith("planner")]
    assert planner_calls[-1] == "planner /api/pipeline/runs/prun-1"


def test_explicit_ended_body_also_ends_run():
    """A stub-style {"ended": true} response ends the run immediately."""

    class ExplicitEnd(FakePipeline):
        def handler(self, request):
            if request.url.path == "/frame" and self.frame_i >= self.n_frames:
                self.calls.append("ingest /frame")
                return httpx.Response(200, json={"ended": True})
            return super().handler(request)

    fake = ExplicitEnd(n_frames=2)
    final = make_loop(fake).run()
    assert final["state"] == "ended"
    assert final["frames"] == 2


def test_stage_failure_marks_run_failed():
    """A stage returning 500 fails the run locally and in the planner."""

    class BrokenPersons(FakePipeline):
        def handler(self, request):
            if request.url.host == "persons":
                self.calls.append("persons /detect")
                return httpx.Response(500, json={"error": "boom"})
            return super().handler(request)

    fake = BrokenPersons(n_frames=2)
    final = make_loop(fake).run()
    assert final["state"] == "failed"
    assert "500" in final["error"]
    assert fake.run_ended["status"] == "failed"


def test_stop_ends_rtsp_style_source():
    """A source that never ends is stopped via stop(); run still settles."""
    fake = FakePipeline(n_frames=10_000)
    loop = make_loop(fake)

    orig = fake.handler

    def stopping_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/frame" and fake.frame_i == 3:
            loop.stop()  # as POST /runs/:id/stop would
        return orig(request)

    loop.client = httpx.Client(transport=httpx.MockTransport(stopping_handler))
    loop.planner.transport = httpx_transport(loop.client)
    final = loop.run()
    assert final["state"] == "ended"
    assert 3 <= final["frames"] <= 5
    assert fake.run_ended["status"] == "ended"
