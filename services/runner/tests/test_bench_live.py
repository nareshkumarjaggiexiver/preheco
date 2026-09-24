"""scripts/bench-live.py against a FAKE console — never the real one.

The bench drives the planner's own API (POST /api/pipeline/control/start,
the run launcher's body) and reports from the settled run record.  This
pins that it sends exactly what the launcher sends, finds the run the start
produced, waits for it to settle, and reports throughput, the count and
every lever counter; the upload path streams a multipart body of the right
length.  A throwaway HTTP server stands in for the console.
"""

import importlib.util
import io
import json
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "bench-live.py"


def load_bench():
    """Import the hyphen-named script as a module."""
    spec = importlib.util.spec_from_file_location("bench_live", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeConsole:
    """Just enough of the planner's /api for one bench run."""

    def __init__(self):
        self.started: list[dict] = []
        self.uploads: list[dict] = []
        self.polls = 0
        self.new_run = False

    def record(self) -> dict:
        """The run record: running for two polls, then settled."""
        self.polls += 1
        if self.polls <= 2:
            return {"id": "prun-9", "status": "running",
                    "stats": [{"stage": "count", "frames": 40 * self.polls, "fps": 15.5}]}
        return {
            "id": "prun-9", "status": "ended", "endReason": "source-ended",
            "startedAt": "2026-09-24T21:00:00.000Z", "endedAt": "2026-09-24T21:02:00.000Z",
            "config": {"levers": {"pipelineOverlap": True},
                       "faceRegion": {"x": 0.25, "y": 0.0, "w": 0.5, "h": 0.6},
                       "models": {"faces": "scrfd_2.5g_kps.onnx"}},
            "results": {"unique": 74, "frames": 1500, "framesCaptured": 1815,
                        "framesSkippedNoMotion": 315, "faceDetectSkippedSettled": 900},
            "stats": [
                {"stage": "count", "frames": 1500, "fps": 12.5,
                 "metrics": {"stepMs": {"count": 1500, "mean": 61.0, "max": 140.0},
                             "detectWaitMs": {"count": 1500, "mean": 4.0, "max": 30.0}}},
                {"stage": "face-detect", "frames": 600, "fps": 5.0,
                 "metrics": {"faceDetectMs": {"count": 600, "mean": 48.0, "max": 90.0}}},
            ],
        }

    def serve(self):
        """Start the server on a free port; returns (server, base url)."""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _json(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):  # noqa: N802 — http.server's name
                if self.path == "/api/events/ev-7/runs":
                    rows = [{"id": "old-1"}] + ([{"id": "prun-9"}] if fake.new_run else [])
                    return self._json(200, rows)
                if self.path == "/api/pipeline/runs/prun-9?include=stats":
                    return self._json(200, fake.record())
                return self._json(404, {"error": self.path})

            def do_POST(self):  # noqa: N802 — http.server's name
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n)
                if self.path == "/api/pipeline/control/start":
                    fake.started.append(json.loads(raw))
                    fake.new_run = True  # the runner creates the row a beat later
                    return self._json(200, {"runId": "run-abc", "state": "starting"})
                if self.path == "/api/pipeline/videos":
                    fake.uploads.append({"length": n, "type": self.headers["content-type"],
                                         "body": raw})
                    return self._json(201, {"id": "vid-up"})
                return self._json(404, {"error": self.path})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def console():
    """A fake console, shut down after the test."""
    fake = FakeConsole()
    server, base = fake.serve()
    yield fake, base
    server.shutdown()


def test_the_bench_sends_the_launchers_body_and_reports_the_levers(console):
    """Start body, run discovery, polling to settlement, and the report."""
    fake, base = console
    bench = load_bench()
    out = io.StringIO()
    with redirect_stdout(out):
        code = bench.main([
            "--console", base, "--event", "ev-7", "--video-id", "vid-3",
            "--pipeline", "heco-faces",
            "--quality", '{"faceReverifyIntervalS": 2}',
            "--model-profile", '{"id": "live", "stages": {"faces": "scrfd-2.5g"}}',
            "--face-region", "0.25,0,0.5,0.6",
        ], sleep=lambda _s: None)
    assert code == 0
    assert fake.started == [{
        "eventId": "ev-7", "mode": "count", "videoId": "vid-3", "lockstep": True,
        "pipelineId": "heco-faces", "quality": {"faceReverifyIntervalS": 2},
        "modelProfile": {"id": "live", "stages": {"faces": "scrfd-2.5g"}},
        "faceRegion": {"x": 0.25, "y": 0.0, "w": 0.5, "h": 0.6},
    }]
    text = out.getvalue()
    assert "run prun-9: ended (source-ended)" in text
    assert "unique 74   frames 1500   wall 120.0 s" in text
    # 1500 processed in 120 s = 12.5 fps; 1815 footage frames = 15.1 fps.
    assert "processed 12.50 fps   footage 15.12 fps   camera 15 fps" in text
    assert "kept camera rate: True" in text
    assert "faceDetectSkippedSettled: 900" in text
    assert "framesDroppedLive: —" in text, "absent is shown as absent, not zero"
    assert "count.detectWaitMs: mean 4.0 ms" in text


def test_the_bench_json_summary_and_a_failed_run(console):
    """--json is machine-readable; a run that did not end cleanly exits 1."""
    fake, base = console
    bench = load_bench()
    settled = fake.record

    def failed():
        r = settled()
        return r if r["status"] == "running" else {**r, "status": "failed",
                                                   "endReason": "source-stalled"}

    fake.record = failed
    out = io.StringIO()
    with redirect_stdout(out):
        code = bench.main(["--console", base, "--event", "ev-7", "--video-id", "v",
                           "--no-lockstep", "--json"], sleep=lambda _s: None)
    assert code == 1
    summary = json.loads(out.getvalue())
    assert summary["status"] == "failed" and summary["keptCameraRate"] is True
    assert summary["counters"]["framesSkippedNoMotion"] == 315
    assert fake.started[0]["lockstep"] is False


def test_the_bench_uploads_a_clip_as_a_streamed_multipart_body(console, tmp_path):
    """--upload ships the file as the launcher does, then runs the new videoId."""
    fake, base = console
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x01fake-mp4" * 1000)
    bench = load_bench()
    with redirect_stdout(io.StringIO()):
        assert bench.main(["--console", base, "--event", "ev-7", "--upload", str(clip)],
                          sleep=lambda _s: None) == 0
    (up,) = fake.uploads
    assert up["type"].startswith("multipart/form-data; boundary=")
    assert b'name="files"; filename="clip.mp4"' in up["body"]
    assert clip.read_bytes() in up["body"]
    assert up["length"] == len(up["body"])
    assert fake.started[0]["videoId"] == "vid-up"
