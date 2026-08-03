#!/usr/bin/env python3
"""End-to-end smoke test for the heco pipeline — no docker required.

Launches all seven services as subprocess uvicorns on their contract ports,
preferring each service's real code (services/<name>/ with its own .venv) and
falling back to a built-in stub when a service is not implemented yet or does
not come healthy.  Generates a synthetic walking-rectangle video, runs a full
run against a FAKE planner (a tiny in-process HTTP server that records every
ingest call), and asserts:

  1. the planner run was created,
  2. stats were posted for every active stage,
  3. the run ended cleanly,
  4. the unique count is >= 0.

HONEST LIMITATION: the synthetic frames contain a moving white rectangle and
NO real faces.  With the real faces/embed services, face-detect finds nothing,
so embed/match legitimately see zero work; with the stubs, plausible face
boxes and embeddings are fabricated so the full chain is exercised.  Either
way this validates PLUMBING — orchestration, timing, stat aggregation,
planner reporting — not accuracy.  The first real-face validation happens at
the POC bench with the real camera (CONTRACTS.md "POC geometry").

Usage:
    python3.12 -m venv .e2e-venv
    .e2e-venv/bin/pip install -r scripts/requirements-e2e.txt
    .e2e-venv/bin/python scripts/e2e_smoke.py

(Internal: `--stub NAME --port N` runs one built-in stub service; the main
process spawns those itself.)
"""

import argparse
import base64
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVICES = {  # contract ports (CONTRACTS.md topology table)
    "ingest": 7101,
    "persons": 7102,
    "tracker": 7103,
    "faces": 7104,
    "embed": 7105,
    "match": 7106,
    "runner": 7100,
}
STAGES_ALWAYS = ("ingest", "person-detect", "track", "face-detect", "quality", "count")
STAGES_FACE_CHAIN = ("embed", "match")


# --------------------------------------------------------------------- video


def make_walking_rectangle_video(path: Path, frames: int = 48) -> None:
    """Write an MJPG video of a white rectangle walking left to right."""
    import cv2
    import numpy as np

    w, h = 320, 240
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), 30, np.uint8)
        x = 20 + int(i * (w - 100) / frames)
        cv2.rectangle(img, (x, 60), (x + 50, 200), (255, 255, 255), -1)
        out.write(img)
    out.release()


# --------------------------------------------------------------- fake planner


class FakePlanner:
    """A tiny HTTP server that records every pipeline-ingest call it gets."""

    def __init__(self) -> None:
        """Bind to an ephemeral port; nothing is served until start()."""
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _record_and_reply(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
                with outer._lock:
                    outer.calls.append({"method": method, "path": self.path, "body": body})
                reply = {"ok": True}
                if method == "POST" and self.path == "/api/pipeline/runs":
                    reply = {"id": "prun-e2e"}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802 — http.server API
                self._record_and_reply("POST")

            def do_PUT(self):  # noqa: N802 — http.server API
                self._record_and_reply("PUT")

            def log_message(self, *_):  # keep the smoke output readable
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]

    def start(self) -> None:
        """Serve in a daemon thread."""
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def recorded(self, method: str, path_suffix: str) -> list[dict]:
        """Return recorded calls matching method + path suffix."""
        with self._lock:
            return [
                c for c in self.calls if c["method"] == method and c["path"].endswith(path_suffix)
            ]


# --------------------------------------------------------------------- stubs


def build_stub_app(name: str):
    """Return a FastAPI app imitating one stage service's contract shape.

    Stubs are deliberately simple: persons finds the bright rectangle by
    thresholding, faces FABRICATES a face box inside each person box (there
    are no real faces in synthetic frames), embed derives a deterministic
    pseudo-embedding from the face width so match sees repeatable identities.
    """
    import cv2
    import numpy as np
    from fastapi import FastAPI

    app = FastAPI(title=f"stub-{name}")
    state: dict = {"frames": [], "i": 0, "face_flip": 0, "age": 0}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": f"stub-{name}", "version": "e2e"}

    def _decode(b64: str) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)

    if name == "ingest":

        @app.post("/open")
        def open_source(body: dict) -> dict:
            cap = cv2.VideoCapture(body.get("path") or body.get("url"))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ok, buf = cv2.imencode(".jpg", frame)
                frames.append(base64.b64encode(buf.tobytes()).decode())
            cap.release()
            state["frames"], state["i"] = frames, 0
            return {"ok": True, "frames": len(frames)}

        @app.get("/frame")
        def frame() -> dict:
            # Real-ingest semantics: serve the LATEST frame; at EOF keep
            # re-serving the last one with a stalled seq (no explicit end).
            i = min(state["i"], len(state["frames"]) - 1)
            if state["i"] < len(state["frames"]):
                state["i"] += 1
            return {
                "imageB64": state["frames"][i],
                "tMs": int(i * (1000.0 / 15.0)),
                "w": 320,
                "h": 240,
                "seq": i,
            }

    elif name == "persons":

        @app.post("/detect")
        def detect(body: dict) -> dict:
            gray = cv2.cvtColor(_decode(body["imageB64"]), cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                boxes.append(
                    {"x": float(x), "y": float(y), "w": float(w), "h": float(h), "conf": 0.9}
                )
            return {"boxes": boxes}

    elif name == "tracker":

        @app.post("/track")
        def track(body: dict) -> dict:
            state["age"] += 1
            tracks = [
                {"id": i + 1, "box": b, "age": state["age"]}
                for i, b in enumerate(body.get("boxes", []))
            ]
            return {"tracks": tracks}

    elif name == "faces":

        @app.post("/detect")
        def faces_detect(body: dict) -> dict:
            faces = []
            for box in body.get("within", []):
                state["face_flip"] ^= 1
                w = 64.0 if state["face_flip"] else 85.0  # sub-canon / canon mix
                faces.append(
                    {
                        "box": {"x": box["x"] + 8, "y": box["y"] + 4, "w": w, "h": w * 1.3},
                        "landmarks": [
                            [box["x"] + 8 + dx, box["y"] + 10 + dy]
                            for dx, dy in ((10, 10), (30, 10), (20, 20), (12, 32), (28, 32))
                        ],
                        "conf": 0.9,
                    }
                )
            return {"faces": faces}

    elif name == "embed":

        @app.post("/embed")
        def embed(body: dict) -> dict:
            out = []
            for f in body.get("faces", []):
                rng = np.random.default_rng(int(f["box"]["w"]))  # width == identity
                v = rng.normal(size=128)
                out.append((v / np.linalg.norm(v)).tolist())
            return {"embeddings": out}

    else:
        raise SystemExit(f"no stub for service {name!r}")

    return app


# ------------------------------------------------------------ orchestration


def wait_health(port: int, timeout_s: float = 20.0) -> bool:
    """Poll GET /health until {ok:true} or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.loads(r.read()).get("ok"):
                    return True
        except OSError:
            time.sleep(0.3)
    return False


def port_free(port: int) -> bool:
    """True if nothing is listening on the port."""
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def launch_real(name: str, port: int, env: dict) -> subprocess.Popen | None:
    """Launch services/<name> with its own venv, or None if not launchable."""
    svc = ROOT / "services" / name
    py = svc / ".venv" / "bin" / "python"
    if not (svc / "app" / "main.py").exists() or not py.exists():
        return None
    cmd = [str(py), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)]
    return subprocess.Popen(
        cmd, cwd=svc, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def launch_stub(name: str, port: int) -> subprocess.Popen:
    """Launch a built-in stub for the service via self-invocation."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--stub", name, "--port", str(port)]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def http_json(method: str, url: str, body: dict | None = None) -> dict:
    """One JSON request with stdlib urllib (the smoke needs no client deps)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run_smoke(force_stubs: set[str]) -> int:
    """Run the whole smoke sequence; returns a process exit code.

    ``force_stubs`` names services to stub even when their real code exists —
    `--force-stubs persons,faces,embed` exercises the full face chain (embed +
    match see work) on synthetic footage, which real detectors, correctly,
    would never give them.
    """
    import os

    for name, port in SERVICES.items():
        if not port_free(port):
            print(f"FATAL: port {port} ({name}) already in use")
            return 2

    planner = FakePlanner()
    planner.start()
    print(f"fake planner listening on 127.0.0.1:{planner.port}")

    tmp = Path(tempfile.mkdtemp(prefix="heco-e2e-"))
    video = tmp / "walking-rectangle.avi"
    make_walking_rectangle_video(video)
    print(f"synthetic video: {video}")

    env = dict(os.environ)
    for name, port in SERVICES.items():
        env[f"HECO_{name.upper()}_URL"] = f"http://127.0.0.1:{port}"
    env["PLANNER_URL"] = f"http://127.0.0.1:{planner.port}"
    env["HECO_MATCH_DATA_DIR"] = str(tmp / "galleries")
    env["HECO_FLUSH_INTERVAL_S"] = "1.0"
    env["HECO_SOURCE_STALL_S"] = "2.0"  # snappy EOF detection for the smoke

    procs: dict[str, subprocess.Popen] = {}
    stubbed: set[str] = set()
    try:
        for name, port in SERVICES.items():
            p = None if name in force_stubs else launch_real(name, port, env)
            if p is not None and wait_health(port):
                procs[name] = p
                print(f"  {name:8s} :{port}  REAL (services/{name})")
                continue
            if p is not None:
                p.terminate()
                print(f"  {name:8s} :{port}  real service failed health — falling back to stub")
            if name == "runner" or name == "match":
                print(f"FATAL: {name} is a real deliverable and must come healthy")
                return 2
            procs[name] = launch_stub(name, port)
            stubbed.add(name)
            if not wait_health(port):
                print(f"FATAL: stub {name} failed health on :{port}")
                return 2
            print(f"  {name:8s} :{port}  STUB (built into e2e_smoke.py)")

        # ------------------------------------------------------------- run
        created = http_json(
            "POST",
            "http://127.0.0.1:7100/runs",
            {
                "eventId": "e2e-smoke",
                "source": {"path": str(video)},
                "plannerUrl": env["PLANNER_URL"],
            },
        )
        run_id = created["runId"]
        print(f"run started: {run_id}")

        deadline = time.monotonic() + 120
        status: dict = {}
        while time.monotonic() < deadline:
            status = http_json("GET", f"http://127.0.0.1:7100/runs/{run_id}")
            if status["state"] in ("ended", "failed"):
                break
            time.sleep(0.5)

        print(f"final status: {json.dumps(status)}")

        # -------------------------------------------------------- asserts
        failures = []
        if status.get("state") != "ended":
            failures.append(
                f"run did not end cleanly: {status.get('state')} ({status.get('error')})"
            )
        if not planner.recorded("POST", "/api/pipeline/runs"):
            failures.append("planner never saw the run-created POST")

        stat_stages = {c["body"].get("stage") for c in planner.recorded("POST", "/stats")}
        required = set(STAGES_ALWAYS)
        full_chain = {"persons", "faces", "embed"} <= stubbed
        if full_chain:
            required |= set(STAGES_FACE_CHAIN)
        missing = required - stat_stages
        if missing:
            failures.append(f"no stats posted for stages: {sorted(missing)}")
        if not full_chain and not (set(STAGES_FACE_CHAIN) & stat_stages):
            print(
                "NOTE: embed/match saw no work — expected with real detectors on "
                "synthetic frames (no real faces). Plumbing before them is verified."
            )

        ended = planner.recorded("PUT", "/api/pipeline/runs/prun-e2e")
        if not ended:
            failures.append("planner never saw the run-ended PUT")
        elif ended[-1]["body"].get("status") != "ended":
            failures.append(f"run-ended PUT had status {ended[-1]['body'].get('status')!r}")

        unique = status.get("unique", -1)
        if not isinstance(unique, int) or unique < 0:
            failures.append(f"unique count must be >= 0, got {unique!r}")

        for b in (c["body"] for c in planner.recorded("POST", "/samples")):
            if len(b.get("samples", [])) > 200:
                failures.append("a samples batch exceeded 200 rows")
                break

        print()
        if failures:
            for f in failures:
                print(f"FAIL: {f}")
            return 1
        print(
            f"PASS: run ended, stats for {sorted(stat_stages)}, "
            f"unique={unique} frames={status.get('frames')} "
            f"subCanonShare={status.get('subCanonShare'):.2f} "
            f"(stubs: {sorted(stubbed) or 'none'})"
        )
        print("Reminder: this validates plumbing, not accuracy — see module docstring.")
        return 0
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        planner.server.shutdown()


def main() -> int:
    """CLI entry: default runs the smoke; --stub runs one stub service."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stub", help="internal: run the named built-in stub service")
    ap.add_argument("--port", type=int, help="internal: stub port")
    ap.add_argument(
        "--force-stubs",
        default="",
        help="comma-separated services to stub even if real code exists "
        "(e.g. persons,faces,embed to exercise the full face chain)",
    )
    args = ap.parse_args()
    if args.stub:
        import uvicorn

        uvicorn.run(
            build_stub_app(args.stub), host="127.0.0.1", port=args.port, log_level="warning"
        )
        return 0
    return run_smoke({s.strip() for s in args.force_stubs.split(",") if s.strip()})


if __name__ == "__main__":
    sys.exit(main())
