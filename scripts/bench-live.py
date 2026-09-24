#!/usr/bin/env python3
"""Run one clip through the console and report whether it kept camera rate.

The bench for tonight's throughput levers.  It drives the site-planner
console's own API — the same POST /api/pipeline/control/start the run
launcher sends (RunLauncher.jsx start()) — so the run goes through exactly
the path an operator's does: the planner resolves the uploaded video to a
finite, lockstepped source, forwards the quality and model profiles, and
the runner stamps the levers it ran with into the run's config.  Then it
polls the run record until the run settles and prints what the levers
bought and what they skipped:

* throughput: processed frames per wall second, and FOOTAGE frames per wall
  second (ingest's framesCaptured when the motion gate reports it — the
  number that has to reach camera rate), against --camera-fps;
* the count: unique, frames, the end reason;
* every lever counter the run record carries (framesSkippedNoMotion,
  framesDroppedLive, ingestBacklogMax, faceDetectSkippedSettled,
  faceRegionUnplaced) and the levers stamped in its config;
* per-stage mean wall times from the run's stats (persons, faces, embed,
  match, the loop's step, and detectWaitMs under the overlap).

Stdlib only, so it runs from any python3 on the laptop or the box.  It
never touches the pipeline directly; point it at a console you mean to run
a count on.

    python3 scripts/bench-live.py --event EV --video-id VID \\
        [--pipeline heco-faces] [--quality '{"faceReverifyIntervalS": 2}'] \\
        [--model-profile '{"id": "live", "stages": {"faces": "scrfd-2.5g"}}'] \\
        [--face-region 0.25,0,0.5,0.6] [--no-lockstep] [--json]
    python3 scripts/bench-live.py --event EV --upload clip.mp4 ...
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

#: Stage metrics worth a line in the report: (stage, metric).
STAGE_METRICS = (
    ("ingest", "frameWaitMs"),
    ("person-detect", "personDetectMs"),
    ("track", "trackMs"),
    ("face-detect", "faceDetectMs"),
    ("embed", "embedMs"),
    ("match", "matchMs"),
    ("count", "stepMs"),
    ("count", "loopMs"),
    ("count", "detectWaitMs"),
)

#: The lever counters the runner puts in a run's results when it measured them.
LEVER_COUNTERS = (
    "framesCaptured",
    "framesSkippedNoMotion",
    "framesDroppedLive",
    "ingestBacklogMax",
    "faceDetectSkippedSettled",
    "faceRegionUnplaced",
)


class Console:
    """The planner's /api, as the run launcher speaks it."""

    def __init__(self, base: str, token: str | None = None, timeout_s: float = 30.0):
        """``base`` is the console origin, e.g. http://localhost:5173."""
        self.api = base.rstrip("/") + "/api"
        self.headers = {"authorization": f"Bearer {token}"} if token else {}
        self.timeout_s = timeout_s

    def call(self, method: str, path: str, body=None):
        """One JSON request; raises RuntimeError with the console's own words."""
        data = json.dumps(body).encode() if body is not None else None
        headers = dict(self.headers)
        if data is not None:
            headers["content-type"] = "application/json"
        req = urllib.request.Request(self.api + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
        return json.loads(raw) if raw else None

    def upload(self, clip: Path) -> str:
        """POST /pipeline/videos (multipart ``files``), streamed; returns the videoId."""
        boundary = f"bench-{uuid.uuid4().hex}"
        head = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
            f"filename=\"{clip.name}\"\r\nContent-Type: video/mp4\r\n\r\n"
        ).encode()
        tail = f"\r\n--{boundary}--\r\n".encode()
        size = clip.stat().st_size

        def body():
            yield head
            with clip.open("rb") as f:
                while chunk := f.read(1 << 20):
                    yield chunk
            yield tail

        req = urllib.request.Request(
            self.api + "/pipeline/videos", data=body(), method="POST",
            headers={**self.headers,
                     "content-type": f"multipart/form-data; boundary={boundary}",
                     "content-length": str(len(head) + size + len(tail))},
        )
        with urllib.request.urlopen(req, timeout=max(self.timeout_s, 600.0)) as r:
            reply = json.loads(r.read() or b"{}")
        if not reply.get("id"):
            raise RuntimeError(f"upload answered without an id: {reply}")
        return reply["id"]

    def run_ids(self, event_id: str) -> set[str]:
        """The event's run ids (a bare list today, a paged {items} tomorrow)."""
        rows = self.call("GET", f"/events/{urllib.parse.quote(str(event_id))}/runs")
        if isinstance(rows, dict):
            rows = rows.get("items") or rows.get("runs") or []
        return {str(r["id"]) for r in rows or [] if isinstance(r, dict) and r.get("id")}

    def run(self, run_id: str) -> dict:
        """The run record with its per-stage stats (never the samples)."""
        return self.call("GET", f"/pipeline/runs/{urllib.parse.quote(run_id)}?include=stats")


def start_body(args) -> dict:
    """The body RunLauncher.jsx start() sends for an uploaded-video count."""
    body = {
        "eventId": args.event,
        "mode": "count",
        "videoId": args.video_id,
        "lockstep": args.lockstep,
    }
    if args.pipeline:
        body["pipelineId"] = args.pipeline
    if args.quality:
        body["quality"] = json.loads(args.quality)
    if args.model_profile:
        body["modelProfile"] = json.loads(args.model_profile)
    if args.face_region:
        x, y, w, h = (float(v) for v in args.face_region.split(","))
        body["faceRegion"] = {"x": x, "y": y, "w": w, "h": h}
    return body


def wait_for_new_run(console: Console, event_id: str, before: set[str],
                     timeout_s: float = 15.0, poll_s: float = 0.25, sleep=time.sleep) -> str:
    """The run row the start produced (the runner creates it a beat later)."""
    deadline = time.monotonic() + timeout_s
    while True:
        fresh = console.run_ids(event_id) - before
        if fresh:
            return sorted(fresh)[0]
        if time.monotonic() >= deadline:
            raise RuntimeError("the pipeline accepted the start but no run row appeared")
        sleep(poll_s)


def _stage(record: dict, stage: str) -> dict:
    for s in record.get("stats") or []:
        if s.get("stage") == stage:
            return s
    return {}


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    parse = lambda v: datetime.fromisoformat(v.replace("Z", "+00:00"))  # noqa: E731
    return max(0.0, (parse(end) - parse(start)).total_seconds())


def summarise(record: dict, camera_fps: float) -> dict:
    """What the run bought and what it skipped, from the settled run record."""
    results = record.get("results") or {}
    count = _stage(record, "count")
    frames = results.get("frames", count.get("frames"))
    wall = _seconds(record.get("startedAt"), record.get("endedAt"))
    captured = results.get("framesCaptured")
    fps = frames / wall if frames is not None and wall else None
    footage_fps = (captured if captured is not None else frames) / wall if wall and (
        captured is not None or frames is not None) else None
    stages = {}
    for stage, metric in STAGE_METRICS:
        m = (_stage(record, stage).get("metrics") or {}).get(metric)
        if m and m.get("count"):
            stages[f"{stage}.{metric}"] = {"mean": m.get("mean"), "max": m.get("max"),
                                           "n": m.get("count")}
    config = record.get("config") or {}
    return {
        "runId": record.get("id"),
        "status": record.get("status"),
        "endReason": record.get("endReason"),
        "unique": results.get("unique"),
        "frames": frames,
        "wallS": wall,
        "fps": fps,
        "footageFps": footage_fps,
        "cameraFps": camera_fps,
        "keptCameraRate": None if footage_fps is None else footage_fps >= camera_fps,
        "countStageFps": count.get("fps"),
        "levers": config.get("levers"),
        "faceRegion": config.get("faceRegion"),
        "models": config.get("models"),
        "devices": config.get("devices"),
        "counters": {k: results[k] for k in LEVER_COUNTERS if k in results},
        "stages": stages,
    }


def _fmt(v, nd=1):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def print_report(s: dict, out=None) -> None:
    """The human table (to stdout unless told otherwise)."""
    out = out or sys.stdout
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"run {s['runId']}: {s['status']} ({s['endReason']})")
    p(f"  unique {s['unique']}   frames {s['frames']}   wall {_fmt(s['wallS'])} s")
    p(f"  processed {_fmt(s['fps'], 2)} fps   footage {_fmt(s['footageFps'], 2)} fps   "
      f"camera {s['cameraFps']:g} fps   kept camera rate: {_fmt(s['keptCameraRate'])}")
    p(f"  count-stage fps (runner's own board): {_fmt(s['countStageFps'], 2)}")
    p(f"  levers {json.dumps(s['levers'])}   faceRegion {json.dumps(s['faceRegion'])}")
    p(f"  models {json.dumps(s['models'])}   devices {json.dumps(s['devices'])}")
    for k in LEVER_COUNTERS:
        p(f"  {k:>26}: {_fmt(s['counters'].get(k))}")
    for k, m in s["stages"].items():
        p(f"  {k:>26}: mean {_fmt(m['mean'])} ms  max {_fmt(m['max'])} ms  (n={m['n']})")


def main(argv=None, sleep=time.sleep) -> int:
    """Start, wait, report; 0 when the run ended cleanly."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--console", default="http://localhost:5173",
                    help="console origin (the planner API lives under /api)")
    ap.add_argument("--event", required=True, help="eventId to count under")
    ap.add_argument("--pipeline", help="pipelineId (omit for the console's default)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-id", help="an already-uploaded video's id")
    src.add_argument("--upload", type=Path, help="upload this clip first, then run it")
    ap.add_argument("--no-lockstep", dest="lockstep", action="store_false",
                    help="let ingest drop frames like a camera (default: every frame)")
    ap.add_argument("--quality", help="quality profile JSON, e.g. '{\"faceReverifyIntervalS\": 2}'")
    ap.add_argument("--model-profile", help="model profile JSON {id?, name?, stages}")
    ap.add_argument("--face-region", help="x,y,w,h normalised 0..1 (sent as faceRegion)")
    ap.add_argument("--camera-fps", type=float, default=15.0)
    ap.add_argument("--poll-s", type=float, default=3.0)
    ap.add_argument("--timeout-s", type=float, default=6 * 3600.0,
                    help="give up waiting after this long (the run keeps going)")
    ap.add_argument("--token", default=os.environ.get("HECO_PLANNER_TOKEN"),
                    help="bearer token (default $HECO_PLANNER_TOKEN; loopback needs none)")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)

    console = Console(args.console, args.token)
    if args.upload:
        args.video_id = console.upload(args.upload)
        print(f"uploaded {args.upload.name} -> videoId {args.video_id}", file=sys.stderr)
    before = console.run_ids(args.event)
    body = start_body(args)
    reply = console.call("POST", "/pipeline/control/start", body) or {}
    run_id = reply.get("plannerRunId") or wait_for_new_run(
        console, args.event, before, sleep=sleep)
    print(f"started {run_id} ({json.dumps(body)})", file=sys.stderr)

    deadline = time.monotonic() + args.timeout_s
    record = console.run(run_id)
    while record.get("status") == "running":
        if time.monotonic() >= deadline:
            print(f"gave up waiting on {run_id} after {args.timeout_s:.0f} s — "
                  "it is still running", file=sys.stderr)
            return 2
        count = _stage(record, "count")
        print(f"  ... {count.get('frames', 0)} frames, count-stage "
              f"{_fmt(count.get('fps'), 2)} fps", file=sys.stderr)
        sleep(args.poll_s)
        record = console.run(run_id)

    summary = summarise(record, args.camera_fps)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary)
    return 0 if summary["status"] == "ended" else 1


if __name__ == "__main__":
    sys.exit(main())
