"""Bench one ingest container the way the runner drives it (stdlib only).

Opens a clip, then polls GET /frame like the runner's poller: a fresh seq is
"processed" for --work-ms (the chain's per-frame cost) and a repeat seq is
re-polled after 20 ms (source_poll_s). Reads the container's cgroup CPU
counter before and after, so the ingest process AND its decoder subprocess
are both charged.

    python3 bench_ingest.py --url http://localhost:7501 --container lever-ingest-cpu \
        --clip /clips/bench-sharon-120s.mp4 --seconds 60 [--gate] [--buffer-s 0] \
        [--work-ms 200] [--label cpu-gate-off]

Prints one JSON line: CPU ms per captured frame, captured fps, published /
skipped / dropped / served, and /frame latency for fresh and repeat polls.
"""

import argparse
import http.client
import json
import subprocess
import time
from urllib.parse import urlparse


def cpu_usec(container: str) -> int:
    """The container's cumulative CPU time (cgroup v2 usage_usec)."""
    out = subprocess.run(
        ["docker", "exec", container, "cat", "/sys/fs/cgroup/cpu.stat"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(next(ln.split()[1] for ln in out.splitlines() if ln.startswith("usage_usec")))


def call(conn: http.client.HTTPConnection, method: str, path: str, body=None):
    """One request; returns (status, raw bytes, seconds)."""
    t0 = time.perf_counter()
    headers = {"content-type": "application/json"} if body is not None else {}
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers=headers)
    res = conn.getresponse()
    raw = res.read()
    return res.status, raw, time.perf_counter() - t0


def pct(xs, q):
    """Percentile q of xs (nearest rank), or None when empty."""
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q / 100.0 * len(xs)))] * 1000.0, 1)


def main() -> None:
    """Run one configuration and print its line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--buffer-s", type=float, default=0.0)
    ap.add_argument("--work-ms", type=float, default=200.0)
    ap.add_argument("--lockstep", action="store_true")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    u = urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    body = {"path": a.clip, "loop": False, "owner": "bench",
            "motionGate": a.gate, "bufferS": a.buffer_s, "lockstep": a.lockstep}
    c0 = cpu_usec(a.container)
    t_open = time.monotonic()
    st, raw, open_s = call(conn, "POST", "/open", body)
    assert st == 200, raw[:300]
    fresh, repeat, last_seq, served, body = [], [], None, 0, {}
    t_end = t_open + a.seconds
    while time.monotonic() < t_end:
        st, raw, dt = call(conn, "GET", "/frame")
        if st == 503:
            time.sleep(0.02)
            continue
        body = json.loads(raw)
        if body.get("ended"):
            break
        if body["seq"] != last_seq:
            last_seq = body["seq"]
            served += 1
            fresh.append(dt)
            time.sleep(a.work_ms / 1000.0)
        else:
            repeat.append(dt)
            time.sleep(0.02)
    wall = time.monotonic() - t_open
    c1 = cpu_usec(a.container)
    st, raw, _ = call(conn, "GET", "/health")
    health = json.loads(raw)
    call(conn, "POST", "/close", {"owner": "bench", "force": True})
    counters = (health.get("capture") or {}).get("counters") or {}
    # Today's loop keeps no counters; its seq IS the capture count.
    captured = counters.get("captured") or body.get("captured") or last_seq or 0
    print(json.dumps({
        "label": a.label,
        "device": health.get("device"),
        "gate": a.gate, "bufferS": a.buffer_s, "lockstep": a.lockstep,
        "openS": round(open_s, 2),
        "wallS": round(wall, 1),
        "captured": captured,
        "capturedFps": round(captured / wall, 2),
        "cpuMsPerCaptured": round((c1 - c0) / 1000.0 / max(captured, 1), 1),
        "cpuCores": round((c1 - c0) / 1e6 / wall, 2),
        "published": counters.get("published"),
        "skipped": counters.get("skipped"),
        "dropped": counters.get("dropped"),
        "served": served,
        "backlogMax": counters.get("backlogMax"),
        "frameMsFresh_p50_p95": [pct(fresh, 50), pct(fresh, 95)],
        "frameMsRepeat_p50_p95": [pct(repeat, 50), pct(repeat, 95)],
    }))


if __name__ == "__main__":
    main()
