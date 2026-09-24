"""A synthetic latency harness for the frame loop's scheduling levers (L3).

Not a test module: ``tests/test_overlap.py`` asserts on it, and running it
directly prints the table the lever was argued from::

    cd services/runner && PYTHONPATH=../../common:../../counting \\
        python -m tests.latency_harness

The stage services are the scripted scene fake with SLEEPS standing in for
the measured costs — persons 30 ms, faces 70 ms, embed 20 ms, match 5 ms per
face — so what is measured is the loop's SCHEDULING, not any model: how
much of the chain the detect worker hides behind the decide half, and how
much running the two detectors side by side hides on top.  A sleep releases
the GIL exactly as a socket wait does, which is what a stage call is to
this process.

Throughput is read the way the loop paces itself: between the first and the
last tracker call (one per frame, always on the loop thread), so pipeline
fill and teardown do not flatter or punish either arm.
"""

import time

import httpx

from tests.test_loop_v1 import make_loop, scripted_verdict
from tests.test_presence import FA, RUN, A, Scene

#: Per-call stage costs, milliseconds (the lever brief's synthetic numbers).
COSTS_MS = {"persons": 30.0, "faces": 70.0, "embed": 20.0, "match": 5.0}


class Slow(Scene):
    """The scene fake with a per-call sleep per stage, and a clock on the tracker."""

    def __init__(self, n_frames: int, costs_ms: dict):
        frames = [{"boxes": [A], "faces": [FA]} for _ in range(n_frames)]
        script = [scripted_verdict("p00001", True, None)] + [
            scripted_verdict("p00001", False, 0.8) for _ in range(n_frames)
        ]
        super().__init__(frames, script)
        self.costs_ms = costs_ms
        self.track_at: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Sleep the stage's cost, then serve the scene."""
        host, path = request.url.host, request.url.path
        if host == "tracker" and path == "/track":
            self.track_at.append(time.perf_counter())
        cost = self.costs_ms.get(host)
        if cost and path in ("/detect", "/embed", "/match"):
            time.sleep(cost / 1000.0)
        return super().handler(request)


def measure(n_frames: int = 16, costs_ms: dict | None = None, **settings) -> dict:
    """Run ``n_frames`` through the loop under ``settings``; frames/s and more."""
    fake = Slow(n_frames, costs_ms or COSTS_MS)
    loop = make_loop(
        fake, RUN,
        # Reporting on the loop thread but rare, so the table is the stages.
        flush_interval_s=3600.0, tap_interval_s=3600.0,
        **settings,
    )
    final = loop.run()
    span = fake.track_at[-1] - fake.track_at[0]
    return {
        "frames": final["frames"],
        "unique": final["unique"],
        "state": final["state"],
        "fps": (len(fake.track_at) - 1) / span if span > 0 else float("inf"),
        "ms_per_frame": 1000.0 * span / max(1, len(fake.track_at) - 1),
    }


#: name -> settings, in the order the table prints them.
ARMS = {
    "serial": {},
    "overlap": {"pipeline_overlap": True},
}


def table(n_frames: int = 24, whole_frame: bool = True) -> dict:
    """Every arm on the same footage; name -> measure()."""
    base = {"faces_whole_frame": whole_frame}
    return {name: measure(n_frames, **base, **arm) for name, arm in ARMS.items()}


if __name__ == "__main__":
    for whole in (True, False):
        print(f"\n{'whole-frame' if whole else 'crop'} face search, "
              f"costs {COSTS_MS} ms per call")
        rows = table(whole_frame=whole)
        serial = rows["serial"]["fps"]
        for name, r in rows.items():
            print(f"  {name:>26}: {r['fps']:5.2f} fps  {r['ms_per_frame']:6.1f} ms/frame"
                  f"  x{r['fps'] / serial:4.2f}  (frames={r['frames']}, unique={r['unique']})")
