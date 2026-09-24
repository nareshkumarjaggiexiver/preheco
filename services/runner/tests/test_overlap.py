"""Stage overlap and parallel detect (lever L3): detection off the loop thread.

HECO_PIPELINE_OVERLAP moves the stateless half of the chain — persons, and
the face search when it needs no tracks — onto ONE worker thread that runs a
frame ahead of the loop.  Everything the count is made of stays on the loop
thread in frame order.  These pin that promise from outside:

* the SAME reasoning as the serial loop, frame for frame: every scenario of
  the call-sequence pin replays with overlap on to the identical ledger,
  status and permanent record, with every host's calls identical and in
  order — only the interleaving between hosts moves;
* tracker and match calls strictly in frame order, the worker never more
  than one frame ahead, the ledger in frame order;
* a stage failure on the worker surfaces for THAT frame exactly as the
  inline call would have — same error, same frames, same tracker calls;
* stop() drains: the worker is joined and no stage call outlives the run;
* the knob is visible (GET /health, the run's config) and plumbed through
  compose;
* and it is worth having: the synthetic latency harness (persons 30 ms,
  faces 70 ms, embed 20 ms, match 5 ms) reads ~1.25x the serial frame rate.

HECO_PARALLEL_DETECT issues persons and the whole-frame face search for one
frame side by side, with or without the overlap; the same promises are
pinned for it, plus that the two calls really do run at the same time and
that on the crop path — where faces must wait for the tracker — it changes
nothing at all.
"""

import json
import re
import threading
import time
from pathlib import Path

import httpx
import pytest
from app import config as cfg

from tests import latency_harness
from tests.test_call_sequence_pinned import (
    NIGHT,
    NIGHT_SCRIPT,
    SCENARIOS,
    STAGE_HOSTS,
    STATUS_KEYS,
    Recorder,
    _fixture,
)
from tests.test_loop_v1 import make_loop
from tests.test_presence import FA, RUN, A, Scene, scene_index

REPO = Path(__file__).resolve().parents[3]


def _per_host(calls: list) -> dict:
    return {h: [c for c in calls if c[0] == h] for h in STAGE_HOSTS if h != "ingest"}


def _ledger(path: Path) -> list:
    return [
        {k: v for k, v in json.loads(line).items() if k != "ms"}
        for line in path.read_text().splitlines()
    ]


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_overlap_reasons_exactly_as_the_pinned_serial_run(name, prefetch, tmp_path):
    """Same calls per host, same ledger, same status, same run record.

    Compared against the fixture captured from the serial loop BEFORE the
    lever existed — not against another overlap run, which could share a
    mistake with it.
    """
    want = _fixture()[name]
    settings, extra = SCENARIOS[name]
    fake = Recorder(NIGHT, NIGHT_SCRIPT)
    golden = tmp_path / "g.jsonl"
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}, **extra}
    final = make_loop(
        fake, request, golden_path=str(golden),
        **{**settings, "frame_prefetch": prefetch, "pipeline_overlap": True},
    ).run()
    assert _per_host(json.loads(json.dumps(fake.stage_calls))) == _per_host(want["calls"])
    assert _ledger(golden) == want["ledger"]
    assert {k: final.get(k) for k in STATUS_KEYS} == want["status"]
    assert json.loads(json.dumps(fake.run_ended)) == want["ended"]
    config = json.loads(json.dumps(fake.run_created["config"]))
    assert config.pop("levers") == {"pipelineOverlap": True}, "the lever is on the record"
    assert config == want["config"], "and nothing else about the config moved"


class Ordered(Scene):
    """Every stage call as (host, frame index) in arrival order.

    The frame is read off what the call carries: the payload for persons,
    faces and embed, tMs for the tracker, and for match the embed call
    before it (both are loop-thread calls, so that pairing is exact).
    """

    def __init__(self, frames, script, match_s: float = 0.0):
        super().__init__(frames, script)
        self.seq: list[tuple[str, int]] = []
        self._embed_frame = -1
        self._lock = threading.Lock()
        # A slow decide half, so a worker that COULD run two ahead would.
        self.match_s = match_s

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Note (host, frame), then serve the scene."""
        host, path = request.url.host, request.url.path
        if host == "match" and path == "/match" and self.match_s:
            time.sleep(self.match_s)
        body = json.loads(request.content) if request.content else {}
        frame = None
        if host in ("persons", "faces", "embed") and "imageB64" in body:
            frame = scene_index(body["imageB64"])
            if host == "embed":
                self._embed_frame = frame
        elif host == "tracker" and path == "/track":
            frame = body["tMs"] // 100
        elif host == "match" and path == "/match":
            frame = self._embed_frame
        if frame is not None:
            with self._lock:
                self.seq.append((host, frame))
        return super().handler(request)


@pytest.mark.parametrize("whole_frame", [False, True])
def test_state_is_touched_in_frame_order_and_detection_runs_one_ahead(whole_frame):
    """Tracker and match strictly in frame order; the worker at most ONE ahead."""
    n = 12
    frames = [{"boxes": [A], "faces": [FA]} for _ in range(n)]
    fake = Ordered(frames, [], match_s=0.01)
    final = make_loop(
        fake, RUN, pipeline_overlap=True, faces_whole_frame=whole_frame,
        frame_prefetch=True,
    ).run()
    assert final["frames"] == n
    tracker = [f for h, f in fake.seq if h == "tracker"]
    assert tracker == list(range(n)), "one tracker call per frame, in frame order"
    match = [f for h, f in fake.seq if h == "match"]
    assert match == sorted(match) and set(match) == set(range(n))
    for host in ("persons", "faces"):
        assert [f for h, f in fake.seq if h == host] == list(range(n)), host
    # One ahead, never two: frame j's detection starts only once frame j-2 is
    # fully decided — so no tracker, embed or match call for frame j-2 (or
    # earlier) may come after it.  The slow match above is what would let a
    # deeper worker run ahead and be caught here (mutation-checked: a
    # two-deep queue fails this).
    for i, (host, frame) in enumerate(fake.seq):
        if host != "persons":
            continue
        late = [
            (h, f) for h, f in fake.seq[i:]
            if h in ("tracker", "embed", "match") and f <= frame - 2
        ]
        assert late == [], f"persons({frame}) ran while frame {late[0][1]} was undecided"


@pytest.mark.parametrize("async_reporting", [False, True])
def test_the_ledger_and_the_taps_see_frames_in_order(async_reporting):
    """Every frame on the ledger once, in order; every tap round in order.

    Synchronous reporting with no interval and no duty guard taps EVERY
    frame, so the tap stream is the frame stream; the async plane is
    drop-not-queue by design, so there the promise is order, not coverage.
    """
    n = 10
    frames = [{"boxes": [A], "faces": [FA]} for _ in range(n)]
    fake = Scene(frames, [])
    make_loop(
        fake, RUN, pipeline_overlap=True, faces_whole_frame=True,
        tap_interval_s=0.0, tap_duty_factor=0.0, async_reporting=async_reporting,
    ).run()
    assert [r["seq"] for r in fake.frame_records] == list(range(n))
    seqs = [t["payload"]["seq"] for t in fake.taps if t["stage"] == "ingest"]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs
    if not async_reporting:
        assert seqs == list(range(n))


class FailsAt(Scene):
    """A detector that answers 500 on one frame (read off the payload)."""

    def __init__(self, frames, host: str, at: int):
        super().__init__(frames, [])
        self.fail_host, self.fail_at = host, at
        self.tracker_calls = 0
        self.after_close: list[str] = []
        self.closed_seen = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        """500 on the chosen host+frame; count tracker calls; else the scene."""
        host, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else {}
        if host == "ingest" and path == "/close":
            self.closed_seen = True
        elif self.closed_seen and path in ("/detect", "/track", "/embed", "/match"):
            self.after_close.append(f"{host} {path}")
        if host == "tracker" and path == "/track":
            self.tracker_calls += 1
        if (
            host == self.fail_host and path == "/detect"
            and scene_index(body.get("imageB64", "")) == self.fail_at
        ):
            return httpx.Response(500, json={"detail": f"{host} fell over"})
        return super().handler(request)


@pytest.mark.parametrize("host", ["persons", "faces"])
def test_a_worker_failure_settles_the_run_exactly_as_the_inline_call_did(host):
    """Same error, same frames counted, same tracker calls, same planner record."""
    frames = [{"boxes": [A], "faces": [FA]} for _ in range(8)]
    arms = {}
    for overlap in (False, True):
        fake = FailsAt(frames, host, at=3)
        final = make_loop(
            fake, RUN, pipeline_overlap=overlap, faces_whole_frame=True,
            frame_prefetch=False,
        ).run()
        arms[overlap] = (fake, final)
    (serial, s_final), (over, o_final) = arms[False], arms[True]
    assert o_final["state"] == s_final["state"] == "failed"
    assert o_final["error"] == s_final["error"] and host in o_final["error"]
    assert o_final["frames"] == s_final["frames"] == 3, "frames 0..2 counted, 3 failed"
    # persons failing means frame 3 never reached the tracker; faces failing
    # means it did (the inline order is persons -> tracker -> faces).
    assert over.tracker_calls == serial.tracker_calls == (4 if host == "faces" else 3)
    assert over.run_ended["status"] == serial.run_ended["status"] == "failed"
    assert over.run_ended["notes"] == serial.run_ended["notes"]
    assert over.after_close == [], "no stage call after the camera was handed back"


class StopsAt(Scene):
    """Calls loop.stop() from inside frame ``at``'s tracker call."""

    def __init__(self, frames, at: int):
        super().__init__(frames, [])
        self.at = at
        self.loop = None
        self.closed = False
        self.after_close: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Stop the run mid-frame; record any stage call after /close."""
        host, path = request.url.host, request.url.path
        if host == "ingest" and path == "/close":
            self.closed = True
        elif self.closed and path in ("/detect", "/track", "/embed", "/match"):
            self.after_close.append(f"{host} {path}")
        if (
            host == "tracker" and path == "/track"
            and json.loads(request.content)["tMs"] // 100 == self.at
        ):
            self.loop.stop()
        return super().handler(request)


@pytest.mark.parametrize("prefetch", [False, True])
def test_stop_drains_the_worker_and_nothing_outlives_the_run(prefetch):
    """The frame being decided finishes; the one being detected is dropped."""
    frames = [{"boxes": [A], "faces": [FA]} for _ in range(30)]
    fake = StopsAt(frames, at=5)
    loop = make_loop(
        fake, RUN, pipeline_overlap=True, faces_whole_frame=True, frame_prefetch=prefetch,
    )
    fake.loop = loop
    final = loop.run()
    assert final["state"] == "ended"
    assert final["frames"] == 6, "frames 0..5: the stop lands after the frame in hand"
    assert fake.after_close == []
    alive = [t.name for t in threading.enumerate() if t.name.startswith("frame-detect")]
    assert alive == [], f"detect worker still alive after run(): {alive}"


def test_overlap_off_is_the_default_and_the_knob_is_visible():
    """Off unless set; GET /health says what this process resolved."""
    assert cfg.Settings().pipeline_overlap is False
    assert cfg.knobs(cfg.Settings())["HECO_PIPELINE_OVERLAP"] is False
    assert cfg.knobs(cfg.Settings(pipeline_overlap=True))["HECO_PIPELINE_OVERLAP"] is True


def test_the_knob_reads_from_the_env_and_empty_means_unset(monkeypatch):
    """compose renders an unset ${VAR-} as "" — that must mean off, not crash."""
    monkeypatch.setenv("HECO_PIPELINE_OVERLAP", "")
    assert cfg.from_env().pipeline_overlap is False
    monkeypatch.setenv("HECO_PIPELINE_OVERLAP", "1")
    assert cfg.from_env().pipeline_overlap is True


def test_health_carries_the_knobs_block():
    """The operator's one place to check a lever is really on."""
    from app import main
    from fastapi.testclient import TestClient

    body = TestClient(main.app).get("/health").json()
    assert body["knobs"]["HECO_PIPELINE_OVERLAP"] is main.manager.settings.pipeline_overlap


def runner_env_passthroughs() -> set[str]:
    """Every HECO_* the runner service passes through compose as ${X-}."""
    text = (REPO / "docker-compose.yml").read_text()
    block = text.split("\n  runner:\n", 1)[1].split("\nvolumes:\n", 1)[0]
    return set(re.findall(r"^\s+(HECO_[A-Z0-9_]+): \$\{\1-\}\s*$", block, re.M))


def test_every_lever_knob_reaches_the_container():
    """A knob with no compose passthrough looks set and changes nothing."""
    assert set(cfg.knobs(cfg.Settings())) <= runner_env_passthroughs()


@pytest.mark.parametrize("overlap", [False, True])
def test_parallel_detect_reasons_exactly_as_the_pinned_serial_run(overlap, tmp_path):
    """The whole-frame scenario, both detectors side by side: same reasoning."""
    want = _fixture()["whole-frame"]
    settings, _extra = SCENARIOS["whole-frame"]
    fake = Recorder(NIGHT, NIGHT_SCRIPT)
    golden = tmp_path / "g.jsonl"
    final = make_loop(
        fake, {"eventId": "ev-1", "source": {"path": "/x.mp4"}}, golden_path=str(golden),
        **{**settings, "parallel_detect": True, "pipeline_overlap": overlap},
    ).run()
    assert _per_host(json.loads(json.dumps(fake.stage_calls))) == _per_host(want["calls"])
    assert _ledger(golden) == want["ledger"]
    assert {k: final.get(k) for k in STATUS_KEYS} == want["status"]
    assert json.loads(json.dumps(fake.run_ended)) == want["ended"]
    config = json.loads(json.dumps(fake.run_created["config"]))
    levers = {"parallelDetect": True, **({"pipelineOverlap": True} if overlap else {})}
    assert config.pop("levers") == levers
    assert config == want["config"]


def test_parallel_detect_changes_nothing_on_the_crop_path(tmp_path):
    """Crops need this frame's tracks first: the full call ORDER is today's."""
    want = _fixture()["crops"]
    settings, _extra = SCENARIOS["crops"]
    fake = Recorder(NIGHT, NIGHT_SCRIPT)
    make_loop(
        fake, {"eventId": "ev-1", "source": {"path": "/x.mp4"}},
        **{**settings, "parallel_detect": True},
    ).run()
    assert json.loads(json.dumps(fake.stage_calls)) == want["calls"]


class Timed(latency_harness.Slow):
    """The slow fake, writing down when each detector call started and ended."""

    def __init__(self, n_frames: int):
        super().__init__(n_frames, {"persons": 30.0, "faces": 70.0})
        self.spans: dict[tuple[str, int], tuple[float, float]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Time persons and faces /detect per frame."""
        host, path = request.url.host, request.url.path
        if host in ("persons", "faces") and path == "/detect":
            frame = scene_index(json.loads(request.content)["imageB64"])
            t = time.perf_counter()
            out = super().handler(request)
            self.spans[(host, frame)] = (t, time.perf_counter())
            return out
        return super().handler(request)


@pytest.mark.parametrize("overlap", [False, True])
def test_persons_and_faces_really_run_side_by_side(overlap):
    """Every frame's two detector calls overlap in time — the whole point."""
    n = 5
    fake = Timed(n)
    final = make_loop(
        fake, RUN, faces_whole_frame=True, parallel_detect=True, pipeline_overlap=overlap,
        flush_interval_s=3600.0, tap_interval_s=3600.0,
    ).run()
    assert final["frames"] == n
    for frame in range(n):
        (p0, p1), (f0, f1) = fake.spans[("persons", frame)], fake.spans[("faces", frame)]
        assert p0 < f1 and f0 < p1, f"frame {frame}: persons and faces ran one after the other"


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("host", ["persons", "faces"])
def test_a_parallel_failure_settles_the_run_exactly_as_the_inline_call_did(host, overlap):
    """Side by side or not, a failing detector fails the same frame the same way."""
    frames = [{"boxes": [A], "faces": [FA]} for _ in range(8)]
    arms = {}
    for parallel in (False, True):
        fake = FailsAt(frames, host, at=3)
        final = make_loop(
            fake, RUN, faces_whole_frame=True, frame_prefetch=False,
            parallel_detect=parallel, pipeline_overlap=overlap and parallel,
        ).run()
        arms[parallel] = (fake, final)
    (serial, s_final), (par, p_final) = arms[False], arms[True]
    assert p_final["error"] == s_final["error"] and host in p_final["error"]
    assert p_final["frames"] == s_final["frames"] == 3
    assert par.tracker_calls == serial.tracker_calls
    assert par.run_ended["notes"] == serial.run_ended["notes"]
    assert par.after_close == []
    alive = [t.name for t in threading.enumerate() if t.name.startswith("face-detect")]
    assert alive == [], f"the parallel detector thread outlived the run: {alive}"


def test_the_levers_are_worth_having():
    """The synthetic harness: each lever buys what the stage costs predict.

    persons 30 + faces 70 + embed 20 + match 5 = 125 ms serial.  Whole
    frame: parallel-only max(30, 70) + 25 = 95 ms; the overlap alone is bound
    by the worker's 30 + 70 = 100 ms; both together by max(70, 25) = 70 ms.
    Crops: the overlap is bound by the loop's 70 + 25 = 95 ms, and parallel
    detect has nothing to parallelise.  Asserted with slack for a busy box;
    `python -m tests.latency_harness` prints the full table.
    """
    whole = latency_harness.table(n_frames=6, whole_frame=True)
    assert {r["frames"] for r in whole.values()} == {6}
    assert len({r["unique"] for r in whole.values()}) == 1, "no arm may change the count"
    assert whole["overlap"]["fps"] >= 1.12 * whole["serial"]["fps"]
    assert whole["parallel-detect"]["fps"] >= 1.12 * whole["serial"]["fps"]
    assert whole["overlap+parallel-detect"]["fps"] >= 1.2 * whole["overlap"]["fps"]
    crops = latency_harness.table(n_frames=6, whole_frame=False, arms=("serial", "overlap"))
    assert crops["overlap"]["unique"] == crops["serial"]["unique"]
    assert crops["overlap"]["fps"] >= 1.15 * crops["serial"]["fps"]
