"""Durable planner writes, from the runner's side (heco_common.planner.Outbox).

THE INCIDENT. The planner's restart dropped every tap and frame the runner
posted meanwhile, and runs were marked failed "the runner went silent" —
best effort meant "lost on failure". The runner now builds its planner client
DURABLE (HECO_PLANNER_DURABLE, default on): what the planner is away for is
kept and delivered in order when it answers.

Two claims are pinned here. With every post landing, a durable run IS today's
run — the pinned call-sequence fixture replays exactly through a durable
client, read and never regenerated. And with the planner restarting under a
live run, nothing the runner reported is lost, duplicated or reordered, and
the run's own record says so.
"""

import threading
import time

import httpx
import pytest
from app import runs as runs_module
from app.config import Settings, from_env, knobs
from app.loop import httpx_file_transport, httpx_transport
from heco_common.planner import QUEUED, PlannerClient

from tests import test_call_sequence_pinned as pinned
from tests.test_loop_v1 import V1Fake, make_loop, real_jpeg_b64
from tests.test_overlap import runner_env_passthroughs

#: The best-effort paths the outbox covers.
BEST_EFFORT = ("/taps", "/frames", "/faces", "/frame-records")
REQUEST = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
#: Every key the outbox may add to a run's status — none of them when unused.
OUTBOX_KEYS = (
    "outboxDelivered", "outboxDropped", "outboxBacklog", "outboxBacklogMax",
    "frameRecordsQueued", "faceCardsQueued", "forensicFramesQueued",
)


def accept_face_cards(fake: V1Fake) -> None:
    """Give the fake the /faces route it lacks (it answers 500 "unscripted").

    Without it every face card is a server error — which a durable client
    retries a few times and then drops, as it must for a message the planner
    cannot take; a test of the healthy path needs the planner to take them.
    """
    original = fake._planner
    fake.face_cards = []

    def handler(request, path, body):
        if request.method == "POST" and path.endswith("/faces"):
            fake.face_cards.append(request.content)
            return httpx.Response(200, json={"ok": True})
        return original(request, path, body)

    fake._planner = handler


def durable(loop, **kw):
    """Swap a make_loop() loop's planner for a DURABLE client on the same fake.

    The retrying path's backoff is not slept (a test about the outbox has no
    use for 0.6 s per failed flush), the outbox's is 1 ms, and a drain can
    never hold a failing test for the production 30 s.
    """
    kw.setdefault("sleep", lambda _s: None)
    kw.setdefault("outbox_sleep", lambda _s: time.sleep(0.001))
    kw.setdefault("drain_timeout_s", 5.0)
    loop.planner = PlannerClient(
        loop.s.planner_url,
        transport=httpx_transport(loop.client),
        file_transport=httpx_file_transport(loop.client),
        durable=True,
        **kw,
    )
    return loop


class Outage:
    """The planner restarting under a live run.

    The first ``n`` attempts at ``paths`` are answered 503 — counted, not
    timed, so the outage hits the same posts whatever the machine's speed —
    and every attempt is written down with its two durable headers and
    whether the planner, in the end, took it.
    """

    def __init__(self, fake: V1Fake, n: int, paths: tuple[str, ...] = BEST_EFFORT) -> None:
        self.left = n
        self.attempts: list[dict] = []
        self._lock = threading.Lock()
        original = fake._planner

        def handler(request, path, body):
            if not path.endswith(paths):
                return original(request, path, body)
            with self._lock:
                failing = self.left > 0
                if failing:
                    self.left -= 1
            if failing:
                response = httpx.Response(503, json={"error": "planner restarting"})
            else:
                response = original(request, path, body)
            with self._lock:
                self.attempts.append({
                    "path": path,
                    "delivery": request.headers.get("x-heco-delivery"),
                    "at": request.headers.get("x-heco-at"),
                    "landed": response.status_code < 400,
                })
            return response

        fake._planner = handler


def test_the_pinned_runs_replay_exactly_through_a_durable_client(monkeypatch, tmp_path):
    """Durable ON with every post landing is today's run: the same stage calls,
    the same decisions, the same status and the same permanent record."""
    built: list[PlannerClient] = []

    def durable_make_loop(fake, request, **kw):
        loop = durable(make_loop(fake, request, **kw))
        built.append(loop.planner)
        return loop

    monkeypatch.setattr(pinned, "make_loop", durable_make_loop)
    fx = pinned._fixture()
    got = pinned.capture_all(tmp_path)
    for name in (*pinned.SCENARIOS, "prefetch"):
        for part in fx[name]:
            assert got[name][part] == fx[name][part], f"{name}: {part} moved"
    assert len(built) == len(pinned.SCENARIOS) + 1 and all(p.durable for p in built)
    assert all(p.outbox_stats()["queued"] == 0 for p in built), "nothing needed the outbox"


def test_a_run_whose_every_post_lands_grows_no_new_keys():
    """Absent is not zero: an unused outbox adds nothing to status or record."""
    fake = V1Fake(n_frames=3, face_widths=(85.0,), image_b64=real_jpeg_b64())
    accept_face_cards(fake)
    loop = durable(make_loop(fake, dict(REQUEST), tap_interval_s=0.0))
    final = loop.run()
    assert final["state"] == "ended" and fake.taps and fake.face_cards
    assert [k for k in OUTBOX_KEYS if k in final] == []
    assert [k for k in OUTBOX_KEYS if k in fake.run_ended["results"]] == []
    assert "outbox" not in fake.run_ended["notes"]
    assert loop.planner.outbox_stats()["queued"] == 0


@pytest.mark.parametrize("async_reporting", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "paths",
    [BEST_EFFORT, BEST_EFFORT + ("/stats", "/samples")],
    ids=["best-effort-down", "everything-down"],
)
def test_a_planner_restart_mid_run_loses_nothing(async_reporting, paths):
    """THE INCIDENT, under a live run: every tap, frame, forensic picture and
    frame record the runner posted while the planner was away lands exactly
    once, in the order it was posted — and the record says the outbox did it."""
    fake = V1Fake(n_frames=8, face_widths=(85.0,), image_b64=real_jpeg_b64())
    accept_face_cards(fake)
    request = {**REQUEST, "forensic": True}
    loop = durable(make_loop(
        fake, request, tap_interval_s=0.0, tap_duty_factor=0.0,
        async_reporting=async_reporting, reporter_poll_s=0.005,
    ))
    outage = Outage(fake, n=12, paths=paths)
    final = loop.run()
    assert final["state"] == "ended"

    stamped = [a for a in outage.attempts if a["path"].endswith(BEST_EFFORT)]
    assert any(not a["landed"] for a in stamped), "the outage really hit the reporting"
    assert all(a["delivery"] and a["at"] for a in stamped), "every durable post is stamped"
    landed = [a["delivery"] for a in stamped if a["landed"]]
    assert len(landed) == len(set(landed)), "nothing delivered twice"
    assert set(landed) == {a["delivery"] for a in stamped}, "nothing lost"
    numbers = [int(d.split(":")[1]) for d in landed]
    assert numbers == sorted(numbers), "delivered in the order it was posted"

    assert [r["seq"] for r in fake.frame_records] == list(range(final["frames"]))
    assert final["frameRecordsDropped"] == 0
    if not async_reporting:
        # Synchronous reporting uploads EVERY frame's forensic picture: each one
        # landed at once or was held — none is reported lost.
        held = final.get("forensicFramesPosted", 0) + final.get("forensicFramesQueued", 0)
        assert held == final["frames"] and "forensicFramesDropped" not in final

    results = fake.run_ended["results"]
    assert results["outboxDelivered"] >= 1 and results["outboxDropped"] == 0
    assert results["outboxBacklogMax"] >= 1
    assert "outboxDelivered=" in fake.run_ended["notes"]
    assert final["outboxBacklog"] == 0, "drained before the record was written"


def test_a_route_the_planner_keeps_failing_does_not_hold_the_rest_behind_it():
    """The fake has no /faces route, so every face card is answered 500. Each
    is retried a few times and dropped as failed — counted — while every tap
    and frame queued behind it lands. Retrying it forever would have turned one
    bad route into a reporting outage for the rest of the run."""
    fake = V1Fake(n_frames=4, face_widths=(85.0,), image_b64=real_jpeg_b64())
    loop = durable(make_loop(fake, dict(REQUEST), tap_interval_s=0.0, tap_duty_factor=0.0))
    cards: list[str] = []
    original = fake._planner

    def handler(request, path, body):
        if path.endswith("/faces"):
            cards.append(request.headers.get("x-heco-delivery"))
        return original(request, path, body)  # 500 "unscripted planner"

    fake._planner = handler
    final = loop.run()
    assert final["state"] == "ended"
    assert cards, "face cards were offered"
    stats = loop.planner.outbox_stats()
    assert stats["dropped"] == {"failed": len(set(cards))} and stats["backlog"] == 0
    tries = loop.planner.outbox.max_faults
    assert all(cards.count(d) == tries for d in set(cards)), "each tried max_faults times"
    assert len(fake.taps) >= 5 * final["frames"], "every tap landed regardless"
    assert fake.run_ended["results"]["outboxDropped"] == len(set(cards))


def test_frame_records_the_outbox_holds_are_not_posted_again():
    """The flush used to re-send a refused batch itself. Behind the outbox that
    would deliver every record twice — and queue a growing copy of the backlog
    every flush of an outage — so a QUEUED batch is handed over, not kept."""
    fake = V1Fake(n_frames=6, face_widths=(85.0,))
    fake.frame_records_fail_first = 2  # a planner restart mid-run
    final = durable(make_loop(fake, dict(REQUEST))).run()

    assert [r["seq"] for r in fake.frame_records] == list(range(6)), "each once, in order"
    assert fake.frame_record_posts == 6 + 2, "two refusals, then every batch exactly once"
    assert final["frameRecordsQueued"] >= 1
    assert final["frameRecordsPosted"] + final["frameRecordsQueued"] == 6
    assert final["frameRecordsDropped"] == 0
    assert fake.run_ended["results"]["frameRecordsQueued"] == final["frameRecordsQueued"]


def test_a_run_end_the_planner_misses_is_delivered_after_the_run_returns():
    """The planner is still restarting when the run ends. The PUT keeps its
    retrying path — and when that fails the end is kept, stamped with the
    moment the run ended, and delivered once the planner answers, long after
    the run thread has returned."""
    fake = V1Fake(n_frames=2, face_widths=(85.0,))
    hold = threading.Event()
    loop = durable(make_loop(fake, dict(REQUEST)), outbox_sleep=lambda _s: hold.wait(5.0))
    puts: list[tuple] = []
    original = fake._planner

    def handler(request, path, body):
        if request.method == "PUT" and path == "/api/pipeline/runs/prun-1":
            puts.append((request.headers.get("x-heco-delivery"), request.headers.get("x-heco-at")))
            if len(puts) <= 3:  # every attempt of the retrying path
                return httpx.Response(503, json={"error": "planner restarting"})
        return original(request, path, body)

    fake._planner = handler
    final = loop.run()
    assert final["state"] == "ended"
    assert "queued" in final["error"] and final["plannerReportErrors"] >= 1
    assert fake.run_ended is None, "not yet: the planner has not answered"

    hold.set()
    assert loop.planner.drain_outbox(5.0)
    assert fake.run_ended["results"]["unique"] == final["unique"]
    assert fake.run_ended["status"] == "ended"
    assert len(puts) == 4 and len(set(puts)) == 1, "one run end, one stamp, four attempts"


def test_the_settle_waits_for_the_outbox_once_not_twice():
    """The count run drains before it writes its record (so the record can say
    what was delivered), then tells end_run not to wait again: a planner that is
    still down costs the settle one drain timeout, not two."""
    fake = V1Fake(n_frames=2, face_widths=(85.0,))
    loop = durable(make_loop(fake, dict(REQUEST)), drain_timeout_s=4.0)
    waits: list[float] = []
    real = loop.planner.outbox.drain
    loop.planner.outbox.drain = lambda timeout: (waits.append(timeout), real(timeout))[1]
    assert loop.run()["state"] == "ended"
    assert waits == [4.0, 0.0]


def test_the_runner_builds_durable_planner_clients_by_default(monkeypatch):
    """HECO_PLANNER_DURABLE (on) and the per-run bound reach the client
    RunManager builds — through the SAME three http clients as before."""
    built: list[PlannerClient] = []
    made: list[float] = []

    class RecordingClient:
        def __init__(self, timeout=None, **kw):
            made.append(timeout)

    class DummyLoop:
        def __init__(self, run_id, request, settings, client, planner, **kw):
            built.append(planner)

        def run(self):
            return {}

    monkeypatch.setattr(runs_module.httpx, "Client", RecordingClient)
    monkeypatch.setattr(runs_module, "RunLoop", DummyLoop)

    mgr = runs_module.RunManager(Settings(planner_outbox_max_bytes=1234))
    mgr.start(dict(REQUEST))
    assert len(made) == 3, "no extra client for the outbox"
    assert built[-1].durable and built[-1].outbox.max_bytes == 1234
    health = mgr.planner_outbox()
    assert health["backlog"] == 0 and health["runsHolding"] == 0
    assert health["bytesLimit"] == 1234 and health["dropped"] == {}

    off = runs_module.RunManager(Settings(planner_durable=False))
    off.start(dict(REQUEST))
    assert built[-1].durable is False and off._planners == {}


def test_a_reaped_run_keeps_its_outbox_on_health_until_it_is_delivered():
    """A run end queued while the planner was down outlives its run's place in
    the registry: /health keeps showing it waiting, and once delivered its
    counts stay in the process totals."""
    state = {"down": True}
    hold = threading.Event()

    def transport(method, url, payload, headers=None):
        if url.endswith("/api/pipeline/runs"):
            return 200, {"id": "prun-9"}
        return (503, {}) if state["down"] else (200, {})

    planner = PlannerClient(
        "http://planner:8787", transport=transport, durable=True,
        outbox_sleep=lambda _s: hold.wait(5.0),
    )
    planner.create_run("ev-1")
    assert planner.offer_tap("match", {"unique": 1}) == QUEUED

    mgr = runs_module.RunManager(Settings(run_retention_s=0.0))
    mgr._planners["run-gone"] = planner  # its RunLoop is already out of the registry
    mgr.reap()
    assert "run-gone" in mgr._planners, "still holding something: kept"
    health = mgr.planner_outbox()
    assert health["runsHolding"] == 1 and health["backlog"] == 1

    state["down"] = False
    hold.set()
    assert planner.drain_outbox(5.0)
    mgr.reap()
    assert mgr._planners == {}, "idle and reaped: let go"
    health = mgr.planner_outbox()
    assert health["delivered"] == 1 and health["backlog"] == 0 and health["runsHolding"] == 0
    assert health["queued"] == 1, "the process totals never run backwards"


def test_health_reports_the_outbox_only_when_durable(monkeypatch):
    """The planner cannot report its own outage; the runner's /health says
    what is waiting to reach it — and says nothing when durability is off."""
    from app import main
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    monkeypatch.setattr(main.manager, "settings", Settings(planner_durable=True), raising=False)
    body = client.get("/health").json()
    assert body["plannerOutbox"]["backlog"] == 0
    assert set(body["plannerOutbox"]) >= {
        "queued", "delivered", "deliveredLate", "retried", "dropped",
        "backlog", "backlogMax", "bytes", "bytesMax", "bytesLimit", "runsHolding",
    }
    assert body["knobs"]["HECO_PLANNER_DURABLE"] is True

    monkeypatch.setattr(main.manager, "settings", Settings(planner_durable=False), raising=False)
    assert "plannerOutbox" not in client.get("/health").json()


def test_the_durable_knobs_read_from_the_env_and_reach_the_container(monkeypatch):
    """On by default, "" means unset (compose's ${VAR-}), 0 turns it off — and
    both knobs have a passthrough, or setting them would change nothing."""
    s = Settings()
    assert s.planner_durable is True and s.planner_outbox_max_bytes == 512 * 1024 * 1024

    monkeypatch.setenv("HECO_PLANNER_DURABLE", "")
    monkeypatch.setenv("HECO_PLANNER_OUTBOX_MAX_BYTES", "")
    s = from_env()
    assert s.planner_durable is True, "empty is unset, and unset is on"
    assert s.planner_outbox_max_bytes == 512 * 1024 * 1024

    monkeypatch.setenv("HECO_PLANNER_DURABLE", "0")
    monkeypatch.setenv("HECO_PLANNER_OUTBOX_MAX_BYTES", "1048576")
    s = from_env()
    assert s.planner_durable is False and s.planner_outbox_max_bytes == 1048576
    assert knobs(s)["HECO_PLANNER_DURABLE"] is False
    assert knobs(s)["HECO_PLANNER_OUTBOX_MAX_BYTES"] == 1048576
    assert {"HECO_PLANNER_DURABLE", "HECO_PLANNER_OUTBOX_MAX_BYTES"} <= runner_env_passthroughs()
