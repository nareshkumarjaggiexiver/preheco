"""The runner against ingest's throughput levers (L1): a gate and a queue.

Ingest's side of lever L1 — the motion gate (INGEST_MOTION_GATE) and the
bounded live buffer (INGEST_BUFFER_S) — changes two things this loop leans
on.  Frames arrive with GAPS in ``seq`` (the gate withholds still frames and
sends a keepalive at least every INGEST_MOTION_KEEPALIVE_S), and with the
buffer on every GET returns and DEQUEUES the oldest unread frame instead of
the newest one.  The frame wire grows five optional fields: ``motion``,
``backlog`` (frames queued behind this one), and the cumulative ``skipped``,
``dropped`` and ``captured``.

What is pinned here: every frame ingest publishes is processed exactly once
and in order, with the prefetcher on or off; a keepalive keeps a still
scene's run alive past the stall window while a genuinely silent source
still stalls; ingest's counters reach the status, the results and the notes;
and an ingest that sends none of the new fields leaves the run exactly as it
was — no key, not a zero.
"""

import json
import time

import httpx
import pytest

from tests.test_loop_v1 import make_loop
from tests.test_presence import FA, RUN, A, Scene, scene_b64

#: The camera decoded 20 frames; the motion gate published these (a still
#: stretch is covered by one keepalive at seq 9) and withheld the other 12.
PUBLISHED = (0, 3, 4, 5, 9, 12, 13, 19)
CAPTURED, SKIPPED = 20, 12


class Queued(Scene):
    """Ingest with the buffer on: each GET dequeues the OLDEST published frame.

    ``counters`` False plays an ingest from before the levers (none of the
    new fields); ``dropped`` is what its buffer discarded.  A file source:
    ``ended`` only once the queue is empty.
    """

    def __init__(self, published=PUBLISHED, counters=True, dropped=0):
        frames = [{"boxes": [A], "faces": [FA]} for _ in range(max(published) + 1)]
        super().__init__(frames, [])
        self.queue = list(published)
        self.counters = counters
        self.dropped = dropped

    def _fields(self, **kw) -> dict:
        if not self.counters:
            return {}
        return {"captured": CAPTURED, "skipped": SKIPPED, "dropped": self.dropped, **kw}

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Serve /frame from the queue; everything else as the scene does."""
        if request.url.host == "ingest" and request.url.path == "/frame":
            self.calls.append("ingest /frame")
            if not self.queue:
                return httpx.Response(200, json={"ended": True, **self._fields(backlog=0)})
            seq = self.queue.pop(0)
            return httpx.Response(200, json={
                "imageB64": scene_b64(seq), "tMs": seq * 66, "w": 160, "h": 120,
                "seq": seq, "ended": False,
                **self._fields(motion=0.01 * seq, backlog=len(self.queue)),
            })
        return super().handler(request)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_every_published_frame_is_processed_once_and_in_order(prefetch, overlap, tmp_path):
    """A FIFO hands out distinct frames with gaps in seq; none is lost or repeated.

    The seq gaps are the motion gate's withheld frames, not a stall and not a
    duplicate — the loop's "new seq?" test must take each queued frame exactly
    once, whichever thread fetched it: the loop, the prefetcher, or the
    detect worker (HECO_PIPELINE_OVERLAP).
    """
    fake = Queued()
    golden = tmp_path / "g.jsonl"
    final = make_loop(
        fake, RUN, frame_prefetch=prefetch, pipeline_overlap=overlap, golden_path=str(golden),
    ).run()
    assert final["state"] == "ended" and final["endReason"] == "source-ended"
    assert final["frames"] == len(PUBLISHED)
    seqs = [json.loads(line)["seq"] for line in golden.read_text().splitlines()]
    assert seqs == list(PUBLISHED), "every published frame once, in capture order"


def test_ingests_counters_reach_the_status_the_results_and_the_notes():
    """What the gate withheld and the buffer threw away is on the permanent record."""
    fake = Queued(dropped=3)
    final = make_loop(fake, RUN, frame_prefetch=False).run()
    assert final["framesCaptured"] == CAPTURED
    assert final["framesSkippedNoMotion"] == SKIPPED
    assert final["framesDroppedLive"] == 3
    # The deepest the queue got: 7 frames behind the first one handed over.
    assert final["ingestBacklogMax"] == len(PUBLISHED) - 1
    results = fake.run_ended["results"]
    assert results["framesCaptured"] == CAPTURED
    assert results["framesSkippedNoMotion"] == SKIPPED
    assert results["framesDroppedLive"] == 3
    assert results["ingestBacklogMax"] == len(PUBLISHED) - 1
    notes = fake.run_ended["notes"]
    assert (
        f"framesCaptured={CAPTURED} framesSkippedNoMotion={SKIPPED} "
        f"framesDroppedLive=3 ingestBacklogMax={len(PUBLISHED) - 1} "
    ) in notes
    # Every results value is a finite number: the planner's door refuses null.
    assert all(isinstance(v, int | float) for v in results.values())


def test_motion_and_backlog_are_charted_on_the_ingest_board():
    """The two readings an engineer tunes the gate and the buffer by."""
    fake = Queued()
    make_loop(fake, RUN, frame_prefetch=False).run()
    ingest = [s for s in fake.stats if s["stage"] == "ingest"][-1]["metrics"]
    assert ingest["motion"]["count"] == len(PUBLISHED)
    assert ingest["motion"]["max"] == pytest.approx(0.19)
    assert ingest["backlog"]["max"] == len(PUBLISHED) - 1
    assert ingest["backlog"]["min"] == 0


def test_an_ingest_without_the_fields_leaves_the_run_as_it_was():
    """Absent is not zero: no key in the results, no words in the notes."""
    fake = Queued(counters=False)
    final = make_loop(fake, RUN, frame_prefetch=False).run()
    for key in ("framesCaptured", "framesSkippedNoMotion", "framesDroppedLive",
                "ingestBacklogMax"):
        assert final[key] is None, f"{key} was never measured"
        assert key not in fake.run_ended["results"]
        assert key not in fake.run_ended["notes"]
    ingest = [s for s in fake.stats if s["stage"] == "ingest"][-1]["metrics"]
    assert "motion" not in ingest and "backlog" not in ingest


def test_malformed_counters_are_not_measurements():
    """A negative, a float or a bool is ignored rather than trusted."""
    class Garbled(Queued):
        def _fields(self, **kw):
            return {"captured": -1, "skipped": 2.5, "dropped": True, "backlog": "7"}

    fake = Garbled()
    final = make_loop(fake, RUN, frame_prefetch=False).run()
    assert final["framesCaptured"] is None
    assert final["framesSkippedNoMotion"] is None
    assert final["framesDroppedLive"] is None
    assert final["ingestBacklogMax"] is None


class Keepalive(Scene):
    """A LIVE source behind the motion gate: an empty room, and keepalives.

    Publishes one keepalive every ``every_s`` of wall time for ``for_s``,
    then nothing at all (the camera died).  Between publications a GET
    answers the last frame again — same seq — the way the slot always has,
    or 503 before the first one exists.  Never ``ended``: a camera.
    """

    def __init__(self, every_s: float, for_s: float, repeat=True):
        n = int(for_s / every_s) + 1
        super().__init__([{"boxes": [], "faces": []} for _ in range(n)], [])
        self.every_s, self.n = every_s, n
        self.repeat = repeat
        self.t0: float | None = None
        self.next_seq = 0
        self.last: dict | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Publish due keepalives, dequeue one, else repeat the last (or 503)."""
        if request.url.host == "ingest" and request.url.path == "/open":
            self.t0 = time.monotonic()
        if request.url.host == "ingest" and request.url.path == "/frame":
            self.calls.append("ingest /frame")
            due = min(self.n, int((time.monotonic() - self.t0) / self.every_s) + 1)
            if self.next_seq < due:
                seq = self.next_seq
                self.next_seq += 1
                self.last = {
                    "imageB64": scene_b64(seq), "tMs": int(seq * self.every_s * 1000),
                    "w": 160, "h": 120, "seq": seq, "motion": 0.0,
                    "backlog": 0, "skipped": seq * 14, "captured": seq * 15 + 1,
                    "dropped": 0,
                }
                return httpx.Response(200, json=self.last)
            if self.last is None or not self.repeat:
                return httpx.Response(503, json={"detail": "no frame yet"})
            return httpx.Response(200, json=self.last)
        return super().handler(request)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("repeat", [True, False])
def test_keepalives_hold_a_still_room_open_past_the_stall_window(repeat, overlap):
    """A still scene is not a dead camera.

    The stall window here is 0.2 s and the room stays still for ~0.6 s: the
    gate's keepalive (every 0.05 s) advances seq, so the run counts on.  Once
    the keepalives stop the silence is real, and the run settles as a stall
    (failed, gallery kept) exactly as before the gate existed.
    """
    fake = Keepalive(every_s=0.05, for_s=0.6, repeat=repeat)
    t = time.monotonic()
    final = make_loop(
        fake, RUN, source_stall_s=0.2, source_poll_s=0.005, frame_prefetch=False,
        pipeline_overlap=overlap,
    ).run()
    assert time.monotonic() - t >= 0.6, "the run outlived the stall window"
    assert final["frames"] == fake.n, "every keepalive was processed"
    assert final["endReason"] == "source-stalled"
    assert final["state"] == "failed"
    assert final["framesSkippedNoMotion"] == (fake.n - 1) * 14


def test_a_runs_gate_and_buffer_overrides_reach_ingest_open_and_absent_adds_nothing():
    """source.motionGate / source.bufferS are ingest /open's per-run overrides.

    The runner's Source model is the only way that dict reaches ingest, so an
    undeclared field would be dropped by validation and the run would count
    without the gate it asked for.  Absent must add no key at all: the OFF
    run opens ingest with the body it always sent.
    """
    from app.main import RunRequest

    on = RunRequest.model_validate({
        "eventId": "ev-1",
        "source": {"url": "rtsp://cam/1", "motionGate": True, "bufferS": 10},
    }).model_dump(exclude_none=True)
    assert on["source"] == {
        "url": "rtsp://cam/1", "loop": False, "isFile": False, "lockstep": False,
        "motionGate": True, "bufferS": 10.0,
    }
    off = RunRequest.model_validate(RUN).model_dump(exclude_none=True)
    assert "motionGate" not in off["source"] and "bufferS" not in off["source"]
    with pytest.raises(ValueError):
        RunRequest.model_validate({"eventId": "e", "source": {"path": "/x", "bufferS": -1}})

    fake = Queued()
    make_loop(fake, {**RUN, "source": {**RUN["source"], "motionGate": False}}).run()
    assert fake.opened["motionGate"] is False, "forwarded verbatim to ingest /open"
