"""The head-covering reads (app.headwear): off the frame loop, bounded, off by default.

After a /match that minted or enrolled a template and logged a body row, the
loop cuts the head's context crop and hands it to a background worker, which
asks embed POST /headwear and writes the logits to match POST
/body-sightings/headwear by bodyId.  Pinned here:

* OFF (the default) cuts nothing, sends nothing and records nothing — the
  status counters stay None and the run record has no headwear key (the
  pinned call-sequence and decision replays hold the rest of the loop);
* ON reads exactly the mints and enrolments that carry a bodyId, never a
  plain match, never staff, and writes each reading onto its own row;
* the worker never blocks the loop: a stuck embed leaves offer() instant, a
  full queue drops and counts, and whatever is still queued when the run
  ends is counted as dropped;
* a failing embed is counted, and the run ends normally.
"""

import base64
import json
import threading
import time
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
from app.config import Settings, from_env, knobs
from app.headwear import COUNTERS, HeadwearWorker, context_crop

from tests.test_loop_v1 import V1Fake, make_loop, scripted_verdict

RUN = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
READING = [0.5, -1.0, 0.25, 0.0, 1.5, -0.5, 0.0, 0.75]


def jpeg_b64(h: int = 240, w: int = 320) -> str:
    """A decodable JPEG frame with some structure in it."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 2) % 256], 2).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Fake(V1Fake):
    """The V1 fake plus embed /headwear and match /body-sightings/headwear."""

    def __init__(self, *args, headwear_status=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.headwear_status = headwear_status
        self.reads: list[dict] = []
        self.writes: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Serve the two headwear routes, recording them; defer the rest."""
        host, path = request.url.host, request.url.path
        if host == "embed" and path == "/headwear":
            self.reads.append(json.loads(request.content))
            if self.headwear_status != 200:
                return httpx.Response(self.headwear_status, json={"detail": "not loaded"})
            return httpx.Response(200, json={"readings": [READING], "model": "abc+def", "ms": 1.0})
        if host == "match" and path == "/body-sightings/headwear":
            self.writes.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True, "written": True})
        return super().handler(request)


def v(key, is_new, cosine, body_id=None, template_added=False, staff=False) -> dict:
    """One scripted /match reply with the fields the reader keys on."""
    out = {**scripted_verdict(key, is_new, cosine, staff=staff), "templateAdded": template_added}
    if body_id is not None:
        out["bodyId"] = body_id
    return out


#: One face a frame: a mint (read), a plain match (not), an enrolment (read),
#: a mint with no body row (not: nothing to write onto), a staff hit (never).
SCRIPT = [
    v("p00001", True, None, body_id=1),
    v("p00001", False, 0.70, body_id=2),
    v("p00001", False, 0.72, body_id=3, template_added=True),
    v("p00002", True, 0.20),
    v("st-1", False, 0.80, body_id=5, staff=True),
]


def run(settings_kw=None, **fake_kw):
    """One scripted run of SCRIPT; (fake, final status)."""
    fake = Fake(n_frames=len(SCRIPT), face_widths=(60.0,), image_b64=jpeg_b64(),
                match_script=list(SCRIPT), **fake_kw)
    final = make_loop(fake, RUN, **(settings_kw or {})).run()
    return fake, final


def test_off_nothing_is_cut_sent_or_recorded():
    """HECO_HEADWEAR off: no read, no write, counters None, no run-record key."""
    fake, final = run()
    assert final["state"] == "ended"
    assert fake.reads == [] and fake.writes == []
    assert all(final[k] is None for k in COUNTERS), "not measured, not zero"
    assert "headwear" not in fake.run_created["config"]
    assert not [k for k in fake.run_ended["results"] if k.startswith("headwear")]
    assert "headwear" not in fake.run_ended["notes"]


def test_on_reads_exactly_the_mints_and_enrolments_with_a_body_row():
    """On: each mint and enrolment with a bodyId is read once, onto its own row."""
    fake, final = run({"headwear": True})
    assert final["state"] == "ended"
    assert [w["bodyId"] for w in fake.writes] == [1, 3], (
        "a plain match, a row-less mint, staff: never")
    for w in fake.writes:
        assert w["runId"] == "prun-1" and w["headwear"] == READING and w["model"] == "abc+def"
    for r in fake.reads:
        (face,) = r["faces"]
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(r["imageB64"]), np.uint8),
                            cv2.IMREAD_COLOR)
        assert r["crop"]["frameW"] == 320 and r["crop"]["frameH"] == 240
        assert face["box"]["x"] == 12 - r["crop"]["x"] and face["box"]["w"] == 60.0
        assert crop.shape[:2] == (
            int(np.ceil(22 + 78.0)) - r["crop"]["y"],
            int(np.ceil(12 + 60 * 1.5)) - r["crop"]["x"],
        ), "0.5 face widths a side, 1.2 heights up (clamped), down to the face bottom"
    assert {k: final[k] for k in COUNTERS} == {
        "headwearQueued": 2, "headwearDropped": 0, "headwearWritten": 2, "headwearFailed": 0}
    assert fake.run_created["config"]["headwear"] == {"queue": 32}
    assert fake.run_ended["results"]["headwearWritten"] == 2
    assert "headwearWritten=2" in fake.run_ended["notes"]


def test_a_failing_embed_is_counted_and_the_run_ends_normally():
    """An embed without the reader (503): counted, remembered, the count untouched."""
    fake, final = run({"headwear": True}, headwear_status=503)
    _, off = run()
    assert final["state"] == "ended" and final["unique"] == off["unique"], "the count is untouched"
    assert fake.writes == []
    assert final["headwearFailed"] == 2 and final["headwearWritten"] == 0
    assert "503" in final["headwearLastError"]


# ---------------------------------------------------------------- the worker


class _Loop:
    """The three things a worker needs from its run loop, observable."""

    def __init__(self, entered: threading.Event, gate: threading.Event):
        self.s = Settings(embed_url="http://embed", match_url="http://match")
        self.status = {k: 0 for k in COUNTERS}
        self._lock = threading.Lock()
        self.entered, self.gate = entered, gate
        self.writes: list[int] = []
        self.log = SimpleNamespace(warning=lambda *a, **k: None)

    def _bump(self, key, by=1):
        with self._lock:
            self.status[key] += by

    def _set(self, **kv):
        with self._lock:
            self.status.update(kv)

    def _post(self, url, body):
        if url == "http://embed/headwear":
            self.entered.set()
            assert self.gate.wait(10), "the test never released the embed call"
            return {"readings": [READING], "model": "m+s"}
        self.writes.append(body["bodyId"])
        return {"ok": True, "written": True}


def test_a_stuck_embed_never_blocks_the_loop_and_a_full_queue_drops_and_counts():
    """offer() is instant while embed hangs; past capacity a read is dropped and counted."""
    entered, gate = threading.Event(), threading.Event()
    loop = _Loop(entered, gate)
    worker = HeadwearWorker(loop, "prun-1", capacity=2, poll_s=0.01)
    worker.start()
    frame = np.zeros((2160, 3840, 3), np.uint8)
    box = {"x": 1800.0, "y": 900.0, "w": 150.0, "h": 195.0}
    assert worker.offer(1, frame, box)
    assert entered.wait(5), "the worker took the first read and is stuck in embed"
    took = []
    for body_id in range(2, 7):
        t0 = time.perf_counter()
        worker.offer(body_id, frame, box)
        took.append(time.perf_counter() - t0)
    assert max(took) < 0.05, f"offer() must never wait on the worker: {took}"
    assert loop.status["headwearQueued"] == 3 and loop.status["headwearDropped"] == 3
    gate.set()
    worker.finish(drain_s=5.0)
    assert loop.writes == [1, 2, 3] and loop.status["headwearWritten"] == 3
    assert not worker.is_alive()


def test_what_is_still_queued_when_the_run_ends_is_counted_dropped():
    """finish() counts the reads it could not make."""
    loop = _Loop(threading.Event(), threading.Event())
    worker = HeadwearWorker(loop, "prun-1", capacity=4)  # never started: nothing drains
    frame = np.zeros((100, 100, 3), np.uint8)
    for body_id in (1, 2):
        worker.offer(body_id, frame, {"x": 40, "y": 50, "w": 20, "h": 26})
    worker.finish(drain_s=0.0)
    assert loop.status["headwearQueued"] == 2 and loop.status["headwearDropped"] == 2


def test_the_context_crop_is_an_owned_copy_with_the_margins_and_the_clamps():
    """0.5 widths a side, 1.2 heights up, to the face bottom; clamped; a copy."""
    frame = np.arange(480 * 640 * 3, dtype=np.uint32).astype(np.uint8).reshape(480, 640, 3)
    crop, box, where = context_crop(frame, {"x": 100.5, "y": 200.25, "w": 40.0, "h": 52.0})
    assert where == {"x": 80, "y": 137, "frameW": 640, "frameH": 480}
    assert crop.shape[:2] == (253 - 137, 161 - 80)
    assert box == {"x": 20.5, "y": 63.25, "w": 40.0, "h": 52.0}
    assert np.array_equal(crop, frame[137:253, 80:161])
    assert not np.shares_memory(crop, frame), "a queued read must not pin the 4K frame"
    _, box, where = context_crop(frame, {"x": 5.0, "y": 10.0, "w": 40.0, "h": 52.0})
    assert (where["x"], where["y"]) == (0, 0) and (box["x"], box["y"]) == (5.0, 10.0)
    for bad in ({"x": 1, "y": 2}, {"x": 1, "y": 2, "w": 0, "h": 5}, None,
                {"x": 900, "y": 900, "w": 10, "h": 10}):
        assert context_crop(frame, bad) is None


def test_the_knobs_read_from_the_env_and_empty_means_off(monkeypatch):
    """compose renders an unset ${VAR-} as empty: that is off / the default."""
    monkeypatch.setenv("HECO_HEADWEAR", "")
    monkeypatch.setenv("HECO_HEADWEAR_QUEUE", "")
    s = from_env()
    assert s.headwear is False and s.headwear_queue == 32
    monkeypatch.setenv("HECO_HEADWEAR", "1")
    monkeypatch.setenv("HECO_HEADWEAR_QUEUE", "0")
    s = from_env()
    assert s.headwear is True and s.headwear_queue == 1, "clamped: a queue holds one at least"
    assert knobs(s)["HECO_HEADWEAR"] is True and knobs(s)["HECO_HEADWEAR_QUEUE"] == 1
