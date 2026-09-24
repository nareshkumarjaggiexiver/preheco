"""A file stream cut mid-run resumes where it was cut — or fails loudly.

THE CASE (2026-09-25). Footage is streamed to ingest over HTTP from the
planner's disk. Restarting the planner mid-run cut the stream at 330 MB of a
629 MB clip; OpenCV's read simply failed, the worker called it the end of the
file, and the run settled as source-ended with 4709 of 9000 frames counted.
Run 8b8b87 lost its last 554 frames the same way. A short count reported as
complete is the failure this pipeline must never have.
"""

import time

import cv2
import numpy as np
import pytest
from app.capture import CaptureWorker

from .conftest import SyntheticCapture

N, CUT = 120, 50


def frame_for(i: int) -> np.ndarray:
    """Frame i carries its own index in its pixels, so order is checkable."""
    img = np.full((48, 64, 3), 40, np.uint8)
    img[0, 0] = (i % 256, i // 256, 7)
    return img


def patched(monkeypatch, *, cut_reopens: int = 0, frame_count: float = N):
    """cv2.VideoCapture serving an N-frame file whose FIRST stream is cut at
    frame CUT; the next ``cut_reopens`` reopens fail to open (the planner still
    restarting); after that a reopened stream serves the rest.  Returns the
    shared state so a test can see what happened."""
    state = {"cut": False, "opens": 0, "failing": cut_reopens, "seeks": []}

    class Cap(SyntheticCapture):
        def __init__(self):
            super().__init__(frame_for, n_frames=N, fps=15.0)
            self.ok = True

        def isOpened(self):  # noqa: N802 — cv2's name
            return self.ok

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return frame_count
            return super().get(prop)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                state["seeks"].append(int(value))
            return super().set(prop, value)

        def _next(self):
            if not state["cut"] and self.i >= CUT:
                state["cut"] = True
                return None  # the stream is cut here: read() just fails
            return super()._next()

    def factory(*_a, **_k):
        state["opens"] += 1
        cap = Cap()
        if state["cut"] and state["failing"] > 0:
            state["failing"] -= 1
            cap.ok = False  # still restarting: this open fails
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    monkeypatch.setattr(CaptureWorker, "RESUME_BACKOFF_S", (0.01,) * 4)
    return state


def drain(worker, budget_s=10.0):
    """Take every frame a lockstep file hands out, until ended or quiet."""
    seen, quiet_since = [], None
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        got = worker.latest()
        if got is not None and (not seen or got[0] != seen[-1]):
            seen.append(got[0])
            quiet_since = None
        elif worker.ended or worker.interrupted:
            if quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since > 0.2:
                break
        time.sleep(0.002)
    return seen


def test_a_cut_stream_resumes_at_the_frame_it_was_cut_and_every_frame_arrives(monkeypatch):
    """Cut at frame 50 while the server restarts (two failed reopens): every
    one of the 120 frames still arrives, in order, and the file ends."""
    state = patched(monkeypatch, cut_reopens=2)
    w = CaptureWorker(source="http://planner/run-videos/x", is_file=True, lockstep=True)
    w.start()
    try:
        seen = drain(w)
    finally:
        w.stop()
    assert seen == list(range(1, N + 1)), f"got {len(seen)} of {N}"
    assert w.ended is True, "and the file really ends at its end"
    assert (w.resumes, w.interrupted) == (1, None)
    assert state["seeks"][-1] == CUT, "resumed at the next frame, not from the top"


def test_a_cut_that_cannot_be_resumed_never_claims_the_file_ended(monkeypatch):
    """The server never comes back: the worker stops producing and says why,
    and `ended` stays false so the runner fails the run out loud."""
    patched(monkeypatch, cut_reopens=99)
    w = CaptureWorker(source="http://planner/run-videos/x", is_file=True, lockstep=True)
    w.start()
    try:
        seen = drain(w)
    finally:
        w.stop()
    assert seen == list(range(1, CUT + 1))
    assert w.ended is False, "a short count must not settle as complete"
    assert "cut at frame 50 of 120" in (w.interrupted or "")
    assert w.describe()["interrupted"] == w.interrupted


def test_a_file_that_really_ends_ends_and_an_unknown_length_keeps_todays_rule(monkeypatch):
    """Frame count unknown (0): nothing to judge a cut by — today's behaviour,
    the read failure is the end (and nothing is re-opened)."""
    state = patched(monkeypatch, frame_count=0.0)
    w = CaptureWorker(source="http://planner/run-videos/x", is_file=True, lockstep=True)
    w.start()
    try:
        seen = drain(w)
    finally:
        w.stop()
    assert seen == list(range(1, CUT + 1)) and w.ended is True
    assert w.resumes == 0 and state["seeks"] == []


@pytest.mark.parametrize("missing", [1, 29])
def test_a_stop_within_the_tolerance_of_the_end_is_the_end(monkeypatch, missing):
    """An NVR file's last frames can fail to decode (a trailing partial GOP):
    stopping within CUT_TOLERANCE_FRAMES of the length is an end, not a cut."""
    state = patched(monkeypatch, frame_count=float(CUT + missing))
    w = CaptureWorker(source="http://planner/run-videos/x", is_file=True, lockstep=True)
    w.start()
    try:
        drain(w)
    finally:
        w.stop()
    assert w.ended is True and w.resumes == 0 and state["seeks"] == []
