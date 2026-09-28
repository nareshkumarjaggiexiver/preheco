"""ingest — FastAPI app on the contract port 7101.

Endpoints (CONTRACTS.md):

* ``POST /open``  {url|path, loop, isFile?, lockstep?, owner?, takeover?} — start
  capturing.  ``isFile`` marks a url as a finite recording (paced, ends at
  EOF) rather than a live stream.  The capture slot is EXCLUSIVE: see "one
  slot, one owner" below.
* ``POST /close`` {owner?, force?} — release the slot at end of run.
* ``GET /frame``  → {tMs, imageB64, w, h, seq, ended, frameRef} — the LATEST
  frame only (drop-not-queue; see app.capture for the policy). With a lever
  armed (app.config) it is the OLDEST UNREAD frame instead, dequeued, and the
  body also carries {motion, backlog, skipped, dropped, captured}.
  ``frameRef`` names the same frame on the shared tmpfs (heco_common.frameref;
  null with no mount), and ``?jpeg=0`` drops the JPEG when a ref was written.
* ``GET /health`` → {ok, model, version, owner, knobs, capture}.

One slot, one owner
-------------------
This service holds exactly ONE capture worker.  It used to let every /open
replace it unconditionally, which made a second run a silent thief: start a
staff enrolment while a count run is live at the gate (the documented
mid-event workflow) and the count run keeps polling /frame, gets the
enrolment's frames, and counts the walk-through — no error anywhere, just a
plausible-looking frame stream and a corrupted count.

So an /open that carries an ``owner`` CLAIMS the slot, and any later /open by
a different owner is refused with 409 naming the holder.  Three escapes, all
deliberate: the same owner may re-open (an idempotent restart), a slot whose
capture thread has died is not a live run and may be claimed, and
``takeover: true`` is the operator's explicit "I know, take it anyway".  An
/open with no owner keeps the old replace-anything behaviour so ad-hoc probes
and the smoke script are unaffected — but the runner always sends one.

Frame encoding happens here, on demand, per request — the capture thread
stores raw arrays so an idle pipeline costs no JPEG work.
"""

import itertools
import os
import threading
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, HTTPException
from heco_common import frameref
from heco_common.config import env_int
from heco_common.gate_auth import install_bearer_gate
from heco_common.imaging import encode_jpeg_b64
from heco_common.schemas import CloseSource, Frame, Health, OpenSource
from pydantic import Field

from . import __version__
from .capture import CaptureError, CaptureWorker
from .config import cv_threads_from_env, levers_from_env


class OpenIngest(OpenSource):
    """POST /open body: the shared OpenSource plus this run's lever overrides.

    Both default to None, which means "whatever the service's env says"
    (app.config) — so a runner that sends neither gets exactly the worker it
    always got. Declared here rather than on the shared model because only
    ingest reads them; the runner forwards its ``source`` dict verbatim.
    """

    #: Override INGEST_MOTION_GATE for this run.
    motionGate: bool | None = None
    #: Override INGEST_BUFFER_S for this run (0 = the newest-frame slot).
    bufferS: float | None = Field(default=None, ge=0)


class _State:
    """Holds the single active capture worker and the run that claimed it."""

    def __init__(self) -> None:
        """Start with an empty, unowned slot."""
        self.worker: CaptureWorker | None = None
        self.owner: str | None = None
        self.lock = threading.Lock()

    def swap(self, new: CaptureWorker | None, owner: str | None = None) -> None:
        """Install a new worker (or None), stopping the previous one.

        The shared transport's frames go with the run that was served them
        (frameref.clear): a closed run has no stage call left in flight — the
        runner closes only after its last frame — and without this its last
        HECO_FRAMES_KEEP frames (24 MB each at 4K) sat on tmpfs until the next
        run's writes happened to retire them, or forever after a restart,
        when the write numbers begin again below theirs.
        """
        old, self.worker, self.owner = self.worker, new, (owner if new else None)
        if old is not None:
            old.stop()
        frameref.clear()

    def held_by_live_run(self) -> bool:
        """True when an owning run's capture thread is still alive.

        A dead thread means the owner crashed or its process went away, so the
        slot is not really held and refusing the next run would just wedge the
        pipeline until somebody restarts ingest.
        """
        return (
            self.worker is not None
            and self.owner is not None
            and self.worker.is_alive()
        )


state = _State()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Size OpenCV's pool (INGEST_CV_THREADS) and release capture on shutdown."""
    threads = cv_threads_from_env()
    if threads is not None:
        cv2.setNumThreads(threads)
    yield
    state.swap(None)


app = FastAPI(title="heco-ingest", version=__version__, lifespan=_lifespan)
# Inbound auth (runbook step 8): armed by HECO_REQUIRE_AUTH=1, this refuses
# LAN callers without a bearer credential — an heco-auth token verified
# locally, or the legacy shared secret while it survives. /health stays
# open for the compose healthchecks. Unarmed, nothing changes.
install_bearer_gate(app)


@app.post("/open")
def open_source(body: OpenIngest) -> dict:
    """Open an RTSP url or a video file, claiming the exclusive capture slot.

    409 when a different, still-live owner holds the slot — the error names
    that run so the operator learns "the gate count is using this camera"
    instead of silently losing it.
    """
    if body.path is not None and not os.path.exists(body.path):
        raise HTTPException(status_code=400, detail=f"no such file: {body.path}")
    try:
        levers = levers_from_env(motion_gate=body.motionGate, buffer_s=body.bufferS)
    except ValueError as exc:
        # A malformed knob in the SERVICE's env, not in the request: refuse
        # loudly with the variable's name rather than guess a default.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    with state.lock:
        holder = state.owner
        if (
            body.owner
            and not body.takeover
            and state.held_by_live_run()
            and holder != body.owner
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"capture slot is held by run {holder!r}; stop that run or "
                    f"re-send with takeover=true to seize its camera"
                ),
            )
        # A url is live unless the caller SAYS it is a finite recording
        # (isFile): pacing and EOF-ends-the-run semantics must not hinge on
        # guessing from the url's shape.
        source, is_file = (
            (body.path, True) if body.path else (body.url, body.isFile)
        )
        try:
            worker = CaptureWorker(
                source=source, is_file=is_file, loop=body.loop, lockstep=body.lockstep,
                levers=levers,
            )
        except CaptureError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        worker.start()
        state.swap(worker, owner=body.owner)
    return {"ok": True, "source": source, "loop": body.loop, "owner": body.owner}


@app.post("/close")
def close_source(body: CloseSource) -> dict:
    """Release the capture slot so the next run can claim the camera.

    Owner-checked: a stale close from a run that already lost the slot must
    not stop the run that holds it now.  ``force`` is the operator override.
    Closing an already-free slot is success — release is idempotent.
    """
    with state.lock:
        if state.worker is None:
            return {"ok": True, "released": False, "owner": None}
        holder = state.owner
        if body.owner and holder and holder != body.owner and not body.force:
            raise HTTPException(
                status_code=409,
                detail=f"capture slot is held by run {holder!r}, not {body.owner!r}",
            )
        state.swap(None)
    return {"ok": True, "released": True, "owner": holder}


#: The last JPEG handed out, keyed by (worker generation, seq, quality).
_jpeg_lock = threading.Lock()
_jpeg_last: tuple[tuple[int, int, int], str] | None = None


def _jpeg_once(worker: CaptureWorker, seq: int, img, quality: int) -> str:
    """Encode a frame once, however many times it is asked for.

    The runner polls /frame every source_poll_s (20 ms) while it waits for
    the next seq, and every one of those polls returned the SAME frame, freshly
    encoded — 14-21 ms of JPEG per poll on a 4K frame. The output for a given
    frame is identical, so encoding it again is pure waste, and the motion
    gate makes that waiting common: a still scene is one frame a second.

    Keyed by the worker's generation as well as seq, because seq restarts at 1
    on every /open: seq alone would hand a new run the old run's first frame.
    """
    global _jpeg_last
    key = (worker.generation, seq, quality)
    with _jpeg_lock:
        if _jpeg_last is not None and _jpeg_last[0] == key:
            return _jpeg_last[1]
    b64 = encode_jpeg_b64(img, quality=quality)
    with _jpeg_lock:
        _jpeg_last = (key, b64)
    return b64


#: The last frame written to the shared transport: ((generation, seq), ref).
#: A frame is immutable for its (generation, seq), so re-writing it on a poll
#: is pure waste — see get_frame.
_ref_lock = threading.Lock()
_ref_last: tuple[tuple[int, int], str | None] | None = None
#: WRITE numbers, one per frame written, never reused in this process. The
#: ref's number is this, not the capture seq, because frameref retires files
#: more than HECO_FRAMES_KEEP NUMBERS behind the newest: numbered by capture
#: seq, a gap wider than the keep between two SERVED frames — a still room
#: under the motion gate publishes one frame a second, 15 seqs apart at
#: 15 fps; an unbuffered camera outrunning a 1.5 fps consumer does it too —
#: retired the frame the runner was still embedding. Numbered by write, the
#: keep means "the last N frames handed out", whatever the gaps.
_ref_ids = itertools.count(1)


def _cached_ref(worker: CaptureWorker, seq: int, img) -> str | None:
    """The shared-transport ref for this frame, writing it at most once.

    Keyed by (generation, seq), as :func:`_jpeg_once` is: seq restarts at 1 on
    every /open, and seq alone would hand a new run the previous run's frame.
    The lock is held across the write so two concurrent polls of one frame
    cannot write it twice. Only frames actually served are ever written — a
    frame the motion gate skipped, or the buffer dropped, never touches tmpfs.
    """
    global _ref_last
    key = (worker.generation, seq)
    with _ref_lock:
        if _ref_last is not None and _ref_last[0] == key:
            return _ref_last[1]
        ref = frameref.write_frame(img, next(_ref_ids))
        _ref_last = (key, ref)
        return ref


# exclude_unset: a field is on the wire only when this handler SET it. With
# every lever off that is exactly the six fields /frame has always carried —
# plus the frameRef this branch passes explicitly (null with no mount) — and
# in lever mode an unmeasured value still travels as an explicit null.
@app.get("/frame", response_model_exclude_unset=True)
def get_frame(jpeg: bool = True) -> Frame:
    """Return the latest captured frame as base64 JPEG.

    409: no source open. 503: source open but no frame decoded yet (a live
    RTSP source can take a moment) — callers should retry shortly.

    Lever mode (a gate or buffer armed): the OLDEST unread frame, taken out
    of the store, plus the counters the runner carries into the run's notes.

    The shared transport, when one is mounted: the SAME pixels written once
    to tmpfs so the three consuming stages can take them without a codec.
    imageB64 is still produced unless the caller declines it — the ref is an
    optimisation and a consumer must always have something to fall back to.
    ONCE PER FRAME, not once per request: the runner polls this endpoint at
    source_poll_s (50 Hz) waiting for the seq to advance, every poll returns
    the SAME frame, and writing on each one wrote 24 MB to tmpfs fifty times a
    second for one frame (ref-only measured SLOWER than the JPEG it replaced,
    2.32 fps against 5.14, until it stopped). With a lever armed the frame is
    written when it is TAKEN, so the buffer's backlog lives in ingest's own
    memory (INGEST_BUFFER_MB) and tmpfs holds only the last HECO_FRAMES_KEEP
    frames handed out.

    ``jpeg=0`` says the caller has verified it can read refs, so the encode is
    waste — 13.9 ms per 4K frame, measured. Honoured ONLY when a ref was
    actually written: a caller that declines the JPEG and gets no ref either
    would receive a frame with no pixels in it at all, which is a far worse
    failure than an encode nobody needed.
    """
    worker = state.worker
    if worker is None:
        raise HTTPException(status_code=409, detail="no source open — POST /open first")
    quality = env_int("INGEST_JPEG_QUALITY", 85)
    if worker.levered:
        served = worker.take()
        if served is None:
            raise HTTPException(status_code=503, detail="no frame captured yet — retry")
        h, w = served.image.shape[:2]
        frame_ref = _cached_ref(worker, served.seq, served.image)
        skip_jpeg = (not jpeg) and frame_ref is not None
        return Frame(
            tMs=served.t_ms,
            imageB64="" if skip_jpeg else _jpeg_once(worker, served.seq, served.image, quality),
            w=w, h=h, seq=served.seq, ended=served.ended, frameRef=frame_ref,
            motion=served.motion, backlog=served.backlog, skipped=served.skipped,
            dropped=served.dropped, captured=served.captured,
        )
    latest = worker.latest()
    if latest is None:
        raise HTTPException(status_code=503, detail="no frame captured yet — retry")
    seq, t_ms, img = latest
    h, w = img.shape[:2]
    frame_ref = _cached_ref(worker, seq, img)
    skip_jpeg = (not jpeg) and frame_ref is not None
    # `ended` is the only signal that separates a played-out file from a camera
    # that blinked: both freeze `seq`. The worker knows which it is (it retries
    # a live stream forever and only sets ended for a finished file), and until
    # now it kept that to itself.
    return Frame(
        tMs=t_ms, imageB64="" if skip_jpeg else _jpeg_once(worker, seq, img, quality),
        w=w, h=h, seq=seq, ended=bool(getattr(worker, "ended", False)),
        frameRef=frame_ref,
    )


@app.get("/health")
def health() -> dict:
    """Liveness + identity; 'model' names the capture backend, not a DNN.

    ``owner`` is included so an operator can see which run holds the camera
    without guessing from a 409.

    ``knobs`` is what the NEXT /open will get from the env (a run may still
    override motionGate/bufferS); a malformed knob turns ``ok`` false, so the
    compose healthcheck refuses the stack instead of letting it count with a
    default nobody chose. ``capture`` is the open worker's own settings and
    counters (null when nothing is open; counters null outside lever mode).

    ``device`` is the decoder truth, the same shape the model services serve:
    ``requested`` (INGEST_DECODER) against ``active`` (what the open worker
    actually decodes with — null while nothing is open) and ``error``, the
    reason when a hardware decoder fell back to cpu.
    """
    body = {
        **Health(ok=True, model="opencv-videocapture", version=__version__).model_dump(),
        "owner": state.owner,
    }
    requested = None
    try:
        knobs = levers_from_env().knobs()
        knobs["cvThreads"] = cv_threads_from_env()
        body["knobs"], requested = knobs, knobs["decoder"]
    except ValueError as exc:
        body["ok"] = False
        body["knobs"] = {"error": str(exc)}
    # What the pool actually is, whoever set it.
    body["cvThreadsActive"] = cv2.getNumThreads()
    worker = state.worker
    body["capture"] = worker.describe() if worker is not None else None
    body["device"] = (
        dict(worker.decoder)
        if worker is not None
        else {"requested": requested, "active": None, "error": None}
    )
    return body
