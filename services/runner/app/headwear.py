"""Head-covering reads for the review queue, made OFF the frame loop.

WHY. The match service can set a turban against a bare head aside only if it
knows which heads were covered, and that takes a SigLIP B/16 read of the head
(embed POST /headwear): two images through a ViT, a few ms on the GPU but
about 0.3 s on a CPU. That must never be the frame loop's cost, so the loop
only CUTS the head's context from the frame it already decoded (a slice and a
copy) and hands it to this worker; the worker encodes it, asks embed, and
writes the 8 logits onto the sighting's body row (match POST
/body-sightings/headwear) by the ``bodyId`` /match returned.

WHEN. Only after a /match that MINTED (``isNew``) or ENROLLED a template
(``templateAdded``) and logged a body row (``bodyId``): the views the gallery
keeps, spread over the visit. On run 8b8b87 that was 734 reads for 74
identities (median 10 each), ~0.75 reads/s, in bursts at the doorway. Never a
staff hit, never every frame.

BOUNDED. The queue holds ``HECO_HEADWEAR_QUEUE`` crops (32; a 4K face's
context crop is ~0.4 MB, 2 MB at the largest). A burst past it is DROPPED and
counted (``headwearDropped``) — a lost read is an absent read, and absent is
not zero. At the end of a run the worker gets a few seconds to finish what is
queued; what it cannot finish is counted as dropped too.

OFF BY DEFAULT (``HECO_HEADWEAR=0``): no worker, no crop, no call, no status
counter, nothing in the run record — the pinned replays hold the loop to it.
"""

from __future__ import annotations

import base64
import math
import queue
import threading
import time

import cv2
import numpy as np

#: The context the runner cuts around a face: 0.5 face widths each side,
#: 1.2 face heights above, down to the face's bottom. The embed service's
#: views need 0.30 / 1.0 (and it refuses a cut that does not cover them).
SIDE_MARGIN = 0.5
UP_MARGIN = 1.2
#: Status counters (0 when the reader is on, None when it is off).
COUNTERS = ("headwearQueued", "headwearDropped", "headwearWritten", "headwearFailed")


def context_crop(frame_img: np.ndarray, box) -> tuple[np.ndarray, dict, dict] | None:
    """Cut a face's head context from the full frame.

    Returns ``(crop, box in the crop's pixels, {x, y, frameW, frameH})`` —
    the crop an owned copy (the frame may be large and must not be kept
    alive by a queued read) — or None for a box that cannot be cut.
    """
    try:
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    height, width = frame_img.shape[:2]
    x0 = max(0, int(math.floor(x - SIDE_MARGIN * w)))
    y0 = max(0, int(math.floor(y - UP_MARGIN * h)))
    x1 = min(width, int(math.ceil(x + w + SIDE_MARGIN * w)))
    y1 = min(height, int(math.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_img[y0:y1, x0:x1].copy()
    return (
        crop,
        {"x": x - x0, "y": y - y0, "w": w, "h": h},
        {"x": x0, "y": y0, "frameW": int(width), "frameH": int(height)},
    )


class HeadwearWorker(threading.Thread):
    """One background thread reading heads for one run; ``offer()`` never blocks.

    The loop calls :meth:`offer` from the frame loop; everything slow — PNG
    encoding, the embed call, the match write — happens here. Every failure
    is counted (``headwearFailed``), remembered (status ``headwearLastError``)
    and logged once per distinct reason; the thread never dies of one.
    """

    def __init__(self, loop, run_id: str, capacity: int, poll_s: float = 0.05) -> None:
        """Bind to a run loop (its HTTP client, status and log) and one gallery."""
        super().__init__(name="heco-headwear", daemon=True)
        self._loop = loop
        self._run_id = run_id
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(capacity)))
        self._poll_s = poll_s
        self._stopping = threading.Event()
        self._said: set[str] = set()

    # ------------------------------------------------------------ the loop side

    def offer(self, body_id: int, frame_img: np.ndarray, box) -> bool:
        """Queue one sighting's head for reading; False (and counted) when it cannot be.

        Called on the frame loop: a slice, a copy and a non-blocking put.
        """
        cut = context_crop(frame_img, box)
        if cut is None:
            self._fail("face box cannot be cut")
            return False
        try:
            self._queue.put_nowait((int(body_id), *cut))
        except queue.Full:
            self._loop._bump("headwearDropped")
            return False
        self._loop._bump("headwearQueued")
        return True

    def finish(self, drain_s: float) -> None:
        """Let the queue drain for up to ``drain_s``, then stop; count the rest dropped."""
        deadline = time.monotonic() + max(0.0, drain_s)
        while not self._queue.empty() and time.monotonic() < deadline and self.is_alive():
            time.sleep(self._poll_s)
        self._stopping.set()
        if self.is_alive():
            self.join(timeout=max(0.1, deadline - time.monotonic()) + 5.0)
        left = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            left += 1
        if left:
            self._loop._bump("headwearDropped", left)

    # ------------------------------------------------------------ the worker side

    def run(self) -> None:
        """Read until stopped; one item at a time, never raising."""
        while not self._stopping.is_set():
            try:
                item = self._queue.get(timeout=self._poll_s)
            except queue.Empty:
                continue
            try:
                self._read(*item)
            except Exception as e:  # noqa: BLE001 — a read is advisory; the thread must live
                self._fail(f"{type(e).__name__}: {e}")

    def _read(self, body_id: int, crop: np.ndarray, box: dict, where: dict) -> None:
        """Encode, ask embed for the logits, write them to the body row."""
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            self._fail("png encode failed")
            return
        s = self._loop.s
        reply = self._loop._post(f"{s.embed_url}/headwear", {
            "imageB64": base64.b64encode(buf.tobytes()).decode("ascii"),
            "faces": [{"box": box}],
            "crop": where,
        })
        readings = reply.get("readings") if isinstance(reply, dict) else None
        reading = readings[0] if isinstance(readings, list) and readings else None
        model = reply.get("model") if isinstance(reply, dict) else None
        if not (
            isinstance(reading, list) and len(reading) == 8
            and all(isinstance(v, int | float) and math.isfinite(v) for v in reading)
            and isinstance(model, str) and model
        ):
            self._fail("embed returned no reading")
            return
        res = self._loop._post(f"{s.match_url}/body-sightings/headwear", {
            "runId": self._run_id, "bodyId": body_id,
            "headwear": [float(v) for v in reading], "model": model,
        })
        if isinstance(res, dict) and res.get("written"):
            self._loop._bump("headwearWritten")
        else:
            self._fail("the body row is gone")

    def _fail(self, reason: str) -> None:
        """Count one failed read and say why — once per distinct reason in the log."""
        self._loop._bump("headwearFailed")
        self._loop._set(headwearLastError=reason[:300])
        key = reason.split(":", 1)[0][:120]
        if key not in self._said:
            self._said.add(key)
            self._loop.log.warning(f"head-covering read failed ({reason}) — counted in "
                                   "headwearFailed; the count is unaffected")
