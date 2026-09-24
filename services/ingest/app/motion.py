"""The L1 motion gate: publish a frame only when something in it moved.

A wedding entrance spends long stretches empty or standing still, and every
frame the pipeline processes there costs a full detection chain (~200 ms per
4K frame measured on the CUDA box) to find the same nothing. The gate compares
a ~1/8-scale luma image of each frame with the last PUBLISHED one and lets the
frame through when enough of it changed, or when the keepalive is due (the
tracker and the runner's stall detection need a heartbeat, and a guest who
stands still is still seen once a second).

Every choice leans toward processing, because a missed guest is the failure
this pipeline fears most and an extra frame only costs time:

* **Compared with the last PUBLISHED frame, not the previous one.** A slow
  walker whose change from one frame to the next stays under the threshold
  still builds up change against the last published frame until it crosses
  it. A frame-to-frame gate could skip him until the keepalive.
* **Each small frame is normalised by its own mean and std** before the
  comparison, so a global exposure step, an auto-iris hunt or a DJ flash is
  not motion: those move every pixel through one affine map, and normalising
  removes it. Anything local (a spotlight sweeping, a coloured wash that
  lights some surfaces more than others) still reads as motion and is
  processed.
* **The first frame, and the first after a reconnect, is always published**
  and reports ``motion=None``, because there is nothing to compare it with.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Downscale for the luma the gate compares: 3840x2160 -> 480x270. INTER_AREA
#: averages each 8x8 block, which takes sensor and codec noise down with it
#: (point sampling kept it and read as motion).
SCALE = 8
#: Never shrink below this width. The gate should still see a small test clip.
_MIN_SMALL_W = 32
#: Floor on the std a small frame is divided by. A near-flat frame (lens cap,
#: black screen) would otherwise have its noise amplified into "motion".
_STD_FLOOR = 1.0


def small_size(w: int, h: int) -> tuple[int, int]:
    """The (w, h) the gate compares at for a frame of this size."""
    factor = max(1, min(SCALE, w // _MIN_SMALL_W))
    return max(1, w // factor), max(1, h // factor)


def small_luma(img: np.ndarray) -> np.ndarray:
    """~1/8-scale grey image of a frame: BGR (HxWx3) or a luma plane (HxW).

    Measured on one 4K frame (laptop, 8 threads): from BGR, resize then grey
    6.0 ms; from an NV12 Y plane, 2.2 ms. The hardware-decode path gets its
    grey image for free, which is half of why L6 exists.
    """
    h, w = img.shape[:2]
    size = small_size(w, h)
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _normalise(small: np.ndarray) -> np.ndarray:
    """Zero mean, unit std: what is left once a global lighting change is gone."""
    z = small.astype(np.float32)
    mean, std = float(z.mean()), float(z.std())
    return (z - mean) / max(std, _STD_FLOOR)


class MotionGate:
    """Decides, frame by frame, whether a frame is worth publishing.

    ``decide`` is called once per decoded frame, in order, with the frame's
    small luma and its clock: footage seconds for a file (frame number / fps,
    so a lockstep run makes the same choices however fast it runs), and
    monotonic seconds since open for a live source.
    """

    def __init__(self, min_frac: float, pixel_thr: float, keepalive_s: float) -> None:
        """Hold the thresholds; nothing has been published yet."""
        self.min_frac = min_frac
        self.pixel_thr = pixel_thr
        self.keepalive_s = keepalive_s
        self._ref: np.ndarray | None = None
        self._last_pub_s = 0.0

    def reset(self) -> None:
        """Forget the reference, so the next frame is published (a new stream)."""
        self._ref = None

    def decide(self, small: np.ndarray, clock_s: float) -> tuple[bool, float | None]:
        """(publish?, motion fraction or None when there was nothing to compare)."""
        z = _normalise(small)
        if self._ref is None or self._ref.shape != z.shape:
            publish, motion = True, None
        else:
            changed = np.count_nonzero(np.abs(z - self._ref) > self.pixel_thr)
            motion = changed / z.size
            quiet_s = clock_s - self._last_pub_s
            # 1e-6: fifteen frames at 14.99992 fps (what the bench clip's
            # container reports) are 1.000005 s, and at 15.00008 fps they
            # would fall just short of 1.0. Neither should push the keepalive
            # back a whole frame.
            publish = motion >= self.min_frac or quiet_s + 1e-6 >= self.keepalive_s
        if publish:
            self._ref = z
            self._last_pub_s = clock_s
        return publish, motion
