"""Decoded frames, and the store where published ones wait for the consumer.

A :class:`RawFrame` keeps a picture in whatever layout its decoder produced
and hands out its BGR image, fitted to INGEST_MAX_WIDTH, at most once — the
runner polls the same frame repeatedly while it waits for the next seq.

:class:`FrameStore` is the L1 buffer. With ``INGEST_BUFFER_S = 0`` it is the
newest-frame slot (capacity one; a new frame overwrites an unread one and the
overwrite is counted as a drop). With ``INGEST_BUFFER_S > 0`` it is a FIFO
holding at most that many seconds of published frames AND at most
``INGEST_BUFFER_MB`` of pixels, oldest dropped first. It is not thread-safe on
its own: the capture worker holds its lock around every call.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class RawFrame:
    """One decoded picture; the cv2 decoder delivers ``bgr`` (HxWx3)."""

    __slots__ = ("layout", "data", "w", "h", "_bgr")

    def __init__(self, layout: str, data: np.ndarray, w: int, h: int) -> None:
        """Wrap decoded pixels; nothing is converted yet."""
        self.layout = layout
        self.data = data
        self.w = w
        self.h = h
        self._bgr: np.ndarray | None = None

    @classmethod
    def from_bgr(cls, bgr: np.ndarray) -> RawFrame:
        """A frame the decoder already delivered as BGR (the cv2 path)."""
        h, w = bgr.shape[:2]
        return cls("bgr", bgr, w, h)

    @property
    def nbytes(self) -> int:
        """Bytes of pixels held while the frame waits in the store."""
        return int(self.data.nbytes)

    def luma_source(self) -> np.ndarray:
        """What the motion gate shrinks to its small grey image."""
        return self.data

    def bgr(self, fit: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        """The BGR image (after ``fit``), computed on first call and kept."""
        if self._bgr is None:
            self._bgr = fit(self.data) if fit is not None else self.data
        return self._bgr


@dataclass(slots=True)
class Item:
    """A published frame waiting in the store."""

    seq: int
    t_ms: int
    #: The gate's clock at capture: footage seconds for a file, monotonic
    #: seconds since open for a live source. The FIFO's age bound reads it.
    clock_s: float
    frame: RawFrame
    motion: float | None


@dataclass(slots=True)
class Served:
    """What GET /frame hands out in lever mode: the frame plus the counters."""

    seq: int
    t_ms: int
    image: np.ndarray
    ended: bool
    motion: float | None
    backlog: int
    captured: int
    skipped: int | None
    dropped: int


class FrameStore:
    """Newest-frame slot (``buffer_s == 0``) or a FIFO bounded in seconds and bytes."""

    def __init__(self, buffer_s: float, cap_bytes: int) -> None:
        """An empty store; ``cap_bytes`` only binds in FIFO mode."""
        self.buffer_s = buffer_s
        self.cap_bytes = cap_bytes
        self._q: deque[Item] = deque()
        self.bytes = 0

    @property
    def fifo(self) -> bool:
        """True when frames queue instead of overwriting each other."""
        return self.buffer_s > 0

    def pending(self) -> int:
        """Frames published and not yet taken."""
        return len(self._q)

    def put(self, item: Item) -> int:
        """Store a published frame; returns how many unread frames it cost.

        The newest frame is never the one dropped: a cap smaller than one
        frame degrades to the newest-frame slot rather than to nothing.
        """
        dropped = 0
        if not self.fifo:
            dropped = len(self._q)  # an unread frame, overwritten
            self._q.clear()
            self.bytes = 0
        self._q.append(item)
        self.bytes += item.frame.nbytes
        while len(self._q) > 1 and item.clock_s - self._q[0].clock_s > self.buffer_s:
            self._drop_oldest()
            dropped += 1
        while len(self._q) > 1 and self.bytes > self.cap_bytes:
            self._drop_oldest()
            dropped += 1
        return dropped

    def take(self) -> Item | None:
        """The oldest unread frame, removed from the store; None when empty."""
        if not self._q:
            return None
        item = self._q.popleft()
        self.bytes -= item.frame.nbytes
        return item

    def _drop_oldest(self) -> None:
        old = self._q.popleft()
        self.bytes -= old.frame.nbytes
