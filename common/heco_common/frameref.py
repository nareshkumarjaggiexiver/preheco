"""Frame handoff through a shared tmpfs, instead of base64 JPEG in JSON.

WHY THIS EXISTS. The POC moved frames as base64 JPEG inside the JSON body,
and said so in the README: "Frames travel as base64 JPEG in JSON — POC scale.
Shared memory arrives only if profiling demands it." It demanded. Measured on
the .94 box against a 4K frame:

    ingest JPEG-encodes it                       13.9 ms
    persons JPEG-decodes it                      17.8 ms
    faces JPEG-decodes it                        17.8 ms
    embed JPEG-decodes it                        17.8 ms
    base64 + JSON of a 2.1 MB payload, x3        ~15 ms
    ---------------------------------------------------
                                                 ~82 ms per frame

on a pipeline whose whole budget was 224 ms. The pixels never leave one box —
they are encoded, expanded 33% by base64, parsed out of JSON and decoded
again, four times, to travel between processes that share a kernel.

THE DESIGN, and why it is files rather than POSIX shared memory. Every frame
is written ONCE to a tmpfs (RAM; no disk touches it) as raw BGR bytes, and
the consumers are handed its path. A tmpfs file read is a page-cache read —
the same memory, no copy through a socket, no codec either way.

TORN READS ARE IMPOSSIBLE HERE, and that is the reason for the naming
scheme rather than a ring of reused slots. Each frame gets its OWN file named
for its sequence number, and old files are unlinked once N newer ones exist.
A reader that opened the file keeps its fd, and POSIX keeps the inode alive
until that fd closes — so a frame being retired while somebody reads it is
simply read to completion. A ring of overwritten slots would need leasing,
generation counters and a fence; this needs none, because the filesystem
already provides exactly the guarantee we want.

WHAT IT COSTS. The stages must now share a host, which HTTP did not require —
HECO_PERSONS_URL exists precisely so a stage could live on another machine.
For a single edge box that was always true in practice; it is a one-way door
and is named here so nobody discovers it by surprise.

FALLBACK IS ALWAYS AVAILABLE. A consumer that cannot read the ref — no mount,
a frame already retired, a mixed-version deploy — falls back to `imageB64`,
which every producer still sends. The transport is an optimisation, never a
correctness dependency.
"""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

import numpy as np

#: Where the shared tmpfs is mounted in every container that speaks this.
#: Unset (or a missing directory) disables the whole path — the services then
#: behave exactly as they did before, on base64 JPEG.
FRAMES_DIR_ENV = "HECO_FRAMES_DIR"

#: How many frames stay readable behind the newest one. A consumer that falls
#: this far behind loses the ref and falls back to imageB64 rather than
#: reading something that is no longer what it asked for. At 4K a frame is
#: 24 MB, so the default keeps ~200 MB of tmpfs in flight.
KEEP_ENV = "HECO_FRAMES_KEEP"
DEFAULT_KEEP = 8

#: Only ever touch files we wrote: the sweeper unlinks by pattern, and a
#: pattern that could match something else is a sweeper that deletes it.
_NAME = re.compile(r"^f(\d+)_(\d+)x(\d+)\.bgr$")


def frames_dir() -> Path | None:
    """The configured shared directory, or None when the path is off."""
    raw = os.environ.get(FRAMES_DIR_ENV)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def _keep() -> int:
    try:
        n = int(os.environ.get(KEEP_ENV) or DEFAULT_KEEP)
    except ValueError:
        return DEFAULT_KEEP
    return max(1, n)


def write_frame(img: np.ndarray, seq: int, directory: Path | None = None) -> str | None:
    """Write one BGR frame to the shared dir; return its ref, or None if off.

    The name carries the shape, so a reader needs nothing but the ref to
    rebuild the array — no sidecar, no second source of truth to disagree
    with the bytes.

    Written to a dot-prefixed temp name and RENAMED into place: rename is
    atomic within a filesystem, so a consumer can never observe a partially
    written frame. The sweep then retires everything older than `keep`.
    """
    directory = directory or frames_dir()
    if directory is None:
        return None
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        return None  # not the contract's BGR uint8 — say nothing, let JPEG carry it
    h, w = img.shape[:2]
    # SWEEP FIRST. Sweeping only after a successful write is a deadlock: a
    # full tmpfs fails the write, the sweep never runs, and nothing is ever
    # retired again — measured live, the mount sat at 98% holding 21 frames
    # against a keep of 8, and every write from then on failed silently into
    # the JPEG fallback. Making room is a precondition for writing, not a
    # reward for having written.
    _sweep(directory, seq, _keep())
    name = f"f{seq}_{w}x{h}.bgr"
    final = directory / name
    tmp = directory / f".{name}.part"
    try:
        with open(tmp, "wb") as fh:
            fh.write(np.ascontiguousarray(img).data)
        os.replace(tmp, final)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        return None
    return name


def read_frame(ref: str, directory: Path | None = None) -> np.ndarray | None:
    """Rebuild a frame from its ref, or None when it cannot be had.

    None is not an error worth raising: every caller has `imageB64` in hand
    and the fallback is the point. Returns a COPY, not a view on the mapped
    file, because the caller keeps the array well past this function and the
    file underneath it is retired on a schedule it does not control.
    """
    directory = directory or frames_dir()
    if directory is None or not ref:
        return None
    match = _NAME.match(ref)
    if match is None:
        return None  # not a name we wrote; never open an arbitrary path
    _, w, h = (int(g) for g in match.groups())
    path = directory / ref
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) != w * h * 3:
        return None  # truncated or mid-write; the JPEG is still good
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()


def _sweep(directory: Path, newest_seq: int, keep: int) -> None:
    """Unlink frames more than `keep` behind the newest.

    Unlinking a file a consumer already opened does not disturb it — the
    inode survives until the last fd closes — so this cannot pull a frame out
    from under a reader mid-read. It only bounds what the tmpfs holds.
    """
    cutoff = newest_seq - keep
    if cutoff < 0:
        return
    try:
        entries = os.listdir(directory)
    except OSError:
        return
    for entry in entries:
        match = _NAME.match(entry)
        if match is None:
            continue
        if int(match.group(1)) <= cutoff:
            with contextlib.suppress(OSError):
                (directory / entry).unlink()
