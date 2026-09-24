"""Calibrate the motion gate for a PLACEMENT before arming it at an event.

Replays a recording's ~1/8-scale luma through ingest's own MotionGate at
several INGEST_MOTION_MIN_FRAC values in one decode pass, and prints what
each would have skipped. With --ledger (a run's decision ledger over the same
recording: one JSON object per frame, frame index = seq - 1) it also prints
what each would have cost: the kept-face frames skipped, and the identities
whose EVERY verdict frame would have been skipped.

    python scripts/gate_replay.py /clips/D02.mp4 --ledger ledger.jsonl \
        --min-frac 0.002 0.01 0.02 0.05 [--hw cuda]

Why this exists: on the Sharon wedding placement the default (0.002) skipped
1 frame in 15,077, because the hall behind the entrance never stops moving,
and 0.05 skipped 43% but lost a guest outright. The number that is safe is a
property of the camera's view, so it is measured per placement, not guessed.
Run from services/ingest (it imports app.motion); needs numpy and ffmpeg.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.motion import MotionGate  # noqa: E402

W, H = 480, 270


def frames(video: str, hw: str | None):
    """Yield the video's frames as 480x270 grey uint8 arrays, in order."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if hw:
        cmd += ["-hwaccel", hw]
    cmd += ["-i", video, "-an", "-vf", f"scale={W}:{H}:flags=area,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    n = W * H
    try:
        while True:
            buf = bytearray(n)
            view, got = memoryview(buf), 0
            while got < n:
                k = proc.stdout.readinto(view[got:])
                if not k:
                    return
                got += k
            yield np.frombuffer(buf, np.uint8).reshape(H, W)
    finally:
        proc.kill()
        proc.wait()


def load_ledger(path: str):
    """(kept-face frame indices, {identity: [frame indices with a verdict]})."""
    faces, guests = set(), {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            i = d["seq"] - 1
            if any(f.get("gate") == "kept" for f in (d.get("faces") or {}).get("faces", [])):
                faces.add(i)
            for v in d.get("verdicts") or []:
                if v.get("personKey"):
                    guests.setdefault(v["personKey"], []).append(i)
    return faces, guests


def main() -> None:
    """Replay once, report every threshold."""
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ledger")
    ap.add_argument("--min-frac", type=float, nargs="+", default=[0.002, 0.01, 0.02, 0.05])
    ap.add_argument("--pixel-thr", type=float, default=0.08)
    ap.add_argument("--keepalive-s", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--hw", default=None, help="e.g. cuda: decode with -hwaccel")
    a = ap.parse_args()

    gates = [MotionGate(m, a.pixel_thr, a.keepalive_s) for m in a.min_frac]
    published = [[] for _ in gates]
    n = 0
    for i, small in enumerate(frames(a.video, a.hw)):
        for g, pub in zip(gates, published, strict=True):
            pub.append(g.decide(small, (i + 1) / a.fps)[0])
        n = i + 1
    faces, guests = load_ledger(a.ledger) if a.ledger else (set(), {})
    print(f"{n} frames" + (f", {len(faces)} with kept faces, {len(guests)} identities"
                           if a.ledger else ""))
    for m, pub in zip(a.min_frac, published, strict=True):
        pub = np.array(pub)
        line = f"min_frac {m:<7} skipped {1 - pub.mean():6.1%}"
        if a.ledger:
            fidx = np.array(sorted(i for i in faces if i < n), dtype=int)
            lost = [k for k, fr in guests.items() if not pub[[i for i in fr if i < n]].any()]
            line += (f"  face frames skipped {1 - pub[fidx].mean():6.1%}"
                     f"  identities with every verdict frame skipped: {len(lost)}")
        print(line)


if __name__ == "__main__":
    main()
