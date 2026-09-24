"""ingest's levers, read from the environment in ONE place.

Every lever here defaults OFF, and OFF runs today's capture loop untouched —
the tests pin that. A knob only counts as a knob when it is read here, passed
through docker-compose.yml as ``${INGEST_X-}`` (an empty value means "not
chosen", see heco_common.config), and visible on GET /health under ``knobs``.

L1 — motion gate + bounded live buffer:

* ``INGEST_MOTION_GATE`` (0|1, default 0) — publish a frame only when
  something in it moved, or when ``INGEST_MOTION_KEEPALIVE_S`` has passed
  since the last published frame. See app.motion for the measure.
* ``INGEST_MOTION_MIN_FRAC`` (default 0.002) — fraction of the ~1/8-scale
  luma pixels that must change for a frame to count as motion.
* ``INGEST_MOTION_PIXEL_THR`` (default 0.08) — how far one small pixel must
  move, in units of that frame's own luma standard deviation.
* ``INGEST_MOTION_KEEPALIVE_S`` (default 1.0) — the longest a quiet scene goes
  without a published frame, so trackers and stall detection stay alive.
* ``INGEST_BUFFER_S`` (default 0 = the newest-frame slot) — > 0 holds up to
  that many seconds of published frames in a FIFO instead of dropping them.
* ``INGEST_BUFFER_MB`` (default 2048) — hard memory cap on that FIFO.

L6 — hardware decode:

* ``INGEST_DECODER`` = ``cpu`` (default: today's cv2.VideoCapture) |
  ``nvdec`` | ``vaapi`` | ``ffmpeg``. The last three decode in an ffmpeg
  subprocess to NV12 (app.ffmpeg_source); ``ffmpeg`` is SOFTWARE decode down
  the same pipe — the cheap gate for a box without a usable GPU, and the path
  the test-suite can run anywhere an ffmpeg binary exists. A decoder that
  cannot start falls back to ``cpu`` loudly (stderr and /health ``device``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from heco_common.config import env_bool, env_float, env_int, env_str

#: Every env var this module reads. The test-suite checks each one is passed
#: through docker-compose.yml, because a knob with no passthrough looks set
#: on the host and changes nothing in the container.
ENV_KNOBS = (
    "INGEST_MOTION_GATE",
    "INGEST_MOTION_MIN_FRAC",
    "INGEST_MOTION_PIXEL_THR",
    "INGEST_MOTION_KEEPALIVE_S",
    "INGEST_BUFFER_S",
    "INGEST_BUFFER_MB",
    "INGEST_DECODER",
)

#: The accepted INGEST_DECODER values; ``cpu`` is today's path.
DECODERS = ("cpu", "nvdec", "vaapi", "ffmpeg")


@dataclass(frozen=True)
class Levers:
    """The resolved L1/L6 settings for one capture worker.

    The defaults ARE today's behaviour, so ``Levers()`` is what a worker gets
    when nobody asked for anything.
    """

    motion_gate: bool = False
    motion_min_frac: float = 0.002
    motion_pixel_thr: float = 0.08
    motion_keepalive_s: float = 1.0
    buffer_s: float = 0.0
    buffer_mb: int = 2048
    decoder: str = "cpu"

    @property
    def armed(self) -> bool:
        """True when any lever leaves today's capture loop."""
        return self.motion_gate or self.buffer_s > 0 or self.decoder != "cpu"

    def knobs(self) -> dict:
        """The /health ``knobs`` block — camelCase, like every wire field."""
        d = asdict(self)
        return {
            "motionGate": d["motion_gate"],
            "motionMinFrac": d["motion_min_frac"],
            "motionPixelThr": d["motion_pixel_thr"],
            "motionKeepaliveS": d["motion_keepalive_s"],
            "bufferS": d["buffer_s"],
            "bufferMb": d["buffer_mb"],
            "decoder": d["decoder"],
        }


def levers_from_env(
    motion_gate: bool | None = None, buffer_s: float | None = None
) -> Levers:
    """Read the levers; ``/open`` overrides win over the env when not None.

    Raises ValueError naming the variable for anything malformed — a typo in
    a knob must refuse loudly, not quietly run with the default.
    """
    gate = env_bool("INGEST_MOTION_GATE", False) if motion_gate is None else bool(motion_gate)
    buf = env_float("INGEST_BUFFER_S", 0.0) if buffer_s is None else float(buffer_s)
    min_frac = env_float("INGEST_MOTION_MIN_FRAC", 0.002)
    pixel_thr = env_float("INGEST_MOTION_PIXEL_THR", 0.08)
    keepalive = env_float("INGEST_MOTION_KEEPALIVE_S", 1.0)
    cap_mb = env_int("INGEST_BUFFER_MB", 2048)
    # "" is unset (compose renders an unchosen knob as the empty string).
    decoder = (env_str("INGEST_DECODER", "") or "cpu").strip().lower()
    if not 0.0 <= min_frac <= 1.0:
        raise ValueError(f"INGEST_MOTION_MIN_FRAC must be in [0, 1], got {min_frac}")
    if pixel_thr <= 0:
        raise ValueError(f"INGEST_MOTION_PIXEL_THR must be > 0, got {pixel_thr}")
    if keepalive <= 0:
        raise ValueError(f"INGEST_MOTION_KEEPALIVE_S must be > 0, got {keepalive}")
    if buf < 0:
        raise ValueError(f"INGEST_BUFFER_S (or bufferS) must be >= 0, got {buf}")
    if cap_mb < 1:
        raise ValueError(f"INGEST_BUFFER_MB must be >= 1, got {cap_mb}")
    if decoder not in DECODERS:
        raise ValueError(f"INGEST_DECODER must be one of {'|'.join(DECODERS)}, got {decoder!r}")
    return Levers(
        motion_gate=gate,
        motion_min_frac=min_frac,
        motion_pixel_thr=pixel_thr,
        motion_keepalive_s=keepalive,
        buffer_s=buf,
        buffer_mb=cap_mb,
        decoder=decoder,
    )
