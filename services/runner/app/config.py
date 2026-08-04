"""Environment-driven configuration for the runner.

Service URLs default to docker-compose DNS names; override any of them for a
bare-metal run (the e2e smoke script points them all at localhost).
"""

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    """All knobs the run loop needs, resolved once at startup (or per test)."""

    ingest_url: str = "http://ingest:7101"
    persons_url: str = "http://persons:7102"
    tracker_url: str = "http://tracker:7103"
    faces_url: str = "http://faces:7104"
    embed_url: str = "http://embed:7105"
    match_url: str = "http://match:7106"
    # The site-planner app on the host machine (see docker-compose extra_hosts).
    planner_url: str = "http://host.docker.internal:8787"
    # The planner's shared secret. Required whenever the planner is exposed
    # beyond loopback — which it refuses to be without one — so any runner
    # reaching the host across the docker bridge needs this set.
    planner_token: str | None = None

    # POC quality gate (CONTRACTS.md "POC geometry"): 2.8 mm camera at 2.0 m,
    # subjects at 2-3 m, expected face widths ~64-85 px.  Faces narrower than
    # quality_min_px never reach the embedder; widths in [min, canon) pass but
    # are flagged sub-canon so every report can state the share of low-pixel
    # evidence behind the unique count.
    quality_min_px: float = 56.0
    quality_canon_px: float = 80.0

    flush_interval_s: float = 2.0  # planner stats/samples cadence
    sample_batch_max: int = 200  # planner ingest contract: batch <= 200 rows

    # THREE timeouts, because the three kinds of call have three different
    # costs of being slow.  A stage call is the product (a 1080p embed can
    # legitimately take seconds), so it keeps the generous budget.  Planner
    # calls are reporting: the planner is an Express app on the operator's
    # laptop, so anything past a couple of seconds is a wedged planner, and
    # waiting on it stalls the frame loop — with ingest's drop-not-queue slot
    # every guest crossing during the stall is simply never seen.  "Best
    # effort" used to mean "errors are swallowed"; it now also means "latency
    # is bounded", which is the half that actually protects the count.
    request_timeout_s: float = 30.0  # stage services (persons/faces/embed/...)
    planner_timeout_s: float = 5.0  # retrying planner calls (run, stats, samples)
    report_timeout_s: float = 2.0  # best-effort planner calls (taps/frames/feedback)

    # Debug taps (annotated frames + structured payloads) cadence, and the
    # operator-feedback poll cadence (CONTRACTS.md v1: ~2 s / ~3 s). Both are
    # best-effort — a planner hiccup never blocks or crashes the loop.
    tap_interval_s: float = 2.0
    feedback_poll_s: float = 3.0

    # Whole-budget ceiling for ONE tap round (up to 5 payloads + 5 JPEGs).
    # Timeouts bound each call; this bounds the round, so a planner that
    # answers slowly-but-successfully cannot cost the loop 10 x report_timeout.
    tap_budget_s: float = 3.0

    # A staff member seen again within this many seconds of their last sighting
    # is the SAME crossing, not a new one (see loop._pipeline_step).
    staff_cooldown_s: float = 5.0

    # ENROL MODE: how many face samples (best by quality) to keep per staff
    # walk-through before writing them to the site staff store.
    enrol_best_n: int = 5

    # Ingest serves the LATEST frame with a monotonically increasing `seq`;
    # a stalled seq is the end-of-source signal (there is no `ended` flag on
    # the real service).  The loop polls every source_poll_s while the seq is
    # unchanged and declares the source ended after source_stall_s of stall.
    source_poll_s: float = 0.02
    source_stall_s: float = 5.0


def from_env() -> Settings:
    """Build Settings from the environment, falling back to compose defaults."""
    s = Settings()
    return replace(
        s,
        ingest_url=os.environ.get("HECO_INGEST_URL", s.ingest_url),
        persons_url=os.environ.get("HECO_PERSONS_URL", s.persons_url),
        tracker_url=os.environ.get("HECO_TRACKER_URL", s.tracker_url),
        faces_url=os.environ.get("HECO_FACES_URL", s.faces_url),
        embed_url=os.environ.get("HECO_EMBED_URL", s.embed_url),
        match_url=os.environ.get("HECO_MATCH_URL", s.match_url),
        planner_url=os.environ.get("PLANNER_URL", s.planner_url),
        planner_token=os.environ.get("HECO_TOKEN") or s.planner_token,
        quality_min_px=float(os.environ.get("HECO_QUALITY_MIN_PX", s.quality_min_px)),
        quality_canon_px=float(os.environ.get("HECO_QUALITY_CANON_PX", s.quality_canon_px)),
        flush_interval_s=float(os.environ.get("HECO_FLUSH_INTERVAL_S", s.flush_interval_s)),
        request_timeout_s=float(os.environ.get("HECO_REQUEST_TIMEOUT_S", s.request_timeout_s)),
        planner_timeout_s=float(os.environ.get("HECO_PLANNER_TIMEOUT_S", s.planner_timeout_s)),
        report_timeout_s=float(os.environ.get("HECO_REPORT_TIMEOUT_S", s.report_timeout_s)),
        tap_budget_s=float(os.environ.get("HECO_TAP_BUDGET_S", s.tap_budget_s)),
        staff_cooldown_s=float(os.environ.get("HECO_STAFF_COOLDOWN_S", s.staff_cooldown_s)),
        source_poll_s=float(os.environ.get("HECO_SOURCE_POLL_S", s.source_poll_s)),
        source_stall_s=float(os.environ.get("HECO_SOURCE_STALL_S", s.source_stall_s)),
        tap_interval_s=float(os.environ.get("HECO_TAP_INTERVAL_S", s.tap_interval_s)),
        feedback_poll_s=float(os.environ.get("HECO_FEEDBACK_POLL_S", s.feedback_poll_s)),
        enrol_best_n=int(os.environ.get("HECO_ENROL_BEST_N", s.enrol_best_n)),
    )
