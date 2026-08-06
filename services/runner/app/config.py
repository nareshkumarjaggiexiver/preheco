"""Environment-driven configuration for the runner.

Service URLs default to docker-compose DNS names; override any of them for a
bare-metal run (the e2e smoke script points them all at localhost).
"""

import os
from dataclasses import dataclass, replace

from heco_common.config import env_float, env_int


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

    # COMPOSITE QUALITY GATE (see app/gate.py).  Box width was the entire
    # gate, and it is the wrong axis twice over: recognition size is read in
    # inter-eye distance (our 56/80 px width floor is only ~24/34 px IED), and
    # size alone cannot see a face turned side-on or smeared by a walking
    # guest.  These three floors add those axes.
    #
    # EVERY ONE DEFAULTS TO 0.0 = NOT ARMED, so the shipped gate is exactly the
    # width-only gate it has always been.  That is deliberate: the gate is the
    # pipeline's only irreversible discard, we have no measured floor for any
    # of the three yet, and a guessed floor costs guests off an invoice.  Arm
    # them per camera once the eval harness can price them.
    quality_min_ied_px: float = 0.0
    quality_min_frontality: float = 0.0
    quality_min_sharpness: float = 0.0

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

    # TAP DUTY-CYCLE GUARD (see loop._maybe_tap).  tap_interval_s alone has
    # two stable states, both measured on the T440 with 4K frames: frames at
    # ~0.43 s -> a ~1-1.5 s tap round every ~5th frame -> 2.3 fps; but ANY
    # transient stall (a docker build on the box, both times) pushes one frame
    # past the ~2 s interval, after which EVERY frame triggers a full round —
    # ~2.5 s/frame, locked at 0.4 fps forever, with the stage timings
    # identical in both states (ingest 42 ms, persons 76, faces 58, embed 74,
    # match 4: the stages were innocent, the untimed round was the whole gap).
    # The guard breaks the second state by construction: a round may fire only
    # when this factor x the PREVIOUS round's measured cost has elapsed since
    # that round ENDED, so a 1 s round forces >= 3 s of counting before the
    # next — the round is bounded at ~1/(factor+1) of loop time (~25% at 3)
    # and the every-frame lock-in is impossible whatever the interval says.
    # 0 disables the guard (config semantics); deferred rounds are counted in
    # the run status as tapRoundsDeferred.
    tap_duty_factor: float = 3.0

    # A staff member seen again within this many seconds of their last sighting
    # is the SAME crossing, not a new one (see loop._pipeline_step).
    staff_cooldown_s: float = 5.0

    # TRACK HEAL (see loop._maybe_heal). The live bench (3 real people, counted
    # 5) minted p00002 on a re-entry frame at cosine 0.3084 against a 0.363
    # threshold — and the SAME physical track matched the correct identity
    # p00001 at 0.69 four seconds later. The evidence that the mint was junk
    # arrived almost immediately; the heal is the loop acting on it: when a
    # track that recently minted a new identity later matches a DIFFERENT
    # existing identity comfortably, the mint is folded back (singleton-only,
    # via /merge onlyIfSingleton).
    #
    # heal_window_s bounds how long after a mint the same track's later match
    # may fold it. Tonight's heal evidence arrived in ~4 s; 20 s covers a slow
    # walk through the frame with margin while keeping the bookkeeping small
    # and the residual risk (a tracker identity swap inside the window — see
    # loop._maybe_heal) short-lived. 0 disables healing entirely.
    heal_window_s: float = 20.0
    # heal_min_cosine is the "comfortably" in the trigger. The measured
    # impostor ceiling on this camera is 0.377 (two DIFFERENT men's templates),
    # while same-person misses measured 0.294/0.308/0.361 — the distributions
    # OVERLAP, so no value of the match threshold separates them and 0.363
    # stays where it is. The heal does not need to separate them: it only needs
    # its own evidence to sit clear of any measured impostor, so the floor is
    # 0.45 — above 0.377 with margin, and far below tonight's 0.69 heal
    # evidence. A cross-identity match below this floor proves nothing and
    # heals nothing.
    heal_min_cosine: float = 0.45
    # HEAL APPEARANCE VETO (see loop._maybe_heal and app/appearance.py). A heal
    # candidate whose remembered torso descriptor scores BELOW this histogram
    # intersection against the current frame's descriptor is refused — the
    # clash is exactly the fingerprint of the heal's documented residual risk,
    # a tracker identity swap handing the track from person A to person B
    # mid-window. 0.50 is REASONED, NOT CALIBRATED (like the M1 margins were at
    # first): identical L1-normalized histograms score 1.0, disjoint colour
    # distributions 0.0, and one event's worth of same-person pairs has not yet
    # been measured to place the floor empirically — calibrate it from real
    # footage before trusting it near the boundary. 0 disables the veto.
    # Appearance AGREEMENT never loosens anything (the 0.45 cosine floor
    # stands): the measured 0.377 impostor pair was two DIFFERENT men both in
    # light shirts, so agreeing torsos prove nothing about identity.
    heal_appearance_clash: float = 0.50

    # ENROL MODE: how many face samples (best by quality) to keep per staff
    # walk-through before writing them to the site staff store.
    enrol_best_n: int = 5

    # How long a SETTLED run stays in the process registry before it is reaped
    # (see RunManager._reap).  Long enough that an operator whose run just
    # finished can still read its final status from GET /runs/:id; short enough
    # that a night of runs cannot accumulate one dead RunLoop each — every one
    # of which pins its last frame's full base64 JPEG.
    run_retention_s: float = 600.0

    # Ingest serves the LATEST frame with a monotonically increasing `seq`;
    # a stalled seq is the end-of-source signal (there is no `ended` flag on
    # the real service).  The loop polls every source_poll_s while the seq is
    # unchanged and declares the source ended after source_stall_s of stall.
    source_poll_s: float = 0.02
    # How long the frame sequence may stand still before the run gives up.
    #
    # This was 5 s, which is shorter than a single RTSP reconnect over venue
    # Wi-Fi — so an ordinary blip mid-event ended the run. 45 s tolerates a
    # PoE bounce or a reconnect cycle while still bounding how long a genuinely
    # dead camera leaves a run open; the planner's own silence detector raises
    # its alarm at 30 s, so the operator sees the problem before the runner
    # acts on it. A stall no longer destroys the gallery either (see
    # RunLoop._next_frame), so the cost of waiting is small and the cost of
    # giving up early is a re-counted room.
    source_stall_s: float = 45.0


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
        quality_min_px=env_float("HECO_QUALITY_MIN_PX", s.quality_min_px),
        quality_canon_px=env_float("HECO_QUALITY_CANON_PX", s.quality_canon_px),
        quality_min_ied_px=env_float("HECO_QUALITY_MIN_IED_PX", s.quality_min_ied_px),
        quality_min_frontality=env_float("HECO_QUALITY_MIN_FRONTALITY", s.quality_min_frontality),
        quality_min_sharpness=env_float("HECO_QUALITY_MIN_SHARPNESS", s.quality_min_sharpness),
        run_retention_s=env_float("HECO_RUN_RETENTION_S", s.run_retention_s),
        flush_interval_s=env_float("HECO_FLUSH_INTERVAL_S", s.flush_interval_s),
        request_timeout_s=env_float("HECO_REQUEST_TIMEOUT_S", s.request_timeout_s),
        planner_timeout_s=env_float("HECO_PLANNER_TIMEOUT_S", s.planner_timeout_s),
        report_timeout_s=env_float("HECO_REPORT_TIMEOUT_S", s.report_timeout_s),
        tap_budget_s=env_float("HECO_TAP_BUDGET_S", s.tap_budget_s),
        tap_duty_factor=env_float("HECO_TAP_DUTY_FACTOR", s.tap_duty_factor),
        staff_cooldown_s=env_float("HECO_STAFF_COOLDOWN_S", s.staff_cooldown_s),
        heal_window_s=env_float("HECO_HEAL_WINDOW_S", s.heal_window_s),
        heal_min_cosine=env_float("HECO_HEAL_MIN_COSINE", s.heal_min_cosine),
        heal_appearance_clash=env_float(
            "HECO_HEAL_APPEARANCE_CLASH", s.heal_appearance_clash
        ),
        source_poll_s=env_float("HECO_SOURCE_POLL_S", s.source_poll_s),
        source_stall_s=env_float("HECO_SOURCE_STALL_S", s.source_stall_s),
        tap_interval_s=env_float("HECO_TAP_INTERVAL_S", s.tap_interval_s),
        feedback_poll_s=env_float("HECO_FEEDBACK_POLL_S", s.feedback_poll_s),
        enrol_best_n=env_int("HECO_ENROL_BEST_N", s.enrol_best_n),
    )
