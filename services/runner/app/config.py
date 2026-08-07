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
    # TRACK-SCOPED IDENTITY LOCK (see loop._note_lock / loop._maybe_lock_fold).
    # The heal CURES a bad mint after the fact; the lock PREVENTS it. Once a
    # verdict on a track has matched an existing identity at >= this cosine,
    # that track is bound to that identity for the heal window, and a LATER
    # mint on the SAME track is folded straight back into it — no need to wait
    # for a second comfortable match to arrive, which on bench 6e1a5d is
    # exactly what never came: p00005 (face 0.212 vs p00001) and p00006 (0.228)
    # survived to the end of the run because the track that had already
    # resolved to p00001 never matched again after minting them.
    #
    # 0.45 is the SAME floor the heal uses and for the same reason: the
    # measured impostor ceiling on this camera is 0.377 (two genuinely
    # different men), so the evidence that binds a track to an identity must
    # sit clear of any impostor pair we have seen. 0 disables the lock
    # entirely (config semantics), and it is disabled implicitly whenever
    # healing is off, because the lock expires on the heal window.
    #
    # WHY IT HAS AN OFF SWITCH AT ALL: a wrong split over-counts and somebody
    # argues about the invoice; a wrong MERGE under-counts silently and nobody
    # ever sees it. The lock makes track identity more authoritative, which
    # amplifies the silent failure if the tracker swaps people — so it is
    # gated, counted (lockedTrackFolds) and guarded by the same clothing bands
    # as the heal.
    track_lock_min_cosine: float = 0.45
    # HEAL APPEARANCE BANDS (see loop._appearance_refuses and app/appearance.py).
    # A fold candidate's remembered torso descriptor is compared with the
    # current frame's by histogram intersection, and the reading falls in one
    # of three bands:
    #
    #   < heal_appearance_clash            CLEAR CLASH — the fold is refused
    #                                      (healVetoedByAppearance), because
    #                                      that is the fingerprint of a tracker
    #                                      identity swap handing the track from
    #                                      person A to person B mid-window;
    #   clash .. heal_appearance_unsure    UNCERTAIN — the fold PROCEEDS and is
    #                                      counted (healUncertainAppearance) so
    #                                      an operator can see how often the
    #                                      system acted on weak corroboration;
    #   >= heal_appearance_unsure          corroborated — proceeds silently.
    #
    # WHY THE CLASH FLOOR DROPPED 0.50 -> 0.35. On bench 6e1a5d a fold was
    # VETOED at intersection 0.4991 against the 0.50 floor — nine
    # ten-thousandths, on a fold that was probably correct (track 2 had linked
    # p00001 and p00005, and the pipeline threw that evidence away). A cliff at
    # the exact centre of the distribution decides nothing well: the v2
    # descriptor read two SURVIVING splits of the same man at 0.797 and 0.875,
    # while a genuinely different pair of men reached 0.747 — mid-range
    # readings simply do not separate people, so only a GENUINE disagreement
    # may block, and everything between is recorded rather than acted on.
    # 0 disables the clash veto (nothing is ever refused on clothing); absent
    # descriptors on either side veto nothing and count nothing (absent is not
    # zero). Both numbers are REASONED, NOT CALIBRATED — one event's worth of
    # same-person pairs still has not been measured — and clothing agreement
    # NEVER loosens the cosine floors, because that 0.747 impostor pair proves
    # agreeing torsos say nothing about identity.
    heal_appearance_clash: float = 0.35
    heal_appearance_unsure: float = 0.55

    # SAME-FRAME SAME-KEY GUARD (see loop._split_same_key).  The clothing floor
    # under which two faces in ONE frame, sitting in two DIFFERENT person
    # boxes, that both resolved to the SAME identity are ruled two people and
    # the weaker one is re-resolved with that identity off the table.  0
    # disables the guard entirely.
    #
    # WHY IT EXISTS (run fa8fc3, 2026-08-07, ground truth TWO men, counted
    # ONE).  Frame kf-577 holds them side by side, both boxed, both labelled
    # p00001, matched at face 0.6874 and 0.6733 — not borderline, no near-miss
    # band, nothing for an operator to notice.  By the end p00001 held five
    # templates whose internal cosines ran 0.2223..0.5232, impostor-level
    # against each other: one identity had grown to contain both men.  Every
    # existing mechanism was structurally blind to it.  Co-presence asserts
    # that two KEYS in a frame are different people and there was only ever
    # one key.  The enrolment veto refuses WRITES, never verdicts, and it is
    # charitable by construction (BEST intersection over the identity's stored
    # views), so once one of the second man's sightings was in, every later
    # one agreed with it and nothing clashed again.  The near-miss banners
    # fire on MINTS and no second mint ever happened.
    #
    # THE EVIDENCE THE GUARD USES IS NOT THE FACE.  Two bodies at different
    # positions in one frame cannot be one guest, whatever the cosine says —
    # the same certainty co-presence already runs on, applied one step
    # earlier, to the case where the matcher collapsed the pair before a
    # second key existed to constrain.
    #
    # CLOTHING IS REQUIRED AS CORROBORATION, and that is the whole reason for
    # a floor rather than a bare box test.  A person and their reflection in a
    # hall mirror, or a face on a hand-held phone screen, ALSO occupy two
    # person boxes — and would be split into two guests by box geometry alone,
    # turning this fix into an over-count.  Those pairs wear the SAME clothes
    # and read high; fa8fc3's two men read 0.130..0.294 across the divide,
    # against 0.624..0.817 within each man.  0.35 is the same CLEAR-CLASH floor
    # heal_appearance_clash uses, for the same reason: mid-range clothing
    # readings do not separate people, so only a genuine disagreement may act.
    # Descriptor absent on either side = no split (absent is not zero); this
    # errs towards the under-count, which is why the guard is a floor and not
    # the whole answer.
    same_frame_clash: float = 0.35

    # CO-PRESENCE SPLITS (see loop._assert_co_presence).  1 = on, 0 = off.
    #
    # THE MEASUREMENT (run 05b3b7, 2026-08-06, ground truth THREE people,
    # counted THREE — the COUNT was right and the NOISE was wrong).  Two
    # "likely duplicate" banners were raised between people who are
    # demonstrably different: p00002 carried "Minted 0.316 from p00001 —
    # clothing agreement 0.94" and p00007 (the operator) carried "Minted 0.360
    # from p00002 — clothing agreement 0.57".  p00001 and p00002 walked in
    # through the main door together; p00007 and p00002 stood in the alley
    # together.  Both banners sat INSIDE the ordinary face near-miss band
    # (0.316 and 0.360 against the 0.363 threshold) and clothing agreement —
    # 0.94 on the first pair — did not save them.
    #
    # NO THRESHOLD FIXES THIS.  The impostor and genuine face distributions
    # overlap on this camera: a measured impostor pair reached 0.377 while
    # same-person misses measured 0.294 / 0.308 / 0.361.  Moving 0.363
    # anywhere trades one error for the other.
    #
    # CO-PRESENCE IS INDEPENDENT AND CERTAIN.  The tap ledger for that run
    # holds a single round in which p00002 and p00007 were matched in the SAME
    # FRAME.  Two faces at different positions in one frame are two different
    # people — about as certain as machine evidence gets — and the pipeline
    # had that fact and did nothing with it.  With this on, every unordered
    # pair of distinct non-staff identities seen in one frame is asserted to
    # the gallery as a cannot_link via POST /split, which both silences the
    # near-miss banner for the pair AND makes /merge refuse it, so no heal or
    # track-lock fold can ever erase one of two co-present guests.
    #
    # THE KNOWN COST, stated rather than hidden: a person holding a PHONE
    # showing their own face (or a mirror, or a printed photo) puts ONE real
    # person's face in the frame twice, and co-presence will assert they are
    # two different people — blocking a fold that was correct.  That is the
    # deliberate side of this project's asymmetry: a blocked fold OVER-counts,
    # which is VISIBLE and an operator can merge; a wrong fold UNDER-counts
    # SILENTLY and nobody ever sees it.  Fixed mirrors and screens are handled
    # by exclusion zones; a hand-held phone is not, and that is the residual.
    # 0 turns the mechanism off completely: nothing recorded, nothing sent.
    copresence_split: int = 1

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
        track_lock_min_cosine=env_float(
            "HECO_TRACK_LOCK_MIN_COSINE", s.track_lock_min_cosine
        ),
        heal_appearance_clash=env_float(
            "HECO_HEAL_APPEARANCE_CLASH", s.heal_appearance_clash
        ),
        heal_appearance_unsure=env_float(
            "HECO_HEAL_APPEARANCE_UNSURE", s.heal_appearance_unsure
        ),
        same_frame_clash=env_float("HECO_SAME_FRAME_CLASH", s.same_frame_clash),
        copresence_split=env_int("HECO_COPRESENCE_SPLIT", s.copresence_split),
        source_poll_s=env_float("HECO_SOURCE_POLL_S", s.source_poll_s),
        source_stall_s=env_float("HECO_SOURCE_STALL_S", s.source_stall_s),
        tap_interval_s=env_float("HECO_TAP_INTERVAL_S", s.tap_interval_s),
        feedback_poll_s=env_float("HECO_FEEDBACK_POLL_S", s.feedback_poll_s),
        enrol_best_n=env_int("HECO_ENROL_BEST_N", s.enrol_best_n),
    )
