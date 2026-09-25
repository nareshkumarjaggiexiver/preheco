"""Environment-driven configuration for the runner.

Service URLs default to docker-compose DNS names; override any of them for a
bare-metal run (the e2e smoke script points them all at localhost).
"""

import os
from dataclasses import dataclass, replace

from heco_common.config import env_bool, env_float, env_int


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
    # The auth service (apps/heco-auth) and this runner's application
    # credential. With these three set the runner mints its own short-lived
    # tokens and refreshes them before they lapse — no restart to rotate, and
    # nothing long-lived on the wire. This is the way in.
    auth_url: str | None = None
    app_id: str | None = None
    app_secret: str | None = None

    # LEGACY. The planner's shared secret, one string for every process and
    # every browser. Kept so an existing lab keeps working untouched while the
    # migration lands; removed at step 7 of the runbook. When the three fields
    # above are set, this is ignored.
    #: Where the minted token is kept so a RESTART during a WAN outage still
    #: has a credential. Must be on a VOLUME, or `docker compose up
    #: --force-recreate` throws it away and the cache never survives the one
    #: event it exists for. Empty disables persistence.
    token_cache_path: str | None = "/srv/state/runner-token.json"

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
    # guest.  These floors add those axes.
    #
    # EVERY ONE DEFAULTS TO OFF, so the shipped gate is exactly the width-only
    # gate it has always been.  That is deliberate: the gate is the pipeline's
    # only irreversible discard, and a guessed floor costs guests off an
    # invoice.  Arm them per camera.
    #
    # WHAT THEY SHOULD BE SET TO, when armed.  The sibling face-detection
    # pipeline priced two of these against a camera of this family and
    # recorded the measurements (docs/planning/13-…): frontal crops scored a
    # nose-offset/eye-distance of 0.05–0.43 and an eye-span/box-width of
    # 0.31–0.49, while the off-angle crops that had spawned phantom
    # identities scored 0.48–1.02 and 0.17–0.29.  Both gaps are clean.
    #
    # Their yaw ratio is our `frontality` inverted — frontality = 1 - yaw —
    # so their 0.45 yaw ceiling is a frontality floor of 0.55, which lands
    # between the two observed bands (frontal 0.57–0.95, off-angle 0.00–0.52).
    # Their eye-span floor of 0.30 transfers directly to quality_min_eye_span.
    #
    # Those are a STARTING POINT for this camera family, not a default: the
    # measurements are somebody else's camera, mount and lighting, and the
    # console's quality profile exists so an operator can arm them and watch
    # what the gate starts rejecting before committing a live count to it.
    # Detector confidence floor (face["conf"]). 0 = unarmed, and unarmed is
    # the default because the right value belongs to the DETECTOR FAMILY:
    # yunet operates at 0.8, scrfd at InsightFace's permissive 0.5. It is the
    # only gate signal not derived from the detector's own landmarks, which is
    # why it is the only one that can refuse a confident hallucination — see
    # heco_counting.gate.REASONS.
    # WHOLE-FRAME FACE DETECTION. Off by default, which keeps the per-person
    # crop path that has always run. On, the runner stops sending `within` and
    # the faces service sees the frame once.
    #
    # The crops exist because letterboxing 4K into a 640 square leaves a 67 px
    # face ~11 network px, under what stride-8 anchors resolve — so the crops
    # were buying RESOLUTION, at one inference per person. Measured on the
    # 4060: seven crops cost 59.5 ms; the same frame at a 1472x832 input costs
    # 20.5 ms and leaves that face 25.7 net px. Same model, same weights, 2.9x
    # less work, and it no longer depends on the person detector having found
    # somebody to crop.
    #
    # Turn it on WITH FACES_SCRFD_INPUT set large and frame-shaped, or it is
    # strictly worse than the crops: at the default 640 square, whole-frame on
    # a 4K source finds almost nothing.
    # REF-ONLY FRAMES. Off by default. On, the runner asks ingest to skip the
    # JPEG encode entirely (`/frame?jpeg=0`) and sends only the shared-memory
    # ref to each stage — removing the 13.9 ms encode and the ~15 ms of
    # base64+JSON for a 2.1 MB payload, on top of the three decodes the ref
    # already saved.
    #
    # It REQUIRES that ingest and all three consuming stages mount the same
    # HECO_FRAMES_DIR. That is why it is opt-in and why the runner PROVES it
    # before using it (see RunLoop._negotiate_ref_only): a stage that cannot
    # read refs and is sent no JPEG has no pixels at all, and the honest place
    # to discover that is one probe at run start, not silently for an hour.
    frames_ref_only: bool = False
    faces_whole_frame: bool = False
    quality_min_conf: float = 0.0
    quality_min_ied_px: float = 0.0
    quality_min_frontality: float = 0.0
    quality_min_sharpness: float = 0.0
    #: Eye separation as a fraction of box width — the pose axis iedPx cannot
    #: see, because IED in pixels grows as a guest approaches while this
    #: collapses in profile at any distance.  Suggested when armed: 0.30.
    quality_min_eye_span: float = 0.0
    #: Reject detections whose 5 landmarks do not describe a face (eyes above
    #: nose above mouth, sane eye span, nose near the eye span).  Boolean, not
    #: a floor: the test is topological.  This is the one gate here that says
    #: "not a face" rather than "not a good enough face" — it catches the
    #: clothing and torso detections that satisfy the detector's own
    #: confidence (measured at 70–91 % on striped shirts) but carry
    #: effectively random landmarks.
    quality_require_landmarks: bool = False

    # FACE RE-VERIFY INTERVAL — how often a track that has ALREADY resolved to
    # an identity is searched for a face again.  0.0 = off (search every track
    # every frame, the original behaviour).
    #
    # Per-person face detection is the dominant per-frame cost: YuNet runs
    # inside every person crop, and each face it finds is then embedded and
    # matched.  Ungated, a group of six standing at a gate pays that six times
    # per frame, forever, to keep re-confirming six identities the run already
    # knows — while the guest walking in behind them, who is the only one who
    # can still change the count, waits behind them for the loop.
    #
    # A track holding an identity lock (RunLoop._locks) has already been
    # resolved.  Re-searching it every frame buys re-confirmation of something
    # settled; searching it every few seconds buys the same at a fraction of
    # the cost.  Tracks WITHOUT a lock — everyone unidentified, which is
    # everyone who can still add to the count — are searched every frame,
    # always.  So the saving is taken entirely out of work that cannot change
    # the answer.
    #
    # The clock is stamped when a track is SCHEDULED for verification, not
    # when a face is successfully found on it.  Stamping on success would let
    # a person who turns away reset nothing and be re-searched every frame for
    # as long as they stay turned — precisely the case the gate exists to stop.
    face_reverify_interval_s: float = 0.0

    # FACE-SEARCH CADENCE (lever L4, HECO_FACE_CADENCE).  Off by default.  On,
    # a frame's face search is SKIPPED when there is at least one person box,
    # every person box is covered one to one (IoU >= 0.5) by a SETTLED track —
    # one holding an identity lock whose last comfortable face match is
    # younger than face_reverify_interval_s — and less than
    # face_cadence_max_gap_s of frame time (tMs: footage on a live camera,
    # processing time on a lockstep replay) has passed since the last search
    # that ran (heco_counting.face_search).  A newcomer, a stale lock, an
    # unconfirmed body, an empty frame: all searched, every frame.
    #
    # WHY.  The whole-frame SCRFD search is the largest single cost of the 4K
    # chain on the CUDA box and saturates the GPU alone (two concurrent
    # inferences measured 1.05x), and most wedding frames are the same
    # identified guests standing where they stood — searching again only
    # re-confirms answers the run holds.  Skipped frames still run the
    # tracker and track-presence co-presence; they are counted in
    # faceDetectSkippedSettled beside `unique`.
    #
    # DEPENDS ON face_reverify_interval_s: at 0 nothing is ever "recently
    # verified", so the cadence skips nothing (said at run start, and visible
    # in GET /health knobs and the run config).  2-3 s is the intended
    # pairing with the 1 s max gap.
    face_cadence: bool = False
    face_cadence_max_gap_s: float = 1.0

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

    # ASYNC REPORTING (app.reporting).  When on — the default — tap rounds,
    # planner flushes and forensic uploads run on their own thread and the
    # frame loop pays only a handover.  Measured on a live 4K camera, the work
    # this moves cost 43-96 ms per frame for tap rounds plus 15-22 ms for
    # flushes; the console keeps its ~2 s cadence, because the fix is to make
    # observability cheap rather than rare.  Set HECO_ASYNC_REPORTING=0 to put
    # it back on the loop: the sync path is kept intact so the two can be
    # A/B'd on one box, and so a bad night at a real gate is one env var from
    # the shape that has been counting all along.
    # GOLDEN DECISION CAPTURE (loop._write_golden).  A path here makes the run
    # ALSO append every frame's decision record to a local JSONL file, so a
    # refactor can be proved to have changed nothing: capture, change, replay
    # the same clip, diff.  `{runId}` in the path is substituted.  Off by
    # default — it is an engineer's instrument, not a production behaviour,
    # and a run that writes a file nobody asked for is a surprise on a box.
    golden_path: str | None = None
    #: Capture per-face EMBEDDINGS into the golden file (doc 15 M3): the
    #: calibration sweep's raw material. Opt-in and golden-only on purpose —
    #: embeddings are biometric data, and the run LEDGER (posted to the
    #: planner) deliberately never carries them; this flag writes them only
    #: into the engineer's local capture file, which already holds the run's
    #: whole reasoning and lives under the same handling.
    golden_embeddings: bool = False
    #: Fetch frame N+1 while frame N is still being processed — ONE frame in
    #: flight, never more (docs/planning memory path-to-15fps: fixed latency,
    #: not the unbounded-buffer divergence). Hides ingest transport plus the
    #: source's frameWaitMs behind the loop's own work — the measured stall
    #: was up to ~75 ms/frame of pure idleness on a live 4K camera. Lockstep
    #: file replays are frame-for-frame identical either way (the poller
    #: returns the NEXT seq whenever it is asked), so golden diffs hold.
    frame_prefetch: bool = True

    # STAGE OVERLAP (lever L3, HECO_PIPELINE_OVERLAP).  Off by default, and
    # off is the loop exactly as it has always run (pinned call for call:
    # tests/test_call_sequence_pinned.py).
    #
    # THE MEASUREMENT.  One 4K camera through the CUDA box, whole-frame
    # SCRFD-10G at 1472x832: ~200 ms a frame with every stage SERIAL in this
    # loop, while the camera delivers a frame every 67 ms.  Persons and faces
    # are the stateless half of that chain — each is a pure function of the
    # frame — and everything else (the tracker, the gallery, the locks,
    # co-presence, the ledger, the taps) is state that must be touched in
    # frame order.  So ONE worker thread fetches frame k+1 and runs its
    # detection (persons, then the face search when it needs no tracks: the
    # whole frame) while this thread runs frame k's track -> gate -> embed ->
    # match -> verdict pass.  At most one frame ahead; every piece of state
    # stays on this thread; a stage failure in the worker is carried back and
    # raised here, at the very point the inline call would have raised it.
    # A crop search still waits for its frame's tracks, so on the crop path
    # only persons moves to the worker.
    pipeline_overlap: bool = False
    # PARALLEL DETECT (lever L3, HECO_PARALLEL_DETECT).  Off by default.  With
    # the whole-frame face search, persons and faces /detect for the SAME
    # frame are issued side by side instead of one after the other — neither
    # needs the other's answer.  Measured persons latency through its HTTP
    # service is 31-43 ms against 8.6 ms of model time, so most of what this
    # hides is transport, not GPU (two concurrent SCRFD inferences measured
    # 1.05x: the GPU is already saturated by one).  Works with or without the
    # overlap; on the crop path it does nothing, because a crop search needs
    # this frame's tracks first.
    parallel_detect: bool = False

    async_reporting: bool = True
    # How long the reporter sleeps when idle.  Short enough that a mint's
    # keyframe is uploaded promptly, long enough that an empty scene does not
    # spin a core: at 50 ms it wakes ~20x a second, and each wake that finds
    # nothing due is a couple of comparisons.
    reporter_poll_s: float = 0.05

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

    # FACE CARDS (see loop._maybe_face_card).  How much wider a face must be,
    # as a multiple of the best card already taken, before it replaces it.
    #
    # A card is the guest register's picture of one guest — their face cut out
    # of the frame — and it exists because whole frames could not answer "which
    # of these three people is p00003?".  The FIRST look at somebody is often
    # the worst: run 0f5c6d minted p00003 from a face of 29 px eye-distance
    # with a phone across it while matching other guests all evening at 44-60,
    # so a card fixed at mint would hand the register exactly the picture that
    # made the identity ambiguous.
    #
    # 1.25 is a compromise between that and cost.  Every replacement must beat
    # the last by this factor, so the number of cards per guest is logarithmic
    # in how much their face grows as they approach the camera — a guest going
    # from 40 px to 200 px costs five uploads, not five hundred.  Lower means
    # sharper cards and more uploads inside the tap round's budget; 1.0 would
    # re-upload on every marginally wider face and is the setting to avoid.
    face_card_improve: float = 1.25

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
    # which is VISIBLE (though not operator-fixable: /merge refuses a
    # cannot_link pair for a human too, and only a fresh run clears the
    # row); a wrong fold UNDER-counts SILENTLY and nobody ever sees it.  Fixed
    # mirrors and screens are handled by exclusion zones; a hand-held phone
    # is not, and that is the residual.
    # 0 turns the mechanism off completely: nothing recorded, nothing sent.
    copresence_split: int = 1

    # TRACK-PRESENCE CO-PRESENCE (see loop._track_presence).  1 = on, 0 = off.
    #
    # Face co-presence above only fires between identities whose FACES were
    # matched in one frame, so a guest standing in the frame with their back
    # to the camera is never asserted co-present with anyone.  THE MEASUREMENT
    # (run f0bfc5, 2026-09-23, Punjab wedding-hall overview camera, 74 guests,
    # a 500-pair review queue): pair #5 was p00048 (girl, yellow top) against
    # p00052 (woman, dark green dress) at face 0.338 — and p00048 was IN THE
    # SAME FRAME as p00052, back-turned.  The ledger held the proof and the
    # loop could not use it.
    #
    # A track that has already resolved to an identity at >=
    # track_lock_min_cosine (the same 0.45 floor the lock and the heal use,
    # above the 0.377 measured impostor ceiling) stays that person until the
    # tracker drops the id or another track CONTESTS the box (IoU >= 0.4 —
    # the geometry of a tracker identity swap).  While bound, the track stands
    # in for the face: over a person box that carries no face verdict this
    # frame, it puts its identity in the frame, and every distinct pair on
    # different bodies goes through the same cannot_link door
    # (loop._assert_co_presence), counted as trackPresenceSplits.  Replayed
    # over f0bfc5's ledger (scratchpad replay.py): 48 pairs proven distinct
    # against 25 from faces alone, pair #5 retired at seq 9283, and 8 of the
    # 500 queued pairs gone — with no threshold moved.  0 turns it off:
    # nothing bound, nothing asserted, and the run behaves exactly as before.
    presence_split: int = 1

    # POST-EMBED FEATURE-NORM FLOOR (see loop._gate_feat_norm).  0 = off, the
    # default.  ArcFace's raw feature norm — the L2 length of the vector
    # BEFORE unit-normalisation — tracks recognisability: a lit, frontal face
    # embeds long; an occluded, blurred or side-on one embeds short (the
    # observation MagFace formalised).  It is the one signal that sees an
    # OCCLUDER that hides much of a face: p00047 on run f0bfc5, a girl half
    # behind a pillar, read 17.7 — under every clear face.  (It does NOT see a
    # dark bar across a face: p00002, behind a railing, read 23.7, a typical
    # clear face — quality_min_balance below is that gate.)  Read after the embedder, so it
    # costs nothing extra; dropped faces are stamped gateReason "featnorm" and
    # counted in gatedByFeatNorm.  A reply without norms (an older embed
    # service) gates nothing and is counted gatedUnmeasured: an armed floor
    # never rejects a face it could not measure.  NOT CALIBRATED — read the
    # norm distribution off a run's ledger (verdicts[].featNorm) before
    # arming, because the gate is the pipeline's only irreversible discard.
    quality_min_feat_norm: float = 0.0
    # HALF-BALANCE FLOOR — the second post-embedder gate: how evenly the two
    # halves of the aligned face were seen (embed align.half_balance, the
    # darker half's brightness over the brighter's).  A face with a dark
    # railing, an arm or a shoulder across half of it reads low while every
    # other signal calls it clean: p00002 on run f0bfc5 (and again on the
    # 2026-09-25 re-runs) read 0.27 against 0.42 for the lowest genuine face
    # of 180 sightings — a Sikh guest head-down with his turban on one side —
    # and 0.51 at the 5th percentile.  Dropped faces are stamped gateReason
    # "balance" and counted gatedByBalance; a face the embedder could not
    # measure (off the frame edge, an older embed service) is kept and counted
    # gatedUnmeasured.  0 = off.
    quality_min_balance: float = 0.0

    # APPEARANCE WHITE BALANCE (heco_counting.appearance.frame_gains).  Off
    # by default.  On: each decoded frame's illuminant is estimated once
    # (shades-of-grey, p = 6, on every 8th pixel, ~2.4 ms at 4K) and the
    # torso descriptor reads its band under neutral light.  Proven on
    # synthetic casts (four garments under red / blue / amber light agree
    # with their neutral read at 0.83-0.99 balanced; raw, 0.00-0.995 and
    # three of the twelve not measurable at all).  Measured on
    # run f0bfc5 it changes nothing: the camera's own AWB already holds the
    # warm hall at gains 1.006 / 0.985 / 1.009 (B/G/R median, every frame
    # within 2%), and within-identity (median 0.90 -> 0.90) and cross-pair
    # separation did not move — so it stays off until a venue with coloured
    # light measures otherwise.
    appearance_wb: bool = False

    # HEAD-COVERING READS (app.headwear, HECO_HEADWEAR).  Off by default.  On,
    # every /match that MINTED or ENROLLED a template and logged a body row
    # has its head read by the embed service's SigLIP reader (POST
    # /headwear) on a background worker, and the 8 logits are written onto
    # that body row (match POST /body-sightings/headwear) for the review
    # queue.  The loop only cuts the context crop and hands it over; a full
    # queue (headwear_queue crops) drops the read and counts it.  Run 8b8b87:
    # 734 such reads for 74 identities, ~0.75/s in bursts.
    headwear: bool = False
    headwear_queue: int = 32

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
        auth_url=os.environ.get("HECO_AUTH_URL") or s.auth_url,
        app_id=os.environ.get("HECO_APP_ID") or s.app_id,
        app_secret=os.environ.get("HECO_APP_SECRET") or s.app_secret,
        token_cache_path=os.environ.get("HECO_TOKEN_CACHE", s.token_cache_path),
        quality_min_px=env_float("HECO_QUALITY_MIN_PX", s.quality_min_px),
        quality_canon_px=env_float("HECO_QUALITY_CANON_PX", s.quality_canon_px),
        frames_ref_only=env_bool("HECO_FRAMES_REF_ONLY", s.frames_ref_only),
        faces_whole_frame=env_bool("HECO_FACES_WHOLE_FRAME", s.faces_whole_frame),
        quality_min_conf=env_float("HECO_QUALITY_MIN_CONF", s.quality_min_conf),
        quality_min_ied_px=env_float("HECO_QUALITY_MIN_IED_PX", s.quality_min_ied_px),
        quality_min_frontality=env_float("HECO_QUALITY_MIN_FRONTALITY", s.quality_min_frontality),
        quality_min_sharpness=env_float("HECO_QUALITY_MIN_SHARPNESS", s.quality_min_sharpness),
        quality_min_eye_span=env_float("HECO_QUALITY_MIN_EYE_SPAN", s.quality_min_eye_span),
        quality_require_landmarks=env_bool(
            "HECO_QUALITY_REQUIRE_LANDMARKS", s.quality_require_landmarks
        ),
        face_reverify_interval_s=env_float(
            "HECO_FACE_REVERIFY_INTERVAL_S", s.face_reverify_interval_s
        ),
        face_cadence=env_bool("HECO_FACE_CADENCE", s.face_cadence),
        face_cadence_max_gap_s=env_float(
            "HECO_FACE_CADENCE_MAX_GAP_S", s.face_cadence_max_gap_s
        ),
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
        face_card_improve=env_float("HECO_FACE_CARD_IMPROVE", s.face_card_improve),
        copresence_split=env_int("HECO_COPRESENCE_SPLIT", s.copresence_split),
        presence_split=env_int("HECO_PRESENCE_SPLIT", s.presence_split),
        quality_min_feat_norm=env_float(
            "HECO_QUALITY_MIN_FEAT_NORM", s.quality_min_feat_norm
        ),
        quality_min_balance=env_float(
            "HECO_QUALITY_MIN_BALANCE", s.quality_min_balance
        ),
        source_poll_s=env_float("HECO_SOURCE_POLL_S", s.source_poll_s),
        source_stall_s=env_float("HECO_SOURCE_STALL_S", s.source_stall_s),
        tap_interval_s=env_float("HECO_TAP_INTERVAL_S", s.tap_interval_s),
        golden_path=os.environ.get("HECO_GOLDEN_PATH") or s.golden_path,
        golden_embeddings=os.environ.get("HECO_GOLDEN_EMBEDDINGS", "") == "1",
        frame_prefetch=os.environ.get("HECO_FRAME_PREFETCH", "1") != "0",
        pipeline_overlap=env_bool("HECO_PIPELINE_OVERLAP", s.pipeline_overlap),
        parallel_detect=env_bool("HECO_PARALLEL_DETECT", s.parallel_detect),
        async_reporting=env_bool("HECO_ASYNC_REPORTING", s.async_reporting),
        reporter_poll_s=env_float("HECO_REPORTER_POLL_S", s.reporter_poll_s),
        feedback_poll_s=env_float("HECO_FEEDBACK_POLL_S", s.feedback_poll_s),
        enrol_best_n=env_int("HECO_ENROL_BEST_N", s.enrol_best_n),
        appearance_wb=env_bool("HECO_APPEARANCE_WB", s.appearance_wb),
        headwear=env_bool("HECO_HEADWEAR", s.headwear),
        headwear_queue=max(1, env_int("HECO_HEADWEAR_QUEUE", s.headwear_queue)),
    )


def knobs(s: Settings) -> dict:
    """The throughput levers (and the appearance switch) as this process
    resolved them, for GET /health.

    Keyed by the variable an operator sets, so "is it on?" is answered
    against the very name in the .env.  The failure this exists for is a
    knob that LOOKED set and changed nothing: three knobs in this repo once
    had no compose passthrough, and nothing anywhere said so.
    """
    return {
        "HECO_PIPELINE_OVERLAP": s.pipeline_overlap,
        "HECO_PARALLEL_DETECT": s.parallel_detect,
        "HECO_FACE_CADENCE": s.face_cadence,
        "HECO_FACE_CADENCE_MAX_GAP_S": s.face_cadence_max_gap_s,
        # The cadence's settledness window: shown beside it because at 0 the
        # cadence skips nothing (a per-run quality profile may override it).
        "HECO_FACE_REVERIFY_INTERVAL_S": s.face_reverify_interval_s,
        # Not a throughput lever: white balance for the review's colour
        # evidence (torso, head, beard), off by default.
        "HECO_APPEARANCE_WB": s.appearance_wb,
        # The head-covering reads for the review (background worker), off by default.
        "HECO_HEADWEAR": s.headwear,
        "HECO_HEADWEAR_QUEUE": s.headwear_queue,
    }
