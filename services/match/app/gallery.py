"""The per-run guest gallery: one SQLite file per run, matched brute-force.

Thin policy layer over :class:`app.store.VectorStore`.  The store owns the
mechanics (cosine scan, float32 BLOB rows, monotonic keys, merge/split/remove);
this module owns the *guest-counting policy*: mint ``p#####`` keys, treat
``cosine >= threshold`` as a re-sighting, decide which re-sightings are worth
keeping as ADDITIONAL views of that guest (multi-template — see
:func:`_should_enrol`), tag sub-canon faces, record operator-attested people
the pipeline missed, and expose the corrections the runner applies from the
feedback loop.

Concurrency and cost: the gallery file is opened ONCE per process
(:func:`app.store.open_store`) and the scan matrix lives in memory, so a match
is a matrix-vector multiply rather than a connect + full-table read.  The
match-then-insert decision runs inside one IMMEDIATE transaction that also
holds the store lock (see :meth:`VectorStore.transaction`), so two concurrent
/match calls can never both insert the same brand-new person.

Lifecycle: ``data/gallery-<runId>.db``.  A run resets its gallery at start, so
the unique count always begins at zero, and the runner deletes it again when
the run ends (:func:`reset` is that entry point too) — these files hold real
guests' face embeddings, so an orphaned one is a retention liability, not just
disk.  :func:`sweep` is the backstop for runs that died without releasing.
"""

import itertools
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .appearance import best_intersection
from .store import Neighbour, VectorStore, as_unit, close_store, open_store

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class BadRunIdError(ValueError):
    """Raised when a runId is unsafe to use as part of a filename."""


@dataclass
class MatchResult:
    """Outcome of one gallery lookup: who, whether new, how close, gallery size."""

    person_key: str
    is_new: bool
    cosine: float | None  # best similarity vs the pre-existing gallery; None if it was empty
    gallery_n: int  # distinct persons after this operation
    sub_canon: bool  # this face was below the 80 px production canon
    template_n: int = 1  # templates this identity holds after the call
    template_added: bool = False  # this sighting was enrolled as an extra view
    # Best torso-histogram intersection vs the matched identity's stored
    # descriptors; None when either side lacks one (absent is not zero), and
    # always None for a fresh mint — there was nothing stored to compare with.
    appearance_sim: float | None = None
    # An enrolment _should_enrol had approved was refused because the torso
    # descriptor clashed.  Never set on the verdict path: appearance vetoes
    # WRITES, not verdicts.
    appearance_vetoed: bool = False
    # Rowid of the template this call ENROLLED (None when nothing was written,
    # and None on a mint — a mint's unit of retraction is the whole key, which
    # the runner already handles).  Handed back so a caller who later proves
    # the enrolment wrong can retract exactly that row via POST /template/forget
    # instead of guessing; the runner's same-frame guard is the only caller.
    template_id: int | None = None
    # Set ONLY on a mint whose best cosine against the pre-existing gallery
    # landed inside one of the TWO near-miss bands:
    # {"key": the near-missed identity, "cosine": that best score,
    #  "appearanceSim": torso intersection vs that identity's stored
    #  descriptors, None when either side lacks one,
    #  "basis": "face" for the strong band [nearmiss_floor .. threshold),
    #  "clothing" for the weak band [weak_floor .. nearmiss_floor) that also
    #  required the torso descriptors to agree}.  Information for the
    #  operator, never behaviour — is_new stays True and nothing is merged
    #  (see _near_miss() for the measured impostor pair that forbids it).
    near_miss: dict | None = None
    # Set when the template THIS CALL WROTE brought its identity within the
    # match threshold of a DIFFERENT identity — the gallery-overlap signal:
    # {"key": the overlapping rival, "cosine": the new template's similarity
    #  to it, "appearanceSim": torso intersection vs the rival's stored
    #  descriptors, None when either side lacks one}.  Same discipline as
    # near_miss: information for the operator, never behaviour.
    overlap: dict | None = None


def db_path(data_dir: Path, run_id: str) -> Path:
    """Return the gallery database path for a run, validating the id first."""
    if not _RUN_ID_RE.match(run_id):
        raise BadRunIdError(f"runId must match {_RUN_ID_RE.pattern!r}")
    return data_dir / f"gallery-{run_id}.db"


def _unlink_db(path: Path) -> None:
    """Close then delete a store file and its WAL sidecars.

    The cached connection is closed FIRST: unlinking a file a live connection
    still holds leaves that connection answering from an inode nobody else can
    reach, so a "fresh" gallery would silently not be fresh.  The ``-wal`` and
    ``-shm`` companions are removed too — a clean close normally takes them,
    but a process killed mid-run leaves them behind, and a stale WAL next to a
    recreated database is how deleted embeddings come back.
    """
    close_store(path)
    for p in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        p.unlink(missing_ok=True)


def reset(data_dir: Path, run_id: str) -> None:
    """Delete the run's gallery file (start-of-run wipe and end-of-run release)."""
    _unlink_db(db_path(data_dir, run_id))


def sweep(data_dir: Path, max_age_s: float) -> list[str]:
    """Delete gallery files untouched for ``max_age_s``; return the run ids.

    The backstop for the release-on-run-end path: a runner that was killed
    mid-event never gets to call :func:`reset`, and every leftover file holds
    guests' face embeddings.  Ops can call this from a cron or by hand
    (``POST /gallery/sweep``); it never touches staff stores, which are
    deliberately persistent.
    """
    cutoff = time.time() - max_age_s
    swept: list[str] = []
    for path in sorted(Path(data_dir).glob("gallery-*.db")):
        try:
            if path.stat().st_mtime > cutoff:
                continue
        except OSError:  # vanished under us — someone else swept it
            continue
        _unlink_db(path)
        swept.append(path.name[len("gallery-") : -len(".db")])
    return swept


def count(data_dir: Path, run_id: str) -> int:
    """Distinct guest count for a run (0 when the gallery does not exist yet)."""
    path = db_path(data_dir, run_id)
    if not path.exists():
        return 0
    store = open_store(path)
    with store.reading():
        return store.distinct_count()


def _should_enrol(
    store: VectorStore,
    key: str,
    embedding: list[float],
    quality: float | None,
    best: float,
    threshold: float,
    cap: int,
    confidence: float,
    margin: float,
    max_cosine: float,
) -> bool:
    """Decide whether a matched sighting is worth keeping as an extra template.

    THE RULE, and why each clause exists.  A matched sighting is enrolled only
    if ALL of these hold:

    1. ``cap > 1``.  A cap of 1 is the pre-M1 behaviour, kept as an off switch.
    2. ``best >= threshold + confidence``.  It matched *comfortably*, not
       barely.  A bare-minimum match is the least certain evidence in the
       system; promoting it to a template would let the identity annex the
       region around a point we are not sure of, and the error would compound
       with every template built on top of it.  This is the drift brake.
    3. ``best < max_cosine``.  It is not a near-duplicate of a view we already
       hold.  Above the ceiling the sighting adds no pose coverage — it would
       spend a capped slot and evict a genuinely different view, making the
       identity NARROWER.  Most frames of a walking guest land here, which is
       also what keeps this write path quiet.
    4. It beats the nearest RIVAL identity by ``margin``.  A sighting sitting
       almost equally close to two people is exactly the one that must not be
       stored: as a template it becomes a bridge, and the next probe near it
       merges two guests into one.  Over-counting costs an argument about an
       invoice; silently merging two paying guests costs the same money and
       nobody can see it happen, so ambiguity is resolved by NOT learning.
    5. If the identity is already at its cap, the sighting must be MORE
       DISTINCTIVE than the closest pair already held — i.e. ``best`` must sit
       below :meth:`VectorStore.max_redundancy`.  Otherwise the newcomer is the
       most redundant view in the set and :meth:`VectorStore.prune_redundant`
       would delete it again on the next line, churning the database for
       nothing.  Passing this test means some existing pair of views is closer
       together than the newcomer is to anything, so one of THEM is evicted and
       the identity's spread widens — the gallery improves its own coverage
       every time it learns.

       This clause used to compare capture QUALITY against the worst view held,
       to match a quality-based eviction.  Both were wrong together, and the
       corridor bench showed why: quality is face width, face width is
       distance, so "keep the best five" resolved to "keep the five frames
       where the guest was nearest the lens".  A walk toward the camera evicted
       every far view in favour of a closer one, and one guest's five templates
       ended up spanning TWO SECONDS of a 140-second crossing — all one
       distance, all one pose.  The same man at 87 px then scored 0.294 against
       a gallery that only knew him between 148 and 329 px, and was counted
       twice.  Distinctiveness is what a template slot is for; quality only
       breaks ties.

    What this deliberately does NOT do: change who counts as a match.  Every
    enrolled template was itself verified at or above the threshold against an
    already-stored template of the same identity, so the identity grows only
    along evidence it already accepted.  The accept region widens — that is the
    entire point of multi-template, and it is how a profile view gets attached
    to the frontal view that first minted the key — but the threshold constant
    is untouched.  Lowering it is M2 and is blocked on impostor data.
    """
    if cap <= 1:
        return False
    if best < threshold + confidence:
        return False
    if best >= max_cosine:
        return False
    rival = store.runner_up(embedding, key)
    if rival is not None and (best - rival.cosine) < margin:
        return False
    if store.count_for(key) >= cap:
        crowded = store.max_redundancy(key)
        if crowded is None or best >= crowded:
            return False
    return True


#: Tolerance on every band comparison below.  Embeddings and descriptors
#: round-trip through float32 storage, so a value an operator AUTHORS as a
#: boundary (0.29, 0.78) reads back a half-ULP under it — measured: a true
#: cosine of 0.29 returns 0.28999999165534973 and fell out of the band.  It
#: failed toward silence (a missed suggestion, never a wrong one), which is
#: why it went unnoticed while the face floor had the same property — but the
#: weak band's entire safety margin is 0.033 wide, so "the knob means what it
#: says" has to be true at the boundary.  1e-6 is far below any tuning step an
#: operator would make and far above float32's ~1e-7 error at these values.
_BAND_EPS = 1e-6


def _near_miss(
    store: VectorStore,
    key: str,
    hit: Neighbour | None,
    appearance: list[float] | None,
    nearmiss_floor: float,
    nearmiss_weak_floor: float,
    nearmiss_clothes: float,
) -> dict | None:
    """Should this mint carry a near-miss suggestion, and on WHOSE evidence?

    Called on the MINT branch only.  ``key`` is the identity the rider would be
    attached to — the key this mint has just taken — and ``hit`` is the best of
    the PRE-EXISTING gallery, read before anything was written, so on this
    branch it always sits below the match threshold and each band only needs
    its floor.

    Ordering note: the torso comparison below reads ONLY ``hit.key``'s stored
    descriptors, and a freshly minted key is by construction never ``hit.key``,
    so this returns exactly what it returned when it ran before the insert.  It
    runs after the mint for one reason: the suppression below has to know whose
    banner this would be, and the key does not exist until it is minted.

    TWO BANDS, one shape, and a ``basis`` field naming which one spoke:

    ``basis="face"`` — cosine in ``[nearmiss_floor .. threshold)``, the
    original v2 band, unchanged in behaviour and now merely labelled.  The
    face got close enough on its own; clothing is reported but not required,
    because at this range the face is the evidence.

    ``basis="clothing"`` — cosine in ``[weak_floor .. nearmiss_floor)`` AND a
    torso intersection that is present and ``>= nearmiss_clothes``.  THE
    MEASUREMENT THAT FORCED THIS BAND, bench 6e1a5d (2026-08-06), ground truth
    ONE person who walked out of frame, came back, and sat down: that person
    produced six tracker ids and two surviving extra guests —

        p00005: face 0.212 vs p00001, clothing 0.797
        p00006: face 0.228 vs p00001, clothing 0.875

    Both sat below the 0.29 face floor, so nothing was flagged, nobody was
    asked, and the count was wrong by two.  Seated and turned-away re-entries
    at this camera's angles land exactly there: the face signal is nearly
    empty while the clothing signal is shouting.

    NOW THE HONEST PART, because this band is the thinnest evidence in the
    system.  The closest measured IMPOSTOR pair — two genuinely different men
    — sat at face 0.377 with clothing 0.503, and another impostor pair reached
    **clothing 0.747**.  The default bar of 0.78 clears that by 0.033.  That
    is a hair, not a margin.  It is defensible for exactly one reason: a
    near-miss is a SUGGESTION A HUMAN CONFIRMS, never a merge — ``is_new``
    stays True, the count moves only on an operator's click.  Said plainly: at
    a venue with uniformed staff, a dress code, or similar traditional dress,
    this band is EXPECTED to produce wrong suggestions, and it is the first
    knob to turn off (``HECO_MATCH_NEARMISS_WEAK_FLOOR=0``, which leaves the
    face band exactly as it was).  Clothing agreement alone never proves
    identity; here it only buys the operator a look.

    ``appearance_sim`` is ``None`` when either side lacks a descriptor, and
    absent is not zero — a descriptor-less mint therefore never enters the
    weak band at all, rather than entering it on an assumed clash or an
    assumed agreement.

    A SETTLED PAIR IS NEVER ASKED ABOUT AGAIN (2026-08-06).  A near-miss rider
    is a QUESTION put to the operator — "is this new guest actually that one?"
    — and :meth:`VectorStore.cannot_link` is the ANSWER already on file for
    that pair.  It gets there two ways, and both are answers: the operator's
    own *false-match* correction, and now the runner asserting CO-PRESENCE —
    two faces at different positions in ONE frame are two different people,
    which is about as certain as machine evidence gets.  Re-asking a settled
    question is precisely the noise that trains an operator to click banners
    away, so a constrained pair returns ``None`` here.  The check sits AFTER
    the band-range early-out — most mints of a real event leave there for free
    — and BEFORE the torso comparison, which is the cheaper order: one indexed
    primary-key lookup against ``appearances_for``'s row read plus a histogram
    intersection.

    THE MEASURED CASE, run 05b3b7 (2026-08-06), ground truth THREE people and
    THREE counted — the count was RIGHT and the banners were wrong:

        p00002: "minted 0.316 from p00001" — clothing agreement 0.94
        p00007: "minted 0.360 from p00002" — clothing agreement 0.57

    p00001 and p00002 walked in through the main door TOGETHER; p00007 (the
    operator) stood in the alley with p00002, and one tap round matched
    **p00002 and p00007 in the same frame**.  The pipeline held proof that
    each pair was two people and argued with itself anyway.

    No threshold move fixes this, which is why it is answered with an
    independent signal rather than a knob: both riders sat at 0.316 and 0.360
    against a 0.363 threshold — INSIDE the ordinary face near-miss band — while
    the measured same-person misses on this camera are 0.294 / 0.308 / 0.361
    and the closest measured impostor pair is 0.377.  The genuine and impostor
    distributions overlap right there; clothing did not save it either (0.94
    and 0.57 on the two false riders).  Co-presence is the only CERTAIN signal
    available, and it is independent of both.

    This is the THIRD thing ``cannot_link`` governs — :meth:`VectorStore.merge`
    refuses a constrained pair, :func:`_overlap_after_write` withholds the
    overlap banner, and now the near-miss rider is withheld too — so the
    constraint is becoming the gallery's single record of "known different",
    which is exactly what it should be: one fact, asserted by a human or by the
    machine, honoured everywhere.

    No knob here on purpose: the suppression is INHERENT to cannot_link (a
    banner that contradicts a recorded fact has no defensible reading), and the
    off switch belongs at the source — the runner's ``HECO_COPRESENCE_SPLIT=0``
    stops co-presence being asserted at all.

    Off switches: ``nearmiss_floor <= 0`` disables BOTH bands (the weak band
    is defined as the region under that floor, so without a floor there is no
    region), ``nearmiss_weak_floor <= 0`` disables only the weak one.  A
    weak floor accidentally set at or above the face floor yields an empty
    band rather than an inverted one, by construction of the comparison below.
    """
    if hit is None or nearmiss_floor <= 0:
        return None
    weak_on = nearmiss_weak_floor > 0
    lowest = min(nearmiss_floor, nearmiss_weak_floor) if weak_on else nearmiss_floor
    if hit.cosine < lowest - _BAND_EPS:
        # Far outside both bands: skip the torso comparison entirely.  Most
        # mints of a real event land here and this is the quiet path.
        return None
    if store.cannot_link(key, hit.key):
        # Settled: an operator or the runner's co-presence assertion has
        # already recorded that these two are different people.  The banner
        # would be the pipeline arguing with a fact it holds (run 05b3b7).
        return None
    appearance_sim = best_intersection(appearance, store.appearances_for(hit.key))
    if hit.cosine >= nearmiss_floor - _BAND_EPS:
        basis = "face"
    elif (
        weak_on
        and hit.cosine >= nearmiss_weak_floor - _BAND_EPS
        and appearance_sim is not None
        and appearance_sim >= nearmiss_clothes - _BAND_EPS
    ):
        basis = "clothing"
    else:
        return None
    return {
        "key": hit.key,
        "cosine": hit.cosine,
        "appearanceSim": appearance_sim,
        "basis": basis,
    }


def match(
    data_dir: Path,
    run_id: str,
    embedding: list[float],
    quality: float | None,
    threshold: float,
    canon_px: float,
    templates_per_person: int = 1,
    template_confidence: float = 0.0,
    template_margin: float = 0.0,
    template_max_cosine: float = 1.0,
    appearance: list[float] | None = None,
    appearance_clash: float = 0.0,
    nearmiss_floor: float = 0.0,
    nearmiss_weak_floor: float = 0.0,
    # 1.0 (not 0.0) is the safe default for a BAR rather than an off switch: a
    # caller that turns the weak floor on without naming a clothes bar gets a
    # band that essentially never fires, instead of one that fires on any
    # clothing whatsoever.  The off switch is nearmiss_weak_floor=0.
    nearmiss_clothes: float = 1.0,
    # Identities this probe has been PROVEN not to be — the runner's same-frame
    # guard, never a heuristic.  Excluded from the scan so the probe resolves
    # against the rest of the gallery (an existing guest if one fits, a fresh
    # mint otherwise) instead of re-landing on the key it cannot belong to.
    exclude_keys: set[str] | None = None,
) -> MatchResult:
    """Match one embedding against the run's gallery; insert if new.

    Rule: best cosine >= threshold means "seen before" (the SFace operating
    point counts equality as a match).  A new person stores the embedding as
    their first template under a fresh monotonic key.  Sub-canon faces
    (quality < canon_px) are matched normally but tagged, per the POC contract.

    MULTI-TEMPLATE (M1).  An identity holds up to ``templates_per_person``
    views, not one.  Before this, a guest was represented forever by the FIRST
    view of them — whatever angle that happened to be — and the corridor bench
    showed what that costs: one man produced THREE gallery identities whose
    views were mutually 0.296-0.347 against a 0.363 threshold, every pair a near
    miss.  The three views were not unrecognisable; they were simply never
    compared with anything except one arbitrary first frame.  Now a re-sighting
    that clears :func:`_should_enrol` is kept as an additional view of the same
    key, so the walk's intermediate poses chain the profile view back to the
    frontal one and the whole crossing resolves to a single guest.

    Comparing a probe against ALL of an identity's templates needs no new
    machinery: :meth:`VectorStore.search` is an argmax over ROWS, and the max
    over rows equals the max over per-identity maxima — so the moment several
    rows share a key, the returned cosine IS that identity's best template
    score.  Verified, not assumed (``test_search_is_per_identity_best``).

    APPEARANCE IS ADVISORY (v1 tie-breaker, 2026-08-06).  ``appearance`` is
    the sighting's torso descriptor (:mod:`app.appearance`).  Exactly three
    things happen with it here, and nothing else:

    1. It is STORED beside any template this call writes — the founding mint
       and every enrolled extra — so later sightings have evidence to compare
       against.
    2. ``appearance_sim`` reports the best histogram intersection against the
       matched identity's ALREADY-stored descriptors, for visibility (the
       runner's taps and counters read it).  ``None`` when either side lacks
       a descriptor — old galleries, no person box, tiny crops: absent is not
       zero — and ``None`` for a fresh mint, which has nothing stored yet.
    3. ENROLMENT VETO (anti-poison): a sighting :func:`_should_enrol` had
       approved as an extra template is NOT kept when its descriptor clashes
       (``appearance_sim < appearance_clash``) with everything the identity
       has stored.  The one thing worse than missing a good template is
       keeping a poisoned one: a template enrolled off a tracker swap or a
       borderline impostor becomes a bridge that silently merges two paying
       guests at every later crossing.

    What appearance NEVER does is touch the verdict.  ``is_new``,
    ``person_key`` and ``cosine`` are decided by the face alone, because the
    measured evidence runs the other way: the closest impostor pair on this
    camera — two DIFFERENT men at cosine 0.377, above the 0.363 threshold —
    were BOTH IN LIGHT SHIRTS, so a torso "rescue" of near-threshold face
    matches would have merged two real guests into one invoice line.  And the
    same-person misses (0.294/0.308/0.361) wore the SAME clothes in every
    frame, so at a wedding full of light shirts a rescue would chain
    strangers together wholesale.  Advisory means: appearance may only refuse
    a WRITE, never make or unmake a match.  ``appearance_clash <= 0`` is the
    off switch, and an absent descriptor on either side never vetoes.

    THE NEAR-MISS FLAG (v2, 2026-08-06), NOW TWO-SIGNAL.  A mint may carry
    ``near_miss = {key, cosine, appearanceSim, basis}`` — the identity it
    almost was, the score, the torso intersection against THAT identity's
    stored descriptors (None when either side lacks one; absent is not zero),
    and which signal spoke.  ``basis="face"`` is the original band
    ``[nearmiss_floor .. threshold)``: measured at face 0.3464 with clothing
    0.562, a split the operator found BY EYE because nothing surfaced it.
    ``basis="clothing"`` is the weaker band ``[weak_floor ..
    nearmiss_floor)``, which additionally requires the clothing to agree at
    ``nearmiss_clothes`` — added because bench 6e1a5d measured one person
    splitting at face 0.212/clothing 0.797 and face 0.228/clothing 0.875,
    both under the face floor, both silently over-counted.  The full argument
    and the honest 0.78-vs-0.747-impostor margin live in :func:`_near_miss`.

    **THE VERDICT IS STILL A MINT, on either basis.**  The flag is a
    suggestion to a human, never an automatic merge, because the measurement
    runs both ways: an IMPOSTOR pair on the same camera sits at face 0.377
    with clothing intersection 0.503, and another impostor pair reached
    clothing 0.747.  Auto-merging on this evidence would fold real strangers
    into one invoice line invisibly; the count must not move without the
    operator clicking.  Out-of-band mints (below both floors, clothing under
    the bar, or an empty gallery) carry ``near_miss = None``; matched
    verdicts and staff hits never carry it at all.  ``nearmiss_floor <= 0``
    switches off both bands; ``nearmiss_weak_floor <= 0`` switches off only
    the clothing one.

    AND A PAIR UNDER CANNOT-LINK CARRIES NO RIDER AT ALL (2026-08-06), on
    either basis.  The rider asks the operator a question; the constraint is
    the answer already on file — from their own *false-match* correction, or
    from the runner asserting co-presence (two faces at different positions in
    one frame are two people).  Run 05b3b7 raised two such banners, at 0.316
    and 0.360 against the 0.363 threshold, between people the pipeline had
    already seen standing side by side.  The suppression is per-pair and
    changes nothing else: ``is_new``, ``person_key``, ``cosine`` and
    ``gallery_n`` are byte-identical with and without the constraint, because
    cannot-link must never change WHO someone is — only what is suggested
    about them.  :func:`_near_miss` carries the full argument.

    The defaults here are the pre-M1 behaviour (cap 1, no enrolment, no
    appearance veto, no near-miss flag of either basis); the service passes
    the real values from :mod:`app.config`, so a caller that only wants the
    old semantics gets them by leaving the knobs alone.
    """
    sub_canon = quality is not None and quality < canon_px
    store = open_store(db_path(data_dir, run_id))

    with store.transaction():
        hit = store.search(embedding, exclude=exclude_keys)
        if hit is not None and hit.cosine >= threshold:
            # Similarity vs what is ALREADY stored, computed before this call
            # writes anything — otherwise an enrolled sighting would be
            # compared against itself and always report 1.0.
            appearance_sim = best_intersection(appearance, store.appearances_for(hit.key))
            added = _should_enrol(
                store,
                hit.key,
                embedding,
                quality,
                hit.cosine,
                threshold,
                templates_per_person,
                template_confidence,
                template_margin,
                template_max_cosine,
            )
            vetoed = False
            if (
                added
                and appearance_clash > 0
                and appearance_sim is not None
                and appearance_sim < appearance_clash
            ):
                # The face said "same person, comfortably"; the torso says the
                # clothes agree with NONE of this identity's stored sightings.
                # Within one event clothing is constant, so the sighting is
                # suspect evidence (tracker swap, near-threshold impostor) and
                # the identity must not grow towards it.  The VERDICT above is
                # untouched — this refuses only the write.
                added, vetoed = False, True
            overlap = None
            template_id = None
            if added:
                template_id = store.add(
                    hit.key,
                    embedding,
                    quality=quality,
                    sub_canon=sub_canon,
                    appearance=appearance,
                )
                # Redundancy, NOT quality — see _should_enrol clause 5 for the
                # measurement that killed the quality rule here.
                store.prune_redundant(hit.key, templates_per_person)
                overlap = _overlap_after_write(store, hit.key, embedding, appearance, threshold)
            return MatchResult(
                hit.key,
                False,
                hit.cosine,
                store.distinct_count(),
                sub_canon,
                store.count_for(hit.key),
                added,
                appearance_sim=appearance_sim,
                appearance_vetoed=vetoed,
                template_id=template_id,
                overlap=overlap,
            )
        key = store.add_auto(
            embedding, quality=quality, sub_canon=sub_canon, prefix="p", appearance=appearance
        )
        # NEAR-MISS: judged against `hit` — the best of the gallery as it stood
        # BEFORE this mint wrote anything — and against the near-missed
        # identity's already-stored descriptors, the same before-the-write
        # discipline as appearance_sim above.  It is evaluated AFTER the insert
        # only because the settled-pair suppression needs the minted key to ask
        # the store whether this pair is already known-different; the insert
        # touches no row _near_miss reads (a fresh key is never hit.key), so
        # the rider is identical either way.
        near_miss = _near_miss(
            store, key, hit, appearance, nearmiss_floor, nearmiss_weak_floor, nearmiss_clothes
        )
        # A mint can NEVER create overlap, by construction: on this branch the
        # best pre-write cosine sits below the threshold, and the nearest
        # rival to the founding view is bounded by that same best.  Overlap is
        # exclusively an ENROLMENT phenomenon — an identity GROWING a view
        # that lands inside someone else's accept region — which is also why
        # the near-miss flag above (strictly BELOW threshold, on either basis)
        # and the overlap flag (at/above threshold, unconditional) never both
        # fire on one verdict.  Pinned by test, not assumed.
        best = hit.cosine if hit is not None else None
        return MatchResult(
            key, True, best, store.distinct_count(), sub_canon, 1, False,
            near_miss=near_miss,
        )


def _overlap_after_write(
    store: VectorStore, key: str, embedding, appearance, threshold: float
) -> dict | None:
    """Did the template just written pull ``key`` into overlap with a rival?

    THE PATTERN THIS SURFACES, measured three times in two days before it had
    a name: a person splits into two identities (a seated face, a blurred
    re-entry, a lens-edge view), and multi-template growth then WIDENS both
    until their templates cross the match threshold — 0.544, 0.452 and 0.376
    on consecutive benches, every pair later confirmed one person by the
    operator.  At that point the gallery itself holds the evidence that two of
    its identities are probably one guest, and nothing surfaced it; the
    operator found each case by reading cosine matrices by hand.

    So: after ANY template write (a mint's founding view or an enrolled
    extra), ask the store for the nearest RIVAL identity to the new template.
    At or above the match threshold, that rival would have MATCHED this very
    sighting had the other identity not existed — the gallery's own standard
    of "same person", which is why this fires regardless of clothing (unlike
    the near-miss banner, which sits below the threshold and needs the
    clothes to agree before it speaks).  The torso intersection rides along
    for the operator's judgement.

    A cannot-link constraint silences the pair for good: someone has already
    said "different people" — the operator by clicking *false-match*, or the
    runner by asserting co-presence (two faces in one frame) — and a banner
    that keeps arguing with a recorded fact is noise.  :func:`_near_miss` now
    withholds its rider on the same test, so the constraint governs all three
    of merge refusal, this banner and that one.  Overlap created by merge()
    re-pointing rows is deliberately not detected here — a merge is itself an
    operator act, and its survivor is pruned back through prune_redundant.

    Information only, never behaviour: nothing is merged, the count does not
    move.  The runner forwards it; the planner turns it into a one-click
    suggestion.
    """
    rival = store.runner_up(embedding, key)
    if rival is None or rival.cosine < threshold:
        return None
    if store.cannot_link(key, rival.key):
        return None
    return {
        "key": rival.key,
        "cosine": rival.cosine,
        "appearanceSim": best_intersection(appearance, store.appearances_for(rival.key)),
    }


def add_manual(data_dir: Path, run_id: str, note: str | None = None) -> tuple[str, int]:
    """Count one person the operator saw and the pipeline did not (*missed*).

    Returns ``(personKey, galleryN)``.  The key carries an ``m`` prefix and no
    embedding, so the addition is permanently distinguishable from an
    automatic detection and can never absorb a later sighting — it is a human
    attestation, recorded as one.
    """
    store = open_store(db_path(data_dir, run_id))
    with store.transaction():
        key = store.add_manual(note)
        return key, store.distinct_count()


def merge(
    data_dir: Path,
    run_id: str,
    keep: str,
    drop: str,
    templates_per_person: int = 0,
    only_if_singleton: bool = False,
) -> tuple[bool, int]:
    """Fold ``drop`` into ``keep`` (a *duplicate* correction).

    Returns ``(merged, galleryN)``; ``merged`` is False (and the count
    unchanged) when a cannot-link constraint or an unknown key blocks it.

    ``only_if_singleton`` additionally refuses the merge unless ``drop`` holds
    EXACTLY one template.  It exists for the runner's track heal, and the
    asymmetry with the operator flow is the point: the caller asserting "this
    key is a junk mint" there is a MACHINE, and a machine's evidence is weaker
    than an operator's.  The heal saw one track mint a key and then match a
    different key; an operator saw two faces.  A drop key that has since
    accumulated more templates has been independently re-sighted — the gallery
    accepted further evidence that this identity is real — so it is no longer
    safely foldable by heuristic, and the refusal (``merged=False``, count
    unchanged) is the correct answer, not an error.  The check runs INSIDE the
    transaction, against the same uncommitted view :meth:`VectorStore.count_for`
    reads, because a template can be enrolled between the runner's decision and
    this merge arriving — checked outside, the heal would fold a key the
    gallery had just re-validated.

    THE CAP APPLIES HERE TOO.  The survivor inherits both identities' views, so
    a merge is the one path that can carry an identity past
    ``templates_per_person``: six single-template people merged one after
    another left six views against a cap of five, and an identity whose accept
    region grows without bound is exactly what causes the NEXT silent merge —
    an under-count, which nobody can see in an invoice figure.

    The survivor is pruned by REDUNDANCY, not by quality.  A merge exists
    because the pipeline failed to join these views, so they are the widest-
    apart evidence the identity has; evicting on quality would discard the
    merged-in views and leave the operator re-merging the same guest at every
    crossing.  :meth:`VectorStore.prune_redundant` carries the full argument.

    ``templates_per_person`` of 0 or 1 prunes nothing, which is the pre-M1
    behaviour — so a caller that has not opted into multi-template gets exactly
    what it got before.
    """
    store = open_store(db_path(data_dir, run_id))
    with store.transaction():
        if only_if_singleton and store.count_for(drop) != 1:
            return False, store.distinct_count()
        merged = store.merge(keep, drop)
        if merged and templates_per_person > 1:
            store.prune_redundant(keep, templates_per_person)
        return merged, store.distinct_count()


def review_duplicates(
    data_dir: Path,
    run_id: str,
    threshold: float,
    floor: float,
    limit: int = 50,
) -> dict:
    """Identity pairs a human should look at, ranked. Never a verdict.

    THE CASE THIS EXISTS FOR (run 0f5c6d, 2026-08-07, ground truth THREE
    people, reported FOUR).  One man — at the back of the room, phone to his
    ear — was minted twice: p00002 and p00003 scored **0.2117** against each
    other, a same-person miss so far below the 0.363 threshold that no
    automatic mechanism could act on it.  And nothing else could separate him
    either: his duplicate pair agreed on clothing at **0.587** while two
    genuinely DIFFERENT men in that same run agreed at **0.538**.  Five
    hundredths apart.  There is no threshold in that gap, and inventing one
    would trade this over-count for a merged guest, which is the failure
    nobody ever notices.

    So this does not decide.  It ASKS, and it asks about as few pairs as the
    evidence allows:

    * **A recorded `cannot_link` disqualifies a pair outright.** That is the
      gallery's record of "known different" — written by co-presence (two
      faces at different positions in one frame) or by the operator's own
      false-match click.  Re-asking a settled question is the noise that
      trains operators to ignore banners.  In the motivating run this alone
      removed both pairs involving p00001.
    * **The face score must sit in ``[floor, threshold)``.** At or above the
      threshold the gallery already calls them one person and no review is
      needed; below the floor they are not similar in any measurable sense.
    * **Clothing RANKS, it never filters.** Given 0.587-against-0.538 above,
      a clothing bar here would be a coin toss wearing a number.  It orders
      the queue so the likeliest pair is read first, and pairs whose torsos
      were never measurable still appear — ranked last among themselves by
      face score — because absent is not zero and an unmeasured guest must
      not fall silently off a review list.

    Returns ``{"pairs": [...], "considered": n, "returned": k, "dropped": d}``.
    ``dropped`` is stated rather than swallowed: a truncated queue that looks
    complete is how a real duplicate goes unreviewed.
    """
    store = open_store(db_path(data_dir, run_id))
    with store.reading():
        keys = store.keys()
        vecs = {k: [as_unit(v) for v in store.vectors_for(k)] for k in keys}
        apps = {k: store.appearances_for(k) for k in keys}
        candidates, considered = [], 0
        for a, b in itertools.combinations(keys, 2):
            considered += 1
            if store.cannot_link(a, b):
                continue
            if not vecs[a] or not vecs[b]:
                continue
            cosine = max(float(x @ y) for x in vecs[a] for y in vecs[b])
            if cosine < floor or cosine >= threshold:
                continue
            pairs = [(p, q) for p in apps[a] for q in apps[b]]
            clothes = (
                max(float(np.minimum(p, q).sum()) for p, q in pairs) if pairs else None
            )
            candidates.append({"a": a, "b": b, "cosine": cosine, "clothes": clothes})

    # Measured clothing first, best agreement leading; unmeasured pairs keep
    # their place in the queue behind them, ordered by face score.
    candidates.sort(
        key=lambda c: (c["clothes"] is None, -(c["clothes"] or 0.0), -c["cosine"])
    )
    kept = candidates[: max(0, limit)]
    return {
        "pairs": kept,
        "considered": considered,
        "returned": len(kept),
        "dropped": len(candidates) - len(kept),
    }


def forget_template(data_dir: Path, run_id: str, template_id: int) -> bool:
    """Retract one enrolled template by rowid; True if it was actually removed.

    Exists for a caller that can only DISPROVE a write after making it: the
    runner matches every face in a frame before it can see that two different
    bodies landed on the same key, by which point the loser has already been
    enrolled into an identity it cannot belong to.  Without this the split
    fixes only the frame in hand — the poisoned template stays in the gallery
    and goes on capturing that person in every later frame where they appear
    ALONE and no same-frame evidence exists.  Run fa8fc3 is the worked example:
    five templates under one key, internally 0.22..0.52 (impostor-level),
    holding two different men.

    False — not an error — when the row is already gone (``prune_redundant``
    fires immediately after every enrolment and may have evicted it) or when
    it is its key's last template (see :meth:`VectorStore.forget_template`).
    """
    store = open_store(db_path(data_dir, run_id))
    with store.transaction():
        return store.forget_template(template_id)


def split(data_dir: Path, run_id: str, a: str, b: str) -> int:
    """Record a *false-match* do-not-merge constraint; returns galleryN.

    Both keys keep their templates, so the distinct count is unchanged — the
    effect is forward-looking, and it is now THREE effects: a later
    :func:`merge` of the pair is refused, the gallery-overlap banner is
    withheld (:func:`_overlap_after_write`), and a near-miss rider naming the
    pair is withheld (:func:`_near_miss`).  One constraint, one meaning: these
    two are known-different.

    TWO WRITERS, same door.  The operator's *false-match* click is one.  The
    other is the runner asserting CO-PRESENCE: every distinct non-staff pair of
    personKeys seen in ONE frame is posted here once (``HECO_COPRESENCE_SPLIT``
    in the runner is that source's off switch), because two faces at different
    positions in a single frame are two different people.  The machine's
    assertion goes through the operator's primitive rather than growing new
    machinery, which is why the three effects above came for free.

    THE KNOWN COST, stated not hidden: a person holding a PHONE showing their
    own face — or a mirror, or a printed photo — puts one real person's face in
    the frame twice, and co-presence will assert those are two people.  That
    blocks a fold which would have been correct (it blocked exactly such a
    phone-face heal on the 2026-08-06 bench).  The trade is deliberate and is
    this project's standing asymmetry: a blocked fold OVER-counts, which is
    visible and an operator can merge; a wrong fold UNDER-counts silently and
    nobody ever sees a guest who was erased.  Fixed mirrors and screens are
    handled by exclusion zones; a hand-held phone is the residual.
    """
    store = open_store(db_path(data_dir, run_id))
    with store.transaction():
        store.split(a, b)
        return store.distinct_count()


def remove(
    data_dir: Path, run_id: str, person_key: str
) -> tuple[list[tuple[bytes, float | None, int]], int]:
    """Lift a person out of the gallery (for *mark-staff*).

    Returns ``(templates, galleryN)`` where ``templates`` are the removed rows
    (blob, quality, sub_canon) for the caller to re-home in the staff store.
    """
    store = open_store(db_path(data_dir, run_id))
    with store.transaction():
        templates = store.remove(person_key)
        return templates, store.distinct_count()
