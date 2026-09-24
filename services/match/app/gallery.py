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
from datetime import datetime
from pathlib import Path

import numpy as np

from .appearance import (
    BEARD_DIM,
    HEAD_DIM,
    HEAD_H_BINS,
    TORSO_DIM,
    beard_class,
    beards_differ,
    best_cross,
    best_intersection,
    head_label,
    intersection,
    self_agreement,
    spread,
)
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
    # Rowid of the body_sightings row this call logged (None when the call
    # carried no usable body).  Same purpose as template_id: the runner's
    # same-frame guard hands it back via POST /template/forget when it proves
    # the sighting was a different body, so the box leaves the wrong
    # identity's stature evidence instead of staying there for the run.
    body_id: int | None = None
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
    # The embed service's reading of this face ({"gender", "genderP", "age"}),
    # the raw feature's L2 norm, and the sighting's containing PERSON box
    # ({"h", "w", "yBottom", "frameH"}, raw detector px).  Stored, never
    # judged: attributes and the norm ride with any template this call
    # writes; the body is logged on every guest call.  Their only reader is
    # the review queue (see review_duplicates).  None = not measured.
    attributes: dict | None = None,
    feat_norm: float | None = None,
    body: dict | None = None,
    # The sighting's head descriptor (40 floats) and beard reading (4),
    # logged on the body row beside the torso; None = not measured.  Their
    # one reader is the review queue.
    head: list[float] | None = None,
    beard: list[float] | None = None,
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

    ATTRIBUTES, FEATURE NORM AND BODY ARE RECORDED, NOT CONSULTED (v4,
    2026-09-24).  ``attributes`` (sex, its probability, age) and ``feat_norm``
    are written beside any template this call stores; ``body`` is appended to
    the sighting log on every guest call whether or not a template is
    written.  Nothing here reads them back.  They exist for
    :func:`review_duplicates`, which uses them to set aside pairs that cannot
    be one person — and even there they only remove questions from a human's
    list, never answer one.  ``None`` for any of them is "not measured" and
    stores NULL.

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
                    attributes=attributes,
                    feat_norm=feat_norm,
                )
                # Redundancy, NOT quality — see _should_enrol clause 5 for the
                # measurement that killed the quality rule here.
                store.prune_redundant(hit.key, templates_per_person)
                overlap = _overlap_after_write(store, hit.key, embedding, appearance, threshold)
            body_id = _log_body(store, hit.key, body, quality, appearance, head, beard)
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
                body_id=body_id,
                overlap=overlap,
            )
        key = store.add_auto(
            embedding, quality=quality, sub_canon=sub_canon, prefix="p",
            appearance=appearance, attributes=attributes, feat_norm=feat_norm,
        )
        body_id = _log_body(store, key, body, quality, appearance, head, beard)
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
            near_miss=near_miss, body_id=body_id,
        )


def _log_body(
    store: VectorStore,
    key: str,
    body: dict | None,
    face_w: float | None,
    appearance: list[float] | None = None,
    head: list[float] | None = None,
    beard: list[float] | None = None,
) -> int | None:
    """Append the sighting's person box to ``key``'s body log; its rowid.

    Tolerant of a partial box on purpose: the runner sends ``None`` when no
    person box contained the face, and a box with a missing or non-positive
    dimension is not a measurement either.  Silence, not an error — the
    verdict this rides on has already been decided and must not fail over a
    logging field.  ``face_w`` is the call's ``quality`` (face box width px),
    stored beside the box so the stature reader can see a head-and-shoulders
    crop for what it is; a non-positive one is stored as unknown.
    ``appearance`` (the torso descriptor), ``head`` and ``beard`` (None when
    unmeasured) are logged on the same row: they are the review queue's
    per-sighting evidence.  A torso needs the person box this row records,
    so a call without a usable body had no torso to log either; its head
    and beard readings are not logged — a face outside every person box is
    rare (1 of run f0bfc5's 2,809 guest verdicts) and the row has nothing
    else to hang them on.
    """
    if not body:
        return None
    try:
        h, w = float(body["h"]), float(body["w"])
        y_bottom, frame_h = float(body["yBottom"]), int(body["frameH"])
    except (KeyError, TypeError, ValueError):
        return None
    if h <= 0 or w <= 0 or frame_h <= 0:
        return None
    fw = float(face_w) if face_w is not None and float(face_w) > 0 else None
    return store.add_body_sighting(
        key, h, w, y_bottom, frame_h, fw, appearance=appearance, head=head, beard=beard
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


def review_order(candidate: dict) -> tuple:
    """Sort key for the review queue: face first, clothes the tiebreak.

    A module-level function rather than an inline lambda so the ORDER itself
    is testable without engineering two probes into an exact cosine tie —
    which the embedding helpers cannot produce deterministically, as the
    first attempt at that test demonstrated by flapping.
    """
    return (
        -candidate["cosine"],
        candidate["clothes"] is None,   # measured before unmeasured, at a tie
        -(candidate["clothes"] or 0.0),
    )


def best_cosine(data_dir: Path, run_id: str, embedding: list[float]) -> float | None:
    """The best guest-gallery cosine for a probe, READ-ONLY — no mint, no enrol.

    Exists for exactly one caller: the staff check in ``/match``. Staff used to
    be resolved FIRST and returned on any hit at threshold, without ever
    consulting the guest gallery — so a weak staff score shadowed a strong
    guest match. Run 27ca33 measured the cost: a guest whose face matched
    p00020 at 0.6476 one frame earlier and 0.5998 one frame later was tagged
    STAFF at 0.4234 in between, and 281 of the run's 339 danger-zone verdicts
    were staff hits hugging the threshold. Deciding "staff or guest" needs both
    scores, and fetching the guest side must not have the side effects a full
    :func:`match` carries.

    None when the gallery does not exist yet or holds nothing — absent is not
    zero, and a first-ever sighting must not lose to a 0.0.
    """
    path = db_path(data_dir, run_id)
    if not path.exists():
        return None
    store = open_store(path)
    with store.reading():
        hit = store.search(embedding)
    return float(hit.cosine) if hit is not None else None


# ------------------------------------------ the review queue's side evidence
#
# Everything from here to review_duplicates() exists to REMOVE questions from
# a human's list, never to answer one.  Run f0bfc5 (2026-09-23, a Punjab
# wedding hall from an overview camera, 74 guests) put 500 pairs in the queue;
# its top five, read by eye, included a man against an elderly woman (#3), a
# black beard against a white one (#1) and a child against an adult (#5).
# Each of those the pipeline could have settled itself, from evidence the
# face score does not carry: sex, age, and how tall the body is.  None of it
# is allowed near a verdict — the count moves only on a face match or a human
# click — but a pair the evidence says cannot be one person need not be asked.


def identity_gender(
    rows: list[tuple[str | None, float | None, float | None]],
) -> tuple[str | None, float | None]:
    """An identity's sex and confidence from its templates' attribute readings.

    Each template with a reading is one vote: its reported sex at its reported
    probability, folded to P(male) so the votes are on one axis.  The identity's
    belief is the mean over voting templates — TEMPLATE-weighted, every view
    counting once — SHRUNK towards 0.5 by two pseudo-votes of 0.5 each:
    ``p_male = (sum(votes) + 1) / (n + 2)``.  The answer is the side that
    favours, with its confidence ``max(p, 1-p)``.

    WHY THE SHRINK (measured on run f0bfc5's 594 re-embedded sightings): a
    plain mean makes one template exactly as confident as its one view, and a
    duplicate is by construction minted on the view that FAILED to match —
    head-down, turned, small — which is the view the attribute head flips on.
    Eight of 37 identities with two or more reads carried BOTH a >= 0.8 male
    read and a >= 0.8 female read (p00048, a girl: eleven M >= 0.8 and one F
    0.92; p00049, an elderly woman: F 0.84 and M 0.87), and 7 of the 72 keys
    held a single template.  With the shrink one M 1.0 template reads 0.67,
    three unanimous read 0.80 (the review bar), five read 0.86, and p00048's
    fifteen reads give 0.765 — under the bar, so a second identity minted on
    her one upright female view could not be set aside on sex.  A frontal
    0.97 M with a head-down 0.55 F reads M at 0.605: the sure view is not
    outvoted, and the unsure one costs enough that the bar will not trust
    the pair.

    ``(None, None)`` when no template carries a reading — a runner without
    the attribute model, or a gallery from before the columns.  Absent is not
    a default sex.
    """
    votes = [
        (p if g == "M" else 1.0 - p)
        for g, p, _ in rows
        if g in ("M", "F") and p is not None
    ]
    if not votes:
        return None, None
    p_male = (float(np.sum(votes)) + 1.0) / (len(votes) + 2.0)
    if p_male >= 0.5:
        return "M", p_male
    return "F", 1.0 - p_male


def identity_age(
    rows: list[tuple[str | None, float | None, float | None]],
) -> float | None:
    """An identity's age: the MEDIAN over its templates' readings, or None.

    Median, not mean, because the attribute head's age error on a small or
    turned face is not symmetric — a 40 px profile is as likely to read 15
    years off as 3 — and one such view must not drag a child into the adult
    band.  None when no template carries a reading.
    """
    ages = [a for _, _, a in rows if a is not None]
    return float(np.median(ages)) if ages else None


#: Standing-box geometry, shared with scratchpad replay.py (the reference).
#: A box is STANDING when it is at least twice as tall as wide (seated and
#: bending bodies are squatter), its bottom edge sits more than 60 px above
#: the frame's bottom (a body cut off by the frame edge has no measurable
#: height), and its top is more than 5 px below the frame's top (same, above).
_STANDING_ASPECT = 2.0
_STANDING_BOTTOM_MARGIN_PX = 60.0
_STANDING_TOP_MARGIN_PX = 5.0
#: A standing body is at least this many FACE WIDTHS tall.  Run f0bfc5's
#: ledger (2808 face-bearing person boxes): full standing adults read p10
#: 10.0 / median 11.0 / p95 12.4 face widths, the smallest child (p00001)
#: min 5.7 / p10 9.9 — and the head-to-waist boxes an occlusion produces
#: (p00052 behind a table, seq 9075-9091: 205x470 px on a 112 px face) read
#: 2.9-4.3, with an aspect of 2.3 that passes the h/w test.  Fifteen of
#: those in a row put her median at 0.59 of adult height against 1.05 from
#: her full boxes.  6.0 drops 47 of the run's 1268 standing boxes, moves no
#: identity's 8-frame median by 0.2, and keeps every child measured.  A row
#: without a face width (written before the column) is not filtered.
_STANDING_MIN_FACE_WIDTHS = 6.0
#: An identity's standing boxes must span at least this many distinct write
#: SECONDS before its median is trusted, on top of the min_n row bar.  Two,
#: not min_n: a run's face-bearing standing sightings arrive in bursts of a
#: few seconds (f0bfc5: 53 identities with >= 8 standing boxes spanned a
#: median of 4 distinct video seconds; only 3 of them spanned 8), so a bar
#: of eight seconds would have measured almost nobody, while a bar of two
#: is exactly what rejects the case it exists for — one occlusion's 15
#: consecutive waist-up frames are one second at 15 fps.
_STATURE_MIN_MOMENTS = 2
#: The perspective fit needs this many standing boxes before it is a fit and
#: not a line through noise; below it every stature is null (not measured).
#: 50 is replay.py's bar and is well under a minute of one guest walking.
_STATURE_FIT_MIN_N = 50
#: Residual band kept on each of the three re-fit passes: a box under 0.6 or
#: over 1.5 of the predicted height is a child, a seated body the aspect test
#: missed, or a tracker box lagging a turn — not a data point for the camera's
#: perspective.
_STATURE_TRIM = (0.6, 1.5)


def _moment(created_at: str | None) -> str | None:
    """The second a row was written, as a label; None when the row has no time.

    Eight boxes are eight measurements only when they are eight different
    moments: fifteen consecutive frames of one occlusion are half a second of
    one pose.  ISO timestamps sort as text, so the label is the prefix up to
    the seconds field (``2026-09-24T21:14:03``).
    """
    if not created_at:
        return None
    return created_at[:19]


def standing_sightings(sightings: list) -> list[tuple[str, float, float, str | None]]:
    """Filter the body log to standing boxes: ``(key, y_bottom, h, moment)`` each.

    A box is standing when it is at least twice as tall as wide, its bottom
    edge sits more than 60 px above the frame's bottom, its top more than
    5 px below the frame's top, and — when the sighting's face width is
    known — it is at least _STANDING_MIN_FACE_WIDTHS face widths tall, which
    is what separates a standing body from a head-and-shoulders crop with a
    standing box's aspect.  Rows are :class:`app.store.BodySighting`
    (``face_w`` / ``created_at`` optional); ``moment`` is the write second.
    """
    out = []
    for row in sightings:
        key, h, w, y_bottom, frame_h = row[0], float(row[1]), float(row[2]), float(row[3]), row[4]
        face_w = row[5] if len(row) > 5 else None
        created_at = row[6] if len(row) > 6 else None
        if w <= 0 or h <= 0:
            continue
        if h / w < _STANDING_ASPECT:
            continue
        if y_bottom >= frame_h - _STANDING_BOTTOM_MARGIN_PX:
            continue
        if (y_bottom - h) <= _STANDING_TOP_MARGIN_PX:
            continue
        if face_w is not None and face_w > 0 and h < _STANDING_MIN_FACE_WIDTHS * face_w:
            continue
        out.append((key, y_bottom, h, _moment(created_at)))
    return out


def stature_fit(standing: list[tuple]) -> tuple[float, float] | None:
    """Fit ``h = a * y_bottom + b`` over standing boxes, robustly; ``(a, b)``.

    The camera looks down the hall, so a standing body's box height is a
    near-linear function of where its feet are: lower in frame = nearer the
    lens = taller box.  Run f0bfc5 fitted h = 0.602 * y_bottom + 300 px over
    1268 standing sightings.  An ordinary least-squares line first, then three
    passes that drop every box outside 0.6..1.5 of the line's prediction and
    refit — children, seated bodies and lagging tracker boxes are exactly the
    outliers a plain fit would bend towards, and they are the ones the
    result is meant to measure AGAINST.  scratchpad replay.py is the
    reference; this is that algorithm, guarded.

    ``None`` under _STATURE_FIT_MIN_N boxes, or when a trim pass would leave
    fewer than two points (the previous pass's line stands).
    """
    if len(standing) < _STATURE_FIT_MIN_N:
        return None
    y = np.array([row[1] for row in standing], dtype=np.float64)
    h = np.array([row[2] for row in standing], dtype=np.float64)
    design = np.vstack([y, np.ones_like(y)]).T
    coef = np.linalg.lstsq(design, h, rcond=None)[0]
    lo, hi = _STATURE_TRIM
    for _ in range(3):
        predicted = design @ coef
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(predicted > 0, h / predicted, np.nan)
        mask = (ratio > lo) & (ratio < hi)
        if mask.sum() < 2:
            break
        coef = np.linalg.lstsq(design[mask], h[mask], rcond=None)[0]
    return float(coef[0]), float(coef[1])


def stature_ratios(sightings: list, min_n: int) -> dict[str, float]:
    """Each identity's median standing-height ratio against the run's fit.

    1.0 is "as tall as the fit says a standing body is at that spot"; the
    fit is dominated by adults, so 1.0 is an average adult and f0bfc5's child
    p00009 read 0.73.  An identity needs ``min_n`` standing boxes spanning
    at least _STATURE_MIN_MOMENTS distinct write seconds to appear at all —
    a single frame mid-stride or half behind a pillar is 30% off, the median
    over eight is not, and eight consecutive frames are one moment, not
    eight — and a box the fit predicts a non-positive height for (above the
    vanishing line, a fit artefact) is skipped.  Rows without a timestamp
    (a pure-function caller) count as one moment each.  Identities with no
    trusted ratio are simply absent from the dict; the caller reads absence
    as None.  Empty when the run has too few standing boxes for a fit.
    """
    standing = standing_sightings(sightings)
    coef = stature_fit(standing)
    if coef is None:
        return {}
    a, b = coef
    by_key: dict[str, list[float]] = {}
    moments: dict[str, set] = {}
    untimed: dict[str, int] = {}
    for key, y_bottom, h, moment in standing:
        predicted = a * y_bottom + b
        if predicted <= 0:
            continue
        by_key.setdefault(key, []).append(h / predicted)
        if moment is None:
            untimed[key] = untimed.get(key, 0) + 1
        else:
            moments.setdefault(key, set()).add(moment)
    need = max(1, min_n)
    return {
        key: float(np.median(ratios))
        for key, ratios in by_key.items()
        if len(ratios) >= need
        and len(moments.get(key, ())) + untimed.get(key, 0) >= min(need, _STATURE_MIN_MOMENTS)
    }


def _side_reasons(
    gender: dict, age: dict, stature: dict,
    gender_min_p: float, age_child_max: float, age_adult_min: float, stature_gap: float,
) -> list[str]:
    """Which of sex, age and stature say this pair cannot be one person.

    Every one that speaks, in the order gender, age, stature; the review
    COUNTS a pair under the first reason only (``excluded`` sums to the
    pairs set aside) and lists them all in ``setAside[].reasons``.  Every
    signal has an off switch at zero, and every signal needs its evidence
    on BOTH sides: one measured identity and one unmeasured is not a
    disagreement, it is one opinion, and absent is not zero.

    * gender — both confident at or above ``gender_min_p`` and different.
      One side sure and the other unsure never excludes; the unsure side is
      exactly the head-down or turned-away view the attribute model gets
      wrong, and it must not vote.
    * age — one median at or below ``age_child_max``, the other at or above
      ``age_adult_min``; the gap between the bands is the model's error
      budget and a pair straddling it is still asked about.
    * stature — both ratios measured and ``|a-b| >= stature_gap``.
    """
    reasons: list[str] = []
    ga, gb, pa, pb = gender["a"], gender["b"], gender["pA"], gender["pB"]
    if (
        gender_min_p > 0
        and ga is not None and gb is not None
        and pa is not None and pb is not None
        and pa >= gender_min_p and pb >= gender_min_p
        and ga != gb
    ):
        reasons.append("gender")
    aa, ab = age["a"], age["b"]
    if (
        age_child_max > 0 and age_adult_min > 0
        and aa is not None and ab is not None
        and (
            (aa <= age_child_max and ab >= age_adult_min)
            or (ab <= age_child_max and aa >= age_adult_min)
        )
    ):
        reasons.append("age")
    sa, sb = stature["a"], stature["b"]
    if stature_gap > 0 and sa is not None and sb is not None and abs(sa - sb) >= stature_gap:
        reasons.append("stature")
    return reasons


#: The clothing rule's floor in TIME: an identity's torso reads must span at
#: least this many seconds (first to last write) before its self-agreement
#: means anything.  Three reads from three consecutive frames are one pose
#: under one light, read three times; run f0bfc5's identities were on camera
#: for 1-5 s a walk, so two seconds is the most the rule can ask and still
#: measure most of them.
_CLOTHES_MIN_SPAN_S = 2.0


def _seconds(created_at: str | None) -> float | None:
    """A row's write time as epoch seconds; None when it has none."""
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(created_at).timestamp()
    except ValueError:
        return None


@dataclass
class IdentityReads:
    """One identity's appearance reads of one kind, ready to compare.

    ``vectors`` are the reads spread over the identity's time on camera
    (:func:`app.appearance.spread`); ``n`` counts every read the log held,
    ``span_s`` is first-to-last write time (0.0 with fewer than two timed
    reads) and ``agreement`` the median pairwise intersection of
    ``vectors`` (None under two).
    """

    vectors: list[np.ndarray]
    n: int
    span_s: float
    agreement: float | None


def identity_reads(rows: list[tuple[float | None, np.ndarray]]) -> IdentityReads:
    """Build :class:`IdentityReads` from ``(time_s, vector)`` rows in any order."""
    ordered = sorted(rows, key=lambda r: (r[0] is None, r[0] or 0.0))
    times = [t for t, _ in ordered if t is not None]
    vectors = [v for _, v in spread(ordered)]
    return IdentityReads(
        vectors=vectors,
        n=len(ordered),
        span_s=(max(times) - min(times)) if len(times) >= 2 else 0.0,
        agreement=self_agreement(vectors),
    )


def torso_reads(
    evidence: list, templates: dict[str, list[tuple[str, np.ndarray]]] | None = None
) -> dict[str, IdentityReads]:
    """Each identity's v3 torso reads from the body log; v2 rows never count.

    A 48-float (v2) descriptor is a different partition over a different
    crop — the chin-down band whose skin and hair made a yellow top agree
    with a green dress at 0.72 — so it is not clothing evidence here, and
    an identity with only v2 rows is simply absent.

    ``templates`` (key -> its templates' ``(created_at, torso)`` rows) is
    the fallback for an identity whose body log holds no v3 torso: until
    match 0.13.0 torsos were written on templates only, and run c84098's
    gallery — where the template reads set aside six pairs, all different
    people — has nothing else.  An identity with body-log reads never mixes
    its templates in: a template's torso is also on the body row of the
    sighting that wrote it, and counting it twice would inflate ``n``.
    """
    by_key: dict[str, list] = {}
    for row in evidence:
        vec = row.appearance
        if vec is None or vec.size != TORSO_DIM:
            continue
        by_key.setdefault(row.key, []).append((_seconds(row.created_at), vec))
    for key, rows in (templates or {}).items():
        if key in by_key:
            continue
        v3 = [(_seconds(ts), vec) for ts, vec in rows if vec.size == TORSO_DIM]
        if v3:
            by_key[key] = v3
    return {k: identity_reads(v) for k, v in by_key.items()}


def _reads_of(evidence: list, field: str, dim: int) -> dict[str, IdentityReads]:
    """Each identity's reads of one body-log column of length ``dim``."""
    by_key: dict[str, list] = {}
    for row in evidence:
        vec = getattr(row, field)
        if vec is None or vec.size != dim:
            continue
        by_key.setdefault(row.key, []).append((_seconds(row.created_at), vec))
    return {k: identity_reads(v) for k, v in by_key.items()}


def head_reads(evidence: list) -> dict[str, IdentityReads]:
    """Each identity's head descriptors (40 floats) from the body log."""
    return _reads_of(evidence, "head", HEAD_DIM)


def beard_reads(evidence: list) -> dict[str, IdentityReads]:
    """Each identity's beard readings (4 floats) from the body log."""
    return _reads_of(evidence, "beard", BEARD_DIM)


#: The head rule's own-testimony bar, as the clothing rule's defaults: three
#: reads over two seconds whose median pairwise intersection is 0.6.  Run
#: f0bfc5: every one of the 45 identities' own head reads agreed at 0.63 or
#: more (median 0.87), so the bar only ever refuses an identity whose reads
#: disagree — a merged pair, or a head half out of frame.
_HEAD_MIN_N = 3
_HEAD_SELF_MIN = 0.6
#: ...and BOTH heads must be HEADWEAR: at least this share of each identity's
#: mean head reading chromatic (hue bins 0..23).  On run f0bfc5 the four
#: turbans read 0.68-0.87, every head of black, grey or white hair 0.03-0.38.
#: Why the rule never sets a covered head against a bare one: at a Punjabi
#: wedding the same guest's head is covered and uncovered within the night —
#: a dupatta drawn over the hair for the ceremony, a rumal or patka for the
#: Gurdwara — so "pink turban against black hair" (#15, head cross 0.41) is
#: exactly what one person minted twice can look like.  A bald scalp reads
#: as its skin colour (0.91 chromatic) and counts as headwear here: the
#: residual, a bald man with a rumal on in one identity only, is rare.
_HEAD_WEAR_MIN = 0.5


def headwear_share(reads: IdentityReads | None) -> float | None:
    """The chromatic share of an identity's mean head reading, or None."""
    if reads is None or not reads.vectors:
        return None
    return float(np.mean([float(v[:HEAD_H_BINS].sum()) for v in reads.vectors]))


def head_apart(
    ha: IdentityReads | None, hb: IdentityReads | None, sim: float | None, clash: float
) -> bool:
    """Do two identities' headwear — turban against turban — say two people?

    Both identities must read one head (:data:`_HEAD_MIN_N` reads over
    _CLOTHES_MIN_SPAN_S agreeing at :data:`_HEAD_SELF_MIN`), both heads must
    be headwear (:data:`_HEAD_WEAR_MIN`), and the best reading of one
    against any of the other must stay under ``clash``.  ``clash <= 0`` is
    off.
    """
    for r in (ha, hb):
        share = headwear_share(r)
        if share is None or share < _HEAD_WEAR_MIN:
            return False
    return clothes_apart(ha, hb, sim, clash, _HEAD_MIN_N, _HEAD_SELF_MIN)


def beards_apart(
    ba: IdentityReads | None, bb: IdentityReads | None, min_n: int
) -> tuple[str | None, str | None, bool]:
    """Both identities' beard classes, and whether they cannot be one face.

    A class (:func:`app.appearance.beard_class`) is shown whenever an
    identity has reads; it may set a pair aside only when BOTH identities
    have at least ``min_n`` reads spanning _CLOTHES_MIN_SPAN_S and their
    confident classes differ as one face cannot (none against any beard,
    dark against white).  ``min_n <= 0`` is off.
    """
    ca = None if ba is None else beard_class(ba.vectors)
    cb = None if bb is None else beard_class(bb.vectors)
    if min_n <= 0 or ba is None or bb is None:
        return ca, cb, False
    for r in (ba, bb):
        if r.n < min_n or r.span_s < _CLOTHES_MIN_SPAN_S:
            return ca, cb, False
    return ca, cb, beards_differ(ca, cb)


def clothes_apart(
    ra: IdentityReads | None,
    rb: IdentityReads | None,
    cross: float | None,
    clash: float,
    min_n: int,
    self_min: float,
) -> bool:
    """Do two identities' clothes say they are two people?

    Only when EACH identity's own torso reads agree with each other —
    at least ``min_n`` reads spanning _CLOTHES_MIN_SPAN_S, median pairwise
    intersection at or above ``self_min`` — and the best any read of one
    manages against any read of the other is still under ``clash``.  The
    self-agreement is what makes the cross meaningful: an identity whose
    own reads disagree (a merged pair of two people, a band that kept
    catching a pillar) has no clothing to compare.  ``clash <= 0`` is off.
    """
    if clash <= 0 or cross is None or ra is None or rb is None:
        return False
    for r in (ra, rb):
        if r.n < min_n or r.span_s < _CLOTHES_MIN_SPAN_S:
            return False
        if r.agreement is None or r.agreement < self_min:
            return False
    return cross < clash


def review_duplicates(
    data_dir: Path,
    run_id: str,
    threshold: float,
    floor: float,
    limit: int = 50,
    gender_min_p: float = 0.0,
    age_child_max: float = 0.0,
    age_adult_min: float = 0.0,
    stature_gap: float = 0.0,
    stature_min_n: int = 8,
    adult_m: float = 1.75,
    clothes_clash: float = 0.0,
    clothes_min_n: int = 3,
    clothes_self_min: float = 0.6,
    head_clash: float = 0.0,
    beard_min_n: int = 0,
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
    * **Clothing RANKS — and, since the v3 torso, may SET ASIDE a clear
      clash.** The 0.587-against-0.538 bench above was the v2 histogram,
      whose band started under the chin and read mostly skin, and it was ONE
      descriptor against one; there a clothing bar was a coin toss.  v3
      reads the cloth, and the rule asks for agreement WITHIN each identity
      first and a clash second: a pair is set aside only when BOTH
      identities have >= ``clothes_min_n`` v3 torso reads spanning two
      seconds that agree with each other at ``clothes_self_min``, and the
      best cross reading is still under ``clothes_clash``
      (:func:`clothes_apart`).  The reads are the BODY LOG's — every
      sighting's torso, not the five a template cap keeps — and a gallery
      written before the body log carried them falls back to its templates'
      (:func:`torso_reads`).  Measured twice the same night: on the Sharon
      re-run c84098 (templates) a person against themselves (early reads vs
      late) never scored under 0.52 while the operator's different-people
      pairs scored 0.10-0.25; on run f0bfc5 (45 identities of the queue's
      first 32 pairs, <= 24 body-log reads each) own reads agree at a median
      0.90 (p10 0.70), the same person split across a gap of seconds to
      minutes at a best cross of 0.77-0.97, and the queue's different-people
      pairs anywhere from 0.10 to 0.97 (two white shirts agree) — the
      default 0.35 sets aside #3, #6, #7 and #23 and none of the nine
      same-person splits.  Everything short of that still only ranks;
      unmeasured torsos still appear, because absent is not zero.  A v2
      (48-d) torso against a v3 (64-d) one is not comparable and reads as
      unmeasured, not as 0.
    * **Sex, age and stature may SET A PAIR ASIDE (v4, 2026-09-24)** — after
      the band test and before the cap, so an excluded pair neither costs a
      slot nor counts as dropped.  Run f0bfc5 flooded the queue with 500
      pairs whose top five included a man against an elderly woman and a
      child against an adult; the pipeline had the evidence to settle those
      and asked anyway.  Each pair carries a ``why`` — both identities' sex
      with confidence, median age, and stature ratio (null wherever not
      measured; absent is not zero) — and the reply's ``excluded`` counts
      say how many pairs each signal set aside, so a queue that was quieted
      is never mistaken for one that was quiet.  The rules and their off
      switches are in :func:`_side_reasons`; the knobs in :mod:`app.config`.
      Nothing set aside is written anywhere — not to ``cannot_link``, which
      remains a human's or co-presence's word — and nothing is merged.

    * **Head and beard may set a pair aside (2026-09-24, night)** — read per
      sighting beside the torso: the head descriptor (turban or hair colour)
      under the same own-testimony rule as clothing at ``head_clash``
      (:func:`head_apart`), and the beard class — none, dark, grey or white
      — when both identities are confident and one face could not show both
      (:func:`beards_apart`).  Run f0bfc5's #1, a maroon turban and black
      beard against a peach turban and white beard, is what they exist for.

    Stature is metres-free on the wire by design: ``stature.a``/``b`` are
    ratios against the run's own perspective fit (1.0 = an average standing
    adult at that spot), ``adultM`` is the anchor that turns a ratio into
    metres for a human to read (``aM``/``bM`` carry that product ready-made),
    and the exclusion gap is on the ratio, so the anchor never moves it.

    EVERY PAIR SET ASIDE STAYS INSPECTABLE.  ``setAside`` lists them —
    ``{a, b, cosine, clothes, why, reasons}``, ranked like the queue and
    capped at ``limit`` — with every signal that spoke in ``reasons`` (the
    pair is counted in ``excluded`` under the first).  An operator who
    disagrees with the machine can read the evidence and merge the pair with
    the ordinary /merge: nothing here wrote a cannot_link or a merge.

    ``why.clothes`` is ``{selfA, selfB, cross, nA, nB}`` over the v3 torso
    reads of each identity's body log (its templates' in a gallery written
    before match 0.13.0): each side's median self-agreement
    (null under two reads), the best cross intersection (null when either
    side has none) and how many reads each side had — reported whether or
    not the clothing rule is on.  ``why.head`` is ``{a, b, sim, selfA,
    selfB, nA, nB}`` — ``a``/``b`` each side's dominant colour
    (:func:`app.appearance.head_label`), ``sim`` the best cross — and
    ``why.beard`` ``{a, b, nA, nB}`` with each side's class; null wherever
    a side has no read.

    Returns ``{"pairs": [...], "considered": n, "returned": k, "dropped": d,
    "excluded": {"gender": g, "age": a, "stature": s, "clothes": c,
    "head": h, "beard": b}, "setAside": [...], "setAsideDropped": s}``.
    ``dropped`` (and ``setAsideDropped``, the set-aside pairs past
    ``limit``) is stated rather than swallowed: a truncated queue that looks
    complete is how a real duplicate goes unreviewed.

    The store's lock — the one every /match of the run waits on — is held
    only to READ: the appearance evidence is compared after it is released
    (a 41,000-row body log took 0.7 s to compare; the live loop's /match
    must not wait on an operator's click).
    """
    store = open_store(db_path(data_dir, run_id))
    excluded = {"gender": 0, "age": 0, "stature": 0, "clothes": 0, "head": 0, "beard": 0}
    with store.reading():
        keys = store.keys()
        vecs = {k: [as_unit(v) for v in store.vectors_for(k)] for k in keys}
        apps = {k: store.appearances_for(k) for k in keys}
        tpl_torsos = {k: store.appearance_rows_for(k) for k in keys}
        attrs = {k: store.attributes_for(k) for k in keys}
        genders = {k: identity_gender(attrs[k]) for k in keys}
        ages = {k: identity_age(attrs[k]) for k in keys}
        statures = stature_ratios(store.body_sightings(), stature_min_n)
        evidence = store.sighting_evidence()
        in_band, considered = [], 0
        for a, b in itertools.combinations(keys, 2):
            considered += 1
            if store.cannot_link(a, b):
                continue
            if not vecs[a] or not vecs[b]:
                continue
            cosine = max(float(x @ y) for x in vecs[a] for y in vecs[b])
            if cosine < floor or cosine >= threshold:
                continue
            scores = [
                s for s in (intersection(p, q) for p in apps[a] for q in apps[b])
                if s is not None
            ]
            in_band.append((a, b, cosine, max(scores) if scores else None))

    torsos = torso_reads(evidence, tpl_torsos)
    heads = head_reads(evidence)
    beards = beard_reads(evidence)
    candidates, set_aside = [], []
    for a, b, cosine, clothes in in_band:
        sa, sb = statures.get(a), statures.get(b)
        ta, tb = torsos.get(a), torsos.get(b)
        cross = None if ta is None or tb is None else best_cross(ta.vectors, tb.vectors)
        ha, hb = heads.get(a), heads.get(b)
        head_sim = None if ha is None or hb is None else best_cross(ha.vectors, hb.vectors)
        ba, bb = beards.get(a), beards.get(b)
        beard_a, beard_b, beard_differ = beards_apart(ba, bb, beard_min_n)
        why = {
            "gender": {
                "a": genders[a][0], "b": genders[b][0],
                "pA": genders[a][1], "pB": genders[b][1],
            },
            "age": {"a": ages[a], "b": ages[b]},
            "stature": {
                "a": sa, "b": sb, "adultM": adult_m,
                "aM": None if sa is None else sa * adult_m,
                "bM": None if sb is None else sb * adult_m,
            },
            "clothes": {
                "selfA": None if ta is None else ta.agreement,
                "selfB": None if tb is None else tb.agreement,
                "cross": cross,
                "nA": 0 if ta is None else ta.n,
                "nB": 0 if tb is None else tb.n,
            },
            "head": {
                "a": None if ha is None else head_label(ha.vectors),
                "b": None if hb is None else head_label(hb.vectors),
                "sim": head_sim,
                "selfA": None if ha is None else ha.agreement,
                "selfB": None if hb is None else hb.agreement,
                "nA": 0 if ha is None else ha.n,
                "nB": 0 if hb is None else hb.n,
            },
            "beard": {
                "a": beard_a, "b": beard_b,
                "nA": 0 if ba is None else ba.n,
                "nB": 0 if bb is None else bb.n,
            },
        }
        reasons = _side_reasons(
            why["gender"], why["age"], why["stature"],
            gender_min_p, age_child_max, age_adult_min, stature_gap,
        )
        if clothes_apart(ta, tb, cross, clothes_clash, clothes_min_n, clothes_self_min):
            reasons.append("clothes")
        if head_apart(ha, hb, head_sim, head_clash):
            reasons.append("head")
        if beard_differ:
            reasons.append("beard")
        entry = {"a": a, "b": b, "cosine": cosine, "clothes": clothes, "why": why}
        if reasons:
            excluded[reasons[0]] += 1
            set_aside.append({**entry, "reasons": reasons})
            continue
        candidates.append(entry)

    # FACE FIRST, clothes as the tiebreak — reversed on 2026-08-13, on a
    # measurement. The clothes-first order was built for run 0f5c6d, where
    # clothing (0.587 vs 0.538) was the only signal separating a genuine
    # duplicate from noise. Run 27ca33 — a lobby full of dark formal wear —
    # inverted it: clothing agreement ~0.9 was GENERIC, 34 of the top-50 pairs
    # had face scores below 0.30 (noise wearing a rank), and the one confirmed
    # same-person pair in the entire queue (p00014/p00015 at 0.3526, the 4th-
    # highest face score in the band) sat at #53 — past the caller's limit,
    # silently dropped, invisible to the operator who then reported exactly
    # that pair as missing.
    #
    # The band is [floor, threshold): the face score IS the distance from
    # "the gallery already calls them one person", so it is the band's own
    # metric and ranks by it. Clothing still breaks ties and is still
    # returned, because at equal face evidence a matching torso is worth the
    # operator's glance first. Neither order finds every needle — 0f5c6d's
    # 0.2117 pair ranks lower under this sort — which is why the LIMIT fix
    # rides with it: nothing is silently dropped any more, so a lower rank is
    # a later look rather than no look at all.
    candidates.sort(key=review_order)
    kept = candidates[: max(0, limit)]
    set_aside.sort(key=review_order)
    return {
        "pairs": kept,
        "considered": considered,
        "returned": len(kept),
        "dropped": len(candidates) - len(kept),
        "excluded": excluded,
        # Every pair the evidence set aside, likeliest first, capped like the
        # queue: an exclusion only takes a pair out of the ranked list — the
        # operator can still see it, and still merge it.
        "setAside": set_aside[: max(0, limit)],
        "setAsideDropped": max(0, len(set_aside) - max(0, limit)),
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


def forget_body_sighting(data_dir: Path, run_id: str, body_id: int) -> bool:
    """Retract one body-log row by rowid; True if a row went.

    The same caller and the same reason as :func:`forget_template`: the box
    was logged under the key /match resolved before the runner could prove
    the sighting was a different body.  Left in place it is a stranger's
    height in the loser's stature median — and the re-ask logs the same box
    again under the corrected key, so the fit would hold it twice.
    """
    store = open_store(db_path(data_dir, run_id))
    with store.transaction():
        return store.forget_body_sighting(body_id)


def split(data_dir: Path, run_id: str, a: str, b: str) -> int:
    """Record a *false-match* do-not-merge constraint; returns galleryN.

    BOTH KEYS MUST EXIST in the run's gallery. The store beneath deliberately
    accepts constraints on any pair of identifiers ("a statement about a pair
    of identifiers, not about stored rows") — and run 27ca33 measured what
    that costs at this boundary: an operator's false-match correction arrived
    with truncated keys (p0004 for p00004), wrote a dead cannot_link row, and
    the operator walked away believing the correction had applied. A
    correction that silently protects nobody is worse than a refused one, so
    the unknown key is named in a ValueError here — the operator's feedback
    ledger records the rejection, and the runner's co-presence path treats
    the 4xx as settled, which for a key that does not exist is the truth.

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
        known = set(store.keys())
        missing = [k for k in (a, b) if k not in known]
        if missing:
            raise ValueError(
                f"unknown person key{'s' if len(missing) > 1 else ''} "
                f"{', '.join(missing)} — the constraint would protect nobody. "
                "Check the key against the register (keys are like p00004)."
            )
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
