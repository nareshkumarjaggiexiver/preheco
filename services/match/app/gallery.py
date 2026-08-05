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

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .store import VectorStore, close_store, open_store

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

    The defaults here are the pre-M1 behaviour (cap 1, no enrolment); the
    service passes the real values from :mod:`app.config`, so a caller that
    only wants the old semantics gets them by leaving the knobs alone.
    """
    sub_canon = quality is not None and quality < canon_px
    store = open_store(db_path(data_dir, run_id))

    with store.transaction():
        hit = store.search(embedding)
        if hit is not None and hit.cosine >= threshold:
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
            if added:
                store.add(hit.key, embedding, quality=quality, sub_canon=sub_canon)
                # Redundancy, NOT quality — see _should_enrol clause 5 for the
                # measurement that killed the quality rule here.
                store.prune_redundant(hit.key, templates_per_person)
            return MatchResult(
                hit.key,
                False,
                hit.cosine,
                store.distinct_count(),
                sub_canon,
                store.count_for(hit.key),
                added,
            )
        key = store.add_auto(embedding, quality=quality, sub_canon=sub_canon, prefix="p")
        best = hit.cosine if hit is not None else None
        return MatchResult(key, True, best, store.distinct_count(), sub_canon, 1, False)


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


def split(data_dir: Path, run_id: str, a: str, b: str) -> int:
    """Record a *false-match* do-not-merge constraint; returns galleryN.

    Both keys keep their templates, so the distinct count is unchanged — the
    effect is forward-looking (a later :func:`merge` of the pair is refused).
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
