"""The per-run guest gallery: one SQLite file per run, matched brute-force.

Thin policy layer over :class:`app.store.VectorStore`.  The store owns the
mechanics (cosine scan, float32 BLOB rows, monotonic keys, merge/split/remove);
this module owns the *guest-counting policy*: mint ``p#####`` keys, treat
``cosine >= threshold`` as a re-sighting, tag sub-canon faces, record
operator-attested people the pipeline missed, and expose the corrections the
runner applies from the feedback loop.

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

from .store import close_store, open_store

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


def match(
    data_dir: Path,
    run_id: str,
    embedding: list[float],
    quality: float | None,
    threshold: float,
    canon_px: float,
) -> MatchResult:
    """Match one embedding against the run's gallery; insert if new.

    Rule: best cosine >= threshold means "seen before" (the SFace operating
    point counts equality as a match).  A new person stores the embedding as
    their first template under a fresh monotonic key.  Sub-canon faces
    (quality < canon_px) are matched normally but tagged, per the POC contract.
    """
    sub_canon = quality is not None and quality < canon_px
    store = open_store(db_path(data_dir, run_id))

    with store.transaction():
        hit = store.search(embedding)
        if hit is not None and hit.cosine >= threshold:
            return MatchResult(hit.key, False, hit.cosine, store.distinct_count(), sub_canon)
        key = store.add_auto(embedding, quality=quality, sub_canon=sub_canon, prefix="p")
        best = hit.cosine if hit is not None else None
        return MatchResult(key, True, best, store.distinct_count(), sub_canon)


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


def merge(data_dir: Path, run_id: str, keep: str, drop: str) -> tuple[bool, int]:
    """Fold ``drop`` into ``keep`` (a *duplicate* correction).

    Returns ``(merged, galleryN)``; ``merged`` is False (and the count
    unchanged) when a cannot-link constraint or an unknown key blocks it.
    """
    store = open_store(db_path(data_dir, run_id))
    with store.transaction():
        merged = store.merge(keep, drop)
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
