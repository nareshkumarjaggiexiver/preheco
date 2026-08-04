"""The per-run guest gallery: one SQLite file per run, matched brute-force.

Thin policy layer over :class:`app.store.VectorStore`.  The store owns the
mechanics (cosine scan, float32 BLOB rows, monotonic keys, merge/split/remove);
this module owns the *guest-counting policy*: mint ``p#####`` keys, treat
``cosine >= threshold`` as a re-sighting, tag sub-canon faces, and expose the
operator corrections the runner applies from the feedback loop.

Concurrency: the match-then-insert decision runs inside one IMMEDIATE
transaction (see :meth:`VectorStore.begin_immediate`) so two concurrent /match
calls can never both insert the same brand-new person.

Files: ``data/gallery-<runId>.db``.  A run resets its gallery at start, so the
unique count always begins at zero.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .store import VectorStore

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


def reset(data_dir: Path, run_id: str) -> None:
    """Delete any existing gallery for the run so it starts empty."""
    db_path(data_dir, run_id).unlink(missing_ok=True)


def count(data_dir: Path, run_id: str) -> int:
    """Distinct guest count for a run (0 when the gallery does not exist yet)."""
    path = db_path(data_dir, run_id)
    if not path.exists():
        return 0
    with VectorStore(path) as store:
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
    path = db_path(data_dir, run_id)

    with VectorStore(path) as store:
        store.begin_immediate()
        hit = store.search(embedding)
        if hit is not None and hit.cosine >= threshold:
            return MatchResult(hit.key, False, hit.cosine, store.distinct_count(), sub_canon)
        key = store.add_auto(embedding, quality=quality, sub_canon=sub_canon, prefix="p")
        best = hit.cosine if hit is not None else None
        return MatchResult(key, True, best, store.distinct_count(), sub_canon)


def merge(data_dir: Path, run_id: str, keep: str, drop: str) -> tuple[bool, int]:
    """Fold ``drop`` into ``keep`` (a *duplicate* correction).

    Returns ``(merged, galleryN)``; ``merged`` is False (and the count
    unchanged) when a cannot-link constraint or an unknown key blocks it.
    """
    with VectorStore(db_path(data_dir, run_id)) as store:
        store.begin_immediate()
        merged = store.merge(keep, drop)
        return merged, store.distinct_count()


def split(data_dir: Path, run_id: str, a: str, b: str) -> int:
    """Record a *false-match* do-not-merge constraint; returns galleryN.

    Both keys keep their templates, so the distinct count is unchanged — the
    effect is forward-looking (a later :func:`merge` of the pair is refused).
    """
    with VectorStore(db_path(data_dir, run_id)) as store:
        store.begin_immediate()
        store.split(a, b)
        return store.distinct_count()


def remove(
    data_dir: Path, run_id: str, person_key: str
) -> tuple[list[tuple[bytes, float | None, int]], int]:
    """Lift a person out of the gallery (for *mark-staff*).

    Returns ``(templates, galleryN)`` where ``templates`` are the removed rows
    (blob, quality, sub_canon) for the caller to re-home in the staff store.
    """
    with VectorStore(db_path(data_dir, run_id)) as store:
        store.begin_immediate()
        templates = store.remove(person_key)
        return templates, store.distinct_count()
