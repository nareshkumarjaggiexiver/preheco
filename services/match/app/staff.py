"""The per-site staff whitelist store: persistent, checked before the gallery.

Venue and catering staff recur across every event at a site, so their
enrolment lives in one persistent file per site — ``data/staff-<siteId>.db`` —
not per run.  It is the same :class:`app.store.VectorStore` the guest gallery
uses; the only differences are lifetime (persistent) and keys (an
operator-chosen ``staff_id``, or an ``anon#####`` key when a track is marked
staff without a roster identity).

Two entry points serve the professional enrolment flow of CONTRACTS.md:

* :func:`enrol` — the runner's ENROL MODE posts the best N face samples of one
  walk-through; they become that staff member's templates.
* :func:`check` — COUNT MODE asks "is this face staff?" *before* the guest
  gallery, so a staff hit is tagged and kept out of the unique guest count.

Staff are excluded from the guest count but never from tracking (CONTRACTS.md):
suppression would corrupt track association and hide occlusions.

Like the gallery, a site store is opened once per process and scanned from
memory (:func:`app.store.open_store`) — :func:`check` runs on every face of
every frame, so a connect-and-full-scan per call was the single hottest cost
in the pipeline.  The ``path.exists()`` guards stay: a site with no enrolments
must not get an empty database file created just by being asked about.
"""

import re
from pathlib import Path

import numpy as np

from .store import Neighbour, open_store

_SITE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class BadSiteIdError(ValueError):
    """Raised when a siteId is unsafe to use as part of a filename."""


def db_path(data_dir: Path, site_id: str) -> Path:
    """Return the staff store path for a site, validating the id first."""
    if not _SITE_ID_RE.match(site_id):
        raise BadSiteIdError(f"siteId must match {_SITE_ID_RE.pattern!r}")
    return data_dir / f"staff-{site_id}.db"


def enrol(
    data_dir: Path,
    site_id: str,
    staff_id: str,
    samples: list[dict],
) -> int:
    """Store one staff member's enrolment samples; return the sample count.

    ``samples`` is the runner's best-by-quality shortlist, each
    ``{"embedding": [...], "quality": <px>}``.  Re-enrolment **supersedes** the
    member's prior templates (they are removed first) so the store always holds
    the latest best-N per person and the reported ``sampleCount`` is exact —
    rather than growing without bound across repeated walk-throughs.
    """
    if not staff_id:
        raise ValueError("staffId is required to enrol")
    store = open_store(db_path(data_dir, site_id))
    with store.transaction():
        store.remove(staff_id)  # supersede any prior enrolment of this member
        for s in samples:
            store.add(
                staff_id,
                s["embedding"],
                quality=s.get("quality"),
                sub_canon=bool(s.get("subCanon", False)),
            )
        return store.count_for(staff_id)


def check(
    data_dir: Path,
    site_id: str,
    embedding: list[float],
    threshold: float,
) -> Neighbour | None:
    """Return the matching staff member (cosine >= threshold), else None.

    A missing store (no staff enrolled at this site yet) is not an error — it
    simply means "not staff", so every guest is counted.
    """
    path = db_path(data_dir, site_id)
    if not path.exists():
        return None
    store = open_store(path)
    with store.reading():
        hit = store.search(embedding)
    if hit is not None and hit.cosine >= threshold:
        return hit
    return None


def absorb(
    data_dir: Path,
    site_id: str,
    staff_id: str | None,
    templates: list[tuple[bytes, float | None, int]],
) -> tuple[str, int]:
    """Add templates lifted from a guest gallery into the staff store.

    Backs *mark-staff*: ``templates`` are the ``(blob, quality, sub_canon)``
    rows :func:`app.gallery.remove` returned.  When ``staff_id`` is given the
    templates join that roster member; otherwise a fresh anonymous key
    (``anon#####``) is minted so the person is still excluded from future guest
    counts without claiming a named identity.  Returns ``(staffKey, count)``.
    """
    store = open_store(db_path(data_dir, site_id))
    with store.transaction():
        key = staff_id or store.mint_key("anon")
        for blob, quality, sub_canon in templates:
            vec = np.frombuffer(blob, dtype=np.float32)
            store.add(key, vec, quality=quality, sub_canon=bool(sub_canon))
        return key, store.count_for(key)


def purge(data_dir: Path, site_id: str, staff_ids: list[str]) -> dict[str, int]:
    """Erase staff members' templates from the site store; return removed counts.

    The erasure half of the consent story: the planner tombstones a deleted
    roster member, the runner relays the ids here, and every template keyed by
    each id is removed from ``staff-<siteId>.db``.  Returns
    ``{staff_id: templates_removed}`` — a zero means the id had no templates
    (never enrolled, or already purged), which is success, not an error:
    erasure is idempotent.
    """
    removed: dict[str, int] = {}
    path = db_path(data_dir, site_id)
    if not path.exists():
        # No store for this site yet — nothing to erase, every id trivially done.
        return {sid: 0 for sid in staff_ids}
    store = open_store(path)
    with store.transaction():  # one write txn for the whole purge
        for sid in staff_ids:
            removed[sid] = len(store.remove(sid))
    return removed
