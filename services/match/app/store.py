"""VectorStore — a SQLite-backed embedding store with a brute-force cosine scan.

This is the reusable core behind both the per-run guest gallery
(``data/gallery-<runId>.db``) and the persistent per-site staff whitelist
(``data/staff-<siteId>.db``): the two differ only in how keys are minted
(guests get an auto-incrementing ``p#####`` key; staff carry an operator-chosen
``staff_id``) and in lifetime, not in mechanics.

Why brute force, and what replaces it
-------------------------------------
At POC scale a store holds hundreds — a busy roster a few thousand — of 128-d
vectors.  A full cosine scan over that with numpy costs well under a
millisecond and is *exact*, dependency-free and trivially correct.  The
drop-in indexed backend past ~5k vectors is **sqlite-vec**: its ``vec0``
virtual table stores the same ``float32`` BLOB this store already writes and
answers ``k``-nearest-neighbour queries with an ANN index, so the migration is
"create a ``vec0`` shadow table, copy the ``vec`` column across, swap
:meth:`search`'s scan for a ``MATCH`` query" — the row schema, the BLOB
encoding and every caller stay put.  Do that only when profiling shows the
scan is a bottleneck (project hard rule: measure before you optimise).

Open once, scan in memory
-------------------------
The scan itself was never the cost.  Profiling the POC hot path (one /match
per face per frame, up to 45/s with a staff store in play) showed the money
going to *re-opening*: every call used to ``sqlite3.connect`` + run the schema
script + ``SELECT`` every row + rebuild the numpy matrix from BLOBs, two or
three times per face.  So a store is now opened **once per file** and kept
open (:func:`open_store`), with the key list and the ``(n, dim)`` float32
matrix held in memory and updated **write-through** on every mutation.  SQLite
remains the durable record — nothing is acknowledged before it is committed —
but a match is a matrix-vector multiply against RAM, not disk I/O.

Because the cached store is shared by FastAPI's request threadpool, every
connection is opened ``check_same_thread=False`` and *all* access (read and
write) is serialised on the store's own lock; :meth:`transaction` is the write
path and rolls the in-memory index back (by dropping it, so the next read
reloads from disk) whenever the SQL transaction rolls back.

Storage
-------
* ``vectors``     one row per stored template: ``key``, the ``float32`` BLOB,
                  its ``dim``, the capture ``quality`` (face width px) and a
                  ``sub_canon`` flag.  A person/staff member owns SEVERAL rows:
                  staff from enrolment, guests from accumulating views as they
                  are re-sighted (:func:`app.gallery.match`), either from a
                  merge.  :meth:`prune_redundant` bounds how many, dropping the
                  least distinctive view so an identity keeps the widest spread
                  it can hold.  :meth:`prune_to_cap` is the older quality-based
                  eviction, kept for callers that genuinely want "the best N
                  photographs"; it is NOT used on the enrolment path any more,
                  because quality is face width and face width is distance —
                  see :meth:`prune_redundant` for the measurement.
                  Since 2026-09-24 a template also carries what the embed
                  service's attribute head read off THAT face — ``gender``,
                  ``gender_p``, ``age`` — and ``feat_norm``, the L2 norm of
                  the raw ArcFace feature (a quality proxy: blurred, occluded
                  and turned-away faces embed short).  All nullable; NULL is
                  "not measured", never a value.
* ``body_sightings`` one row per guest /match call that carried the
                  sighting's containing PERSON box: ``h``, ``w``, ``y_bottom``
                  and ``frame_h`` in raw detector pixels, plus ``face_w`` (the
                  call's ``quality``, the face box width) so a box that is
                  only a head-and-shoulders can be told from a standing body
                  by its height in face widths.  NOT a template and
                  not pruned with them — it is the raw material for the
                  review queue's stature estimate (:func:`app.gallery.
                  stature_ratios`), which needs every standing box in the run
                  to fit the camera's perspective, and a person's median box
                  height against that fit needs more than the five views a
                  template cap keeps.  Re-keyed by :meth:`merge`, deleted by
                  :meth:`remove`, so a row always names a live identity.
                  Since 2026-09-24 (night) the row also carries the
                  sighting's APPEARANCE — the torso descriptor, the head
                  descriptor (40 floats) and the beard reading (4) — so the
                  review queue judges an identity's clothing, headwear and
                  beard on every sighting it had, not on the five its
                  template cap kept (:func:`app.gallery.review_duplicates`).
                  NULL = not measured.
* ``cannot_link`` "these are two different people" constraints, stored
                  order-independent.  Written by an operator's *false-match*
                  correction and by the runner asserting CO-PRESENCE (two faces
                  at different positions in one frame are two people).
                  :meth:`merge` refuses to fold a constrained pair — that is
                  how "raise the pair's internal distance, no auto-merge later"
                  is realised in a brute-force store — and the policy layer
                  reads the same rows to withhold both operator-facing
                  "probably a duplicate" banners (:func:`app.gallery.
                  _overlap_after_write` and :func:`app.gallery._near_miss`).
                  It is deliberately the ONE record of "known different".
* ``manual``      operator-attested people who were counted but never matched
                  (a *missed* correction).  They own no vector on purpose —
                  there is no face to store — so they can never be matched
                  into, but they DO count towards :meth:`distinct_count`.  The
                  ``m#####`` key prefix is what makes a human-added person
                  distinguishable from an automatically detected ``p#####``
                  one in any later audit.
* ``meta``        small integer counters (the monotonic key sequence), so keys
                  never collide even after merges and removals shrink the
                  distinct-person count.

Vectors are L2-normalised on the way in, so the dot product IS the cosine
similarity and :meth:`search` is a single matrix-vector multiply.
"""

import contextlib
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

#: SQLite schema for one store file.  ``IF NOT EXISTS`` makes open idempotent.
#: ``appearance`` (2026-08-06) is the optional torso-appearance descriptor
#: captured WITH the face template (float32[48] or float32[64] BLOB — v2 or
#: v3, see :mod:`app.appearance`; NULL when the runner could not measure one).
#: ``gender``/``gender_p``/``age``/``feat_norm`` (2026-09-24) are the embed
#: service's per-face attribute readings and raw-feature norm, NULL when not
#: measured.  The nullable columns sit LAST, in the order they were added, so a
#: freshly created table matches the column order the ALTERs in __init__
#: produce on an older file.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL,
    vec        BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    quality    REAL,
    sub_canon  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    appearance BLOB,
    gender     TEXT,
    gender_p   REAL,
    age        REAL,
    feat_norm  REAL
);
CREATE INDEX IF NOT EXISTS idx_vectors_key ON vectors(key);
CREATE TABLE IF NOT EXISTS body_sightings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL,
    h          REAL NOT NULL,
    w          REAL NOT NULL,
    y_bottom   REAL NOT NULL,
    frame_h    INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    face_w     REAL,
    appearance BLOB,
    head       BLOB,
    beard      BLOB
);
CREATE INDEX IF NOT EXISTS idx_body_sightings_key ON body_sightings(key);
CREATE TABLE IF NOT EXISTS cannot_link (
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS manual (
    key        TEXT PRIMARY KEY,
    note       TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS store_meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""


#: Nullable columns ``vectors`` has gained since it first shipped, in the
#: order they were added, with their types — the in-place migration list.  A
#: file from before any of them opens, gets the missing ones ALTERed in, and
#: its old rows read NULL for each: "not measured", which every reader treats
#: as absent (never zero, never a default sex or age).
_VECTOR_COLUMNS_ADDED = (
    ("appearance", "BLOB"),   # 2026-08-06 torso descriptor
    ("gender", "TEXT"),       # 2026-09-24 attribute head: "M" | "F"
    ("gender_p", "REAL"),     # ... probability of that reported sex, 0..1
    ("age", "REAL"),          # ... estimated age in years
    ("feat_norm", "REAL"),    # ... L2 norm of the raw embedding feature
)
#: Same for ``body_sightings``, which shipped without ``face_w`` for a day
#: and without the sighting's appearance readings until that night.
_BODY_COLUMNS_ADDED = (
    ("face_w", "REAL"),       # 2026-09-24 evening: the sighting's face width px
    ("appearance", "BLOB"),   # 2026-09-24 night: torso descriptor, float32[48|64]
    ("head", "BLOB"),         # ... head descriptor, float32[40]
    ("beard", "BLOB"),        # ... beard reading, float32[4]
)


class BodySighting(NamedTuple):
    """One row of ``body_sightings``: a sighting's containing person box.

    ``face_w`` and ``created_at`` ride along for the stature reader — the
    first to reject a head-and-shoulders box, the second to count how many
    distinct MOMENTS an identity was seen standing (eight consecutive frames
    are half a second of one pose, not eight measurements).
    """

    key: str
    h: float
    w: float
    y_bottom: float
    frame_h: int
    face_w: float | None = None
    created_at: str | None = None


class SightingEvidence(NamedTuple):
    """One body-log row's appearance readings, for the review queue.

    ``created_at`` is the write time (ISO text); ``appearance`` (the torso
    descriptor, 48 or 64 long), ``head`` (40) and ``beard`` (4) are float32
    arrays, each None when that sighting did not carry it — absent is not
    zero.  A row with none of the three is never returned.
    """

    key: str
    created_at: str
    appearance: np.ndarray | None
    head: np.ndarray | None = None
    beard: np.ndarray | None = None


#: Which embedder's vectors this process writes and expects. The catalog id
#: (pipeline.json models.embed), not a filename — one string, shared by every
#: store this service opens. Overridable per deploy so a future stack running
#: a different embedder names itself, and every store it touches is stamped.
# Empty means unset, the service-wide convention: compose renders ${VAR-} as
# an empty string, and an empty id would stamp every store with '' and move
# staff lookups to staff-<site>--.db — every enrolled member billed as a
# guest, silently (review finding, 2026-08-14).
EMBEDDER_ID = os.environ.get("HECO_EMBEDDER_ID", "").strip() or "sface-2021dec"


class EmbedderMismatchError(RuntimeError):
    """A store holds vectors from a DIFFERENT embedder than this process runs.

    The failure this makes loud (docs/planning/15-model-configurations.md M3):
    two embedders with the SAME dimension pass every dim check and then match
    garbage SILENTLY — cosine between vectors from different spaces is noise
    wearing a decimal point, and against a persistent staff store it produces
    confidently wrong bills. Refusal names both identities and the way out.
    """


@dataclass
class Neighbour:
    """The nearest stored template to a query: whose it is and how close."""

    key: str
    cosine: float


def as_unit(embedding: list[float] | np.ndarray) -> np.ndarray:
    """Return ``embedding`` as an L2-normalised ``float32`` vector.

    SFace embeddings arrive unit-length already; re-normalising is a cheap
    no-op that keeps the cosine maths correct for any caller (and for the
    synthetic vectors the tests feed in).  Raises ValueError on a degenerate
    vector (too short, or all-zero — no direction to compare).
    """
    v = np.asarray(embedding, dtype=np.float32)
    if v.ndim != 1 or v.size < 8:
        raise ValueError("embedding must be a flat vector of at least 8 floats")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("embedding must not be the zero vector")
    return v / n


def _pair(a: str, b: str) -> tuple[str, str]:
    """Order a key pair so a constraint is stored once, lookup-order-free."""
    return (a, b) if a <= b else (b, a)


class VectorStore:
    """A single embedding store file: SQLite on disk, the scan matrix in RAM.

    Two ways to use it.  Long-lived (the service path) — :func:`open_store`
    hands back the one cached instance per file and writes go through a
    transaction::

        store = open_store(path)
        with store.transaction():
            hit = store.search(vec)
            if hit is None or hit.cosine < threshold:
                key = store.add_auto(vec, quality=71.0)

    Short-lived (tests, one-off tools) — the context manager owns the whole
    connection and closes it on the way out::

        with VectorStore(path) as store:
            ...

    ``cached=True`` is what :func:`open_store` sets: it stops :meth:`__exit__`
    from closing a connection other callers still hold.
    """

    def __init__(self, path: Path, timeout: float = 5.0, cached: bool = False) -> None:
        """Open ``path`` (creating the schema if new) with a busy timeout."""
        self.path = Path(path)
        self.cached = cached
        # Shared by FastAPI's request threadpool when cached, so the connection
        # must not police its creating thread; `_lock` provides the real safety.
        self.conn = sqlite3.connect(self.path, timeout=timeout, check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA busy_timeout = 5000")
        # WAL + synchronous=NORMAL: every new guest is an INSERT + COMMIT on the
        # frame loop's critical path, and a full fsync per commit costs more
        # than the whole cosine scan.  NORMAL still survives a process crash
        # (the WAL is replayed); only an OS/power failure can lose the last few
        # commits — an acceptable trade for a gallery that is deleted at the end
        # of the run anyway, and for a staff store that can be re-enrolled.
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(_SCHEMA)
        # Migrate OLDER files in place (2026-08-06, extended 2026-09-24):
        # ``CREATE TABLE IF NOT EXISTS`` never touches an existing table, so a
        # gallery or staff store created before a column existed opens without
        # it.  ADD COLUMN is the entire migration — each new column defaults to
        # NULL, and NULL is precisely what "not measured" means everywhere
        # (absent is not zero, the codebase-wide convention), so a pre-existing
        # file keeps working, no veto fires on its rows, and no review
        # exclusion can read a sex or an age into a template that never had
        # one.  ``body_sightings`` is created whole by CREATE IF NOT EXISTS
        # (an empty table is "nothing measured") and gets the same ADD COLUMN
        # treatment for a column it shipped without.
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(vectors)")}
        for name, sql_type in _VECTOR_COLUMNS_ADDED:
            if name not in cols:
                self.conn.execute(f"ALTER TABLE vectors ADD COLUMN {name} {sql_type}")
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(body_sightings)")}
        for name, sql_type in _BODY_COLUMNS_ADDED:
            if name not in cols:
                self.conn.execute(f"ALTER TABLE body_sightings ADD COLUMN {name} {sql_type}")
        # THE EMBEDDER GUARD (doc 15 M3). Every store knows WHOSE vectors it
        # holds, stamped at first open and checked at every open after:
        #   * a new or legacy-unstamped file ADOPTS this process's embedder —
        #     legacy adoption is correct, not lenient, because every store in
        #     existence predates the second embedder by construction;
        #   * a stamped file that disagrees REFUSES, loudly, naming both ids
        #     and the way out. A same-dimension different embedder passes the
        #     dim check and matches garbage silently — this is the only gate
        #     between that and a confidently wrong bill.
        stored = self.conn.execute(
            "SELECT v FROM store_meta WHERE k = 'embedderId'"
        ).fetchone()
        if stored is None:
            # OR IGNORE + re-read: two processes adopting one unstamped
            # legacy file must both land on ONE stamp, not an IntegrityError.
            self.conn.execute(
                "INSERT OR IGNORE INTO store_meta (k, v) VALUES ('embedderId', ?)",
                (EMBEDDER_ID,),
            )
            self.conn.commit()
            stored = self.conn.execute(
                "SELECT v FROM store_meta WHERE k = 'embedderId'"
            ).fetchone()
            if stored[0] != EMBEDDER_ID:
                self.conn.close()
                raise EmbedderMismatchError(
                    f"{self.path.name} was stamped `{stored[0]}` by a concurrent open; "
                    f"this process runs `{EMBEDDER_ID}` — same refusal, same way out."
                )
            self.embedder_id = stored[0]
        elif stored[0] != EMBEDDER_ID:
            self.conn.close()
            raise EmbedderMismatchError(
                f"{self.path.name} holds vectors from embedder `{stored[0]}`; this "
                f"process runs `{EMBEDDER_ID}`. Cosines across embedding spaces are "
                "noise — refuse rather than bill on them. Either point the stack "
                f"back at `{stored[0]}`, or use a fresh store for `{EMBEDDER_ID}` "
                "(staff re-enrolment is a real operational step, not a file rename)."
            )
        else:
            self.embedder_id = stored[0]
        # In-memory index, loaded lazily and kept write-through afterwards.
        # None means "not loaded / invalidated" — the next read reloads it.
        self._keys: list[str] | None = None
        self._ids: list[int] = []  # SQLite rowid per index row, parallel to _keys
        self._manual: list[str] = []
        self._buf: np.ndarray | None = None  # (capacity, dim) float32
        self._n = 0  # rows of _buf in use
        self._dim = 0

    # ---------------------------------------------------------- lifecycle

    def __enter__(self) -> "VectorStore":
        """Enter the context; the store is already open."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Commit on clean exit, roll back on error; close if we own the file."""
        with self._lock:
            try:
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
                    self._invalidate()
            finally:
                if not self.cached:
                    self.conn.close()

    def close(self) -> None:
        """Commit and close for good (cached stores, at run end or sweep)."""
        with self._lock:
            with contextlib.suppress(sqlite3.Error):
                self.conn.commit()
            self.conn.close()
            self._invalidate()

    def begin_immediate(self) -> "VectorStore":
        """Start an IMMEDIATE transaction (write lock now) for search+insert.

        Kept for the short-lived ``with VectorStore(...)`` form; the cached
        service path uses :meth:`transaction`, which additionally repairs the
        in-memory index when the SQL transaction rolls back.
        """
        self.conn.execute("BEGIN IMMEDIATE")
        return self

    @contextlib.contextmanager
    def transaction(self):
        """Serialise + wrap one unit of work in an IMMEDIATE transaction.

        Holds the store lock for the whole block, so the read-then-write window
        a match decision needs cannot interleave with another request's writer.
        On any exception the SQL transaction is rolled back AND the in-memory
        index is dropped, because a partially applied write-through would
        otherwise leave RAM claiming rows the database no longer has.
        """
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self
            except BaseException:
                self.conn.rollback()
                self._invalidate()
                raise
            self.conn.commit()

    @contextlib.contextmanager
    def reading(self):
        """Serialise a read-only unit of work (no transaction needed).

        Reads answer from the in-memory index, but still need the lock: a
        concurrent :meth:`transaction` may be appending to it.
        """
        with self._lock:
            yield self

    # ------------------------------------------------------------- index

    def _invalidate(self) -> None:
        """Forget the in-memory index; the next read reloads it from SQLite."""
        self._keys = None
        self._ids = []
        self._buf = None
        self._n = 0
        self._manual = []

    def _index(self) -> None:
        """Load the key list + scan matrix from SQLite if not already held.

        The rowid of each template is carried alongside its key, so a caller
        that wants to delete *particular* templates (:meth:`prune_to_cap`) can
        write the removal through to the resident matrix instead of dropping
        the whole index and paying for a reload on the match hot path.
        """
        if self._keys is not None:
            return
        rows = self.conn.execute("SELECT id, key, vec, dim FROM vectors ORDER BY id").fetchall()
        self._keys = [r[1] for r in rows]
        self._ids = [int(r[0]) for r in rows]
        self._manual = [
            r[0] for r in self.conn.execute("SELECT key FROM manual").fetchall()
        ]
        if not rows:
            self._buf, self._n, self._dim = None, 0, 0
            return
        self._dim = int(rows[0][3])
        mat = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32)
        self._buf = mat.reshape(len(rows), self._dim).copy()  # writable, growable
        self._n = len(rows)

    def _append_row(self, key: str, vec: np.ndarray, row_id: int) -> None:
        """Write-through one new template into the in-memory index.

        Capacity doubles rather than reallocating per insert, so a run that
        counts N guests does O(N) copying in total, not O(N²).
        """
        self._index()
        if self._buf is None:
            self._dim = int(vec.size)
            self._buf = np.empty((8, self._dim), dtype=np.float32)
            self._n = 0
        if self._n == self._buf.shape[0]:
            bigger = np.empty((self._buf.shape[0] * 2, self._dim), dtype=np.float32)
            bigger[: self._n] = self._buf[: self._n]
            self._buf = bigger
        self._buf[self._n] = vec
        self._n += 1
        self._keys.append(key)
        self._ids.append(int(row_id))

    def _keep_mask(self, mask: list[bool]) -> None:
        """Write-through a row-level deletion described by a keep/drop mask."""
        keep = np.array(mask, dtype=bool)
        self._keys = [k for k, m in zip(self._keys, mask, strict=True) if m]
        self._ids = [i for i, m in zip(self._ids, mask, strict=True) if m]
        if self._buf is not None and self._n:
            kept = self._buf[: self._n][keep]
            self._buf = kept.copy() if kept.size else None
            self._n = int(keep.sum())

    def _drop_rows(self, key: str) -> None:
        """Write-through the removal of every row belonging to ``key``."""
        self._index()
        if not self._keys:
            return
        self._keep_mask([k != key for k in self._keys])

    def _drop_ids(self, doomed: set[int]) -> None:
        """Write-through the removal of specific template rowids."""
        self._index()
        if not self._keys:
            return
        self._keep_mask([i not in doomed for i in self._ids])

    # ------------------------------------------------------------- reads

    def search(
        self, embedding: list[float] | np.ndarray, exclude: set[str] | None = None
    ) -> Neighbour | None:
        """Return the nearest stored template by cosine, or None if empty.

        Rows are unit vectors, so the dot product is the cosine and the whole
        scan is one ``(n, dim) @ (dim,)`` multiply against the resident matrix
        — no SQLite round trip, no BLOB rebuild.  The best row's key is
        returned even when several templates share it (multi-template person).

        ``exclude`` removes whole identities from the scan before the argmax,
        for the one caller that has already PROVEN the probe cannot be them:
        the runner's same-frame guard, which re-asks about a face that matched
        a key another body in the same frame was simultaneously wearing (see
        ``RunLoop._split_same_key``).  Masking rather than re-querying keeps
        this on the same single multiply, and returning the runner-up KEY —
        not merely "no" — matters: the second man may legitimately be an
        already-known guest, and forcing a blind mint would split him instead.
        None when every remaining row is excluded, which the caller reads as
        "nobody else is close" and mints.
        """
        self._index()
        if not self._n:
            return None
        q = as_unit(embedding)
        if q.size != self._dim:
            raise ValueError(f"embedding dim {q.size} != store dim {self._dim}")
        sims = self._buf[: self._n] @ q
        if exclude:
            live = np.fromiter(
                (k not in exclude for k in self._keys[: self._n]),
                dtype=bool,
                count=self._n,
            )
            if not live.any():
                return None
            sims = np.where(live, sims, -np.inf)
        i = int(np.argmax(sims))
        return Neighbour(key=self._keys[i], cosine=float(sims[i]))

    def runner_up(self, embedding: list[float] | np.ndarray, key: str) -> Neighbour | None:
        """Best-scoring template belonging to some key OTHER than ``key``.

        The second opinion :meth:`search` deliberately does not give: "how close
        was this probe to the nearest *rival* identity?".  A caller deciding
        whether to enrol a probe as a new template needs it, because a probe
        that sits almost equally close to two identities is the one sighting
        that must NOT become a template — storing it builds a bridge between
        two people (see :func:`app.gallery.match`).  Returns None when no other
        key has a template.

        Cost: one argsort of the already-computed similarity vector, then a walk
        down it that stops at the first foreign key — at most (templates per
        identity) steps.  This runs only on the enrol-decision path, never on
        the plain match path.  If profiling ever puts the argsort on the budget,
        the vectorised form is to keep an int32 key-code column parallel to the
        scan matrix and take ``sims[codes != code].max()``.
        """
        self._index()
        if not self._n:
            return None
        q = as_unit(embedding)
        if q.size != self._dim:
            raise ValueError(f"embedding dim {q.size} != store dim {self._dim}")
        sims = self._buf[: self._n] @ q
        for i in np.argsort(-sims):
            if self._keys[i] != key:
                return Neighbour(key=self._keys[i], cosine=float(sims[i]))
        return None

    def distinct_count(self) -> int:
        """Distinct people in the store: matched keys plus manual additions.

        Manual (operator-attested) entries have no template, so they are
        invisible to :meth:`search` but are real people and must show up in the
        unique count — that is the whole point of the *missed* correction.
        """
        self._index()
        return len(set(self._keys)) + len(self._manual)

    def keys(self) -> list[str]:
        """All distinct keys with templates, ascending — small at POC scale."""
        self._index()
        return sorted(set(self._keys))

    def manual_keys(self) -> list[str]:
        """Keys of operator-attested people (no template), ascending."""
        self._index()
        return sorted(self._manual)

    def vectors_for(self, key: str) -> list[np.ndarray]:
        """Every stored template for one key as float32 arrays."""
        rows = self.conn.execute(
            "SELECT vec, dim FROM vectors WHERE key = ?", (key,)
        ).fetchall()
        return [np.frombuffer(v, dtype=np.float32).reshape(d) for v, d in rows]

    def appearances_for(self, key: str) -> list[np.ndarray]:
        """Every stored torso-appearance descriptor for one key (NULLs skipped).

        The read half of the advisory tie-breaker: :func:`app.gallery.match`
        compares an incoming sighting's descriptor against THESE to compute
        ``appearanceSim`` and to decide the enrolment veto.  NULL rows —
        templates enrolled before the column existed, or sightings whose torso
        the runner could not measure (no person box, crop under 24 px, fewer
        than 100 unmasked pixels) — are skipped rather than decoded as zeros,
        because a zero histogram would score intersection 0.0 against
        everything and read as a maximal CLASH: absent is not zero, and
        under-counting is this pipeline's dominant failure mode.

        Answered from SQLite through ``idx_vectors_key``, not from the
        resident index, which deliberately does not hold appearance at all —
        the scan matrix is the face-only cosine machine and stays that way
        (see :meth:`add` for the two-white-shirts measurement behind that).
        """
        rows = self.conn.execute(
            "SELECT appearance FROM vectors WHERE key = ? AND appearance IS NOT NULL"
            " ORDER BY id ASC",
            (key,),
        ).fetchall()
        return [np.frombuffer(r[0], dtype=np.float32) for r in rows]

    def appearance_rows_for(self, key: str) -> list[tuple[str, np.ndarray]]:
        """Every stored torso descriptor for one key WITH its write time.

        :meth:`appearances_for` without the timestamp is enough to compare two
        sightings; the review queue's clothing set-aside also has to know the
        reads were taken at different moments, because five reads from one
        second of one crossing agree with each other trivially and prove
        nothing about how consistently this person's clothing reads.
        """
        rows = self.conn.execute(
            "SELECT created_at, appearance FROM vectors"
            " WHERE key = ? AND appearance IS NOT NULL ORDER BY id ASC",
            (key,),
        ).fetchall()
        return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]

    def attributes_for(self, key: str) -> list[tuple[str | None, float | None, float | None]]:
        """Each template's ``(gender, gender_p, age)`` for one key, NULLs kept.

        The raw material for the review queue's per-identity sex and age
        (:func:`app.gallery.identity_gender`, :func:`app.gallery.identity_age`).
        Rows are returned even when every field is NULL — a template written
        by a runner with no attribute model, or before the columns existed —
        so a caller can see how many views it is judging from; it is the
        caller's job to aggregate over the non-NULL ones and answer None
        when there are none.  Never a default sex, never age 0.
        """
        rows = self.conn.execute(
            "SELECT gender, gender_p, age FROM vectors WHERE key = ? ORDER BY id ASC",
            (key,),
        ).fetchall()
        return [
            (
                None if g is None else str(g),
                None if p is None else float(p),
                None if a is None else float(a),
            )
            for g, p, a in rows
        ]

    def feat_norms_for(self, key: str) -> list[float]:
        """Every recorded raw-feature norm for one key's templates (NULLs skipped)."""
        rows = self.conn.execute(
            "SELECT feat_norm FROM vectors WHERE key = ? AND feat_norm IS NOT NULL"
            " ORDER BY id ASC",
            (key,),
        ).fetchall()
        return [float(r[0]) for r in rows]

    def body_sightings(self) -> list[BodySighting]:
        """Every body sighting in the store, oldest first.

        ALL of them, not one key's: the stature estimate first fits the
        camera's perspective — box height against box bottom-y — over every
        standing box the run produced, and only then reads one identity's
        boxes against that fit.  At POC scale this is a few thousand small
        rows, read once per review call, which is an operator's click and
        not the frame loop.  ``face_w`` is None on rows written before the
        column existed; ``created_at`` is the ISO timestamp of the write.
        """
        rows = self.conn.execute(
            "SELECT key, h, w, y_bottom, frame_h, face_w, created_at"
            " FROM body_sightings ORDER BY id ASC"
        ).fetchall()
        return [
            BodySighting(
                str(k), float(h), float(w), float(yb), int(fh),
                None if fw is None else float(fw), str(ts),
            )
            for k, h, w, yb, fh, fw, ts in rows
        ]

    def body_sightings_count(self, key: str) -> int:
        """How many body sightings one key has accumulated."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM body_sightings WHERE key = ?", (key,)
        ).fetchone()
        return int(row[0])

    def sighting_evidence(self) -> list[SightingEvidence]:
        """Every body-log row that carries an appearance reading, oldest first.

        The review queue's clothing, headwear and beard evidence, per
        SIGHTING, where the template column keeps at most the cap's five —
        and on run f0bfc5 those sat inside two seconds for 20 of 44
        identities (median span 2.0 s).  Rows with none of the three are
        skipped in SQL; a NULL column comes back as None, never as zeros.
        """
        rows = self.conn.execute(
            "SELECT key, created_at, appearance, head, beard FROM body_sightings"
            " WHERE appearance IS NOT NULL OR head IS NOT NULL OR beard IS NOT NULL"
            " ORDER BY id ASC"
        ).fetchall()

        def arr(blob):
            return None if blob is None else np.frombuffer(blob, dtype=np.float32)

        return [
            SightingEvidence(str(k), str(ts), arr(a), arr(h), arr(b))
            for k, ts, a, h, b in rows
        ]

    def count_for(self, key: str) -> int:
        """How many templates a single key owns.

        Answered from SQLite through ``idx_vectors_key`` rather than by counting
        the resident key list: since M1 this runs on EVERY match (the verdict
        reports how many views the identity now holds), and a Python-level scan
        of one key list costs ~27 us at 2.5k templates against ~7 us indexed —
        i.e. it grew with gallery size, which is the exact cost "open once, scan
        in memory" was introduced to remove.  Reads on this connection see the
        open transaction's own uncommitted writes, so the answer includes a
        template inserted moments ago in the same match.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE key = ?", (key,)
        ).fetchone()
        return int(row[0])

    def qualities_for(self, key: str) -> list[float | None]:
        """Capture quality (face width px) of each template a key owns.

        Ordered best first, with quality-less templates (``NULL``, e.g. a
        sample enrolled before quality was recorded) last — the same order
        :meth:`prune_to_cap` keeps.  A caller can therefore read the WORST held
        quality off the end of this list and decide, before paying for an
        insert, whether a new sighting would survive eviction at all.
        """
        rows = self.conn.execute(
            "SELECT quality FROM vectors WHERE key = ?"
            " ORDER BY (quality IS NULL) ASC, quality DESC, id ASC",
            (key,),
        ).fetchall()
        return [None if r[0] is None else float(r[0]) for r in rows]

    # ------------------------------------------------------------ writes

    def mint_key(self, prefix: str = "p", width: int = 5) -> str:
        """Return the next monotonic key (``p00001``, ``p00002``, …).

        Backed by a ``meta`` counter, not by ``COUNT`` — so a key is never
        reused after a merge or removal shrinks the distinct count, which would
        otherwise silently collide a fresh person onto a retired key.
        """
        cur = self.conn.execute("SELECT v FROM meta WHERE k = ?", (f"seq:{prefix}",)).fetchone()
        nxt = (cur[0] if cur else 0) + 1
        self.conn.execute(
            "INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (f"seq:{prefix}", nxt),
        )
        return f"{prefix}{nxt:0{width}d}"

    def add(
        self,
        key: str,
        embedding: list[float] | np.ndarray,
        quality: float | None = None,
        sub_canon: bool = False,
        appearance: list[float] | np.ndarray | None = None,
        attributes: dict | None = None,
        feat_norm: float | None = None,
    ) -> int:
        """Store one template under an explicit key; returns its rowid.

        The rowid is returned so a caller who later PROVES the write was wrong
        can retract exactly it (:meth:`forget_template`) rather than guessing
        at "the most recent row" — the runner's same-frame guard is that
        caller.  Callers with nothing to retract simply ignore it.

        ``appearance`` is the sighting's optional torso-appearance descriptor
        (48 floats for v2 — 12×3 Hue×Saturation bins plus 3 brightness bins —
        or 64 for v3, both L1-normalised; see :mod:`app.appearance`),
        stored as a float32 BLOB beside the face vector — beside, not inside:
        it never joins the resident scan matrix, because the cosine scan
        answers WHO and clothing must have no voice in that answer (the
        closest measured impostor pair on this camera, cosine 0.377, was two
        DIFFERENT men BOTH IN LIGHT SHIRTS — an appearance term in the scan
        would pull exactly such pairs together and merge two real guests).
        ``None`` stores NULL, which every reader treats as "not measured",
        never as a zero histogram.  Staff enrolment always passes ``None``:
        the staff store deliberately has no appearance handling.

        ``attributes`` is the embed service's reading of THIS face —
        ``{"gender": "M"|"F", "genderP": 0..1, "age": years}`` — and
        ``feat_norm`` the L2 norm of its raw feature before normalisation.
        Stored per template, beside the vector, for the same reason as the
        descriptor and with the same discipline: they never enter the scan
        matrix and never touch a verdict.  Their one reader is the review
        queue, which uses them to set aside pairs that cannot be one person
        (a man and a woman, a child and an adult).  ``None`` for either, or
        a missing field, stores NULL — a runner without an attribute model
        writes rows the review queue reads as "not measured".
        """
        v = as_unit(embedding)
        blob = None if appearance is None else np.asarray(appearance, dtype=np.float32).tobytes()
        gender = gender_p = age = None
        if attributes:
            g = attributes.get("gender")
            gender = None if g is None else str(g)
            gp = attributes.get("genderP")
            gender_p = None if gp is None else float(gp)
            a = attributes.get("age")
            age = None if a is None else float(a)
        norm = None if feat_norm is None else float(feat_norm)
        self._index()  # load BEFORE the insert, or the load would see it twice
        cur = self.conn.execute(
            "INSERT INTO vectors (key, vec, dim, quality, sub_canon, created_at, appearance,"
            " gender, gender_p, age, feat_norm)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key, v.tobytes(), v.size, quality, int(sub_canon), _now(), blob,
                gender, gender_p, age, norm,
            ),
        )
        row_id = int(cur.lastrowid)
        self._append_row(key, v, row_id)
        return row_id

    def add_auto(
        self,
        embedding: list[float] | np.ndarray,
        quality: float | None = None,
        sub_canon: bool = False,
        prefix: str = "p",
        appearance: list[float] | np.ndarray | None = None,
        attributes: dict | None = None,
        feat_norm: float | None = None,
    ) -> str:
        """Mint a fresh monotonic key, store the template under it, return it."""
        key = self.mint_key(prefix)
        self.add(key, embedding, quality, sub_canon, appearance, attributes, feat_norm)
        return key

    def add_body_sighting(
        self, key: str, h: float, w: float, y_bottom: float, frame_h: int,
        face_w: float | None = None,
        appearance: list[float] | np.ndarray | None = None,
        head: list[float] | np.ndarray | None = None,
        beard: list[float] | np.ndarray | None = None,
    ) -> int:
        """Record one sighting's containing PERSON box under ``key``; its rowid.

        Written on EVERY guest /match call that carried a body, matched or
        minted, enrolled or not — this is a sighting log, not a template.
        The review queue's stature estimate needs the run's whole population
        of standing boxes to fit the camera's perspective, and one identity's
        median over many boxes to be robust against a single mid-stride or
        half-occluded frame; five capped templates would give it neither.
        Raw detector pixels, so the fit is in the geometry the detector saw.
        ``face_w`` is the face box width of the sighting (the /match call's
        ``quality``), kept so the stature reader can reject a box that is
        only a head-and-shoulders: run f0bfc5's p00052 had 15 consecutive
        boxes 4.2 face widths tall (a waist-up crop through an occlusion)
        against her 11-face-width standing boxes, and the median over those
        frames read 0.59 of adult height.  Never enters the scan matrix;
        never touches a verdict.  The rowid comes back so the runner's
        same-frame guard can retract the row if the sighting turns out to
        belong to a different body (:meth:`forget_body_sighting`).

        ``appearance``, ``head`` and ``beard`` are the sighting's readings
        (torso descriptor, head descriptor, beard fractions), stored as
        float32 BLOBs; None stores NULL.  They ride on this row (the torso
        also on any template the call wrote) because a template is written
        on a handful of calls and the review queue judges an identity's
        appearance across all of them — and on this row the same-frame
        guard's retraction takes them too.
        """
        def blob(v):
            return None if v is None else np.asarray(v, dtype=np.float32).tobytes()

        cur = self.conn.execute(
            "INSERT INTO body_sightings (key, h, w, y_bottom, frame_h, created_at, face_w,"
            " appearance, head, beard)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key, float(h), float(w), float(y_bottom), int(frame_h), _now(),
                None if face_w is None else float(face_w),
                blob(appearance), blob(head), blob(beard),
            ),
        )
        return int(cur.lastrowid)

    def forget_body_sighting(self, row_id: int) -> bool:
        """Delete ONE body sighting by rowid; True if a row went.

        The undo half of :meth:`add_body_sighting`, for the same caller as
        :meth:`forget_template`: a sighting is logged under the key /match
        resolved BEFORE the runner can see that two bodies in one frame
        landed on that key.  The loser's box would otherwise stay in the
        wrong identity's stature evidence, and its re-ask logs the same box
        again under the corrected key.  No last-row refusal here — a body
        log is not a template and an identity with no boxes is simply "not
        measured".
        """
        cur = self.conn.execute("DELETE FROM body_sightings WHERE id = ?", (row_id,))
        return cur.rowcount > 0

    def add_manual(self, note: str | None = None, prefix: str = "m") -> str:
        """Record one operator-attested person with no template; return its key.

        Backs the *missed* correction: the operator watched somebody the
        pipeline did not count and says so.  There is no embedding to store —
        the face was never captured — so this person can never be matched into
        and never absorbs a later sighting; they simply exist in the count.
        The ``m`` prefix keeps them separable from detected people forever.
        """
        self._index()  # load BEFORE the insert, or the load would see it twice
        key = self.mint_key(prefix)
        self.conn.execute(
            "INSERT INTO manual (key, note, created_at) VALUES (?, ?, ?)",
            (key, note, _now()),
        )
        self._manual.append(key)
        return key

    def prune_to_cap(self, key: str, cap: int) -> int:
        """Evict a key's WORST templates until it holds at most ``cap``.

        "Worst" is lowest capture quality (face width px).  Ties, and templates
        with no recorded quality at all, break towards the OLDEST row.

        NOT THE ENROLMENT RULE ANY MORE.  This looks like the obvious policy —
        keep a guest's best photographs — and it is wrong for a gallery, which
        :meth:`prune_redundant` now handles instead.  Quality here is face
        width, and face width is distance from the lens, so "keep the best
        five" resolves to "keep the five frames where they stood nearest the
        camera": five near-duplicates of one moment, and no record of the same
        person further away.  Measured on the corridor bench — one guest's five
        surviving templates spanned two seconds of a 140-second walk.

        Retained because "the best N captures" is still the right question
        somewhere (an operator-supervised enrolment picking presentable
        samples, say), and because deleting a tested, documented primitive to
        express a policy change would leave the reasoning nowhere.

        Returns how many templates were evicted (0 when already within cap).
        """
        if cap < 1:
            raise ValueError("cap must be at least 1")
        self._index()  # load BEFORE the delete, so the index still has the rows
        rows = self.conn.execute(
            "SELECT id FROM vectors WHERE key = ?"
            " ORDER BY (quality IS NULL) ASC, quality DESC, id ASC",
            (key,),
        ).fetchall()
        doomed = {int(r[0]) for r in rows[cap:]}
        if not doomed:
            return 0
        self.conn.executemany("DELETE FROM vectors WHERE id = ?", [(i,) for i in doomed])
        self._drop_ids(doomed)
        return len(doomed)

    def max_redundancy(self, key: str) -> float | None:
        """The highest nearest-sibling cosine among a key's templates.

        In other words: how close together the two most similar views this
        identity holds actually are.  It is the number :func:`app.gallery.
        _should_enrol` compares an incoming sighting against to decide whether
        storing it would be churn — if the newcomer sits FURTHER from its
        nearest sibling than this, then some existing pair is more redundant
        than it is, so :meth:`prune_redundant` will evict one of them and the
        newcomer survives.  If it sits closer, the newcomer is itself the most
        redundant view and would be deleted on the next line.

        ``None`` when the key holds fewer than two templates — nothing to be
        redundant with, so no sighting can be rejected on these grounds.
        """
        vecs = self.vectors_for(key)
        if len(vecs) < 2:
            return None
        mat = np.stack([as_unit(v) for v in vecs])
        sims = mat @ mat.T
        np.fill_diagonal(sims, -np.inf)
        return float(sims.max())

    def prune_redundant(self, key: str, cap: int) -> int:
        """Evict a key's most REDUNDANT templates until it holds at most ``cap``.

        The rule :meth:`prune_to_cap` uses — drop the lowest capture quality —
        is right for automatic enrolment, where every stored view already
        matched the ones beside it and the only remaining question is which
        picture is sharper.  After a MERGE it is wrong, and destructively so.

        An operator merges two identities precisely because the pipeline could
        NOT join them: their views are, by construction, the pair that failed to
        match each other.  Evicting on quality there would keep whichever of the
        two was photographed better and discard the very views that caused the
        split — so the next crossing at that pose mints a fresh key and the
        operator merges the same guest again, and again, every night.  A
        correction that has to be re-applied is not a correction.

        So this evicts the template that adds least: the one whose cosine to its
        nearest surviving sibling is highest.  The survivor keeps the WIDEST
        spread of views that fits in ``cap``, which is what the operator's merge
        asserted this person looks like.  Ties break towards evicting the lower
        quality and then the newer row, so the founding view — the anchor the
        identity was minted on — is the last thing to go.

        One eviction at a time, re-scoring in between, because redundancy is
        relative to what is still there: drop two near-twins in one pass and the
        pose they both covered disappears with them.

        Returns how many templates were evicted (0 when already within cap).
        """
        if cap < 1:
            raise ValueError("cap must be at least 1")
        rows = self.conn.execute(
            "SELECT id, vec, dim, quality FROM vectors WHERE key = ? ORDER BY id ASC",
            (key,),
        ).fetchall()
        if len(rows) <= cap:
            return 0
        live = [
            (
                int(r[0]),
                as_unit(np.frombuffer(r[1], dtype=np.float32).reshape(r[2])),
                None if r[3] is None else float(r[3]),
            )
            for r in rows
        ]
        doomed: set[int] = set()
        while len(live) > cap:
            mat = np.stack([v for _, v, _ in live])
            sims = mat @ mat.T
            np.fill_diagonal(sims, -np.inf)
            nearest = sims.max(axis=1)
            # Most redundant first.  Tie-break keys are NEGATED quality (so the
            # lower quality sorts higher = more evictable, and a quality-less
            # row is the most evictable of all) then row id (newer goes first).
            worst = max(
                range(len(live)),
                key=lambda i: (
                    float(nearest[i]),
                    -(live[i][2] if live[i][2] is not None else -1.0),
                    live[i][0],
                ),
            )
            doomed.add(live[worst][0])
            live.pop(worst)
        self._index()  # load BEFORE the delete, so the index still has the rows
        self.conn.executemany("DELETE FROM vectors WHERE id = ?", [(i,) for i in doomed])
        self._drop_ids(doomed)
        return len(doomed)

    def forget_template(self, row_id: int) -> bool:
        """Retract ONE enrolled template by rowid; True if it was removed.

        The undo half of :meth:`add`, for the caller that discovers a write was
        wrong only AFTER making it: the runner matches every face in a frame
        before it can see that two different bodies landed on one key, and by
        then the loser's sighting has already been enrolled into the identity
        it does not belong to.  Leaving it there is not cosmetic — a poisoned
        template goes on pulling that person into the wrong identity in every
        LATER frame, including the frames where they stand alone and no
        same-frame evidence exists to catch it again.  That is precisely how
        run fa8fc3 counted two men as one.

        REFUSES to remove a key's LAST template, returning False.  A key with
        no vectors would still hold its slot in the distinct count while being
        unmatchable forever — a guest who exists but can never be recognised
        again.  Retiring a whole identity is :meth:`remove`'s job, not this
        one's.  Also False when the row is already gone, which is ordinary:
        ``prune_redundant`` runs immediately after every enrolment and may
        have evicted this very row before anyone asked to forget it.
        """
        row = self.conn.execute(
            "SELECT key FROM vectors WHERE id = ?", (row_id,)
        ).fetchone()
        if row is None:
            return False
        held = self.conn.execute(
            "SELECT count(*) FROM vectors WHERE key = ?", (row[0],)
        ).fetchone()[0]
        if held <= 1:
            return False
        self._index()  # load BEFORE the delete, so the index still has the row
        self.conn.execute("DELETE FROM vectors WHERE id = ?", (row_id,))
        self._drop_ids({row_id})
        return True

    # ------------------------------------- operator corrections (pure ops)

    def cannot_link(self, a: str, b: str) -> bool:
        """True if the pair carries a do-not-merge constraint.

        Three callers now, all of them treating the row as "known different":
        :meth:`merge` (refuse the fold), and the two banner paths in
        :mod:`app.gallery` that withhold a duplicate suggestion for a pair
        somebody has already settled.  One indexed primary-key lookup, so it is
        cheap enough to sit on the mint path.
        """
        lo, hi = _pair(a, b)
        row = self.conn.execute(
            "SELECT 1 FROM cannot_link WHERE a = ? AND b = ?", (lo, hi)
        ).fetchone()
        return row is not None

    def split(self, a: str, b: str) -> None:
        """Record that a and b are different people (*false-match* / co-presence).

        Realises "raise the pair's internal distance (no auto-merge later)" as
        a persistent cannot-link constraint that :meth:`merge` honours and that
        both duplicate-suggestion banners consult.  Both keys keep every
        template they had, so the distinct count is unchanged.

        Keys are NOT required to exist.  A constraint is a statement about a
        pair of identifiers, not about stored rows: :meth:`merge` already
        refuses an unknown key on its own terms, and :meth:`remove` clears a
        retired key's rows.  Refusing to record the assertion would mean losing
        it, and losing it is how two people get folded into one.
        """
        if a == b:
            raise ValueError("cannot split a key from itself")
        lo, hi = _pair(a, b)
        self.conn.execute(
            "INSERT OR IGNORE INTO cannot_link (a, b) VALUES (?, ?)", (lo, hi)
        )

    def merge(self, keep: str, drop: str) -> bool:
        """Fold ``drop``'s templates into ``keep`` (a *duplicate* correction).

        ``drop``'s rows are re-pointed to ``keep`` and ``drop`` ceases to exist,
        so the distinct count falls by exactly one.  Refuses (returns False,
        changes nothing) when the two keys are under a cannot-link constraint,
        when either key is unknown, or when they are the same key — the caller
        then leaves the unique count untouched and reports the correction
        rejected.

        A pure op: the survivor inherits EVERY view both keys held and may now
        exceed the template cap, because a cap is policy and this layer does not
        hold it.  :func:`app.gallery.merge` prunes afterwards, in the same
        transaction, via :meth:`prune_redundant`.
        """
        if keep == drop:
            return False
        if self.cannot_link(keep, drop):
            return False
        if self.count_for(keep) == 0 or self.count_for(drop) == 0:
            return False
        self.conn.execute("UPDATE vectors SET key = ? WHERE key = ?", (keep, drop))
        # The attribute columns ride with their rows.  Body sightings are
        # re-keyed the same way: the operator has said these were one person,
        # so every box either key produced is now that person's stature
        # evidence — the sighting log must never name a key that no longer
        # exists, or the survivor's stature would be read off half its data.
        self.conn.execute("UPDATE body_sightings SET key = ? WHERE key = ?", (keep, drop))
        # Carry the retired key's constraints forward onto the survivor.
        self.conn.execute(
            "UPDATE OR IGNORE cannot_link SET a = ? WHERE a = ?", (keep, drop)
        )
        self.conn.execute(
            "UPDATE OR IGNORE cannot_link SET b = ? WHERE b = ?", (keep, drop)
        )
        self.conn.execute("DELETE FROM cannot_link WHERE a = b")
        self._index()
        self._keys = [keep if k == drop else k for k in self._keys]
        return True

    def remove(self, key: str) -> list[tuple[bytes, float | None, int]]:
        """Delete a key's templates, returning them (blob, quality, sub_canon).

        Used by *mark-staff*: the guest person is lifted out of this gallery
        (distinct count −1) and its returned templates are re-added to the site
        staff store under a staff id.  Returns an empty list for an unknown key.

        Appearance descriptors are DELIBERATELY not in the returned tuples —
        nor are the attribute readings or the body sightings: the only caller
        re-homes these rows into the staff store, and staff flows carry no
        appearance, attribute or stature handling anywhere (staff identity is
        operator-attested, never inferred from clothing, sex, age or height),
        so all of it dies with the gallery row instead of leaking into a
        persistent per-site file.
        """
        self._index()  # load BEFORE the delete, so the index still has the rows
        rows = self.conn.execute(
            "SELECT vec, quality, sub_canon FROM vectors WHERE key = ?", (key,)
        ).fetchall()
        if rows:
            self.conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            self.conn.execute("DELETE FROM cannot_link WHERE a = ? OR b = ?", (key, key))
            # The person's box log goes with them: a retired key must not
            # keep feeding the stature fit under a name nothing else holds,
            # and the staff store the rows re-home to keeps no such log.
            self.conn.execute("DELETE FROM body_sightings WHERE key = ?", (key,))
            self._drop_rows(key)
        return [(bytes(v), q, int(sc)) for v, q, sc in rows]


# ------------------------------------------------------------- open cache

#: One open store per file for the whole process — the fix for "re-open and
#: full-scan per /match call".  Keyed by resolved path so two spellings of the
#: same file cannot end up with two connections (and two divergent indexes).
_CACHE: dict[Path, VectorStore] = {}
_CACHE_LOCK = threading.Lock()


def open_store(path: Path, timeout: float = 5.0) -> VectorStore:
    """Return the process-wide open store for ``path``, opening it once.

    The returned store is SHARED: never close it directly and never use it as
    a context manager — use :meth:`VectorStore.transaction` for writes and
    :meth:`VectorStore.reading` for reads, and :func:`close_store` when the
    file's life is over (run ended, gallery swept, store deleted).
    """
    p = Path(path)
    with _CACHE_LOCK:
        store = _CACHE.get(p)
        if store is None:
            store = _CACHE[p] = VectorStore(p, timeout=timeout, cached=True)
        return store


def close_store(path: Path) -> None:
    """Close and forget the cached store for ``path`` (idempotent).

    MUST be called before the file is deleted: an open connection to an
    unlinked inode keeps answering from a database nobody can see.
    """
    p = Path(path)
    with _CACHE_LOCK:
        store = _CACHE.pop(p, None)
    if store is not None:
        store.close()


def close_all_stores() -> None:
    """Close every cached store (service shutdown, test teardown)."""
    with _CACHE_LOCK:
        stores = list(_CACHE.values())
        _CACHE.clear()
    for store in stores:
        store.close()


def _now() -> str:
    """Return an ISO-8601 UTC timestamp for a stored row."""
    return datetime.now(UTC).isoformat()
