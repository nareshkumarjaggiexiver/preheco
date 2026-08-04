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

Storage
-------
* ``vectors``     one row per stored template: ``key``, the ``float32`` BLOB,
                  its ``dim``, the capture ``quality`` (face width px) and a
                  ``sub_canon`` flag.  A person/staff member may own several
                  rows (multi-template) after enrolment or a merge.
* ``cannot_link`` operator "these are two different people" constraints
                  (from a *false-match* correction), stored order-independent.
                  :meth:`merge` refuses to fold a constrained pair — that is
                  how "raise the pair's internal distance, no auto-merge later"
                  is realised in a brute-force store.
* ``meta``        small integer counters (the monotonic key sequence), so keys
                  never collide even after merges and removals shrink the
                  distinct-person count.

Vectors are L2-normalised on the way in, so the dot product IS the cosine
similarity and :meth:`search` is a single matrix-vector multiply.
"""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

#: SQLite schema for one store file.  ``IF NOT EXISTS`` makes open idempotent.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL,
    vec        BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    quality    REAL,
    sub_canon  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_key ON vectors(key);
CREATE TABLE IF NOT EXISTS cannot_link (
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v INTEGER NOT NULL
);
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
    """A single embedding store file, opened for one unit of work.

    Use as a context manager so the connection is committed and closed even on
    error::

        with VectorStore(path) as store:
            hit = store.search(vec)
            if hit is None or hit.cosine < threshold:
                key = store.add_auto(vec, quality=71.0)

    A fresh instance per request mirrors the connection-per-operation style the
    service started with; :meth:`begin_immediate` gives a caller the
    read-then-write atomicity the match decision needs (two concurrent /match
    calls cannot both insert the same brand-new person).
    """

    def __init__(self, path: Path, timeout: float = 5.0) -> None:
        """Open ``path`` (creating the schema if new) with a busy timeout."""
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, timeout=timeout)
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.executescript(_SCHEMA)

    # ---------------------------------------------------------- lifecycle

    def __enter__(self) -> "VectorStore":
        """Enter the context; the store is already open."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Commit on clean exit, roll back on error, then close."""
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()

    def begin_immediate(self) -> "VectorStore":
        """Start an IMMEDIATE transaction (write lock now) for search+insert.

        Returned so it reads well as ``with store.begin_immediate():`` — the
        surrounding :meth:`__exit__` still owns the final commit/close, but the
        IMMEDIATE lock is taken up front so the match-then-insert window cannot
        interleave with another writer.
        """
        self.conn.execute("BEGIN IMMEDIATE")
        return self

    # ------------------------------------------------------------- reads

    def _rows(self) -> list[tuple[str, bytes, int]]:
        """All (key, vec-bytes, dim) rows — the whole table for a scan."""
        return self.conn.execute("SELECT key, vec, dim FROM vectors").fetchall()

    def search(self, embedding: list[float] | np.ndarray) -> Neighbour | None:
        """Return the nearest stored template by cosine, or None if empty.

        Rows are unit vectors, so the dot product is the cosine and the whole
        scan is one ``(n, dim) @ (dim,)`` multiply.  The best row's key is
        returned even when several templates share it (multi-template person).
        """
        rows = self._rows()
        if not rows:
            return None
        q = as_unit(embedding)
        dim = rows[0][2]
        if q.size != dim:
            raise ValueError(f"embedding dim {q.size} != store dim {dim}")
        mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
        mat = mat.reshape(len(rows), dim)
        sims = mat @ q
        i = int(np.argmax(sims))
        return Neighbour(key=rows[i][0], cosine=float(sims[i]))

    def distinct_count(self) -> int:
        """Number of distinct keys (persons / staff members) in the store."""
        return int(self.conn.execute("SELECT COUNT(DISTINCT key) FROM vectors").fetchone()[0])

    def keys(self) -> list[str]:
        """All distinct keys, ascending — small at POC scale."""
        rows = self.conn.execute("SELECT DISTINCT key FROM vectors ORDER BY key").fetchall()
        return [r[0] for r in rows]

    def vectors_for(self, key: str) -> list[np.ndarray]:
        """Every stored template for one key as float32 arrays."""
        rows = self.conn.execute(
            "SELECT vec, dim FROM vectors WHERE key = ?", (key,)
        ).fetchall()
        return [np.frombuffer(v, dtype=np.float32).reshape(d) for v, d in rows]

    def count_for(self, key: str) -> int:
        """How many templates a single key owns."""
        return int(
            self.conn.execute("SELECT COUNT(*) FROM vectors WHERE key = ?", (key,)).fetchone()[0]
        )

    # ------------------------------------------------------------ writes

    def mint_key(self, prefix: str = "p", width: int = 5) -> str:
        """Return the next monotonic key (``p00001``, ``p00002`, …).

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
    ) -> None:
        """Store one template under an explicit key (used for staff ids)."""
        v = as_unit(embedding)
        self.conn.execute(
            "INSERT INTO vectors (key, vec, dim, quality, sub_canon, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (key, v.tobytes(), v.size, quality, int(sub_canon), _now()),
        )

    def add_auto(
        self,
        embedding: list[float] | np.ndarray,
        quality: float | None = None,
        sub_canon: bool = False,
        prefix: str = "p",
    ) -> str:
        """Mint a fresh monotonic key, store the template under it, return it."""
        key = self.mint_key(prefix)
        self.add(key, embedding, quality, sub_canon)
        return key

    # ------------------------------------- operator corrections (pure ops)

    def cannot_link(self, a: str, b: str) -> bool:
        """True if the pair carries a do-not-merge constraint."""
        lo, hi = _pair(a, b)
        row = self.conn.execute(
            "SELECT 1 FROM cannot_link WHERE a = ? AND b = ?", (lo, hi)
        ).fetchone()
        return row is not None

    def split(self, a: str, b: str) -> None:
        """Record a *false-match* correction: a and b are different people.

        Realises "raise the pair's internal distance (no auto-merge later)" as
        a persistent cannot-link constraint that :meth:`merge` honours.  Both
        keys keep every template they had, so the distinct count is unchanged.
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
        """
        if keep == drop:
            return False
        if self.cannot_link(keep, drop):
            return False
        if self.count_for(keep) == 0 or self.count_for(drop) == 0:
            return False
        self.conn.execute("UPDATE vectors SET key = ? WHERE key = ?", (keep, drop))
        # Carry the retired key's constraints forward onto the survivor.
        self.conn.execute(
            "UPDATE OR IGNORE cannot_link SET a = ? WHERE a = ?", (keep, drop)
        )
        self.conn.execute(
            "UPDATE OR IGNORE cannot_link SET b = ? WHERE b = ?", (keep, drop)
        )
        self.conn.execute("DELETE FROM cannot_link WHERE a = b")
        return True

    def remove(self, key: str) -> list[tuple[bytes, float | None, int]]:
        """Delete a key's templates, returning them (blob, quality, sub_canon).

        Used by *mark-staff*: the guest person is lifted out of this gallery
        (distinct count −1) and its returned templates are re-added to the site
        staff store under a staff id.  Returns an empty list for an unknown key.
        """
        rows = self.conn.execute(
            "SELECT vec, quality, sub_canon FROM vectors WHERE key = ?", (key,)
        ).fetchall()
        if rows:
            self.conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            self.conn.execute("DELETE FROM cannot_link WHERE a = ? OR b = ?", (key, key))
        return [(bytes(v), q, int(sc)) for v, q, sc in rows]


def _now() -> str:
    """Return an ISO-8601 UTC timestamp for a stored row."""
    return datetime.now(UTC).isoformat()
