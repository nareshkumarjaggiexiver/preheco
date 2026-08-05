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
                  merge.  Two evictions bound how many, and they differ on
                  purpose: :meth:`prune_to_cap` (after enrolment) drops the
                  lowest quality, :meth:`prune_redundant` (after a merge) drops
                  the least distinctive view — see that method for why keeping
                  the quality rule there would undo the operator's correction.
* ``cannot_link`` operator "these are two different people" constraints
                  (from a *false-match* correction), stored order-independent.
                  :meth:`merge` refuses to fold a constrained pair — that is
                  how "raise the pair's internal distance, no auto-merge later"
                  is realised in a brute-force store.
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
import sqlite3
import threading
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
CREATE TABLE IF NOT EXISTS manual (
    key        TEXT PRIMARY KEY,
    note       TEXT,
    created_at TEXT NOT NULL
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

    def search(self, embedding: list[float] | np.ndarray) -> Neighbour | None:
        """Return the nearest stored template by cosine, or None if empty.

        Rows are unit vectors, so the dot product is the cosine and the whole
        scan is one ``(n, dim) @ (dim,)`` multiply against the resident matrix
        — no SQLite round trip, no BLOB rebuild.  The best row's key is
        returned even when several templates share it (multi-template person).
        """
        self._index()
        if not self._n:
            return None
        q = as_unit(embedding)
        if q.size != self._dim:
            raise ValueError(f"embedding dim {q.size} != store dim {self._dim}")
        sims = self._buf[: self._n] @ q
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
    ) -> None:
        """Store one template under an explicit key (used for staff ids)."""
        v = as_unit(embedding)
        self._index()  # load BEFORE the insert, or the load would see it twice
        cur = self.conn.execute(
            "INSERT INTO vectors (key, vec, dim, quality, sub_canon, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (key, v.tobytes(), v.size, quality, int(sub_canon), _now()),
        )
        self._append_row(key, v, int(cur.lastrowid))

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

        "Worst" is lowest capture quality (face width px), so an identity that
        keeps being re-sighted accumulates its best views rather than its most
        recent — a most-recent policy would let a run of bad frames flush the
        good evidence out of a guest's record, and the count is an invoice
        figure.  Ties, and templates with no recorded quality at all, break
        towards the OLDEST row: the earliest template is the one the identity
        was founded on, and keeping it anchors the identity against drift.

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
        """
        self._index()  # load BEFORE the delete, so the index still has the rows
        rows = self.conn.execute(
            "SELECT vec, quality, sub_canon FROM vectors WHERE key = ?", (key,)
        ).fetchall()
        if rows:
            self.conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            self.conn.execute("DELETE FROM cannot_link WHERE a = ? OR b = ?", (key, key))
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
