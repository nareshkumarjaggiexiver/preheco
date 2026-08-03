"""SQLite-backed embedding gallery, one database file per run.

POC scale: a run holds hundreds of people, not millions, so matching is a
brute-force cosine scan over every stored template.  That is deliberate — it
is exact, dependency-free, and fast enough (<1 ms for a few thousand 128-d
vectors with numpy).  A vector index (FAISS/pgvector) is a scale decision for
later, taken on profiling evidence, per the project rule.

Concurrency: a connection is opened per operation with a busy timeout, and the
match-then-insert decision runs inside one IMMEDIATE transaction so two
concurrent /match calls cannot both insert the same brand-new person.
"""

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_key TEXT PRIMARY KEY,
    sub_canon  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS templates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_key TEXT NOT NULL REFERENCES persons(person_key),
    embedding  BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    quality    REAL,
    sub_canon  INTEGER NOT NULL DEFAULT 0
);
"""


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


def _connect(path: Path) -> sqlite3.Connection:
    """Open the gallery database, creating the schema if needed."""
    conn = sqlite3.connect(path, timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(_SCHEMA)
    return conn


def reset(data_dir: Path, run_id: str) -> None:
    """Delete any existing gallery for the run so it starts empty."""
    path = db_path(data_dir, run_id)
    path.unlink(missing_ok=True)


def _as_unit(embedding: list[float]) -> np.ndarray:
    """Return the embedding as an L2-normalised float32 vector.

    SFace embeddings arrive normalised already; normalising again is a cheap
    no-op that makes the cosine maths safe for any caller.
    """
    v = np.asarray(embedding, dtype=np.float32)
    if v.ndim != 1 or v.size < 8:
        raise ValueError("embedding must be a flat vector of at least 8 floats")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("embedding must not be the zero vector")
    return v / n


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
    point counts equality as a match).  New persons store the embedding as
    their first template.  Sub-canon faces (quality < canon_px) are matched
    normally but tagged, per the POC contract.
    """
    v = _as_unit(embedding)
    sub_canon = quality is not None and quality < canon_px
    path = db_path(data_dir, run_id)

    with _connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("SELECT person_key, embedding, dim FROM templates").fetchall()

        best_key: str | None = None
        best_cos: float | None = None
        if rows:
            dim = rows[0][2]
            if v.size != dim:
                raise ValueError(f"embedding dim {v.size} != gallery dim {dim}")
            mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
            mat = mat.reshape(len(rows), dim)
            sims = mat @ v  # rows are unit vectors, so the dot product is cosine
            i = int(np.argmax(sims))
            best_key, best_cos = rows[i][0], float(sims[i])

        if best_key is not None and best_cos is not None and best_cos >= threshold:
            n = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            return MatchResult(best_key, False, best_cos, n, sub_canon)

        n_before = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        person_key = f"p{n_before + 1:05d}"
        conn.execute(
            "INSERT INTO persons (person_key, sub_canon) VALUES (?, ?)",
            (person_key, int(sub_canon)),
        )
        conn.execute(
            "INSERT INTO templates (person_key, embedding, dim, quality, sub_canon)"
            " VALUES (?, ?, ?, ?, ?)",
            (person_key, v.tobytes(), v.size, quality, int(sub_canon)),
        )
        return MatchResult(person_key, True, best_cos, n_before + 1, sub_canon)
