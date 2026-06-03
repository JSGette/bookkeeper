"""SQLite storage backend for bookkeeper: documents, chunks, vectors, state."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_write_lock = threading.RLock()

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    path TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    preamble TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    indexed_at TEXT NOT NULL,
    UNIQUE(source, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_heading TEXT NOT NULL DEFAULT '',
    start_line INTEGER NOT NULL DEFAULT 0,
    end_line INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT '',
    doc_preamble TEXT NOT NULL DEFAULT '',
    indexed_at TEXT NOT NULL,
    vector BLOB NOT NULL,
    session_summary TEXT,
    full_session_path TEXT,
    session_turn_count INTEGER
);

CREATE TABLE IF NOT EXISTS source_state (
    source_name TEXT PRIMARY KEY,
    commit_hash TEXT,
    mtimes TEXT,
    indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS deprecated (
    path TEXT PRIMARY KEY,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_state (
    transcript_hash TEXT PRIMARY KEY,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hnsw_cache (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data BLOB NOT NULL,
    chunk_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    source_filter TEXT,
    top_k INTEGER NOT NULL,
    result_count INTEGER NOT NULL,
    results_json TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    reranker_used INTEGER NOT NULL DEFAULT 0,
    rrf_k INTEGER NOT NULL DEFAULT 0,
    score_kind TEXT NOT NULL DEFAULT 'rrf',
    query_embedding BLOB
);

-- Backs the dedupe pre-check (`get_recent_query_embeddings`) which fires
-- on every search. Without this, the WHERE timestamp >= ? ORDER BY
-- timestamp DESC LIMIT 5 degrades to a full table scan and pins the
-- per-connection mutex once query_log accumulates real history.
CREATE INDEX IF NOT EXISTS idx_query_log_timestamp ON query_log(timestamp);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the database and ensure schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    with _write_lock:
        for stmt in _SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        _migrate(conn)
        conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns/tables introduced after initial schema. Caller holds _write_lock."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "summary" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
        log.info("Migrated: added documents.summary column")

    qlog_cols = {r[1] for r in conn.execute("PRAGMA table_info(query_log)").fetchall()}
    if qlog_cols and "reranker_used" not in qlog_cols:
        conn.execute("ALTER TABLE query_log ADD COLUMN reranker_used INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE query_log ADD COLUMN rrf_k INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE query_log ADD COLUMN score_kind TEXT NOT NULL DEFAULT 'rrf'")
        log.info("Migrated: added query_log.reranker_used / rrf_k / score_kind columns")
    if qlog_cols and "query_embedding" not in qlog_cols:
        conn.execute("ALTER TABLE query_log ADD COLUMN query_embedding BLOB")
        log.info("Migrated: added query_log.query_embedding column")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def upsert_document(
    conn: sqlite3.Connection,
    source: str,
    path: str,
    title: str,
    preamble: str,
    content: str,
    summary: str = "",
) -> int:
    with _write_lock:
        conn.execute(
            """INSERT INTO documents (source, path, title, preamble, content, summary, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source, path) DO UPDATE SET
                   title=excluded.title,
                   preamble=excluded.preamble,
                   content=excluded.content,
                   summary=excluded.summary,
                   indexed_at=excluded.indexed_at""",
            (source, path, title, preamble, content, summary, _now()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM documents WHERE source=? AND path=?", (source, path),
        ).fetchone()
        return row[0]


def get_document_id(conn: sqlite3.Connection, source: str, path: str) -> int | None:
    with _write_lock:
        row = conn.execute(
            "SELECT id FROM documents WHERE source=? AND path=?", (source, path),
        ).fetchone()
    return row[0] if row else None


def get_document_content(conn: sqlite3.Connection, source: str, path: str) -> str | None:
    with _write_lock:
        row = conn.execute(
            "SELECT content FROM documents WHERE source=? AND path=?", (source, path),
        ).fetchone()
    return row[0] if row else None


def get_document_summary(conn: sqlite3.Connection, source: str, path: str) -> str | None:
    with _write_lock:
        row = conn.execute(
            "SELECT summary FROM documents WHERE source=? AND path=?", (source, path),
        ).fetchone()
    return row[0] if row else None


def delete_source_documents(conn: sqlite3.Connection, source: str) -> None:
    """Delete all documents (and cascading chunks) for a source."""
    with _write_lock:
        conn.execute("DELETE FROM documents WHERE source=?", (source,))
        conn.commit()


# ---------------------------------------------------------------------------
# Chunks + vectors
# ---------------------------------------------------------------------------

def upsert_chunks(
    conn: sqlite3.Connection,
    doc_id: int,
    chunks: list[dict],
    vectors: np.ndarray,
) -> None:
    """Replace all chunks for a document with new ones."""
    with _write_lock:
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        for i, chunk in enumerate(chunks):
            vec_blob = vectors[i].astype(np.float32).tobytes()
            conn.execute(
                """INSERT INTO chunks
                   (doc_id, section_heading, start_line, end_line, text, tags,
                    description, doc_preamble, indexed_at, vector,
                    session_summary, full_session_path, session_turn_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    chunk.get("section_heading", ""),
                    chunk.get("start_line", 0),
                    chunk.get("end_line", 0),
                    chunk.get("text", ""),
                    json.dumps(chunk.get("tags", [])),
                    chunk.get("description", ""),
                    chunk.get("doc_preamble", ""),
                    chunk.get("indexed_at", _now()),
                    vec_blob,
                    chunk.get("session_summary"),
                    chunk.get("full_session_path"),
                    chunk.get("session_turn_count"),
                ),
            )
        conn.commit()


def append_chunks(
    conn: sqlite3.Connection,
    doc_id: int,
    chunks: list[dict],
    vectors: np.ndarray,
) -> None:
    """Append chunks to a document without deleting existing ones."""
    with _write_lock:
        for i, chunk in enumerate(chunks):
            vec_blob = vectors[i].astype(np.float32).tobytes()
            conn.execute(
                """INSERT INTO chunks
                   (doc_id, section_heading, start_line, end_line, text, tags,
                    description, doc_preamble, indexed_at, vector,
                    session_summary, full_session_path, session_turn_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    chunk.get("section_heading", ""),
                    chunk.get("start_line", 0),
                    chunk.get("end_line", 0),
                    chunk.get("text", ""),
                    json.dumps(chunk.get("tags", [])),
                    chunk.get("description", ""),
                    chunk.get("doc_preamble", ""),
                    chunk.get("indexed_at", _now()),
                    vec_blob,
                    chunk.get("session_summary"),
                    chunk.get("full_session_path"),
                    chunk.get("session_turn_count"),
                ),
            )
        conn.commit()


def load_all_chunks(
    conn: sqlite3.Connection,
    source: str | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Load all chunk metadata and vectors. Returns (metadata_list, vector_matrix)."""
    _cols = """\
            c.id, d.source, d.path, d.title, c.section_heading,
            c.start_line, c.end_line, c.text, c.tags, c.description,
            c.doc_preamble, c.indexed_at, c.vector,
            c.session_summary, c.full_session_path, c.session_turn_count,
            d.summary"""
    with _write_lock:
        if source:
            rows = conn.execute(
                f"SELECT {_cols} FROM chunks c JOIN documents d ON c.doc_id = d.id"
                " WHERE d.source = ? ORDER BY c.id",
                (source,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_cols} FROM chunks c JOIN documents d ON c.doc_id = d.id"
                " ORDER BY c.id",
            ).fetchall()

    metadata: list[dict] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        meta: dict = {
            "chunk_id": row[0],
            "source": row[1],
            "path": row[2],
            "doc_title": row[3],
            "section_heading": row[4],
            "start_line": row[5],
            "end_line": row[6],
            "text": row[7],
            "tags": json.loads(row[8]),
            "description": row[9],
            "indexed_at": row[11],
        }
        if row[10]:
            meta["doc_preamble"] = row[10]
        if row[13]:
            meta["session_summary"] = row[13]
        if row[14]:
            meta["full_session_path"] = row[14]
        if row[15] is not None:
            meta["session_turn_count"] = row[15]
        if row[16]:
            meta["doc_summary"] = row[16]
        metadata.append(meta)
        vectors.append(np.frombuffer(row[12], dtype=np.float32))

    if vectors:
        vec_matrix = np.vstack(vectors)
    else:
        vec_matrix = np.empty((0, 384), dtype=np.float32)
    return metadata, vec_matrix


def chunk_count(conn: sqlite3.Connection) -> int:
    with _write_lock:
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# HNSW cache
# ---------------------------------------------------------------------------

def load_hnsw_cache(conn: sqlite3.Connection) -> tuple[bytes, int] | None:
    with _write_lock:
        row = conn.execute("SELECT data, chunk_count FROM hnsw_cache WHERE id=1").fetchone()
    if row is None:
        return None
    return row[0], row[1]


def save_hnsw_cache(conn: sqlite3.Connection, data: bytes, num_chunks: int) -> None:
    with _write_lock:
        conn.execute(
            """INSERT INTO hnsw_cache (id, data, chunk_count, updated_at)
               VALUES (1, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   data=excluded.data,
                   chunk_count=excluded.chunk_count,
                   updated_at=excluded.updated_at""",
            (data, num_chunks, _now()),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Source state
# ---------------------------------------------------------------------------

def load_source_state(conn: sqlite3.Connection, source_name: str) -> dict:
    with _write_lock:
        row = conn.execute(
            "SELECT commit_hash, mtimes, indexed_at FROM source_state WHERE source_name=?",
            (source_name,),
        ).fetchone()
    if row is None:
        return {}
    state: dict = {}
    if row[0]:
        state["commit_hash"] = row[0]
    if row[1]:
        try:
            state["mtimes"] = json.loads(row[1])
        except json.JSONDecodeError:
            pass
    if row[2]:
        state["indexed_at"] = row[2]
    return state


def save_source_state(conn: sqlite3.Connection, source_name: str, state: dict) -> None:
    with _write_lock:
        conn.execute(
            """INSERT INTO source_state (source_name, commit_hash, mtimes, indexed_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(source_name) DO UPDATE SET
                   commit_hash=excluded.commit_hash,
                   mtimes=excluded.mtimes,
                   indexed_at=excluded.indexed_at""",
            (
                source_name,
                state.get("commit_hash"),
                json.dumps(state.get("mtimes", {})),
                state.get("indexed_at", _now()),
            ),
        )
        conn.commit()


def load_all_source_state(conn: sqlite3.Connection) -> dict[str, dict]:
    """Load state for all sources, keyed by source name."""
    with _write_lock:
        rows = conn.execute("SELECT source_name, commit_hash, mtimes, indexed_at FROM source_state").fetchall()
    result: dict[str, dict] = {}
    for name, commit, mtimes_json, indexed_at in rows:
        state: dict = {}
        if commit:
            state["commit_hash"] = commit
        if mtimes_json:
            try:
                state["mtimes"] = json.loads(mtimes_json)
            except json.JSONDecodeError:
                pass
        if indexed_at:
            state["indexed_at"] = indexed_at
        result[name] = state
    return result


# ---------------------------------------------------------------------------
# Deprecated
# ---------------------------------------------------------------------------

def load_deprecated(conn: sqlite3.Connection) -> dict[str, str]:
    with _write_lock:
        rows = conn.execute("SELECT path, reason FROM deprecated").fetchall()
    return dict(rows)


def save_deprecated(conn: sqlite3.Connection, path: str, reason: str) -> None:
    with _write_lock:
        conn.execute(
            """INSERT INTO deprecated (path, reason) VALUES (?, ?)
               ON CONFLICT(path) DO UPDATE SET reason=excluded.reason""",
            (path, reason),
        )
        conn.commit()


def remove_deprecated(conn: sqlite3.Connection, path: str) -> bool:
    with _write_lock:
        cur = conn.execute("DELETE FROM deprecated WHERE path=?", (path,))
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Transcript state
# ---------------------------------------------------------------------------

def load_transcript_state(conn: sqlite3.Connection, transcript_path: str) -> dict:
    import hashlib
    h = hashlib.sha256(transcript_path.encode()).hexdigest()[:16]
    with _write_lock:
        row = conn.execute(
            "SELECT state_json FROM transcript_state WHERE transcript_hash=?", (h,),
        ).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


def save_transcript_state(conn: sqlite3.Connection, transcript_path: str, state: dict) -> None:
    import hashlib
    h = hashlib.sha256(transcript_path.encode()).hexdigest()[:16]
    with _write_lock:
        conn.execute(
            """INSERT INTO transcript_state (transcript_hash, state_json)
               VALUES (?, ?)
               ON CONFLICT(transcript_hash) DO UPDATE SET state_json=excluded.state_json""",
            (h, json.dumps(state)),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Query audit log
# ---------------------------------------------------------------------------

def log_query(
    conn: sqlite3.Connection,
    query: str,
    source_filter: str | None,
    top_k: int,
    result_count: int,
    results_json: str,
    latency_ms: int,
    reranker_used: bool = False,
    rrf_k: int = 0,
    score_kind: str = "rrf",
    query_embedding: bytes | None = None,
) -> int:
    """Insert one row into the audit log. Returns the row id so callers can
    reference prior queries (e.g. for the repeated-query dedupe hint)."""
    with _write_lock:
        cur = conn.execute(
            """INSERT INTO query_log (timestamp, query, source_filter, top_k,
                                      result_count, results_json, latency_ms,
                                      reranker_used, rrf_k, score_kind,
                                      query_embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), query, source_filter, top_k, result_count, results_json, latency_ms,
             1 if reranker_used else 0, rrf_k, score_kind, query_embedding),
        )
        conn.commit()
        row_id = cur.lastrowid
        assert row_id is not None
        return row_id


def get_recent_queries(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    with _write_lock:
        rows = conn.execute(
            """SELECT id, timestamp, query, source_filter, top_k,
                      result_count, results_json, latency_ms,
                      reranker_used, rrf_k, score_kind
               FROM query_log ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0], "timestamp": r[1], "query": r[2],
            "source_filter": r[3], "top_k": r[4], "result_count": r[5],
            "results_json": r[6], "latency_ms": r[7],
            "reranker_used": bool(r[8]), "rrf_k": r[9], "score_kind": r[10],
        }
        for r in rows
    ]


def get_recent_query_embeddings(
    conn: sqlite3.Connection,
    source_filter: str | None,
    since_seconds: int = 120,
    limit: int = 5,
) -> list[dict]:
    """Fetch recent queries (with embeddings) for the dedupe check.

    Filters by source_filter equality (NULL matches NULL) so we don't dedupe
    across unrelated source scopes. Excludes rows already served from cache
    so a single user query can't dedupe against itself in a chain.
    """
    cutoff = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - since_seconds),
    )
    with _write_lock:
        if source_filter is None:
            rows = conn.execute(
                """SELECT id, timestamp, query, results_json, query_embedding,
                          reranker_used, rrf_k, score_kind
                   FROM query_log
                   WHERE source_filter IS NULL
                     AND timestamp >= ?
                     AND query_embedding IS NOT NULL
                     AND score_kind != 'cached'
                   ORDER BY id DESC LIMIT ?""",
                (cutoff, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, timestamp, query, results_json, query_embedding,
                          reranker_used, rrf_k, score_kind
                   FROM query_log
                   WHERE source_filter = ?
                     AND timestamp >= ?
                     AND query_embedding IS NOT NULL
                     AND score_kind != 'cached'
                   ORDER BY id DESC LIMIT ?""",
                (source_filter, cutoff, limit),
            ).fetchall()
    return [
        {
            "id": r[0], "timestamp": r[1], "query": r[2],
            "results_json": r[3], "query_embedding": r[4],
            "reranker_used": bool(r[5]), "rrf_k": r[6], "score_kind": r[7],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Garbage collection
# ---------------------------------------------------------------------------

def gc_source(conn: sqlite3.Connection, source: str, live_paths: set[str]) -> int:
    """Delete documents (and cascading chunks) for paths no longer on disk.

    Returns number of documents removed.
    """
    with _write_lock:
        rows = conn.execute(
            "SELECT id, path FROM documents WHERE source=?", (source,),
        ).fetchall()
        stale_ids = [doc_id for doc_id, path in rows if path not in live_paths]
        if not stale_ids:
            return 0
        placeholders = ",".join("?" * len(stale_ids))
        conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", stale_ids)
        conn.commit()
    return len(stale_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
