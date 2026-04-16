"""SQLite storage backend for agents-rag: documents, chunks, vectors, state."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

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
    latency_ms INTEGER NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the database and ensure schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns/tables introduced after initial schema."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "summary" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
        log.info("Migrated: added documents.summary column")


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
    row = conn.execute(
        "SELECT id FROM documents WHERE source=? AND path=?", (source, path),
    ).fetchone()
    return row[0] if row else None


def get_document_content(conn: sqlite3.Connection, source: str, path: str) -> str | None:
    row = conn.execute(
        "SELECT content FROM documents WHERE source=? AND path=?", (source, path),
    ).fetchone()
    return row[0] if row else None


def get_document_summary(conn: sqlite3.Connection, source: str, path: str) -> str | None:
    row = conn.execute(
        "SELECT summary FROM documents WHERE source=? AND path=?", (source, path),
    ).fetchone()
    return row[0] if row else None


def delete_source_documents(conn: sqlite3.Connection, source: str) -> None:
    """Delete all documents (and cascading chunks) for a source."""
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
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# HNSW cache
# ---------------------------------------------------------------------------

def load_hnsw_cache(conn: sqlite3.Connection) -> tuple[bytes, int] | None:
    row = conn.execute("SELECT data, chunk_count FROM hnsw_cache WHERE id=1").fetchone()
    if row is None:
        return None
    return row[0], row[1]


def save_hnsw_cache(conn: sqlite3.Connection, data: bytes, num_chunks: int) -> None:
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
    rows = conn.execute("SELECT path, reason FROM deprecated").fetchall()
    return dict(rows)


def save_deprecated(conn: sqlite3.Connection, path: str, reason: str) -> None:
    conn.execute(
        """INSERT INTO deprecated (path, reason) VALUES (?, ?)
           ON CONFLICT(path) DO UPDATE SET reason=excluded.reason""",
        (path, reason),
    )
    conn.commit()


def remove_deprecated(conn: sqlite3.Connection, path: str) -> bool:
    cur = conn.execute("DELETE FROM deprecated WHERE path=?", (path,))
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Transcript state
# ---------------------------------------------------------------------------

def load_transcript_state(conn: sqlite3.Connection, transcript_path: str) -> dict:
    import hashlib
    h = hashlib.sha256(transcript_path.encode()).hexdigest()[:16]
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
) -> None:
    conn.execute(
        """INSERT INTO query_log (timestamp, query, source_filter, top_k,
                                  result_count, results_json, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (_now(), query, source_filter, top_k, result_count, results_json, latency_ms),
    )
    conn.commit()


def get_recent_queries(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT id, timestamp, query, source_filter, top_k,
                  result_count, results_json, latency_ms
           FROM query_log ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {
            "id": r[0], "timestamp": r[1], "query": r[2],
            "source_filter": r[3], "top_k": r[4], "result_count": r[5],
            "results_json": r[6], "latency_ms": r[7],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
