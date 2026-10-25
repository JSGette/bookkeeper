"""SQLite storage backend for bookkeeper: documents, chunks, vectors, state."""

from __future__ import annotations

import json
import hashlib
import functools
import logging
import sqlite3
import threading
import time
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4

import numpy as np

log = logging.getLogger(__name__)

_write_lock = threading.RLock()
SCHEMA_VERSION = 1


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Serialize a complete unit of work; nested writers never commit their caller."""
    with _write_lock:
        name = "bk_" + uuid4().hex
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield conn
            conn.execute(f"RELEASE SAVEPOINT {name}")
        except BaseException:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            conn.execute(f"RELEASE SAVEPOINT {name}")
            raise


def _atomic(fn):
    @functools.wraps(fn)
    def wrapped(conn, *args, **kwargs):
        with transaction(conn):
            return fn(conn, *args, **kwargs)
    return wrapped


_EVIDENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corpus_state (
    id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO corpus_state VALUES (1,0);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS document_revisions (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, path TEXT NOT NULL,
    content TEXT NOT NULL, content_hash TEXT NOT NULL, completeness TEXT NOT NULL,
    metadata_json TEXT NOT NULL, observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_revisions_source_path ON document_revisions(source,path);
CREATE TABLE IF NOT EXISTS evidence_segments (
    id TEXT PRIMARY KEY, revision_id TEXT NOT NULL REFERENCES document_revisions(id),
    text TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding_spaces (
    id TEXT PRIMARY KEY, dimension INTEGER NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingestion_heads (
    identity TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES ingestion_jobs(id),
    sequence INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, scope TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_revisions (
    id TEXT PRIMARY KEY, memory_id TEXT NOT NULL REFERENCES memory_entries(id),
    content TEXT NOT NULL, evidence_ids TEXT NOT NULL, valid_from TEXT, valid_to TEXT,
    status TEXT NOT NULL, previous_id TEXT REFERENCES memory_revisions(id),
    observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_revisions_entry ON memory_revisions(memory_id);
"""

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

def init_db(db_path: str, *, allow_legacy_upgrade: bool = True) -> sqlite3.Connection:
    """Open (or create) the database and ensure schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        conn.close()
        raise ValueError(f"Unsupported database schema {version}; maximum is {SCHEMA_VERSION}")
    if not allow_legacy_upgrade and version == 0 and conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
    ).fetchone():
        conn.close()
        raise ValueError("Legacy database requires a verified copy: run bookkeeper migrate --source OLD --destination NEW, then set db_path to NEW")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        with transaction(conn):
            for stmt in _SCHEMA.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            _migrate(conn)
    except BaseException:
        conn.close()
        raise
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
    for stmt in _EVIDENCE_SCHEMA.split(";"):
        if stmt.strip():
            conn.execute(stmt)
    additions = {
        "documents": [("revision_id", "TEXT"), ("completeness", "TEXT NOT NULL DEFAULT 'legacy_unknown'")],
        "chunks": [("evidence_id", "TEXT"), ("embedding_space_id", "TEXT NOT NULL DEFAULT 'legacy_unknown'")],
        "hnsw_cache": [("fingerprint", "TEXT")],
        "source_state": [("state_json", "TEXT")],
    }
    for table, columns in additions.items():
        present = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, declaration in columns:
            if name not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
    # Retain historical rows/IDs and mark their provenance honestly.
    for row in conn.execute("SELECT id,source,path,title,preamble,content,summary,indexed_at FROM documents WHERE revision_id IS NULL").fetchall():
        doc_id, source, path, title, preamble, content, summary, observed = row
        completeness = "legacy_summary" if source == "sessions" else "legacy_unknown"
        revision = _record_revision(conn, source, path, content, completeness,
                                    {"title": title, "preamble": preamble, "summary": summary}, observed)
        conn.execute("UPDATE documents SET revision_id=?, completeness=? WHERE id=?", (revision, completeness, doc_id))
    for row in conn.execute("SELECT c.id,c.text,c.section_heading,c.start_line,c.end_line,c.tags,c.description,c.doc_preamble,c.indexed_at,c.session_summary,c.full_session_path,c.session_turn_count,d.revision_id FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.evidence_id IS NULL").fetchall():
        metadata = dict(zip(["legacy_chunk_id", "text", "section_heading", "start_line", "end_line", "tags", "description", "doc_preamble", "indexed_at", "session_summary", "full_session_path", "session_turn_count"], row[:-1]))
        evidence_id = _record_segment(conn, row[-1], row[1], metadata)
        conn.execute("UPDATE chunks SET evidence_id=? WHERE id=?", (evidence_id,row[0]))
    # Triggers also invalidate caches when legacy-compatible SQL writers change rows.
    for table in ("documents", "chunks", "deprecated", "memory_revisions"):
        for event in ("INSERT", "UPDATE", "DELETE"):
            conn.execute(f"CREATE TRIGGER IF NOT EXISTS bk_{table}_{event.lower()} AFTER {event} ON {table} BEGIN UPDATE corpus_state SET generation=generation+1 WHERE id=1; END")
    checksum = hashlib.sha256(_EVIDENCE_SCHEMA.encode()).hexdigest()
    previous = conn.execute("SELECT checksum FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)).fetchone()
    if previous and previous[0] != checksum:
        raise ValueError("Migration checksum mismatch")
    conn.execute("INSERT OR IGNORE INTO schema_migrations VALUES (?,?,?)", (SCHEMA_VERSION,checksum,_now()))
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@_atomic
def upsert_document(
    conn: sqlite3.Connection,
    source: str,
    path: str,
    title: str,
    preamble: str,
    content: str,
    summary: str = "",
    *, completeness: str = "source", metadata: dict | None = None,
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
        row = conn.execute(
            "SELECT id FROM documents WHERE source=? AND path=?", (source, path),
        ).fetchone()
        revision = _record_revision(conn, source, path, content, completeness,
                                    {"title": title, "preamble": preamble, "summary": summary, **(metadata or {})}, _now())
        conn.execute("UPDATE documents SET revision_id=?, completeness=? WHERE id=?", (revision, completeness, row[0]))
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


@_atomic
def delete_source_documents(conn: sqlite3.Connection, source: str) -> None:
    """Delete all documents (and cascading chunks) for a source."""
    with _write_lock:
        conn.execute("DELETE FROM documents WHERE source=?", (source,))


# ---------------------------------------------------------------------------
# Chunks + vectors
# ---------------------------------------------------------------------------

def _validate_vectors(conn, doc_id, chunks, vectors, replacing, space):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or (len(chunks) and vectors.shape[1] == 0):
        raise ValueError("Expected one finite, nonempty vector per chunk")
    if not np.isfinite(vectors).all():
        raise ValueError("Vectors must be finite")
    registered = conn.execute("SELECT dimension FROM embedding_spaces WHERE id=?", (space,)).fetchone()
    if space != "legacy_unknown" and registered is None:
        raise ValueError("Embedding space must be registered before writing vectors")
    if registered and len(chunks) and registered[0] != vectors.shape[1]:
        raise ValueError("Vector dimension does not match its registered embedding space")
    for chunk in chunks:
        if not isinstance(chunk.get("text", ""), str):
            raise ValueError("Chunk text must be a string")
        json.dumps(chunk)  # Validate metadata before deleting anything.
    if len(chunks):
        rows = conn.execute("SELECT DISTINCT length(vector)/4 FROM chunks WHERE embedding_space_id=?" +
                            (" AND doc_id!=?" if replacing else ""),
                            (space, doc_id) if replacing else (space,)).fetchall()
        if any(row[0] != vectors.shape[1] for row in rows):
            raise ValueError("Vector dimensions differ within an embedding space")
    return vectors


def _write_chunks(conn, doc_id, chunks, vectors, *, replacing, embedding_space=None):
    space = embedding_space or "legacy_unknown"
    vectors = _validate_vectors(conn, doc_id, chunks, vectors, replacing, space)
    doc = conn.execute("SELECT revision_id FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        raise ValueError("Unknown document")
    if replacing:
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    for chunk, vector in zip(chunks, vectors):
        evidence_id = _record_segment(conn, doc[0], chunk.get("text", ""), chunk)
        conn.execute("""INSERT INTO chunks
            (doc_id,section_heading,start_line,end_line,text,tags,description,doc_preamble,
             indexed_at,vector,session_summary,full_session_path,session_turn_count,evidence_id,embedding_space_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (doc_id,chunk.get("section_heading",""),chunk.get("start_line",0),chunk.get("end_line",0),
             chunk.get("text",""),json.dumps(chunk.get("tags",[])),chunk.get("description",""),
             chunk.get("doc_preamble",""),chunk.get("indexed_at",_now()),vector.tobytes(),
             chunk.get("session_summary"),chunk.get("full_session_path"),chunk.get("session_turn_count"),evidence_id,space))


@_atomic
def upsert_chunks(conn, doc_id, chunks, vectors, *, embedding_space=None):
    """Validate and atomically replace a document's derived chunks; retain evidence history."""
    _write_chunks(conn, doc_id, chunks, vectors, replacing=True, embedding_space=embedding_space)


@_atomic
def append_chunks(conn, doc_id, chunks, vectors, *, embedding_space=None):
    _write_chunks(conn, doc_id, chunks, vectors, replacing=False, embedding_space=embedding_space)


def load_all_chunks(
    conn: sqlite3.Connection,
    source: str | None = None,
    *, embedding_profile: dict | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Load all chunk metadata and vectors. Returns (metadata_list, vector_matrix)."""
    _cols = """\
            c.id, d.source, d.path, d.title, c.section_heading,
            c.start_line, c.end_line, c.text, c.tags, c.description,
            c.doc_preamble, c.indexed_at, c.vector,
            c.session_summary, c.full_session_path, c.session_turn_count,
            d.summary, c.evidence_id, c.embedding_space_id, COALESCE(e.revision_id,d.revision_id), COALESCE(r.completeness,d.completeness)"""
    with _write_lock:
        if source:
            rows = conn.execute(
                f"SELECT {_cols} FROM chunks c JOIN documents d ON c.doc_id = d.id LEFT JOIN evidence_segments e ON e.id=c.evidence_id LEFT JOIN document_revisions r ON r.id=e.revision_id"
                " WHERE d.source = ? ORDER BY c.id",
                (source,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_cols} FROM chunks c JOIN documents d ON c.doc_id = d.id LEFT JOIN evidence_segments e ON e.id=c.evidence_id LEFT JOIN document_revisions r ON r.id=e.revision_id"
                " ORDER BY c.id",
            ).fetchall()

    metadata: list[dict] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        meta: dict = {
            "chunk_id": row[0],
            "evidence_id": row[17],
            "embedding_space_id": row[18],
            "revision_id": row[19],
            "completeness": row[20],
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
        raw = row[12]
        if isinstance(raw,bytes) and raw and len(raw) % 4 == 0:
            vector = np.frombuffer(raw,dtype=np.float32)
            vectors.append(vector if np.isfinite(vector).all() else np.empty(0,dtype=np.float32))
        else:
            vectors.append(np.empty(0,dtype=np.float32))

    if vectors:
        with _write_lock:
            spaces = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT id,metadata_json FROM embedding_spaces")}
        eligible = [bool(len(v)) and (embedding_profile is None or spaces.get(m["embedding_space_id"]) == embedding_profile)
                    for m,v in zip(metadata,vectors)]
        dimensions = {len(v) for v, ok in zip(vectors, eligible) if ok}
        dimension = next(iter(dimensions)) if len(dimensions) == 1 else 0
        for m, ok in zip(metadata, eligible):
            m["dense_eligible"] = bool(ok and dimension)
        vec_matrix = np.vstack([v if ok and dimension else np.zeros(dimension, dtype=np.float32)
                                for v,ok in zip(vectors,eligible)])
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

def load_hnsw_cache(conn: sqlite3.Connection, fingerprint: str | None = None) -> tuple[bytes, int] | None:
    with _write_lock:
        row = conn.execute("SELECT data, chunk_count, fingerprint FROM hnsw_cache WHERE id=1").fetchone()
    if row is None or (fingerprint is not None and row[2] != fingerprint):
        return None
    return row[0], row[1]


@_atomic
def save_hnsw_cache(conn: sqlite3.Connection, data: bytes, num_chunks: int, fingerprint: str | None = None) -> None:
    with _write_lock:
        conn.execute(
            """INSERT INTO hnsw_cache (id, data, chunk_count, updated_at, fingerprint)
               VALUES (1, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   data=excluded.data,
                   chunk_count=excluded.chunk_count,
                   updated_at=excluded.updated_at, fingerprint=excluded.fingerprint""",
            (data, num_chunks, _now(), fingerprint),
        )


# ---------------------------------------------------------------------------
# Source state
# ---------------------------------------------------------------------------

def load_source_state(conn: sqlite3.Connection, source_name: str) -> dict:
    with _write_lock:
        row = conn.execute(
            "SELECT commit_hash, mtimes, indexed_at, state_json FROM source_state WHERE source_name=?",
            (source_name,),
        ).fetchone()
    if row is None:
        return {}
    state: dict = json.loads(row[3]) if row[3] else {}
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


@_atomic
def save_source_state(conn: sqlite3.Connection, source_name: str, state: dict) -> None:
    with _write_lock:
        conn.execute(
            """INSERT INTO source_state (source_name, commit_hash, mtimes, indexed_at, state_json)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(source_name) DO UPDATE SET
                   commit_hash=excluded.commit_hash,
                   mtimes=excluded.mtimes,
                   indexed_at=excluded.indexed_at, state_json=excluded.state_json""",
            (
                source_name,
                state.get("commit_hash"),
                json.dumps(state.get("mtimes", {})),
                state.get("indexed_at", _now()),
                json.dumps(state),
            ),
        )


def load_all_source_state(conn: sqlite3.Connection) -> dict[str, dict]:
    """Load state for all sources, keyed by source name."""
    with _write_lock:
        names = [row[0] for row in conn.execute("SELECT source_name FROM source_state")]
        return {name: load_source_state(conn, name) for name in names}


# ---------------------------------------------------------------------------
# Deprecated
# ---------------------------------------------------------------------------

def load_deprecated(conn: sqlite3.Connection) -> dict[str, str]:
    with _write_lock:
        rows = conn.execute("SELECT path, reason FROM deprecated").fetchall()
    return dict(rows)


@_atomic
def save_deprecated(conn: sqlite3.Connection, path: str, reason: str) -> None:
    with _write_lock:
        conn.execute(
            """INSERT INTO deprecated (path, reason) VALUES (?, ?)
               ON CONFLICT(path) DO UPDATE SET reason=excluded.reason""",
            (path, reason),
        )


@_atomic
def remove_deprecated(conn: sqlite3.Connection, path: str) -> bool:
    with _write_lock:
        cur = conn.execute("DELETE FROM deprecated WHERE path=?", (path,))
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


@_atomic
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


# ---------------------------------------------------------------------------
# Query audit log
# ---------------------------------------------------------------------------

@_atomic
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

@_atomic
def gc_source(conn: sqlite3.Connection, source: str, live_paths: set[str]) -> int:
    """Delete documents (and cascading chunks) for paths no longer on disk.

    Returns number of documents removed.
    """
    with _write_lock:
        rows = conn.execute(
            "SELECT id, path FROM documents WHERE source=?", (source,),
        ).fetchall()
        canonical_live = {path.replace("\\", "/") for path in live_paths}
        stale_ids = [doc_id for doc_id, path in rows if path.replace("\\", "/") not in canonical_live]
        if not stale_ids:
            return 0
        placeholders = ",".join("?" * len(stale_ids))
        conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", stale_ids)
    return len(stale_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _record_revision(conn, source, path, content, completeness, metadata, observed):
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    identity = _digest([source, path, content_hash, completeness, metadata])
    conn.execute("INSERT OR IGNORE INTO document_revisions VALUES (?,?,?,?,?,?,?,?)",
                 (identity, source, path, content, content_hash, completeness,
                  json.dumps(metadata, ensure_ascii=False, sort_keys=True), observed))
    return identity


def _record_segment(conn, revision, text, metadata):
    stable = {k: v for k, v in metadata.items()
              if k not in {"indexed_at", "embedding_text", "full_content", "transcript_state"}}
    identity = _digest([revision, text, stable])
    conn.execute("INSERT OR IGNORE INTO evidence_segments VALUES (?,?,?,?,?)",
                 (identity, revision, text, json.dumps(stable, ensure_ascii=False), _now()))
    return identity


def corpus_generation(conn) -> int:
    with _write_lock:
        return conn.execute("SELECT generation FROM corpus_state WHERE id=1").fetchone()[0]


def index_fingerprint(metadata, vectors) -> str:
    digest = hashlib.sha256(np.asarray(vectors, dtype=np.float32).tobytes())
    digest.update(json.dumps([(m.get("chunk_id"), m.get("evidence_id"), m.get("embedding_space_id"))
                              for m in metadata], separators=(",", ":")).encode())
    digest.update(str(vectors.shape).encode())
    return digest.hexdigest()


@_atomic
def register_embedding_space(conn, metadata: dict, dimension: int) -> str:
    if dimension <= 0:
        raise ValueError("Embedding dimension must be positive")
    identity = _digest([metadata, dimension])
    conn.execute("INSERT OR IGNORE INTO embedding_spaces VALUES (?,?,?)",
                 (identity, dimension, json.dumps(metadata, sort_keys=True)))
    return identity


def get_evidence(conn, evidence_id: str) -> dict | None:
    with _write_lock:
        row = conn.execute("""SELECT e.id,e.text,e.metadata_json,r.id,r.source,r.path,
            r.content,r.content_hash,r.completeness,r.observed_at FROM evidence_segments e
            JOIN document_revisions r ON e.revision_id=r.id WHERE e.id=?""", (evidence_id,)).fetchone()
        if not row:
            return None
        deprecated = conn.execute("SELECT reason FROM deprecated WHERE path=?", (row[5],)).fetchone()
        current = conn.execute("SELECT revision_id FROM documents WHERE source=? AND path=?", (row[4],row[5])).fetchone()
    result = dict(zip(("id","text","metadata_json","revision_id","source","path","content",
                       "content_hash","completeness","observed_at"), row))
    result["metadata"] = json.loads(result.pop("metadata_json"))
    result["trust"] = "source_data"
    result["is_current"] = bool(current and current[0] == row[3])
    if deprecated:
        result.update(deprecated=True, deprecation_notice=deprecated[0])
    return result


@_atomic
def enqueue_ingestion(conn, chunks: list[dict]) -> str:
    payload = json.dumps(chunks, ensure_ascii=False, sort_keys=True)
    # Same pending snapshot is idempotent, but A -> B -> A is a new event.
    heads = []
    for chunk in chunks:
        row = conn.execute("SELECT job_id FROM ingestion_heads WHERE identity=?",
                           (_ingestion_identity(chunk),)).fetchone()
        heads.append(row[0] if row else None)
    identity = _digest([payload,heads])
    conn.execute("INSERT OR IGNORE INTO ingestion_jobs(id,payload_json,created_at) VALUES (?,?,?)",
                 (identity,payload,_now()))
    return identity


def pending_ingestion(conn) -> list[dict]:
    with _write_lock:
        rows = conn.execute("SELECT id,payload_json FROM ingestion_jobs WHERE status!='done' ORDER BY rowid").fetchall()
    return [{"id": r[0], "chunks": json.loads(r[1])} for r in rows]


def _ingestion_identity(chunk):
    return _digest(["transcript", chunk["transcript_path"]] if chunk.get("transcript_path")
                   else ["document", chunk["source"], chunk["path"]])


@_atomic
def accept_ingestion(conn, job_id: str, chunk: dict) -> bool:
    """Reserve an ordered live projection inside the caller's evidence transaction."""
    identity = _ingestion_identity(chunk)
    sequence = conn.execute("SELECT rowid FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone()
    if not sequence:
        raise ValueError("Unknown ingestion job")
    head = conn.execute("SELECT sequence FROM ingestion_heads WHERE identity=?", (identity,)).fetchone()
    if head and head[0] >= sequence[0]:
        return False
    conn.execute("INSERT INTO ingestion_heads VALUES (?,?,?) ON CONFLICT(identity) DO UPDATE SET job_id=excluded.job_id,sequence=excluded.sequence",
                 (identity,job_id,sequence[0]))
    return True


@_atomic
def finish_ingestion(conn, job_id: str, error: str | None = None):
    conn.execute("UPDATE ingestion_jobs SET status=?,attempts=attempts+1,error=? WHERE id=?",
                 ("failed" if error else "done", error, job_id))


@_atomic
def save_memory(conn, *, kind: str, scope: str, content: str, evidence_ids: list[str],
                memory_id: str | None = None, expected_revision: str | None = None,
                valid_from: str | None = None, valid_to: str | None = None,
                status: str = "active") -> dict:
    """Save an unverified, evidence-backed assertion; never convert source text to authority."""
    if kind not in {"fact", "decision", "preference", "procedure"}:
        raise ValueError("Unknown memory kind")
    if status not in {"active", "superseded", "conflicted", "retracted"}:
        raise ValueError("Unknown memory status")
    if not scope or not content.strip() or not evidence_ids:
        raise ValueError("Scope, content and evidence are required")
    from datetime import datetime, timezone
    def instant(value):
        if not value:
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    start, end = instant(valid_from), instant(valid_to)
    if start and end and start >= end:
        raise ValueError("Validity interval must be increasing")
    for evidence_id in evidence_ids:
        evidence = get_evidence(conn, evidence_id)
        if not evidence or evidence["source"] != scope:
            raise ValueError("Evidence must exist within the memory scope")
    memory_id = memory_id or uuid4().hex
    existing = conn.execute("SELECT kind,scope FROM memory_entries WHERE id=?", (memory_id,)).fetchone()
    if existing and existing != (kind,scope):
        raise ValueError("Memory kind and scope are immutable")
    latest = conn.execute("SELECT id FROM memory_revisions WHERE memory_id=? ORDER BY rowid DESC LIMIT 1", (memory_id,)).fetchone()
    previous = latest[0] if latest else None
    if expected_revision != previous:
        raise ValueError("Memory revision conflict; reread before updating")
    if previous:
        old = conn.execute("SELECT content,evidence_ids,valid_from,valid_to,status FROM memory_revisions WHERE id=?", (previous,)).fetchone()
        if old == (content,json.dumps(evidence_ids),valid_from,valid_to,status):
            return {"id":memory_id,"revision_id":previous,"verification":"unverified","status":status,"no_op":True}
    revision = _digest([memory_id, previous, content, evidence_ids, valid_from, valid_to, status])
    conn.execute("INSERT OR IGNORE INTO memory_entries VALUES (?,?,?,?)", (memory_id,kind,scope,_now()))
    conn.execute("INSERT INTO memory_revisions VALUES (?,?,?,?,?,?,?,?,?)",
                 (revision,memory_id,content,json.dumps(evidence_ids),valid_from,valid_to,status,previous,_now()))
    return {"id": memory_id, "revision_id": revision, "verification": "unverified", "status": status}


def list_memories(conn, scope: str, *, include_history: bool = False, limit: int = 50) -> list[dict]:
    with _write_lock:
        rows = conn.execute("""SELECT m.id,m.kind,m.scope,r.id,r.content,r.evidence_ids,
             r.valid_from,r.valid_to,r.status,r.previous_id,r.observed_at FROM memory_entries m
             JOIN memory_revisions r ON r.memory_id=m.id WHERE m.scope=?""" +
             ("" if include_history else " AND r.rowid=(SELECT MAX(x.rowid) FROM memory_revisions x WHERE x.memory_id=m.id)") +
             " ORDER BY r.rowid DESC LIMIT ?", (scope, max(1,min(limit,200)))).fetchall()
    results = []
    for row in rows:
        entry = dict(zip(("id","kind","scope","revision_id","content","evidence_ids","valid_from",
                          "valid_to","status","previous_revision","observed_at"), row))
        entry["evidence_ids"] = json.loads(entry["evidence_ids"])
        entry["verification"] = "unverified"
        evidence = [get_evidence(conn,key) for key in entry["evidence_ids"]]
        entry["sources_current"] = all(item and item["is_current"] and not item.get("deprecated") for item in evidence)
        results.append(entry)
    return results
