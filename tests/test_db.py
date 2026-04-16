"""Tests for agents_rag.db: schema init, CRUD, HNSW cache, state management."""

from __future__ import annotations

import numpy as np

from agents_rag.db import (
    chunk_count,
    get_document_content,
    get_document_id,
    init_db,
    load_all_chunks,
    load_deprecated,
    load_hnsw_cache,
    load_source_state,
    load_transcript_state,
    remove_deprecated,
    save_deprecated,
    save_hnsw_cache,
    save_source_state,
    save_transcript_state,
    upsert_chunks,
    upsert_document,
    delete_source_documents,
)


def test_init_db_creates_tables(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "documents" in tables
    assert "chunks" in tables
    assert "source_state" in tables
    assert "deprecated" in tables
    assert "transcript_state" in tables
    assert "hnsw_cache" in tables
    conn.close()


def test_upsert_document_and_get(tmp_db):
    doc_id = upsert_document(tmp_db, "src", "a.md", "Title", "Intro", "Full content")
    assert doc_id is not None
    assert get_document_id(tmp_db, "src", "a.md") == doc_id
    assert get_document_content(tmp_db, "src", "a.md") == "Full content"


def test_upsert_document_replaces(tmp_db):
    upsert_document(tmp_db, "src", "a.md", "V1", "", "old")
    upsert_document(tmp_db, "src", "a.md", "V2", "", "new")
    assert get_document_content(tmp_db, "src", "a.md") == "new"


def test_upsert_chunks_and_load(tmp_db):
    doc_id = upsert_document(tmp_db, "src", "a.md", "", "", "content")
    vecs = np.random.randn(2, 384).astype(np.float32)
    chunks = [
        {"text": "chunk 1", "section_heading": "s1", "tags": ["a"]},
        {"text": "chunk 2", "section_heading": "s2"},
    ]
    upsert_chunks(tmp_db, doc_id, chunks, vecs)

    meta, loaded_vecs = load_all_chunks(tmp_db)
    assert len(meta) == 2
    assert loaded_vecs.shape == (2, 384)
    assert np.allclose(loaded_vecs, vecs, atol=1e-5)
    assert meta[0]["text"] == "chunk 1"
    assert meta[0]["tags"] == ["a"]


def test_upsert_chunks_replaces_old(tmp_db):
    doc_id = upsert_document(tmp_db, "src", "a.md", "", "", "content")
    vecs1 = np.random.randn(3, 384).astype(np.float32)
    upsert_chunks(tmp_db, doc_id, [{"text": f"c{i}"} for i in range(3)], vecs1)
    assert chunk_count(tmp_db) == 3

    vecs2 = np.random.randn(1, 384).astype(np.float32)
    upsert_chunks(tmp_db, doc_id, [{"text": "replaced"}], vecs2)
    assert chunk_count(tmp_db) == 1

    meta, _ = load_all_chunks(tmp_db)
    assert meta[0]["text"] == "replaced"


def test_delete_source_cascades(tmp_db):
    doc_id = upsert_document(tmp_db, "src", "a.md", "", "", "content")
    vecs = np.random.randn(2, 384).astype(np.float32)
    upsert_chunks(tmp_db, doc_id, [{"text": "c1"}, {"text": "c2"}], vecs)
    assert chunk_count(tmp_db) == 2

    delete_source_documents(tmp_db, "src")
    assert chunk_count(tmp_db) == 0
    assert get_document_id(tmp_db, "src", "a.md") is None


def test_load_all_chunks_with_source_filter(tmp_db):
    d1 = upsert_document(tmp_db, "a", "f1.md", "", "", "")
    d2 = upsert_document(tmp_db, "b", "f2.md", "", "", "")
    v1 = np.random.randn(1, 384).astype(np.float32)
    v2 = np.random.randn(1, 384).astype(np.float32)
    upsert_chunks(tmp_db, d1, [{"text": "x"}], v1)
    upsert_chunks(tmp_db, d2, [{"text": "y"}], v2)

    meta_a, _ = load_all_chunks(tmp_db, source="a")
    meta_b, _ = load_all_chunks(tmp_db, source="b")
    assert len(meta_a) == 1
    assert meta_a[0]["source"] == "a"
    assert len(meta_b) == 1
    assert meta_b[0]["source"] == "b"


def test_hnsw_cache_roundtrip(tmp_db):
    assert load_hnsw_cache(tmp_db) is None

    blob = b"\x00\x01\x02\x03"
    save_hnsw_cache(tmp_db, blob, 42)

    result = load_hnsw_cache(tmp_db)
    assert result is not None
    loaded_blob, count = result
    assert loaded_blob == blob
    assert count == 42


def test_source_state_roundtrip(tmp_db):
    assert load_source_state(tmp_db, "test") == {}

    state = {"commit_hash": "abc123", "mtimes": {"a.md": 1.0}, "indexed_at": "2024-01-01T00:00:00Z"}
    save_source_state(tmp_db, "test", state)

    loaded = load_source_state(tmp_db, "test")
    assert loaded["commit_hash"] == "abc123"
    assert loaded["mtimes"] == {"a.md": 1.0}


def test_deprecated_roundtrip(tmp_db):
    assert load_deprecated(tmp_db) == {}

    save_deprecated(tmp_db, "old.md", "outdated")
    assert load_deprecated(tmp_db) == {"old.md": "outdated"}

    assert remove_deprecated(tmp_db, "old.md") is True
    assert load_deprecated(tmp_db) == {}
    assert remove_deprecated(tmp_db, "old.md") is False


def test_transcript_state_roundtrip(tmp_db):
    path = "/some/transcript.txt"
    assert load_transcript_state(tmp_db, path) == {}

    state = {"turns_processed": 5, "byte_offset": 1234}
    save_transcript_state(tmp_db, path, state)

    loaded = load_transcript_state(tmp_db, path)
    assert loaded["turns_processed"] == 5
    assert loaded["byte_offset"] == 1234
