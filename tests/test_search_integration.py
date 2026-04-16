"""Integration tests: build index from fixture docs, query, verify results."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agents_rag.db import load_all_chunks, upsert_chunks, upsert_document
from agents_rag.searcher import Searcher
from tests.conftest import make_source_cfg


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic fake embedder keyed on text content."""
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % (2**31))
        vecs.append(rng.randn(384).astype(np.float32))
    return np.array(vecs)


def _build_test_index(doc_tree: Path, conn: sqlite3.Connection) -> None:
    """Build a small index from doc_tree using fake embeddings."""
    from agents_rag.indexer import reindex_source

    cfg_docs = make_source_cfg(doc_tree, name="docs", file_types=[".md"])
    cfg_code = make_source_cfg(doc_tree, name="code", file_types=[".py"])

    for cfg in [cfg_docs, cfg_code]:
        reindex_source(cfg, conn, use_gemma=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_returns_results(mock_search_embed, mock_idx_embed, doc_tree, tmp_db):
    _build_test_index(doc_tree, tmp_db)

    searcher = Searcher(tmp_db)
    assert searcher.load() is True

    results = searcher.query("installation guide", top_k=5)
    assert len(results) > 0
    assert all("source" in r and "path" in r for r in results)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_source_filter(mock_search_embed, mock_idx_embed, doc_tree, tmp_db):
    _build_test_index(doc_tree, tmp_db)

    searcher = Searcher(tmp_db)
    searcher.load()

    docs_results = searcher.query("hello function", top_k=10, source="docs")
    code_results = searcher.query("hello function", top_k=10, source="code")

    assert all(r["source"] == "docs" for r in docs_results)
    assert all(r["source"] == "code" for r in code_results)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_deprecation_flag(mock_search_embed, mock_idx_embed, doc_tree, tmp_db):
    from agents_rag.db import save_deprecated

    _build_test_index(doc_tree, tmp_db)
    save_deprecated(tmp_db, "guide.md", "outdated content")

    searcher = Searcher(tmp_db)
    searcher.load()

    results = searcher.query("installation", top_k=10)
    guide_results = [r for r in results if r["path"] == "guide.md"]
    for r in guide_results:
        assert r.get("deprecated") is True
        assert "outdated" in r.get("deprecation_notice", "").lower()


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_score_gap_keeps_at_least_one(mock_search_embed, mock_idx_embed, doc_tree, tmp_db):
    _build_test_index(doc_tree, tmp_db)

    searcher = Searcher(tmp_db)
    searcher.load()

    results = searcher.query("completely unrelated xyzzy query gibberish", top_k=5)
    assert len(results) >= 1


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_chunk_count_and_stats(mock_search_embed, mock_idx_embed, doc_tree, tmp_db):
    _build_test_index(doc_tree, tmp_db)

    searcher = Searcher(tmp_db)
    searcher.load()

    assert searcher.chunk_count > 0
    stats = searcher.source_stats()
    assert "docs" in stats
    assert "code" in stats
    assert stats["docs"] > 0
    assert stats["code"] > 0


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_reload_picks_up_changes(mock_search_embed, mock_idx_embed, doc_tree, tmp_db):
    _build_test_index(doc_tree, tmp_db)

    searcher = Searcher(tmp_db)
    searcher.load()
    count_before = searcher.chunk_count

    # Add an extra chunk directly via DB
    doc_id = upsert_document(tmp_db, "extra", "new.md", "", "", "brand new doc")
    extra_vecs = _fake_embed(["brand new doc"])
    upsert_chunks(tmp_db, doc_id, [
        {"source": "extra", "path": "new.md", "text": "brand new doc",
         "section_heading": "", "tags": [], "description": ""},
    ], extra_vecs)

    assert searcher.reload() is True
    assert searcher.chunk_count == count_before + 1
