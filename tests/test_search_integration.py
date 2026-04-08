"""Integration tests: build index from fixture docs, query, verify results."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from agents_rag.indexer import save_index
from agents_rag.searcher import Searcher, save_deprecated
from tests.conftest import make_source_cfg


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic fake embedder keyed on text content."""
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % (2**31))
        vecs.append(rng.randn(384).astype(np.float32))
    return np.array(vecs)


def _build_test_index(doc_tree: Path, tmp_cache: str) -> None:
    """Build a small index from doc_tree using fake embeddings."""
    from agents_rag.indexer import reindex_source

    cfg_docs = make_source_cfg(doc_tree, name="docs", file_types=[".md"])
    cfg_code = make_source_cfg(doc_tree, name="code", file_types=[".py"])

    meta_all = []
    vecs_all = []

    for cfg in [cfg_docs, cfg_code]:
        meta, vecs, _ = reindex_source(cfg, tmp_cache, use_gemma=False)
        meta_all.extend(meta)
        if len(vecs) > 0:
            vecs_all.append(vecs)

    merged_vecs = np.vstack(vecs_all) if vecs_all else np.empty((0, 384))
    save_index(tmp_cache, merged_vecs, meta_all)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_returns_results(mock_search_embed, mock_idx_embed, doc_tree, tmp_cache):
    """Querying a populated index returns non-empty results."""
    _build_test_index(doc_tree, tmp_cache)

    searcher = Searcher(tmp_cache)
    assert searcher.load() is True

    results = searcher.query("installation guide", top_k=5)
    assert len(results) > 0
    assert all("source" in r and "path" in r for r in results)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_source_filter(mock_search_embed, mock_idx_embed, doc_tree, tmp_cache):
    """Filtering by source returns only chunks from that source."""
    _build_test_index(doc_tree, tmp_cache)

    searcher = Searcher(tmp_cache)
    searcher.load()

    docs_results = searcher.query("hello function", top_k=10, source="docs")
    code_results = searcher.query("hello function", top_k=10, source="code")

    assert all(r["source"] == "docs" for r in docs_results)
    assert all(r["source"] == "code" for r in code_results)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_deprecation_flag(mock_search_embed, mock_idx_embed, doc_tree, tmp_cache):
    """Deprecated docs get flagged in results."""
    _build_test_index(doc_tree, tmp_cache)

    save_deprecated(tmp_cache, {"guide.md": "outdated content"})

    searcher = Searcher(tmp_cache)
    searcher.load()

    results = searcher.query("installation", top_k=10)
    guide_results = [r for r in results if r["path"] == "guide.md"]
    for r in guide_results:
        assert r.get("deprecated") is True
        assert "outdated" in r.get("deprecation_notice", "").lower()


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_score_gap_keeps_at_least_one(mock_search_embed, mock_idx_embed, doc_tree, tmp_cache):
    """Score-gap filtering should never drop all results."""
    _build_test_index(doc_tree, tmp_cache)

    searcher = Searcher(tmp_cache)
    searcher.load()

    results = searcher.query("completely unrelated xyzzy query gibberish", top_k=5)
    assert len(results) >= 1


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_chunk_count_and_stats(mock_search_embed, mock_idx_embed, doc_tree, tmp_cache):
    """Searcher reports correct chunk counts and source stats."""
    _build_test_index(doc_tree, tmp_cache)

    searcher = Searcher(tmp_cache)
    searcher.load()

    assert searcher.chunk_count > 0
    stats = searcher.source_stats()
    assert "docs" in stats
    assert "code" in stats
    assert stats["docs"] > 0
    assert stats["code"] > 0


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_reload_picks_up_changes(mock_search_embed, mock_idx_embed, doc_tree, tmp_cache):
    """Reload after saving new data picks up the changes."""
    _build_test_index(doc_tree, tmp_cache)

    searcher = Searcher(tmp_cache)
    searcher.load()
    count_before = searcher.chunk_count

    # Add extra chunks directly
    from agents_rag.indexer import load_index
    vecs, meta = load_index(tmp_cache)
    extra_meta = [{"source": "extra", "path": "new.md", "text": "brand new doc",
                   "section_heading": "", "tags": [], "description": ""}]
    extra_vecs = _fake_embed(["brand new doc"])
    merged_vecs = np.vstack([vecs, extra_vecs])
    save_index(tmp_cache, merged_vecs, meta + extra_meta)

    assert searcher.reload() is True
    assert searcher.chunk_count == count_before + 1
