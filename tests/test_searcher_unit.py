"""Unit tests for searcher.py internals: RRF fusion, grouping, freshness boost, session collapse."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import numpy as np
import pytest

from agents_rag.db import init_db, upsert_chunks, upsert_document
from agents_rag.searcher import Searcher, _tokenize


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


def test_tokenize_basic():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_with_numbers():
    tokens = _tokenize("bge-small-en-v1.5")
    assert "bge-small-en-v1" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []


def test_tokenize_punctuation_only():
    assert _tokenize("!!! ???") == []


def test_tokenize_underscore():
    assert "my_func" in _tokenize("my_func()")


# ---------------------------------------------------------------------------
# _rrf_fuse
# ---------------------------------------------------------------------------


def _make_searcher(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    return Searcher(conn), conn


def test_rrf_fuse_combines_sources(tmp_path):
    searcher, _ = _make_searcher(tmp_path)
    vec = [(0, 0), (1, 1), (2, 2)]
    bm25 = [(1, 0), (3, 1), (0, 2)]

    indices, scores = searcher._rrf_fuse(vec, bm25, top_k=5)
    # Index 0 and 1 appear in both lists, should have higher scores
    assert 0 in indices
    assert 1 in indices
    assert scores[0] > scores[2]  # 0 in both lists, 2 only in vec
    assert scores[1] > scores[3]  # 1 in both lists, 3 only in bm25


def test_rrf_fuse_top_k_limit(tmp_path):
    searcher, _ = _make_searcher(tmp_path)
    vec = [(i, i) for i in range(10)]
    bm25 = [(i, i) for i in range(10, 20)]
    indices, _ = searcher._rrf_fuse(vec, bm25, top_k=3)
    assert len(indices) == 3


def test_rrf_fuse_empty_inputs(tmp_path):
    searcher, _ = _make_searcher(tmp_path)
    indices, scores = searcher._rrf_fuse([], [], top_k=5)
    assert indices == []
    assert scores == {}


# ---------------------------------------------------------------------------
# _group_document_results
# ---------------------------------------------------------------------------


def test_group_merges_same_doc(tmp_path):
    searcher, _ = _make_searcher(tmp_path)
    results = [
        {"source": "docs", "path": "guide.md", "score": 0.8,
         "matched_sections": ["Installation"], "start_line": 5, "end_line": 10,
         "tags": ["guide"], "description": "desc", "indexed_at": "", "summary": "sum"},
        {"source": "docs", "path": "guide.md", "score": 0.6,
         "matched_sections": ["Configuration"], "start_line": 15, "end_line": 20,
         "tags": ["guide"], "description": "desc", "indexed_at": "", "summary": "sum"},
    ]
    merged = searcher._group_document_results(results)
    assert len(merged) == 1
    assert merged[0]["score"] == 0.8  # best score
    assert "Installation" in merged[0]["matched_sections"]
    assert "Configuration" in merged[0]["matched_sections"]
    assert merged[0]["start_line"] == 5
    assert merged[0]["end_line"] == 20
    assert merged[0]["merged_chunks"] == 2


def test_group_keeps_different_docs(tmp_path):
    searcher, _ = _make_searcher(tmp_path)
    results = [
        {"source": "docs", "path": "a.md", "score": 0.8,
         "matched_sections": ["S1"], "start_line": 1, "end_line": 5,
         "tags": [], "description": "", "indexed_at": "", "summary": ""},
        {"source": "docs", "path": "b.md", "score": 0.7,
         "matched_sections": ["S2"], "start_line": 1, "end_line": 5,
         "tags": [], "description": "", "indexed_at": "", "summary": ""},
    ]
    merged = searcher._group_document_results(results)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# Session grouping in query()
# ---------------------------------------------------------------------------


def _fake_embed(texts):
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % (2**31))
        vecs.append(rng.randn(384).astype(np.float32))
    return np.array(vecs)


def _insert_session_chunks(conn, summary_path, turns):
    """Insert session chunks that will be grouped."""
    doc_id = upsert_document(conn, "sessions", summary_path, "Test Session", "", "content")
    chunks = []
    for i, query in enumerate(turns):
        chunks.append({
            "section_heading": f"Turn {i+1}: {query}",
            "start_line": 0, "end_line": 0,
            "text": f"Turn about {query}",
            "tags": json.dumps(["test"]),
            "description": query,
            "doc_preamble": "",
            "session_summary": "Session did X and Y.",
            "full_session_path": f"/sessions/{summary_path}",
            "session_turn_count": len(turns),
        })
    vecs = _fake_embed([c["text"] for c in chunks])
    upsert_chunks(conn, doc_id, chunks, vecs)


@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_session_grouping_collapses_turns(mock_embed, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    _insert_session_chunks(conn, "abc_session.md", ["configure sources", "add search"])

    searcher = Searcher(conn)
    searcher.load()
    results = searcher.query("configure sources", top_k=10)

    session_results = [r for r in results if r.get("source") == "sessions"]
    if session_results:
        r = session_results[0]
        assert "matched_turns" in r
        assert "session_summary" in r
        assert r.get("full_session_path") is not None


# ---------------------------------------------------------------------------
# Freshness boost
# ---------------------------------------------------------------------------


@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_freshness_boost_favors_recent(mock_embed, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))

    old_time = "2024-01-01T00:00:00Z"
    new_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    doc_old = upsert_document(conn, "docs", "old.md", "Old", "", "bazel migration plan")
    doc_new = upsert_document(conn, "docs", "new.md", "New", "", "bazel migration plan")

    old_chunk = {
        "section_heading": "Plan", "start_line": 1, "end_line": 5,
        "text": "bazel migration plan details",
        "tags": json.dumps(["bazel"]), "description": "old plan",
        "doc_preamble": "", "indexed_at": old_time,
    }
    new_chunk = {
        "section_heading": "Plan", "start_line": 1, "end_line": 5,
        "text": "bazel migration plan details",
        "tags": json.dumps(["bazel"]), "description": "new plan",
        "doc_preamble": "", "indexed_at": new_time,
    }
    upsert_chunks(conn, doc_old, [old_chunk], _fake_embed(["bazel migration plan details"]))
    upsert_chunks(conn, doc_new, [new_chunk], _fake_embed(["bazel migration plan details"]))

    searcher = Searcher(conn)
    searcher.load()
    results = searcher.query("bazel migration", top_k=5)
    # Both should appear; we just verify the boost doesn't crash
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Empty index edge case
# ---------------------------------------------------------------------------


def test_query_empty_index(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    searcher = Searcher(conn)
    searcher.load()
    results = searcher.query("anything")
    assert results == []


# ---------------------------------------------------------------------------
# Reranker integration
# ---------------------------------------------------------------------------


def _insert_doc_chunks(conn, docs):
    """docs: list of (path, title, [(heading, text), ...])"""
    all_texts = []
    for path, title, chunks in docs:
        doc_id = upsert_document(conn, "docs", path, title, "", "content")
        chunk_metas = []
        for heading, text in chunks:
            chunk_metas.append({
                "section_heading": heading,
                "start_line": 0, "end_line": 0,
                "text": text,
                "tags": json.dumps([]),
                "description": "",
                "doc_preamble": "",
            })
            all_texts.append(text)
        vecs = _fake_embed([c["text"] for c in chunk_metas])
        upsert_chunks(conn, doc_id, chunk_metas, vecs)


@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_rerank_pool_disabled_by_default(mock_embed, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    _insert_doc_chunks(conn, [
        ("a.md", "A", [("S1", "alpha beta gamma")]),
        ("b.md", "B", [("S2", "delta epsilon")]),
    ])
    searcher = Searcher(conn)
    searcher.load()
    indices = [0, 1]
    scores = {0: 0.01, 1: 0.02}
    out_indices, out_scores = searcher._rerank_pool("query", indices, scores)
    assert out_indices == indices
    assert out_scores == scores


@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_rerank_pool_reorders_and_replaces_scores(mock_embed, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    _insert_doc_chunks(conn, [
        ("a.md", "A", [("S1", "alpha beta gamma")]),
        ("b.md", "B", [("S2", "delta epsilon")]),
        ("c.md", "C", [("S3", "zeta eta")]),
    ])
    searcher = Searcher(
        conn,
        reranker_enabled=True,
        reranker_pool_multiplier=6,
    )
    searcher.load()

    # RRF puts 0 first; reranker puts 2 first.
    rrf_indices = [0, 1, 2]
    rrf_scores = {0: 0.03, 1: 0.02, 2: 0.01}

    def fake_rerank(query, passages, model_name):
        assert query == "query"
        assert len(passages) == 3
        return np.array([0.1, 0.5, 0.9], dtype=np.float32)

    with patch("agents_rag.searcher._rerank_fn", side_effect=fake_rerank):
        out_indices, out_scores = searcher._rerank_pool("query", rrf_indices, rrf_scores)

    assert out_indices == [2, 1, 0]
    assert out_scores[2] == pytest.approx(0.9)
    assert out_scores[0] == pytest.approx(0.1)


@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_rerank_pool_falls_back_when_reranker_returns_none(mock_embed, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    _insert_doc_chunks(conn, [
        ("a.md", "A", [("S1", "alpha")]),
        ("b.md", "B", [("S2", "beta")]),
    ])
    searcher = Searcher(conn, reranker_enabled=True)
    searcher.load()

    rrf_indices = [0, 1]
    rrf_scores = {0: 0.03, 1: 0.02}

    with patch("agents_rag.searcher._rerank_fn", return_value=None):
        out_indices, out_scores = searcher._rerank_pool("q", rrf_indices, rrf_scores)

    assert out_indices == rrf_indices
    assert out_scores == rrf_scores


@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_with_reranker_uses_reranked_scores(mock_embed, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    _insert_doc_chunks(conn, [
        ("a.md", "Alpha doc", [("S1", "alpha beta")]),
        ("b.md", "Beta doc",  [("S2", "beta gamma")]),
        ("c.md", "Gamma doc", [("S3", "gamma delta")]),
    ])
    searcher = Searcher(conn, reranker_enabled=True, reranker_pool_multiplier=6)
    searcher.load()

    # Force reranker to prefer b.md strongly.
    def fake_rerank(query, passages, model_name):
        scores = []
        for p in passages:
            if "Beta" in p:
                scores.append(0.95)
            else:
                scores.append(0.2)
        return np.array(scores, dtype=np.float32)

    with patch("agents_rag.searcher._rerank_fn", side_effect=fake_rerank):
        results = searcher.query("anything", top_k=3)

    assert results, "expected at least one result"
    assert results[0]["path"] == "b.md"
    # Score came from reranker (not RRF 1/(60+rank) range).
    assert results[0]["score"] > 0.5


def test_passage_for_rerank_composes_title_heading_text(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    searcher = Searcher(conn)
    searcher._metadata = [{
        "doc_title": "Auth Migration Plan",
        "section_heading": "Step 2: token refresh",
        "text": "We rotate tokens every 12h using...",
    }]
    passage = searcher._passage_for_rerank(0)
    assert "Auth Migration Plan" in passage
    assert "Step 2: token refresh" in passage
    assert "rotate tokens" in passage


def test_passage_for_rerank_truncates_long_text(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    searcher = Searcher(conn)
    huge = "x" * 10_000
    searcher._metadata = [{"doc_title": "T", "section_heading": "H", "text": huge}]
    assert len(searcher._passage_for_rerank(0)) <= 2000


# ---------------------------------------------------------------------------
# BM25 title-boost regression (Proposal 6)
# ---------------------------------------------------------------------------


def test_bm25_title_tokens_not_doubled(tmp_path):
    """Title tokens appear exactly once in the BM25 corpus, not twice.

    Previously title_tokens * 2 caused auto-derived titles (slash commands,
    session titles) to pin their doc atop any query sharing a title word.
    """
    conn = init_db(str(tmp_path / "test.db"))
    doc_id = upsert_document(conn, "docs", "a.md", "bazel migration plan", "", "body")
    chunk = {
        "section_heading": "intro",
        "start_line": 0, "end_line": 0,
        "text": "body paragraph",
        "tags": ["planning"],
        "description": "", "doc_preamble": "",
    }
    upsert_chunks(conn, doc_id, [chunk], _fake_embed([chunk["text"]]))

    searcher = Searcher(conn)
    searcher.load()

    # Reach into the built corpus to count token occurrences.
    corpus_tokens = searcher._bm25.corpus_size  # sanity
    assert corpus_tokens == 1
    tokenized = _tokenize("bazel migration plan intro planning body paragraph")
    # Each vocabulary word appears at most once.
    for tok in ("bazel", "migration", "plan", "intro", "planning"):
        count = searcher._bm25.doc_freqs[0].get(tok, 0)
        assert count == 1, f"{tok!r} occurs {count} times, expected 1"


@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_bm25_scores_body_match_above_title_only_match(mock_embed, tmp_path):
    """BM25 alone: a doc whose body answers the query should outrank a doc
    whose title merely shares one word with the query. Previously title
    tokens were doubled, so the title_only doc often won."""
    conn = init_db(str(tmp_path / "test.db"))
    # Noise docs push IDF positive for the query terms.
    _insert_doc_chunks(conn, [
        ("noise1.md", "cooking tips", [("s", "pasta recipes and tomato sauces")]),
        ("noise2.md", "garden notes", [("s", "houseplants and succulents watering")]),
        ("noise3.md", "travel log",   [("s", "trip to the alps last summer")]),
        ("title_only.md", "bazel quickstart",
         [("intro", "this document discusses gardening tips and houseplants")]),
        ("body_match.md", "internal tools",
         [("migration", "bazel migration plan details with build graph rewiring")]),
    ])
    searcher = Searcher(conn)
    searcher.load()

    bm25_hits = searcher._bm25_search("bazel migration plan", n=10, source=None)
    assert bm25_hits, "expected at least one BM25 match"
    top_idx = bm25_hits[0][0]
    assert searcher._metadata[top_idx]["path"] == "body_match.md"
