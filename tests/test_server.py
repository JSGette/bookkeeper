"""Tests for agents_rag/server.py: config parsing, app factory, tool-level logic."""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from agents_rag.config import load_config, parse_sources, validate_config


# ---------------------------------------------------------------------------
# Config parsing and validation
# ---------------------------------------------------------------------------


def test_validate_config_valid():
    config = {"sources": [{"name": "test", "path": "/tmp"}]}
    errors = validate_config(config)
    path_errors = [e for e in errors if "missing required" in e]
    assert path_errors == []


def test_validate_config_missing_name():
    config = {"sources": [{"path": "/tmp"}]}
    errors = validate_config(config)
    assert any("name" in e for e in errors)


def test_validate_config_missing_path():
    config = {"sources": [{"name": "test"}]}
    errors = validate_config(config)
    assert any("path" in e for e in errors)


def test_validate_config_bad_sessions():
    config = {"sessions": "not a dict"}
    errors = validate_config(config)
    assert any("sessions" in e for e in errors)


def test_parse_sources_empty():
    assert parse_sources({}) == []


def test_parse_sources_basic(tmp_path):
    src = {"name": "t", "path": str(tmp_path)}
    result = parse_sources({"sources": [src]})
    assert len(result) == 1
    assert result[0].name == "t"
    assert result[0].index_enabled is True


def test_load_config_from_env(tmp_path):
    cfg = tmp_path / "test.yaml"
    cfg.write_text("sources: []\n")
    with patch.dict(os.environ, {"AGENTS_RAG_CONFIG": str(cfg)}):
        config = load_config()
    assert config == {"sources": []}


def test_load_usage_guide_handles_read_error(monkeypatch):
    from agents_rag import server

    monkeypatch.setattr("pathlib.Path.is_file", lambda _self: True)

    def _raise_read_error(_self, encoding="utf-8"):  # pragma: no cover - called via monkeypatch
        raise OSError("disk read error")

    monkeypatch.setattr("pathlib.Path.read_text", _raise_read_error)
    out = server._load_usage_guide()
    assert "Usage guide file not found." in out


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _fake_embed(texts):
    vecs = []
    for t in texts:
        seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "little")
        rng = np.random.RandomState(seed)
        vecs.append(rng.randn(384).astype(np.float32))
    return np.array(vecs)


@pytest.fixture(autouse=True)
def _offline_models(monkeypatch):
    """Patch inference before fixtures index data, as well as during test calls."""
    from agents_rag import indexer, reranker, searcher, server

    monkeypatch.setattr(indexer, "embed_texts", _fake_embed)
    monkeypatch.setattr(searcher, "embed_texts", _fake_embed)
    monkeypatch.setattr(server, "embed_texts", _fake_embed)
    monkeypatch.setattr(server.ollama_client, "status", lambda: {"available": False})
    monkeypatch.setattr(reranker, "warm", lambda model_name: False)
    monkeypatch.setattr(reranker, "_get_reranker", lambda model_name: None)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_create_app_registers_tools(mock_s, mock_i, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n\nContent.")

    config = {
        "sources": [{"name": "test", "path": str(docs), "file_types": [".md"]}],
        "cache_dir": str(tmp_path / "cache"),
        "use_gemma": False,
        "sessions": {
            "transcript_dirs": [],
            "output_dir": str(tmp_path / "sessions"),
            "watch_interval_minutes": 0,
        },
    }

    from agents_rag.server import create_app
    import asyncio
    mcp = create_app(config_override=config)
    assert mcp.name == "bookkeeper"
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    expected = {
        "query_knowledge_base", "get_full_document", "reindex",
        "save_session", "list_sources", "deprecate_doc",
        "query_audit_log", "distill_transcripts", "status", "gc",
        "get_context_block",
    }
    assert expected.issubset(tool_names), f"Missing: {expected - tool_names}"


# ---------------------------------------------------------------------------
# Tool functions tested via direct calls (bypassing FastMCP dispatch)
# ---------------------------------------------------------------------------


@pytest.fixture
def server_env(tmp_path):
    """Set up a server environment with pre-indexed data for tool testing."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(textwrap.dedent("""\
        # Guide

        ## Setup

        Run pip install to set things up.

        ## Usage

        Call the query tool.
    """))
    sessions_dir = tmp_path / "sessions_out"
    sessions_dir.mkdir()

    config = {
        "sources": [{"name": "test", "path": str(docs), "file_types": [".md"]}],
        "cache_dir": str(tmp_path / "cache"),
        "use_gemma": False,
        "sessions": {
            "transcript_dirs": [],
            "output_dir": str(sessions_dir),
            "watch_interval_minutes": 0,
        },
    }

    from agents_rag import db
    from agents_rag.config import parse_sources
    from agents_rag.indexer import reindex_all
    from agents_rag.searcher import Searcher

    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    db_path = os.path.join(cache_dir, "bookkeeper.db")
    conn = db.init_db(db_path)
    sources = parse_sources(config)
    reindex_all(sources, conn, use_gemma=False)

    searcher = Searcher(conn)
    searcher.load()

    yield {
        "conn": conn,
        "searcher": searcher,
        "sources": sources,
        "config": config,
        "tmp_path": tmp_path,
        "sessions_dir": sessions_dir,
        "db": db,
    }
    conn.close()


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_knowledge_base_returns_results(mock_s, mock_i, server_env):
    env = server_env
    results = env["searcher"].query("setup guide", top_k=5)
    assert len(results) > 0
    for r in results:
        assert "source" in r
        assert "path" in r


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_empty_after_delete(mock_s, mock_i, server_env):
    env = server_env
    env["db"].delete_source_documents(env["conn"], "test")
    env["searcher"].reload()
    results = env["searcher"].query("anything", top_k=5)
    assert results == []


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_get_full_document(mock_s, mock_i, server_env):
    env = server_env
    content = env["db"].get_document_content(env["conn"], "test", "guide.md")
    assert content is not None
    assert "Guide" in content


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_get_full_document_not_found(mock_s, mock_i, server_env):
    env = server_env
    content = env["db"].get_document_content(env["conn"], "test", "nope.md")
    assert content is None


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_deprecate_roundtrip(mock_s, mock_i, server_env):
    env = server_env
    env["db"].save_deprecated(env["conn"], "guide.md", "outdated")
    deprecated = env["db"].load_deprecated(env["conn"])
    assert "guide.md" in deprecated

    env["db"].remove_deprecated(env["conn"], "guide.md")
    deprecated = env["db"].load_deprecated(env["conn"])
    assert "guide.md" not in deprecated


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_audit_log(mock_s, mock_i, server_env):
    env = server_env
    env["db"].log_query(
        env["conn"], "test query", None, 5, 2, "[]", 42,
    )
    entries = env["db"].get_recent_queries(env["conn"], limit=10)
    assert len(entries) >= 1
    assert entries[0]["query"] == "test query"
    # Defaults preserved when caller doesn't pass scoring metadata.
    assert entries[0]["reranker_used"] is False
    assert entries[0]["rrf_k"] == 0
    assert entries[0]["score_kind"] == "rrf"


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_audit_log_records_scoring_path(mock_s, mock_i, server_env):
    env = server_env
    env["db"].log_query(
        env["conn"], "rerank query", "test", 5, 1, "[]", 123,
        reranker_used=True, rrf_k=10, score_kind="rerank",
    )
    entries = env["db"].get_recent_queries(env["conn"], limit=1)
    assert entries[0]["reranker_used"] is True
    assert entries[0]["rrf_k"] == 10
    assert entries[0]["score_kind"] == "rerank"


def test_query_log_migrates_legacy_query_embedding(tmp_path):
    """A pre-existing query_log without the embedding column should be migrated in place."""
    import sqlite3
    db_path = str(tmp_path / "legacy_embed.db")
    legacy = sqlite3.connect(db_path)
    legacy.execute("""CREATE TABLE query_log (
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
        score_kind TEXT NOT NULL DEFAULT 'rrf'
    )""")
    legacy.commit()
    legacy.close()

    from agents_rag.db import init_db, log_query, get_recent_query_embeddings
    conn = init_db(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(query_log)").fetchall()}
    assert "query_embedding" in cols

    log_query(conn, "after", None, 5, 0, "[]", 1, query_embedding=b"\x00" * 8)
    rows = get_recent_query_embeddings(conn, None, since_seconds=60, limit=5)
    assert len(rows) == 1
    assert rows[0]["query"] == "after"
    assert rows[0]["query_embedding"] == b"\x00" * 8
    conn.close()


def test_get_recent_query_embeddings_skips_cached_and_old(tmp_path):
    """Dedupe lookup must ignore cached rows (to prevent self-dedupe chains)
    and rows outside the time window."""
    from agents_rag.db import init_db, log_query, get_recent_query_embeddings
    conn = init_db(str(tmp_path / "test.db"))

    log_query(conn, "fresh", None, 5, 0, "[]", 1, query_embedding=b"\x01" * 8)
    log_query(conn, "cached", None, 5, 0, "[]", 1,
              score_kind="cached", query_embedding=b"\x02" * 8)
    log_query(conn, "no-embedding", None, 5, 0, "[]", 1)
    log_query(conn, "wrong-source", "other", 5, 0, "[]", 1,
              query_embedding=b"\x03" * 8)

    rows = get_recent_query_embeddings(conn, None, since_seconds=60, limit=5)
    queries = [r["query"] for r in rows]
    assert "fresh" in queries
    assert "cached" not in queries
    assert "no-embedding" not in queries
    assert "wrong-source" not in queries
    conn.close()


def test_query_log_migrates_legacy_db(tmp_path):
    """A pre-existing query_log without scoring columns should be migrated in place."""
    import sqlite3
    db_path = str(tmp_path / "legacy.db")
    legacy = sqlite3.connect(db_path)
    legacy.execute("""CREATE TABLE query_log (
        id INTEGER PRIMARY KEY,
        timestamp TEXT NOT NULL,
        query TEXT NOT NULL,
        source_filter TEXT,
        top_k INTEGER NOT NULL,
        result_count INTEGER NOT NULL,
        results_json TEXT NOT NULL,
        latency_ms INTEGER NOT NULL
    )""")
    legacy.execute(
        "INSERT INTO query_log (timestamp, query, source_filter, top_k, "
        "result_count, results_json, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("2026-04-01T00:00:00Z", "legacy", None, 5, 0, "[]", 99),
    )
    legacy.commit()
    legacy.close()

    from agents_rag.db import init_db, get_recent_queries, log_query
    conn = init_db(db_path)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(query_log)").fetchall()}
    assert {"reranker_used", "rrf_k", "score_kind"}.issubset(cols)

    entries = get_recent_queries(conn, limit=10)
    legacy_row = [e for e in entries if e["query"] == "legacy"][0]
    assert legacy_row["reranker_used"] is False
    assert legacy_row["rrf_k"] == 0
    assert legacy_row["score_kind"] == "rrf"

    log_query(conn, "after-migration", None, 5, 0, "[]", 1,
              reranker_used=True, rrf_k=10, score_kind="rerank")
    entries = get_recent_queries(conn, limit=1)
    assert entries[0]["reranker_used"] is True
    assert entries[0]["score_kind"] == "rerank"
    conn.close()


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_create_app_warms_reranker_when_enabled(mock_s, mock_i, tmp_path):
    """Server startup should kick off a background warm-up call when the reranker is on."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "g.md").write_text("# G\n\nx.")

    config = {
        "sources": [{"name": "t", "path": str(docs), "file_types": [".md"]}],
        "cache_dir": str(tmp_path / "cache"),
        "use_gemma": False,
        "sessions": {
            "transcript_dirs": [], "output_dir": str(tmp_path / "s"),
            "watch_interval_minutes": 0,
        },
        "reranker": {"enabled": True, "model": "fake-model"},
    }

    warm_calls: list[str] = []
    warm_event = __import__("threading").Event()

    def fake_warm(model_name):
        warm_calls.append(model_name)
        warm_event.set()
        return True

    from agents_rag.server import create_app
    with patch("agents_rag.server.reranker.warm", side_effect=fake_warm):
        create_app(config_override=config)
        assert warm_event.wait(timeout=5)

    assert warm_calls == ["fake-model"]


def _build_app(tmp_path, extra_config: dict | None = None):
    """Spin up a real create_app() against a fake docs corpus."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n\n## Setup\n\nRun pip install.\n\n## Usage\n\nCall the query tool.\n")
    config = {
        "sources": [{"name": "test", "path": str(docs), "file_types": [".md"]}],
        "cache_dir": str(tmp_path / "cache"),
        "use_gemma": False,
        "sessions": {
            "transcript_dirs": [], "output_dir": str(tmp_path / "sessions"),
            "watch_interval_minutes": 0,
        },
        "reranker": {"enabled": False},
    }
    if extra_config:
        config.update(extra_config)

    from agents_rag.server import create_app
    from agents_rag.config import parse_sources
    from agents_rag.indexer import reindex_all
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Pre-index so search has data before the first query.
    from agents_rag import db
    conn = db.init_db(os.path.join(cache_dir, "bookkeeper.db"))
    reindex_all(parse_sources(config), conn, use_gemma=False)
    conn.close()
    return create_app(config_override=config)


def _call(mcp, name, **kwargs):
    """Invoke a registered MCP tool by name and return its string payload."""
    import asyncio
    result = asyncio.run(mcp.call_tool(name, kwargs))
    return result.content[0].text


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_knowledge_base_returns_envelope(mock_s, mock_i, tmp_path):
    """The tool now returns {results, meta} not a bare list — verify the shape
    and that meta carries the new confidence/weak_match fields."""
    mcp = _build_app(tmp_path)
    payload = _call(mcp, "query_knowledge_base", query="setup", max_results=3)
    envelope = json.loads(payload)
    assert "results" in envelope
    assert "meta" in envelope
    meta = envelope["meta"]
    assert meta["scoring"] in {"rerank", "rrf"}
    assert "confidence" in meta
    assert "weak_match" in meta
    if envelope["results"]:
        for r in envelope["results"]:
            assert r["confidence"] in {"high", "medium", "low"}


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_knowledge_base_exact_cache_preserves_scoring(mock_s, mock_i, tmp_path):
    """Exact cache hits keep the original scoring path and confidence metadata."""
    mcp = _build_app(tmp_path, {
        "search": {"dedupe_similarity_threshold": 0.5, "dedupe_window_seconds": 60},
    })

    first = json.loads(_call(mcp, "query_knowledge_base", query="setup guide", max_results=3))
    assert first["meta"].get("cache_hit") in (None, False)

    second = json.loads(_call(mcp, "query_knowledge_base", query="setup guide", max_results=3))
    assert second["meta"].pop("cache_hit") is True
    assert second == first


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_query_knowledge_base_skips_dedupe_across_source_filters(mock_s, mock_i, tmp_path):
    """Queries with different source filters must NOT dedupe against each other —
    the agent may legitimately be running the same query against two scopes."""
    mcp = _build_app(tmp_path, {
        "search": {"dedupe_similarity_threshold": 0.5, "dedupe_window_seconds": 60},
    })
    a = json.loads(_call(mcp, "query_knowledge_base", query="setup guide", max_results=3))
    b = json.loads(_call(mcp, "query_knowledge_base", query="setup guide", max_results=3, source="test"))
    assert a["meta"].get("cache_hit") in (None, False)
    assert b["meta"].get("cache_hit") in (None, False)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_exact_cache_hit_avoids_retrieval(mock_s, mock_i, tmp_path, monkeypatch):
    mcp = _build_app(tmp_path, {
        "search": {"dedupe_similarity_threshold": 0.5, "dedupe_window_seconds": 60},
    })

    from agents_rag.searcher import Searcher
    _call(mcp, "query_knowledge_base", query="setup guide", max_results=3)
    with patch.object(Searcher, "query_with_meta", side_effect=AssertionError("unexpected retrieval")):
        hit = json.loads(_call(mcp, "query_knowledge_base", query="setup guide", max_results=3))
    assert hit["meta"]["cache_hit"] is True


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_create_app_skips_warmup_when_reranker_disabled(mock_s, mock_i, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "g.md").write_text("# G\n\nx.")

    config = {
        "sources": [{"name": "t", "path": str(docs), "file_types": [".md"]}],
        "cache_dir": str(tmp_path / "cache"),
        "use_gemma": False,
        "sessions": {
            "transcript_dirs": [], "output_dir": str(tmp_path / "s"),
            "watch_interval_minutes": 0,
        },
        "reranker": {"enabled": False},
    }

    warm_calls: list[str] = []

    from agents_rag.server import create_app
    with patch("agents_rag.server.reranker.warm", side_effect=lambda m: warm_calls.append(m)):
        create_app(config_override=config)
        # Give any spurious thread a chance to fire.
        time.sleep(0.1)

    assert warm_calls == []


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_list_sources(mock_s, mock_i, server_env):
    env = server_env
    stats = env["searcher"].source_stats()
    assert "test" in stats
    assert stats["test"] > 0


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_gc_removes_stale(mock_s, mock_i, server_env):
    env = server_env
    # guide.md exists on disk, so gc should not remove it
    docs_path = env["config"]["sources"][0]["path"]
    root = Path(docs_path)
    live = {str(f.relative_to(root)) for f in root.rglob("*.md") if f.is_file()}

    removed = env["db"].gc_source(env["conn"], "test", live)
    assert removed == 0

    # Now gc with empty live set — should remove the doc
    removed = env["db"].gc_source(env["conn"], "test", set())
    assert removed >= 1


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_save_session_file(mock_s, mock_i, server_env):
    """Test the sync session save logic directly."""
    env = server_env
    sessions_dir = str(env["sessions_dir"])
    os.makedirs(sessions_dir, exist_ok=True)

    import re
    slug = re.sub(r"[^a-z0-9]+", "-", "test session".lower())[:40].strip("-")
    date_str = time.strftime("%Y%m%d", time.gmtime())
    filename = f"{date_str}_{slug}.md"
    filepath = os.path.join(sessions_dir, filename)

    parts = [
        "# test session", "",
        f"**Saved**: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
        "## Summary", "", "Did some testing.", "",
    ]
    Path(filepath).write_text("\n".join(parts))
    assert Path(filepath).exists()
    content = Path(filepath).read_text()
    assert "test session" in content
    assert "Did some testing." in content


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.distill.distill_blob", return_value={"digest": "ok", "profile": "fast"})
def test_prepare_cloud_context_during_reindex_has_no_errors(_mock_d, mock_s, mock_i, tmp_path):
    mcp = _build_app(tmp_path)

    started = json.loads(_call(mcp, "reindex"))
    assert started["status"] in {"started", "already_running"}

    def _call_prepare(i: int) -> dict:
        payload = _call(
            mcp,
            "prepare_cloud_context",
            query=f"setup {i}",
            include_full_docs=True,
            user_prompt="u" * 800,
        )
        return json.loads(payload)

    with ThreadPoolExecutor(max_workers=4) as pool:
        packets = list(pool.map(_call_prepare, range(6)))

    for packet in packets:
        assert "markdown" not in packet
        assert "prior_work" in packet

    deadline = time.time() + 10
    while time.time() < deadline:
        st = json.loads(_call(mcp, "status"))
        if not st.get("reindex_running"):
            break
        time.sleep(0.1)
    assert json.loads(_call(mcp, "status")).get("reindex_running") is False


def test_exact_cache_preserves_low_confidence_and_weak_match(tmp_path):
    from agents_rag.searcher import Searcher

    mcp = _build_app(tmp_path)
    weak_results = [{"source": "test", "path": "guide.md", "score": 0.01,
                     "confidence": "low", "evidence": "Unrelated evidence"}]
    weak_meta = {"reranker_used": False, "rrf_k": 10, "score_kind": "rrf",
                 "confidence": "low", "weak_match": True, "confidence_calibrated": False}
    with patch.object(Searcher, "query_with_meta", return_value=(weak_results, weak_meta)) as retrieve:
        first = json.loads(_call(mcp, "query_knowledge_base", query="unsupported topic"))
        second = json.loads(_call(mcp, "query_knowledge_base", query="unsupported topic"))
    assert retrieve.call_count == 1
    assert second["meta"].pop("cache_hit") is True
    assert second == first
    assert second["meta"]["weak_match"] is True
    assert second["meta"]["confidence"] == "low"
    assert second["results"][0]["confidence"] == "low"


def test_exact_cache_isolates_query_limit_scope_and_corpus_generation(tmp_path, monkeypatch):
    from agents_rag import db
    from agents_rag.searcher import Searcher

    mcp = _build_app(tmp_path)
    # Even indistinguishable query vectors must not substitute another query's answer.
    monkeypatch.setattr("agents_rag.searcher.embed_texts",
                        lambda texts: np.ones((len(texts), 384), dtype=np.float32))
    original = Searcher.query_with_meta
    with patch.object(Searcher, "query_with_meta", autospec=True, side_effect=original) as retrieve:
        base = {"query": "setup v1", "max_results": 1, "source": "test"}
        _call(mcp, "query_knowledge_base", **base)
        cached = json.loads(_call(mcp, "query_knowledge_base", **base))
        assert cached["meta"]["cache_hit"] is True
        assert retrieve.call_count == 1
        for changes in ({"query": "setup v2"}, {"query": "not setup v1"},
                        {"max_results": 2}, {"source": None}):
            response = json.loads(_call(mcp, "query_knowledge_base", **{**base, **changes}))
            assert response["meta"].get("cache_hit") in (None, False)
        assert retrieve.call_count == 5
        conn = db.init_db(str(tmp_path / "cache" / "bookkeeper.db"))
        try:
            db.upsert_document(conn, "test", "guide.md", "Changed guide", "", "Changed source evidence")
        finally:
            conn.close()
        changed = json.loads(_call(mcp, "query_knowledge_base", **base))
        assert changed["meta"].get("cache_hit") in (None, False)
        assert changed["meta"]["generation"] > cached["meta"]["generation"]
        assert retrieve.call_count == 6


def test_deprecation_invalidates_exact_cached_response(tmp_path):
    mcp = _build_app(tmp_path)
    first = json.loads(_call(mcp, "query_knowledge_base", query="setup"))
    assert first["results"]
    assert not first["results"][0].get("deprecated")
    hit = json.loads(_call(mcp, "query_knowledge_base", query="setup"))
    assert hit["meta"]["cache_hit"] is True

    assert json.loads(_call(mcp, "deprecate_doc", path="guide.md", reason="Use guide v2"))["status"] == "deprecated"
    deprecated = json.loads(_call(mcp, "query_knowledge_base", query="setup"))
    assert deprecated["meta"].get("cache_hit") in (None, False)
    assert all(result["deprecated"] for result in deprecated["results"])
    assert "guide v2" in deprecated["results"][0]["deprecation_notice"]
    _call(mcp, "deprecate_doc", path="guide.md", undeprecate=True)
    restored = json.loads(_call(mcp, "query_knowledge_base", query="setup"))
    assert restored["meta"].get("cache_hit") in (None, False)
    assert not restored["results"][0].get("deprecated")


@pytest.mark.parametrize("path_kind", ["absolute", "traversal", "symlink"])
def test_full_document_rejects_files_outside_source_roots(tmp_path, path_kind):
    mcp = _build_app(tmp_path)
    outside = tmp_path / "private.txt"
    outside.write_text("OUTSIDE_PRIVATE_MARKER", encoding="utf-8")
    if path_kind == "absolute":
        requested = str(outside)
    elif path_kind == "traversal":
        requested = "../private.txt"
    else:
        link = tmp_path / "docs" / "escape.txt"
        try:
            link.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"Filesystem symlinks unavailable: {exc}")
        requested = "escape.txt"
    response = _call(mcp, "get_full_document", path=requested, source="test")
    assert "OUTSIDE_PRIVATE_MARKER" not in response
    assert "error" in json.loads(response)
    allowed = tmp_path / "docs" / "local.txt"
    allowed.write_text("ALLOWED_LOCAL_MARKER", encoding="utf-8")
    assert _call(mcp, "get_full_document", path="local.txt", source="test") == "ALLOWED_LOCAL_MARKER"


def test_full_document_requires_scope_for_ambiguous_paths(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "guide.md").write_text("# Other\n\nOTHER_SOURCE_MARKER", encoding="utf-8")
    mcp = _build_app(tmp_path, {"sources": [
        {"name": "test", "path": str(tmp_path / "docs"), "file_types": [".md"]},
        {"name": "other", "path": str(other), "file_types": [".md"]},
    ]})
    ambiguous = json.loads(_call(mcp, "get_full_document", path="guide.md"))
    assert "ambiguous" in ambiguous["error"].lower()
    assert "Run pip install" in _call(mcp, "get_full_document", path="guide.md", source="test")
    assert "OTHER_SOURCE_MARKER" in _call(mcp, "get_full_document", path="guide.md", source="other")
    wrong_scope = json.loads(_call(mcp, "get_full_document", path="guide.md", source="unknown"))
    assert "error" in wrong_scope


@pytest.mark.parametrize("include_markdown", [False, True])
def test_prepare_cloud_context_caps_serialized_tokens_and_preserves_short_task(tmp_path, include_markdown):
    import tiktoken

    mcp = _build_app(tmp_path)
    user_prompt = "Keep café IDs unchanged; never delete data."
    tool_outputs = [{"name": f"tool-{i}", "stdout": ("变量 🚀 café {version: 'v2'}\n" * 200)}
                    for i in range(5)]
    payload = _call(mcp, "prepare_cloud_context", query="setup", user_prompt=user_prompt,
                    max_context_tokens=600, include_markdown=include_markdown,
                    tool_outputs_json=json.dumps(tool_outputs, ensure_ascii=False))
    packet = json.loads(payload)
    actual_tokens = len(tiktoken.get_encoding("cl100k_base").encode(payload, disallowed_special=()))
    assert actual_tokens <= 600
    assert packet["meta"]["token_count"] == actual_tokens
    assert packet["user_prompt"] == user_prompt
    assert packet["meta"]["omitted_count"] > 0
    assert ("markdown" in packet) is include_markdown
    if include_markdown:
        assert user_prompt in packet["markdown"]


def test_prepare_cloud_context_uses_exact_immutable_evidence(tmp_path):
    mcp = _build_app(tmp_path)
    results = json.loads(_call(mcp, "query_knowledge_base", query="setup"))["results"]
    assert results
    spans = {span["evidence_id"]: span for hit in results for span in hit["evidence_spans"]}
    packet = json.loads(_call(mcp, "prepare_cloud_context", query="setup", max_context_tokens=4000))
    assert packet["prior_work"]
    for entry in packet["prior_work"]:
        expected = spans[entry["evidence_id"]]
        assert entry["text"] == expected["text"]
        assert entry["revision_id"] == expected["revision_id"]
        assert entry["completeness"] == expected["completeness"]


def _wait_until(predicate, *, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for background work")


def test_failed_transcript_embedding_retains_job_and_defers_checkpoint_until_retry(tmp_path, sample_transcript, monkeypatch):
    from agents_rag import db, server

    mcp = _build_app(tmp_path, {"sessions": {
        "transcript_dirs": [str(sample_transcript)], "output_dir": str(tmp_path / "sessions"),
        "watch_interval_minutes": 0,
    }})
    transcript = next(sample_transcript.glob("*.txt"))
    raw_content = transcript.read_bytes().decode("utf-8")
    conn = db.init_db(str(tmp_path / "cache" / "bookkeeper.db"))

    def fail_embedding(_texts):
        raise RuntimeError("synthetic embedding failure")

    monkeypatch.setattr(server, "embed_texts", fail_embedding)
    try:
        scan = json.loads(_call(mcp, "distill_transcripts"))
        assert scan["new_turns_indexed"] == 1
        _wait_until(lambda: conn.execute("SELECT 1 FROM ingestion_jobs WHERE status='failed'").fetchone())
        assert db.load_transcript_state(conn, str(transcript)) == {}
        assert conn.execute("SELECT COUNT(*) FROM documents WHERE source='sessions'").fetchone()[0] == 0
        assert json.loads(_call(mcp, "status"))["pending_ingestion_jobs"] == 1

        monkeypatch.setattr(server, "embed_texts", _fake_embed)
        assert json.loads(_call(mcp, "retry_ingestion"))["queued"] == 1
        _wait_until(lambda: json.loads(_call(mcp, "status"))["pending_ingestion_jobs"] == 0)
        state = db.load_transcript_state(conn, str(transcript))
        assert state["byte_offset"] == len(transcript.read_bytes())
        assert state["content_hash"] == hashlib.sha256(transcript.read_bytes()).hexdigest()
        path, content, completeness = conn.execute(
            "SELECT path,content,completeness FROM documents WHERE source='sessions'",
        ).fetchone()
        assert content == raw_content
        assert completeness == "transcript"
        assert _call(mcp, "get_full_document", path=path, source="sessions") == raw_content
        job = conn.execute("SELECT status,attempts,error FROM ingestion_jobs").fetchone()
        assert job == ("done", 2, None)
        assert json.loads(_call(mcp, "retry_ingestion"))["queued"] == 0
    finally:
        conn.close()


def test_retry_older_failed_snapshot_cannot_overwrite_newer_accepted_transcript(tmp_path, sample_transcript, monkeypatch):
    from agents_rag import db, server

    mcp = _build_app(tmp_path, {"sessions": {
        "transcript_dirs": [str(sample_transcript)], "output_dir": str(tmp_path / "sessions"),
        "watch_interval_minutes": 0,
    }})
    transcript = next(sample_transcript.glob("*.txt"))
    conn = db.init_db(str(tmp_path / "cache" / "bookkeeper.db"))
    try:
        with patch("agents_rag.server.embed_texts", side_effect=RuntimeError("first snapshot failed")):
            _call(mcp, "distill_transcripts")
            _wait_until(lambda: conn.execute("SELECT 1 FROM ingestion_jobs WHERE status='failed'").fetchone())
        with transcript.open("a", encoding="utf-8") as stream:
            stream.write("\nLATEST_ACCEPTED_EVIDENCE_MARKER\n")
        latest_content = transcript.read_bytes().decode("utf-8")
        _call(mcp, "distill_transcripts")
        _wait_until(lambda: conn.execute("SELECT COUNT(*) FROM ingestion_jobs WHERE status='done'").fetchone()[0] >= 1)
        expected_state = db.load_transcript_state(conn, str(transcript))
        assert expected_state["byte_offset"] == len(transcript.read_bytes())
        before = conn.execute("SELECT path,content,revision_id FROM documents WHERE source='sessions'").fetchone()
        assert before[1] == latest_content

        _call(mcp, "retry_ingestion")
        _wait_until(lambda: not db.pending_ingestion(conn))
        after = conn.execute("SELECT path,content,revision_id FROM documents WHERE source='sessions'").fetchone()
        assert after == before
        assert db.load_transcript_state(conn, str(transcript)) == expected_state
        assert _call(mcp, "get_full_document", path=after[0], source="sessions") == latest_content
    finally:
        conn.close()


def test_server_refuses_legacy_database_upgrade_without_mutating_original(tmp_path):
    import sqlite3
    from agents_rag.server import create_app

    legacy_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy_path)
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO documents(content) VALUES ('preserve original evidence')")
    conn.commit()
    conn.close()
    original = legacy_path.read_bytes()
    with pytest.raises(ValueError, match="Legacy database requires"):
        create_app(config_override={"sources": [], "db_path": str(legacy_path),
                                    "cache_dir": str(tmp_path / "cache"), "use_gemma": False})
    assert legacy_path.read_bytes() == original
    assert not Path(str(legacy_path) + "-wal").exists()


@pytest.mark.parametrize("pinned_revision", ["pinned-commit", None])
def test_restart_resolves_complete_embedding_profile_before_semantic_query(tmp_path, monkeypatch, pinned_revision):
    from types import SimpleNamespace
    from agents_rag import db, indexer, searcher, server

    monkeypatch.setenv("EMBEDDING_MODEL", "fixture-model")
    if pinned_revision is None:
        monkeypatch.delenv("EMBEDDING_REVISION", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_REVISION", pinned_revision)
    model = SimpleNamespace(tokenizer=SimpleNamespace(name_or_path="fixture-tokenizer"), max_seq_length=512)
    monkeypatch.setattr(indexer, "_model", model)
    saved_profile = indexer.embedding_profile()
    path = tmp_path / "bookkeeper.db"
    conn = db.init_db(str(path))
    space = db.register_embedding_space(conn, saved_profile, 2)
    doc = db.upsert_document(conn, "test", "a.md", "", "", "automobile fuel")
    db.upsert_chunks(conn, doc, [{"text": "automobile fuel"}],
                     np.array([[1.0, 0.0]], dtype=np.float32), embedding_space=space)
    conn.close()
    # Model objects are absent after restart, even when the requested revision is pinned.
    monkeypatch.setattr(indexer, "_model", None)

    def resolve_model():
        monkeypatch.setattr(indexer, "_model", model)
        return len, 512

    config = {"sources": [{"name": "test", "path": str(tmp_path)}], "db_path": str(path),
              "reranker": {"enabled": False}, "use_gemma": False,
              "sessions": {"watch_interval_minutes": 0, "output_dir": str(tmp_path / "sessions")}}
    with patch.object(server, "embedding_token_budget", side_effect=resolve_model) as resolve, \
         patch.object(searcher, "embed_texts", return_value=np.array([[1.0, 0.0]], dtype=np.float32)) as embed:
        app = server.create_app(config)
        result = json.loads(_call(app, "query_knowledge_base", query="vehicle gasoline"))
        assert [item["path"] for item in result["results"]] == ["a.md"]
        assert result["results"][0]["cosine_similarity"] == pytest.approx(1.0)
        assert result["results"][0]["lexical_support"] is False
        assert resolve.call_count == 1
        assert embed.call_count == 1
        _call(app, "query_knowledge_base", query="vehicle gasoline")
        assert resolve.call_count == 1
        assert embed.call_count == 1


@pytest.mark.parametrize("space_kind", ["legacy", "different_revision", "different_model"])
def test_restart_keeps_unknown_or_unrelated_embeddings_lexical(tmp_path, monkeypatch, space_kind):
    from agents_rag import db, indexer, server

    monkeypatch.setenv("EMBEDDING_MODEL", "fixture-model")
    monkeypatch.setenv("EMBEDDING_REVISION", "pinned-commit")
    monkeypatch.setattr(indexer, "_model", None)
    path = tmp_path / "bookkeeper.db"
    conn = db.init_db(str(path))
    space = None
    if space_kind != "legacy":
        saved_profile = {**indexer.embedding_profile(), "tokenizer": "fixture-tokenizer", "max_seq_length": 512}
        saved_profile["revision" if space_kind == "different_revision" else "model"] = "different"
        space = db.register_embedding_space(conn, saved_profile, 2)
    doc = db.upsert_document(conn, "test", "a.md", "", "", "automobile fuel")
    db.upsert_chunks(conn, doc, [{"text": "automobile fuel"}],
                     np.array([[1.0, 0.0]], dtype=np.float32), embedding_space=space)
    conn.close()
    config = {"sources": [{"name": "test", "path": str(tmp_path)}], "db_path": str(path),
              "reranker": {"enabled": False}, "use_gemma": False,
              "sessions": {"watch_interval_minutes": 0, "output_dir": str(tmp_path / "sessions")}}
    with patch.object(server, "embedding_token_budget", side_effect=AssertionError("Unrelated vectors must stay lexical")) as resolve:
        app = server.create_app(config)
        result = json.loads(_call(app, "query_knowledge_base", query="vehicle gasoline"))
        assert result["results"] == []
        resolve.assert_not_called()
        lexical = json.loads(_call(app, "query_knowledge_base", query="automobile"))
        assert [item["path"] for item in lexical["results"]] == ["a.md"]
        assert lexical["results"][0]["cosine_similarity"] is None
