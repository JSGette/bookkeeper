"""Tests for agents_rag/server.py: config parsing, app factory, tool-level logic."""

from __future__ import annotations

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
        rng = np.random.RandomState(hash(t) % (2**31))
        vecs.append(rng.randn(384).astype(np.float32))
    return np.array(vecs)


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
def test_query_knowledge_base_dedupes_near_duplicate(mock_s, mock_i, tmp_path):
    """The second of two near-identical queries within the dedupe window must
    be served from cache and carry repeated_query=True in its meta."""
    mcp = _build_app(tmp_path, {
        "search": {"dedupe_similarity_threshold": 0.5, "dedupe_window_seconds": 60},
    })

    first = json.loads(_call(mcp, "query_knowledge_base", query="setup guide", max_results=3))
    assert first["meta"].get("repeated_query") is None or first["meta"]["repeated_query"] is False

    second = json.loads(_call(mcp, "query_knowledge_base", query="setup guide", max_results=3))
    assert second["meta"]["repeated_query"] is True
    assert second["meta"]["scoring"] == "cached"
    assert second["meta"]["previous_query"] == "setup guide"
    assert isinstance(second["meta"]["hint"], str)
    assert "rephras" in second["meta"]["hint"].lower()


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
    assert a["meta"].get("repeated_query") in (None, False)
    assert b["meta"].get("repeated_query") in (None, False)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
@patch("agents_rag.searcher.embed_texts", side_effect=_fake_embed)
def test_dedupe_lookup_uses_read_only_connection(mock_s, mock_i, tmp_path, monkeypatch):
    mcp = _build_app(tmp_path, {
        "search": {"dedupe_similarity_threshold": 0.5, "dedupe_window_seconds": 60},
    })

    from agents_rag import db as db_mod

    seen_query_only = []
    orig = db_mod.get_recent_query_embeddings

    def _wrapped(conn, source_filter, since_seconds=120, limit=5):
        flag = conn.execute("PRAGMA query_only").fetchone()[0]
        seen_query_only.append(int(flag))
        return orig(conn, source_filter, since_seconds=since_seconds, limit=limit)

    monkeypatch.setattr("agents_rag.server.db.get_recent_query_embeddings", _wrapped)

    _call(mcp, "query_knowledge_base", query="setup guide", max_results=3)
    _call(mcp, "query_knowledge_base", query="setup guide", max_results=3)
    assert seen_query_only
    assert all(v == 1 for v in seen_query_only)


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
        assert "markdown" in packet
        assert "prior_work" in packet

    deadline = time.time() + 10
    while time.time() < deadline:
        st = json.loads(_call(mcp, "status"))
        if not st.get("reindex_running"):
            break
        time.sleep(0.1)
    assert json.loads(_call(mcp, "status")).get("reindex_running") is False
