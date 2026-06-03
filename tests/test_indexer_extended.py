"""Extended tests for indexer.py: git delta path, reindex_all, summarize_document fallback."""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from agents_rag.db import init_db, load_all_chunks
from agents_rag.indexer import (
    _git_changed_files,
    _git_head,
    _git_is_ancestor,
    extract_authored_summary,
    reindex_all,
    reindex_source,
    summarize_document,
)
from tests.conftest import make_source_cfg


def _fake_embed(texts):
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % (2**31))
        vecs.append(rng.randn(384).astype(np.float32))
    return np.array(vecs)


# ---------------------------------------------------------------------------
# Git helpers (using real temp repos)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repo with initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)

    (repo / "doc.md").write_text("# Initial\n\n## Section\n\nContent.")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True)

    return repo


def test_git_head(git_repo):
    head = _git_head(str(git_repo))
    assert head is not None
    assert len(head) == 40  # full SHA


def test_git_head_nonrepo(tmp_path):
    assert _git_head(str(tmp_path)) is None


def test_git_is_ancestor(git_repo):
    head = _git_head(str(git_repo))
    assert _git_is_ancestor(str(git_repo), head) is True


def test_git_is_ancestor_bad_hash(git_repo):
    assert _git_is_ancestor(str(git_repo), "0" * 40) is False


def test_git_changed_files(git_repo):
    old_head = _git_head(str(git_repo))

    (git_repo / "new.md").write_text("# New\n\nContent.")
    subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "add new"], cwd=git_repo, capture_output=True, check=True)

    new_head = _git_head(str(git_repo))
    changed = _git_changed_files(str(git_repo), old_head, new_head, [".md"])
    assert changed is not None
    assert "new.md" in changed


def test_git_changed_files_no_changes(git_repo):
    head = _git_head(str(git_repo))
    changed = _git_changed_files(str(git_repo), head, head, [".md"])
    assert changed is not None
    assert changed == []


# ---------------------------------------------------------------------------
# summarize_document fallback
# ---------------------------------------------------------------------------


def test_summarize_no_gemma():
    content = "# Title\n\nFirst paragraph here.\n\n## Section\n\nMore stuff."
    result = summarize_document(content, use_gemma=False)
    assert result == "First paragraph here."


def test_summarize_empty_content():
    result = summarize_document("", use_gemma=False)
    assert result == ""


# ---------------------------------------------------------------------------
# Authored-summary passthrough (Proposal 1)
# ---------------------------------------------------------------------------


def test_extract_authored_summary_basic():
    content = textwrap.dedent("""
        # My Session

        **Saved**: 2026-04-24

        ## Summary

        We refactored the indexer to skip redundant gemma passes over docs
        that already carry an authored Summary section.

        ## Decisions

        - Keep Claude summaries.
    """).strip()
    assert extract_authored_summary(content).startswith("We refactored the indexer")


def test_extract_authored_summary_missing():
    assert extract_authored_summary("# Title\n\nBody only, no heading.") == ""


def test_extract_authored_summary_empty_body():
    content = "## Summary\n\n## Decisions\n- x"
    assert extract_authored_summary(content) == ""


def test_extract_authored_summary_truncates_to_500():
    body = "x" * 1000
    content = f"## Summary\n\n{body}\n\n## Decisions\n- y"
    assert len(extract_authored_summary(content)) == 500


def test_summarize_document_prefers_authored_summary():
    """Gemma must NOT be called when a ## Summary section is present."""
    content = "# Title\n\n## Summary\n\nAuthored text here.\n\n## Decisions\n- d"
    with patch("httpx.post") as mock_post:
        result = summarize_document(content, use_gemma=True)
    assert result == "Authored text here."
    mock_post.assert_not_called()


def test_summarize_document_falls_back_to_gemma_without_summary_section():
    content = "# Title\n\nRaw body text without summary heading."
    with patch("httpx.post") as mock_post:
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.json.return_value = {
            "message": {"content": '{"summary": "gemma output"}'},
        }
        result = summarize_document(content, use_gemma=True)
    assert result == "gemma output"
    mock_post.assert_called_once()


# ---------------------------------------------------------------------------
# reindex_source with git delta
# ---------------------------------------------------------------------------


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_git_delta(mock_embed, git_repo, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    cfg = make_source_cfg(git_repo, name="gitrepo", file_types=[".md"])

    # First full reindex
    meta1, vecs1, changed1 = reindex_source(cfg, conn, use_gemma=False)
    assert changed1 is True
    assert len(meta1) > 0

    # Add a new file
    (git_repo / "new.md").write_text("# New\n\n## Info\n\nNew content here.")
    subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "add new"], cwd=git_repo, capture_output=True, check=True)

    # Delta reindex
    meta2, vecs2, changed2 = reindex_source(cfg, conn, use_gemma=False)
    assert changed2 is True
    assert len(meta2) > len(meta1)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_git_no_changes(mock_embed, git_repo, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    cfg = make_source_cfg(git_repo, name="gitrepo", file_types=[".md"])

    reindex_source(cfg, conn, use_gemma=False)
    _, _, changed = reindex_source(cfg, conn, use_gemma=False)
    assert changed is False


# ---------------------------------------------------------------------------
# reindex_all
# ---------------------------------------------------------------------------


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_all(mock_embed, doc_tree, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    cfg_docs = make_source_cfg(doc_tree, name="docs", file_types=[".md"])
    cfg_code = make_source_cfg(doc_tree, name="code", file_types=[".py"])

    meta, vecs, count = reindex_all([cfg_docs, cfg_code], conn, use_gemma=False)
    assert count > 0
    assert len(meta) == count


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_all_with_filter(mock_embed, doc_tree, tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    cfg_docs = make_source_cfg(doc_tree, name="docs", file_types=[".md"])
    cfg_code = make_source_cfg(doc_tree, name="code", file_types=[".py"])

    # Reindex all first
    reindex_all([cfg_docs, cfg_code], conn, use_gemma=False)

    # Then filtered
    meta, _, count = reindex_all(
        [cfg_docs, cfg_code], conn, use_gemma=False, source_filter="docs",
    )
    doc_sources = {m["source"] for m in meta}
    assert "docs" in doc_sources


