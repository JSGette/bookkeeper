"""Shared fixtures for agents-rag tests."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache(tmp_path: Path) -> str:
    """Empty cache directory for index storage."""
    d = tmp_path / "cache"
    d.mkdir()
    return str(d)


@pytest.fixture
def doc_tree(tmp_path: Path) -> Path:
    """Small document tree with markdown and code files for chunking/indexing tests."""
    root = tmp_path / "docs"
    root.mkdir()

    (root / "guide.md").write_text(textwrap.dedent("""\
        # User Guide

        Introduction paragraph.

        ## Installation

        Run `pip install agents-rag`.

        ## Configuration

        Edit config.yaml to set sources.
    """))

    (root / "faq.md").write_text(textwrap.dedent("""\
        # FAQ

        ## How do I reindex?

        Call the reindex tool.

        ## How do I search?

        Call query_knowledge_base with a query string.
    """))

    (root / "empty.md").write_text("")

    sub = root / "sub"
    sub.mkdir()
    (sub / "utils.py").write_text(textwrap.dedent("""\
        def hello():
            return "world"

        def add(a, b):
            return a + b
    """))

    (sub / "notes.md").write_text(textwrap.dedent("""\
        # Sub Notes

        Some nested content here.
    """))

    return root


@pytest.fixture
def sample_transcript(tmp_path: Path) -> Path:
    """Minimal Cursor-style transcript file."""
    t = tmp_path / "transcripts"
    t.mkdir()
    content = textwrap.dedent("""\
        user:

        <user_query>
        How do I configure sources?
        </user_query>

        assistant:

        Edit config.yaml and add a sources entry.

        [Tool call] Read
          path: /some/config.yaml

        user:

        <user_query>
        Now add BM25 search
        </user_query>

        assistant:

        Added BM25 hybrid search using rank-bm25 library.

        [Tool call] Write
          path: /some/searcher.py
        [Tool call] Read
          path: /some/indexer.py
    """)
    (t / "abc12345-test-transcript.txt").write_text(content)
    return t


MINIMAL_SOURCE_CFG = {
    "name": "test",
    "path": "",  # must be overridden with actual path
    "file_types": [".md"],
    "skip": [],
    "min_lines": 0,
    "generate_index": False,
    "max_chunk_tokens": 512,
}


def make_source_cfg(root: Path, **overrides) -> "SourceConfig":
    """Build a SourceConfig pointing at root, with optional overrides."""
    from agents_rag.chunker import SourceConfig

    kw = {**MINIMAL_SOURCE_CFG, "path": str(root), **overrides}
    return SourceConfig(**kw)
