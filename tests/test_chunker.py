"""Tests for chunking: markdown heading splits, oversized splits, code fallback, skip patterns."""

from __future__ import annotations

import textwrap
from pathlib import Path

from agents_rag.chunker import (
    Chunk,
    _chunk_fixed_size,
    _estimate_tokens,
    _should_skip,
    chunk_code,
    chunk_markdown,
    chunk_source,
)
from tests.conftest import make_source_cfg


# ---------------------------------------------------------------------------
# _should_skip
# ---------------------------------------------------------------------------

def test_should_skip_match():
    assert _should_skip("vendor/foo.go", ["vendor/*"]) is True
    assert _should_skip(".git/config", [".git/*"]) is True


def test_should_skip_no_match():
    assert _should_skip("src/main.py", ["vendor/*"]) is False
    assert _should_skip("README.md", ["*.go"]) is False


# ---------------------------------------------------------------------------
# chunk_markdown — heading-based splitting
# ---------------------------------------------------------------------------

def test_markdown_sections(doc_tree: Path):
    chunks = chunk_markdown("test", doc_tree, "guide.md", max_tokens=512)
    headings = [c.section_heading for c in chunks]
    assert "Installation" in headings
    assert "Configuration" in headings
    assert all(c.source == "test" for c in chunks)
    assert all(c.doc_title == "User Guide" for c in chunks)


def test_markdown_preamble(doc_tree: Path):
    """Content before the first ## becomes a preamble chunk."""
    chunks = chunk_markdown("test", doc_tree, "guide.md", max_tokens=512)
    preamble = [c for c in chunks if c.section_heading == ""]
    assert len(preamble) == 1
    assert "Introduction" in preamble[0].text


def test_markdown_empty_file(doc_tree: Path):
    chunks = chunk_markdown("test", doc_tree, "empty.md", max_tokens=512)
    assert chunks == []


def test_markdown_no_headings(doc_tree: Path):
    """A markdown file with no ## headings produces a single chunk."""
    chunks = chunk_markdown("test", doc_tree, "sub/notes.md", max_tokens=512)
    assert len(chunks) == 1
    assert chunks[0].section_heading == ""
    assert "nested content" in chunks[0].text


# ---------------------------------------------------------------------------
# Oversized section splitting
# ---------------------------------------------------------------------------

def test_markdown_oversized_splits(tmp_path: Path):
    """A section exceeding max_tokens gets split into smaller chunks."""
    big_section = "## Big Section\n\n" + "\n\n".join(
        f"Paragraph {i} with enough text to fill tokens." for i in range(100)
    )
    content = f"# Doc Title\n\n{big_section}"
    (tmp_path / "big.md").write_text(content)

    # max_tokens=50 forces splitting (~200 chars)
    chunks = chunk_markdown("test", tmp_path, "big.md", max_tokens=50)
    assert len(chunks) > 1
    assert all(c.doc_title == "Doc Title" for c in chunks)


# ---------------------------------------------------------------------------
# Code chunking (fixed-size fallback)
# ---------------------------------------------------------------------------

def test_code_fixed_size_fallback(doc_tree: Path):
    """Without treesitter, code files get fixed-size chunks."""
    chunks = chunk_code("test", doc_tree, "sub/utils.py", max_tokens=512)
    assert len(chunks) >= 1
    assert chunks[0].source == "test"
    assert "hello" in chunks[0].text


def test_code_empty_file(tmp_path: Path):
    (tmp_path / "blank.py").write_text("")
    chunks = chunk_code("test", tmp_path, "blank.py", max_tokens=512)
    assert chunks == []


def test_fixed_size_overlap(tmp_path: Path):
    """Fixed-size chunker produces overlapping chunks for large files."""
    lines = [f"line {i}: some content here for padding" for i in range(200)]
    (tmp_path / "big.py").write_text("\n".join(lines))

    # Very small token limit to force many chunks
    chunks = _chunk_fixed_size("\n".join(lines), "test", "big.py", max_tokens=30)
    assert len(chunks) > 3
    # Chunks should have overlapping line ranges
    for i in range(1, len(chunks)):
        assert chunks[i].start_line <= chunks[i - 1].end_line


# ---------------------------------------------------------------------------
# chunk_source (full pipeline)
# ---------------------------------------------------------------------------

def test_chunk_source_filters_by_file_types(doc_tree: Path):
    cfg = make_source_cfg(doc_tree, file_types=[".md"])
    chunks = chunk_source(cfg)
    paths = {c.path for c in chunks}
    assert all(p.endswith(".md") for p in paths)
    assert not any(p.endswith(".py") for p in paths)


def test_chunk_source_with_code(doc_tree: Path):
    cfg = make_source_cfg(doc_tree, file_types=[".md", ".py"])
    chunks = chunk_source(cfg)
    paths = {c.path for c in chunks}
    assert any(p.endswith(".py") for p in paths)
    assert any(p.endswith(".md") for p in paths)


def test_chunk_source_skip(doc_tree: Path):
    cfg = make_source_cfg(doc_tree, file_types=[".md"], skip=["sub/*"])
    chunks = chunk_source(cfg)
    assert not any(c.path.startswith("sub/") for c in chunks)


def test_chunk_source_min_lines(doc_tree: Path):
    """min_lines filters out short files."""
    cfg = make_source_cfg(doc_tree, file_types=[".md"], min_lines=100)
    chunks = chunk_source(cfg)
    assert chunks == []


# ---------------------------------------------------------------------------
# embedding_text property
# ---------------------------------------------------------------------------

def test_embedding_text_with_heading():
    c = Chunk(source="s", path="p", doc_title="Title", section_heading="Install",
              start_line=1, end_line=5, text="body text")
    assert c.embedding_text == "Title — Install\nbody text"


def test_embedding_text_without_heading():
    c = Chunk(source="s", path="p", doc_title="Title", section_heading="",
              start_line=1, end_line=5, text="body text")
    assert c.embedding_text == "Title\nbody text"
