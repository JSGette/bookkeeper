"""Tests for chunking: markdown heading splits, oversized splits, code fallback, skip patterns."""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents_rag.chunker import (
    Chunk,
    _chunk_fixed_size,
    _estimate_tokens,
    _should_skip,
    chunk_code,
    chunk_file,
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
    chunks = chunk_markdown("test", doc_tree, "guide.md", max_tokens=512, max_whole_doc_tokens=0)
    headings = [c.section_heading for c in chunks]
    assert "Installation" in headings
    assert "Configuration" in headings
    assert all(c.source == "test" for c in chunks)
    assert all(c.doc_title == "User Guide" for c in chunks)


def test_markdown_preamble(doc_tree: Path):
    """Content before the first ## becomes a preamble chunk."""
    chunks = chunk_markdown("test", doc_tree, "guide.md", max_tokens=512, max_whole_doc_tokens=0)
    preamble = [c for c in chunks if c.section_heading == ""]
    assert len(preamble) == 1
    assert "Introduction" in preamble[0].text


def test_markdown_empty_file(doc_tree: Path):
    chunks = chunk_markdown("test", doc_tree, "empty.md", max_tokens=512)
    assert chunks == []


def test_markdown_no_headings(doc_tree: Path):
    """A markdown file with no ## headings produces a single chunk."""
    chunks = chunk_markdown("test", doc_tree, "sub/notes.md", max_tokens=512, max_whole_doc_tokens=0)
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

    # max_tokens=50 forces splitting; max_whole_doc_tokens=0 disables bypass
    chunks = chunk_markdown("test", tmp_path, "big.md", max_tokens=50, max_whole_doc_tokens=0)
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


def test_embedding_text_with_preamble():
    c = Chunk(source="s", path="p", doc_title="Title", section_heading="Install",
              start_line=1, end_line=5, text="body text",
              doc_preamble="This doc covers installation and config.")
    assert c.embedding_text == "Title | This doc covers installation and config. — Install\nbody text"


def test_small_doc_bypass(doc_tree: Path):
    """Documents under max_whole_doc_tokens are returned as a single chunk."""
    chunks = chunk_markdown("test", doc_tree, "guide.md", max_tokens=512, max_whole_doc_tokens=5000)
    assert len(chunks) == 1
    assert "Installation" in chunks[0].text
    assert "Configuration" in chunks[0].text


@pytest.mark.parametrize("extension", [".md", ".py", ".txt"])
@pytest.mark.parametrize("body", ["oversized paragraph " * 300, "界🙂 café e\u0301 " * 100])
def test_long_single_line_is_bounded_and_lossless(tmp_path, monkeypatch, extension, body):
    monkeypatch.setattr("agents_rag.chunker._try_treesitter_chunk", lambda *args: None)
    name = "source" + extension
    (tmp_path / name).write_text(body, encoding="utf-8")
    chunks = chunk_file("test", tmp_path, name, max_tokens=24)
    assert len(chunks) > 1
    assert all(_estimate_tokens(chunk.text) <= 24 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == body
    assert all(chunk.start_line == chunk.end_line == 1 for chunk in chunks)
    assert "\ufffd" not in "".join(chunk.text for chunk in chunks)


def test_small_document_preference_cannot_bypass_hard_limit(tmp_path):
    body = "word " * 100
    (tmp_path / "doc.md").write_text(body)
    chunks = chunk_markdown("test", tmp_path, "doc.md", max_tokens=12, max_whole_doc_tokens=1500)
    assert len(chunks) > 1
    assert all(_estimate_tokens(chunk.text) <= 12 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == body


def test_markdown_preserves_title_preamble_whitespace_and_exact_lines(tmp_path):
    body = "# Title\n## First\n" + ("long paragraph value " * 40) + "\n\n\n## Last\nend\n"
    (tmp_path / "doc.md").write_text(body)
    chunks = chunk_markdown("test", tmp_path, "doc.md", 20, max_whole_doc_tokens=0)
    assert "".join(chunk.text for chunk in chunks) == body
    assert chunks[0].text == "# Title\n"
    offset = 0
    for chunk in chunks:
        assert chunk.text == body[offset:offset + len(chunk.text)]
        assert chunk.start_line == body.count("\n", 0, offset) + 1
        offset += len(chunk.text)
        assert chunk.end_line == body.count("\n", 0, offset - 1) + 1
        assert _estimate_tokens(chunk.text) <= 20


def test_preamble_without_title_keeps_first_line(tmp_path):
    body = "First sentence.\nSecond sentence.\n## Section\nbody"
    (tmp_path / "doc.md").write_text(body)
    chunks = chunk_markdown("test", tmp_path, "doc.md", 100, max_whole_doc_tokens=0)
    assert chunks[-1].doc_preamble == "First sentence.\nSecond sentence."
    assert "".join(chunk.text for chunk in chunks) == body


def test_markdown_fenced_headings_remain_in_their_section(tmp_path):
    body = "# Title\n## Real\n```markdown\n## Example only\n" + "x " * 150 + "\n```\n## End\ndone\n"
    (tmp_path / "doc.md").write_text(body)
    chunks = chunk_markdown("test", tmp_path, "doc.md", 20, max_whole_doc_tokens=0)
    assert {chunk.section_heading for chunk in chunks} == {"", "Real", "End"}
    assert "".join(chunk.text for chunk in chunks) == body
    assert all(_estimate_tokens(chunk.text) <= 20 for chunk in chunks)


def test_embedding_budget_includes_prefix_and_tokenizer_special_tokens(tmp_path):
    title = "界🙂 lengthy title " * 12
    preamble = "Reusable context " * 30
    body = f"# {title}\n{preamble}\n## Section heading\n" + "Evidence " * 100
    (tmp_path / "doc.md").write_text(body, encoding="utf-8")
    # Deliberately differs from tiktoken and includes tokenizer special tokens.
    counter = lambda text: len(text.encode("utf-8")) + 2
    chunks = chunk_source(make_source_cfg(tmp_path, max_chunk_tokens=70),
                          count_tokens=counter, embedding_max_tokens=42)
    assert len(chunks) > 1
    assert "".join(chunk.text for chunk in chunks) == body
    assert all(counter(chunk.text) <= 70 for chunk in chunks)
    assert all(counter(chunk.embedding_text) <= 42 for chunk in chunks)
    assert all(chunk.doc_title == title.strip() for chunk in chunks)
    assert all(chunk.doc_preamble == preamble.strip() for chunk in chunks)


def test_tiny_embedding_budget_drops_prefix_not_source_characters(tmp_path):
    body = "界界界"
    (tmp_path / "doc.md").write_text(body, encoding="utf-8")
    counter = lambda text: len(text.encode("utf-8")) + 2
    chunks = chunk_file("test", tmp_path, "doc.md", max_tokens=10,
                        count_tokens=counter, embedding_max_tokens=5)
    assert [chunk.embedding_text for chunk in chunks] == list(body)
    assert "".join(chunk.text for chunk in chunks) == body
    with pytest.raises(ValueError, match="source character"):
        chunk_file("test", tmp_path, "doc.md", max_tokens=10,
                   count_tokens=counter, embedding_max_tokens=4)


def test_treesitter_output_is_rebounded_and_omitted_gaps_recovered(tmp_path, monkeypatch):
    code = "# import context\n\n" + "def example():\n    return '" + "value " * 100 + "'\n\n# trailing context\n"
    (tmp_path / "doc.py").write_text(code)
    syntax_chunk = SimpleNamespace(text=code[code.index("def example"):code.index("\n\n# trailing")],
                                   section_heading="example")
    monkeypatch.setattr("agents_rag.chunker._try_treesitter_chunk", lambda *args: [syntax_chunk])
    chunks = chunk_code("test", tmp_path, "doc.py", 25,
                        count_tokens=len, embedding_max_tokens=30)
    assert "".join(chunk.text for chunk in chunks) == code
    assert all(len(chunk.text) <= 25 and len(chunk.embedding_text) <= 30 for chunk in chunks)
    assert all(chunk.section_heading == "example" for chunk in chunks)


def test_fixed_size_counts_newlines_and_keeps_bounded_overlap():
    code = "".join(f"line_{i:03d}\n" for i in range(30))
    chunks = _chunk_fixed_size(code, "test", "doc.txt", 27,
                               count_tokens=len, embedding_max_tokens=30)
    covered = set()
    previous_start = -1
    for chunk in chunks:
        assert len(chunk.text) <= 27
        assert len(chunk.embedding_text) <= 30
        start = code.index(chunk.text, previous_start + 1)
        assert start > previous_start
        previous_start = start
        covered.update(range(start, start + len(chunk.text)))
        assert chunk.start_line == code.count("\n", 0, start) + 1
        assert chunk.end_line == code.count("\n", 0, start + len(chunk.text) - 1) + 1
    assert covered == set(range(len(code)))
    assert any(current.start_line <= previous.end_line
               for previous, current in zip(chunks, chunks[1:]))


def test_literal_special_token_is_source_text(tmp_path):
    body = "literal <|endoftext|> marker " * 20
    (tmp_path / "doc.md").write_text(body)
    chunks = chunk_file("test", tmp_path, "doc.md", 20)
    assert "".join(chunk.text for chunk in chunks) == body
    assert all(_estimate_tokens(chunk.text) <= 20 for chunk in chunks)


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_chunk_limit_is_rejected(tmp_path, limit):
    (tmp_path / "doc.md").write_text("evidence")
    with pytest.raises(ValueError, match="positive"):
        chunk_file("test", tmp_path, "doc.md", limit)
