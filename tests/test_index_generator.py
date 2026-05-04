"""Tests for agents_rag/index_generator.py: pure functions for INDEX.md generation."""

from __future__ import annotations

from pathlib import Path

from agents_rag.index_generator import (
    _section_for_path,
    generate_index,
    parse_existing_index,
    update_index_file,
)


# ---------------------------------------------------------------------------
# parse_existing_index
# ---------------------------------------------------------------------------


def test_parse_nonexistent_file(tmp_path):
    assert parse_existing_index(str(tmp_path / "nope.md")) == {}


def test_parse_existing_index(tmp_path):
    index_md = tmp_path / "INDEX.md"
    index_md.write_text(
        "# Title\n\n"
        "## Root\n\n"
        "| Document | Description | Tags |\n"
        "|----------|-------------|------|\n"
        "| [guide.md](guide.md) | A user guide | #guide |\n"
        "| [plan.md](plan.md) | Migration plan | #migration |\n"
        "\n"
        "## plans/\n\n"
        "| Document | Description | Tags |\n"
        "|----------|-------------|------|\n"
        "| [deep.md](plans/deep.md) | Deep analysis | #deep |\n"
    )
    entries = parse_existing_index(str(index_md))
    assert "guide.md" in entries
    assert entries["guide.md"]["description"] == "A user guide"
    assert entries["guide.md"]["tags"] == "#guide"
    assert entries["guide.md"]["section"] == "Root"
    assert "plans/deep.md" in entries
    assert entries["plans/deep.md"]["section"] == "plans/"


# ---------------------------------------------------------------------------
# _section_for_path
# ---------------------------------------------------------------------------


def test_section_root_file():
    assert _section_for_path("guide.md") == "Root"


def test_section_nested():
    assert _section_for_path("plans/migration.md") == "plans/"


def test_section_deeply_nested():
    assert _section_for_path("a/b/c/file.md") == "a/b/c/"


# ---------------------------------------------------------------------------
# generate_index
# ---------------------------------------------------------------------------


def test_generate_index_from_scratch(tmp_path):
    metadata = [
        {"path": "guide.md", "description": "Guide desc", "tags": ["guide"]},
        {"path": "plans/plan.md", "description": "Plan desc", "tags": ["plan"]},
    ]
    content = generate_index(str(tmp_path), metadata, str(tmp_path / "INDEX.md"))
    assert "# AGENTS" in content
    assert "## Root" in content
    assert "[guide.md](guide.md)" in content
    assert "Guide desc" in content
    assert "## plans/" in content
    assert "[plan.md](plans/plan.md)" in content


def test_generate_index_preserves_existing(tmp_path):
    index_md = tmp_path / "INDEX.md"
    index_md.write_text(
        "# Title\n\n"
        "## Root\n\n"
        "| Document | Description | Tags |\n"
        "|----------|-------------|------|\n"
        "| [guide.md](guide.md) | Hand-written | #custom |\n"
    )
    metadata = [
        {"path": "guide.md", "description": "Generated desc", "tags": ["gen"]},
        {"path": "new.md", "description": "New file", "tags": ["new"]},
    ]
    content = generate_index(str(tmp_path), metadata, str(index_md))
    assert "Hand-written" in content  # preserved
    assert "#custom" in content
    assert "New file" in content  # added


def test_generate_index_removes_deleted(tmp_path):
    index_md = tmp_path / "INDEX.md"
    index_md.write_text(
        "# Title\n\n## Root\n\n"
        "| Document | Description | Tags |\n"
        "|----------|-------------|------|\n"
        "| [deleted.md](deleted.md) | Gone | #old |\n"
    )
    metadata = []  # deleted.md is gone from index
    content = generate_index(str(tmp_path), metadata, str(index_md))
    assert "deleted.md" not in content


# ---------------------------------------------------------------------------
# update_index_file
# ---------------------------------------------------------------------------


def test_update_index_file_creates(tmp_path):
    metadata = [{"path": "test.md", "description": "desc", "tags": ["t"]}]
    assert update_index_file(str(tmp_path), metadata) is True
    assert (tmp_path / "INDEX.md").exists()


def test_update_index_file_no_change(tmp_path):
    metadata = [{"path": "test.md", "description": "desc", "tags": ["t"]}]
    update_index_file(str(tmp_path), metadata)
    assert update_index_file(str(tmp_path), metadata) is False
