"""Tests for agents_rag/tagger.py: pure functions and tag_file merge logic."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agents_rag import tagger

from agents_rag.tagger import (
    _clean_description,
    _clean_tags,
    description_from_content,
    load_overrides,
    tag_file,
    tags_from_path,
)


# ---------------------------------------------------------------------------
# tags_from_path
# ---------------------------------------------------------------------------


def test_tags_from_simple_file():
    assert tags_from_path("plans/migration.md") == ["plans", "migration"]


def test_tags_from_nested_path():
    tags = tags_from_path("sub/dir/my_script.py")
    assert "sub" in tags
    assert "dir" in tags
    assert "my-script" in tags


def test_tags_strips_prefixes():
    tags = tags_from_path("docs/guide.md")
    assert "docs" not in tags
    assert "guide" in tags


def test_tags_strips_common_names():
    tags = tags_from_path("readme.md")
    assert tags == []


def test_tags_deduplicates():
    tags = tags_from_path("plans/plans/thing.md")
    assert tags.count("plans") == 1


def test_tags_short_components_ignored():
    tags = tags_from_path("a/b/c.md")
    assert tags == []


# ---------------------------------------------------------------------------
# description_from_content
# ---------------------------------------------------------------------------


def test_description_first_paragraph():
    text = "# Title\n\nThis is the first paragraph.\n\n## Section\n\nMore stuff."
    assert description_from_content(text) == "This is the first paragraph."


def test_description_skips_headings_and_tables():
    text = "# Title\n| col | col |\n--- \n* bullet\n- dash\nActual content here."
    assert description_from_content(text) == "Actual content here."


def test_description_empty_file():
    assert description_from_content("") == ""


def test_description_only_headings():
    assert description_from_content("# Title\n## Subtitle\n### Sub") == ""


def test_description_truncated_to_200():
    long_line = "x" * 300
    assert len(description_from_content(long_line)) == 200


# ---------------------------------------------------------------------------
# _clean_description
# ---------------------------------------------------------------------------


def test_clean_strips_generic_opener():
    assert _clean_description("This document describes the migration plan for Bazel.") == \
        "Describes the migration plan for Bazel."


def test_clean_strips_comprehensive():
    assert _clean_description("A comprehensive guide to setting up CI.") == \
        "Guide to setting up CI."


def test_clean_detects_garbage():
    assert _clean_description("no specific information available") == ""


def test_clean_short_garbage():
    assert _clean_description("short") == ""


def test_clean_capitalizes():
    assert _clean_description("migrating from make to bazel is recommended.")[0] == "M"


def test_clean_truncates_to_200():
    assert len(_clean_description("Valid " + "x" * 300)) == 200


# ---------------------------------------------------------------------------
# _clean_tags
# ---------------------------------------------------------------------------


def test_clean_tags_removes_filler():
    tags = _clean_tags(["bazel", "data", "report", "migration", "overview"])
    assert "bazel" in tags
    assert "migration" in tags
    assert "data" not in tags
    assert "report" not in tags
    assert "overview" not in tags


def test_clean_tags_deduplicates():
    tags = _clean_tags(["Bazel", "bazel", "BAZEL"])
    assert tags == ["bazel"]


def test_clean_tags_ignores_short():
    tags = _clean_tags(["a", "", "ok", "go"])
    assert tags == ["ok", "go"]


def test_clean_tags_max_10():
    tags = _clean_tags([f"tag{i}" for i in range(20)])
    assert len(tags) == 10


# ---------------------------------------------------------------------------
# load_overrides
# ---------------------------------------------------------------------------


def test_load_overrides_empty_dir(tmp_path):
    assert load_overrides(str(tmp_path)) == {}


def test_load_overrides_reads_yaml(tmp_path):
    meta = {
        "files": {
            "guide.md": {
                "description": "Custom description",
                "tags": ["custom-tag"],
            }
        }
    }
    import yaml
    (tmp_path / ".bookkeeper-meta.yaml").write_text(yaml.dump(meta))
    overrides = load_overrides(str(tmp_path))
    assert "guide.md" in overrides
    assert overrides["guide.md"]["description"] == "Custom description"
    assert overrides["guide.md"]["tags"] == ["custom-tag"]


def test_load_overrides_nested(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    meta = {"files": {"util.py": {"description": "desc", "tags": ["util"]}}}
    import yaml
    (sub / ".bookkeeper-meta.yaml").write_text(yaml.dump(meta))
    overrides = load_overrides(str(tmp_path))
    assert "sub/util.py" in overrides


# ---------------------------------------------------------------------------
# tag_file (layer merge priority)
# ---------------------------------------------------------------------------


def test_tag_file_no_gemma():
    desc, tags = tag_file(
        "plans/migration.md", "Migrate all modules to Bazel.",
        "/source", "/cache", overrides={}, use_gemma=False,
    )
    assert desc == "Migrate all modules to Bazel."
    assert "plans" in tags
    assert "migration" in tags


def test_tag_file_override_wins():
    overrides = {
        "guide.md": {"description": "Override desc", "tags": ["override-tag"]}
    }
    desc, tags = tag_file(
        "guide.md", "Content here.", "/source", "/cache",
        overrides=overrides, use_gemma=False,
    )
    assert desc == "Override desc"
    assert "override-tag" in tags


@patch("agents_rag.tagger.generate_tags_gemma")
def test_tag_file_gemma_merged(mock_gemma):
    mock_gemma.return_value = {"description": "LLM desc", "tags": ["llm-tag"]}
    desc, tags = tag_file(
        "plans/thing.md", "Some content.", "/source", "/cache",
        overrides={}, use_gemma=True,
    )
    assert desc == "LLM desc"
    assert "llm-tag" in tags
    assert "plans" in tags  # path tags still present


@patch("agents_rag.tagger.generate_tags_gemma")
def test_tag_file_gemma_override_priority(mock_gemma):
    mock_gemma.return_value = {"description": "LLM desc", "tags": ["llm-tag"]}
    overrides = {"thing.md": {"description": "Manual", "tags": ["manual"]}}
    desc, tags = tag_file(
        "thing.md", "Content.", "/source", "/cache",
        overrides=overrides, use_gemma=True,
    )
    assert desc == "Manual"
    assert tags[0] == "manual"  # override tags come first


@pytest.mark.parametrize("setting", ["prompt", "examples", "model", "profile_limits"])
def test_generated_tag_cache_misses_when_producer_configuration_changes(tmp_path, monkeypatch, setting):
    examples = tmp_path / "examples.jsonl"
    example = {"excerpt": "Example source", "description": "Original example description", "tags": ["original-example"]}
    examples.write_text(json.dumps(example) + "\n", encoding="utf-8")
    monkeypatch.setattr(tagger, "_tagger_prompt", "Original tagger instructions")
    monkeypatch.setattr(tagger, "_tagger_examples_path", str(examples))
    original_profile = replace(tagger.ollama_client.get_profile("fast"), name="offline-model-v1")
    profiles = {**tagger.ollama_client._profiles, "fast": original_profile}
    monkeypatch.setattr(tagger.ollama_client, "_profiles", profiles)
    generated = Mock(side_effect=[
        {"description": "Original generated description for this source.", "tags": ["old-semantic-tag"]},
        {"description": "Updated generated description for this source.", "tags": ["new-semantic-tag"]},
    ])
    monkeypatch.setattr(tagger.ollama_client, "chat_json", generated)
    source = "The source text remains exactly the same."
    cache = str(tmp_path / "cache")

    first = tagger.generate_tags_gemma(source, "guide.md", cache)
    assert tagger.generate_tags_gemma(source, "guide.md", cache) == first
    assert generated.call_count == 1

    if setting == "prompt":
        monkeypatch.setattr(tagger, "_tagger_prompt", "Changed tagger instructions")
    elif setting == "examples":
        # Editing the existing examples file must invalidate cached generations.
        example["description"] = "Changed example description"
        examples.write_text(json.dumps(example) + "\n", encoding="utf-8")
    elif setting == "model":
        monkeypatch.setattr(tagger.ollama_client, "_profiles", {**profiles, "fast": replace(original_profile, name="offline-model-v2")})
    else:
        monkeypatch.setattr(tagger.ollama_client, "_profiles", {**profiles, "fast": replace(original_profile, max_input_chars=1234)})

    changed = tagger.generate_tags_gemma(source, "guide.md", cache)
    assert generated.call_count == 2
    assert changed != first
    assert changed["tags"] == ["new-semantic-tag"]
    assert tagger.generate_tags_gemma(source, "guide.md", cache) == changed
    assert generated.call_count == 2
