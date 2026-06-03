"""Tests for agents_rag/config.py: shared config loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from agents_rag.config import load_config, parse_sources, validate_config


def test_validate_empty_config():
    assert validate_config({}) == []


def test_validate_valid_sources(tmp_path):
    (tmp_path / "subdir").mkdir()
    config = {"sources": [{"name": "test", "path": str(tmp_path / "subdir")}]}
    assert validate_config(config) == []


def test_validate_missing_keys():
    config = {"sources": [{}]}
    errors = validate_config(config)
    assert len(errors) == 1
    assert "name" in errors[0]
    assert "path" in errors[0]


def test_validate_nonexistent_path(tmp_path):
    config = {"sources": [{"name": "test", "path": str(tmp_path / "nope")}]}
    errors = validate_config(config)
    assert any("does not exist" in e for e in errors)


def test_validate_bad_source_type():
    config = {"sources": ["not a dict"]}
    errors = validate_config(config)
    assert any("expected a mapping" in e for e in errors)


def test_parse_sources_with_defaults(tmp_path):
    config = {"sources": [{"name": "t", "path": str(tmp_path)}]}
    sources = parse_sources(config)
    assert len(sources) == 1
    s = sources[0]
    assert s.name == "t"
    assert s.index_enabled is True
    assert s.file_types == [".md"]
    assert s.max_chunk_tokens == 512


def test_parse_sources_with_overrides(tmp_path):
    config = {"sources": [{
        "name": "t", "path": str(tmp_path),
        "index_enabled": False,
        "file_types": [".go", ".py"], "max_chunk_tokens": 1024,
        "skip": ["vendor/*"], "min_lines": 10,
    }]}
    sources = parse_sources(config)
    s = sources[0]
    assert s.file_types == [".go", ".py"]
    assert s.index_enabled is False
    assert s.max_chunk_tokens == 1024
    assert s.skip == ["vendor/*"]
    assert s.min_lines == 10


def test_load_config_env_override(tmp_path):
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("use_gemma: false\n")
    with patch.dict(os.environ, {"AGENTS_RAG_CONFIG": str(cfg)}):
        config = load_config()
    assert config["use_gemma"] is False


def test_load_config_missing_returns_empty():
    with patch.dict(os.environ, {"AGENTS_RAG_CONFIG": "/nonexistent/path.yaml"}):
        config = load_config()
    assert config == {}
