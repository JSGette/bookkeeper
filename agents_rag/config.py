"""Shared configuration loading and validation for bookkeeper."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from .chunker import SourceConfig

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
]

_REQUIRED_SOURCE_KEYS = {"name", "path"}


def load_config() -> dict:
    """Load config from AGENTS_RAG_CONFIG env var or default paths."""
    config_path = os.environ.get("AGENTS_RAG_CONFIG")
    search = [config_path] if config_path else _DEFAULT_CONFIG_PATHS
    for p in search:
        if p and Path(p).exists():
            log.info("Loading config from %s", p)
            return yaml.safe_load(Path(p).read_text()) or {}
    log.warning("No config found, using defaults")
    return {}


def validate_config(config: dict) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    for i, s in enumerate(config.get("sources", [])):
        if not isinstance(s, dict):
            errors.append(f"sources[{i}]: expected a mapping, got {type(s).__name__}")
            continue
        missing = _REQUIRED_SOURCE_KEYS - s.keys()
        if missing:
            errors.append(f"sources[{i}]: missing required keys: {', '.join(sorted(missing))}")
        if "path" in s and not Path(os.path.expanduser(s["path"])).is_dir():
            errors.append(f"sources[{i}] ({s.get('name', '?')}): path does not exist: {s['path']}")
    sessions = config.get("sessions", {})
    if sessions and not isinstance(sessions, dict):
        errors.append(f"sessions: expected a mapping, got {type(sessions).__name__}")
    return errors


def parse_sources(config: dict) -> list[SourceConfig]:
    """Parse config dict into SourceConfig list."""
    sources = []
    for s in config.get("sources", []):
        sources.append(SourceConfig(
            name=s["name"],
            path=os.path.expanduser(s["path"]),
            index_enabled=bool(s.get("index_enabled", True)),
            skip=s.get("skip", []),
            min_lines=s.get("min_lines", 0),
            generate_index=s.get("generate_index", False),
            file_types=s.get("file_types", [".md"]),
            max_chunk_tokens=s.get("max_chunk_tokens", 512),
            max_whole_doc_tokens=s.get("max_whole_doc_tokens", 1500),
        ))
    return sources


def resolve_database_path(config: dict) -> str:
    """Honor explicit selection and never silently replace a legacy database with an empty one."""
    if config.get("db_path"):
        return str(Path(config["db_path"]).expanduser().resolve())
    cache = Path(config.get("cache_dir", "~/.cache/bookkeeper")).expanduser().resolve()
    target = cache / "bookkeeper.db"
    if target.exists():
        return str(target)
    candidates = [cache / "agents-rag.db"]
    if cache == Path("~/.cache/bookkeeper").expanduser().resolve():
        candidates.extend([Path("~/.cache/agents-rag/agents-rag.db").expanduser(),
                           Path("~/.cache/agents-rag/bookkeeper.db").expanduser()])
    legacy = [str(p) for p in candidates if p.is_file()]
    if legacy or (cache / "chunks.json").exists() or (cache / "index.npz").exists():
        raise ValueError("Legacy data found; select db_path explicitly or run bookkeeper migrate "
                         "--from SOURCE --to DESTINATION before starting. Candidates: " + ", ".join(legacy or [str(cache)]))
    return str(target)
