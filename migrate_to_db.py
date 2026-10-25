#!/usr/bin/env python3
"""Compatibility entrypoint for copy-only legacy index migration."""
from __future__ import annotations
import json
from pathlib import Path
from agents_rag.migration import migrate as migrate_copy


def migrate(cache_dir: str, source_roots: dict[str,str] | None = None) -> None:
    # Raw supplied artifacts are preserved; source files are not guessed or overwritten.
    result = migrate_copy(cache_dir, Path(cache_dir) / "bookkeeper.db")
    print(json.dumps(result,indent=2))


def main() -> int:
    from agents_rag.config import load_config
    config = load_config()
    migrate(str(Path(config.get("cache_dir","~/.cache/bookkeeper")).expanduser()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
