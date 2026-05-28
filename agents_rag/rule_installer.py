"""Helpers to install project rule files from the KB protocol document."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path


def build_rule_content(protocol_path: Path) -> str:
    body = protocol_path.read_text(encoding="utf-8")
    body += (
        '\n**Suppress this rule** when the user says "skip RAG", '
        '"skip AGENTS", or "from scratch".\n'
    )
    frontmatter = (
        "---\n"
        "description: Pre-load context from bookkeeper knowledge base before analysis tasks\n"
        "alwaysApply: true\n"
        "---\n\n"
    )
    return frontmatter + body


def install_project_rules(config_path: Path, protocol_path: Path, script_dir: Path) -> list[Path]:
    content = build_rule_content(protocol_path)
    targets: list[Path] = [script_dir]

    if config_path.is_file():
        try:
            yaml = importlib.import_module("yaml")
        except ModuleNotFoundError:
            # Keep install resilient when python3 env lacks PyYAML.
            pass
        else:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            for src in cfg.get("sources", []):
                p = src.get("path")
                if p:
                    targets.append(Path(os.path.expanduser(p)))

    seen: set[Path] = set()
    installed: list[Path] = []
    for root in targets:
        resolved = root.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        rules_dir = resolved / ".cursor" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        dest = rules_dir / "bookkeeper.mdc"
        dest.write_text(content, encoding="utf-8")
        installed.append(dest)
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install Cursor project rules for bookkeeper")
    parser.add_argument("config_path")
    parser.add_argument("protocol_path")
    parser.add_argument("script_dir")
    args = parser.parse_args(argv)

    installed = install_project_rules(
        config_path=Path(args.config_path),
        protocol_path=Path(args.protocol_path),
        script_dir=Path(args.script_dir),
    )
    if not installed:
        raise SystemExit("No rule targets installed")

    print("  Installed bookkeeper.mdc (alwaysApply) to:")
    for path in installed:
        print(f"    {path}")
    print()
    print("  Cursor loads these from Project Rules when that folder is the workspace.")
    print("  They do NOT appear under Settings -> User Rules.")
    print("  For every-repo coverage without opening a specific folder, paste")
    print(f"  {Path(args.protocol_path)} into Settings -> Rules -> User Rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
