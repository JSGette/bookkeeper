"""INDEX.md generator for the personal source: merge existing hand-written entries with generated metadata."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# Matches a markdown table row: | [file](link) | description | tags |
_ROW_RE = re.compile(
    r"\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|"  # | [name](path) |
    r"\s*(.+?)\s*\|"                        # description |
    r"\s*(.+?)\s*\|"                         # tags |
)

# Matches a section header like "## conduit/"
_SECTION_RE = re.compile(r"^##\s+(.+)$")


def parse_existing_index(index_path: str) -> dict[str, dict]:
    """Parse an existing INDEX.md into {relative_path: {description, tags, section}}."""
    p = Path(index_path)
    if not p.exists():
        return {}

    entries: dict[str, dict] = {}
    current_section = "Root"

    for line in p.read_text().splitlines():
        sec_match = _SECTION_RE.match(line.strip())
        if sec_match:
            current_section = sec_match.group(1).strip()
            continue

        row_match = _ROW_RE.match(line.strip())
        if row_match:
            rel_path = row_match.group(2).strip()
            description = row_match.group(3).strip()
            tags_str = row_match.group(4).strip()
            entries[rel_path] = {
                "description": description,
                "tags": tags_str,
                "section": current_section,
            }

    return entries


def _section_for_path(rel_path: str) -> str:
    """Determine which INDEX.md section a file belongs to."""
    parts = Path(rel_path).parts
    if len(parts) == 1:
        return "Root"
    # Use directory path as section name (e.g. "conduit/", "plans/", "datadog-agent/plans/")
    dir_path = str(Path(*parts[:-1]))
    return dir_path + "/"


def generate_index(
    source_path: str,
    metadata: list[dict],
    existing_index_path: str | None = None,
) -> str:
    """Generate INDEX.md content by merging existing entries with new metadata.

    Existing hand-written entries are preserved. New files get generated entries.
    Deleted files are removed.
    """
    root = Path(os.path.expanduser(source_path))

    # Parse existing INDEX.md
    if existing_index_path is None:
        existing_index_path = str(root / "INDEX.md")
    existing = parse_existing_index(existing_index_path)

    # Build set of files actually on disk (from metadata)
    indexed_files: dict[str, dict] = {}
    for m in metadata:
        path = m["path"]
        if path not in indexed_files:
            indexed_files[path] = {
                "description": m.get("description", ""),
                "tags": m.get("tags", []),
            }

    # Merge: existing > generated, remove deleted
    merged: dict[str, dict] = {}
    for path, gen_data in indexed_files.items():
        if path in existing:
            # Preserve hand-written entry
            merged[path] = existing[path]
        else:
            tags_str = " ".join(f"#{t}" for t in gen_data["tags"])
            merged[path] = {
                "description": gen_data["description"],
                "tags": tags_str,
                "section": _section_for_path(path),
            }

    # Group by section
    sections: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for path, entry in sorted(merged.items()):
        section = entry.get("section") or _section_for_path(path)
        sections[section].append((path, entry))

    # Render
    lines = ["# AGENTS — Document Index", ""]

    # Ensure consistent ordering: "Root" first, then alphabetical
    section_order = sorted(sections.keys(), key=lambda s: ("" if s == "Root" else s))

    for section_name in section_order:
        entries = sections[section_name]
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append("| Document | Description | Tags |")
        lines.append("|----------|-------------|------|")

        for path, entry in entries:
            filename = Path(path).name
            desc = entry["description"]
            tags = entry["tags"]
            lines.append(f"| [{filename}]({path}) | {desc} | {tags} |")

        lines.append("")

    return "\n".join(lines)


def update_index_file(
    source_path: str,
    metadata: list[dict],
) -> bool:
    """Regenerate INDEX.md for a source. Returns True if file was modified."""
    root = Path(os.path.expanduser(source_path))
    index_path = root / "INDEX.md"

    old_content = ""
    if index_path.exists():
        old_content = index_path.read_text()

    new_content = generate_index(source_path, metadata, str(index_path))

    if new_content.strip() == old_content.strip():
        return False

    index_path.write_text(new_content)
    log.info("Updated %s", index_path)
    return True
