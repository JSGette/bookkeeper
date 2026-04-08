"""Multi-format chunker: markdown (heading-based), source code (treesitter), fallback (fixed-size)."""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# ~4 chars per token for rough estimation
_CHARS_PER_TOKEN = 4
_DEFAULT_MAX_CHUNK_TOKENS = 512

_MD_EXTENSIONS = {".md", ".mdx", ".markdown"}

_CODE_EXTENSIONS = {
    ".py", ".go", ".rs", ".js", ".ts", ".tsx", ".jsx", ".bzl",
    ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json",
}


@dataclass
class Chunk:
    source: str
    path: str
    doc_title: str
    section_heading: str
    start_line: int
    end_line: int
    text: str
    tags: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def embedding_text(self) -> str:
        if self.section_heading and self.section_heading != self.doc_title:
            return f"{self.doc_title} — {self.section_heading}\n{self.text}"
        return f"{self.doc_title}\n{self.text}"


@dataclass
class SourceConfig:
    name: str
    path: str
    skip: list[str] = field(default_factory=list)
    min_lines: int = 0
    generate_index: bool = False
    file_types: list[str] = field(default_factory=lambda: [".md"])
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS


def _should_skip(rel_path: str, skip_patterns: list[str]) -> bool:
    for pattern in skip_patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------

def _extract_title(lines: list[str]) -> str:
    for line in lines[:20]:
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            return m.group(1).strip()
    return ""


def _split_oversized(text: str, heading: str, max_tokens: int) -> list[str]:
    """Split text that exceeds max_tokens at ### boundaries, then paragraph boundaries."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    # Try splitting at ### headings
    parts = re.split(r"(?=^###\s+)", text, flags=re.MULTILINE)
    if len(parts) > 1:
        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if _estimate_tokens(part) <= max_tokens:
                result.append(part)
            else:
                result.extend(_split_by_paragraphs(part, max_tokens))
        return result

    return _split_by_paragraphs(text, max_tokens)


def _split_by_paragraphs(text: str, max_tokens: int) -> list[str]:
    """Split text at paragraph boundaries (double newlines) with overlap."""
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current: list[str] = []
    current_tokens = 0
    overlap_chars = 50 * _CHARS_PER_TOKEN

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            # Keep last paragraph as overlap
            overlap = current[-1] if current else ""
            current = [overlap] if _estimate_tokens(overlap) < overlap_chars // _CHARS_PER_TOKEN else []
            current_tokens = _estimate_tokens("\n\n".join(current))
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


def chunk_markdown(source: str, root: Path, rel_path: str, max_tokens: int) -> list[Chunk]:
    full_path = root / rel_path
    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = content.splitlines()
    if not lines:
        return []

    doc_title = _extract_title(lines) or Path(rel_path).stem

    sections: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        if re.match(r"^##\s+", line):
            heading = re.sub(r"^##\s+", "", line).strip()
            sections.append((heading, i))

    chunks = []

    if not sections:
        for part in _split_oversized(content, "", max_tokens):
            chunks.append(Chunk(
                source=source, path=rel_path, doc_title=doc_title,
                section_heading="", start_line=1, end_line=len(lines),
                text=part,
            ))
        return chunks

    if sections[0][1] > 0:
        preamble_lines = lines[: sections[0][1]]
        preamble_text = "\n".join(preamble_lines).strip()
        if preamble_text and len(preamble_lines) > 2:
            for part in _split_oversized(preamble_text, "", max_tokens):
                chunks.append(Chunk(
                    source=source, path=rel_path, doc_title=doc_title,
                    section_heading="", start_line=1, end_line=sections[0][1],
                    text=part,
                ))

    for idx, (heading, start) in enumerate(sections):
        end = sections[idx + 1][1] if idx + 1 < len(sections) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()
        if not section_text:
            continue
        for part in _split_oversized(section_text, heading, max_tokens):
            chunks.append(Chunk(
                source=source, path=rel_path, doc_title=doc_title,
                section_heading=heading, start_line=start + 1, end_line=end,
                text=part,
            ))

    return chunks


# ---------------------------------------------------------------------------
# Source code chunker (treesitter with fixed-size fallback)
# ---------------------------------------------------------------------------

def _try_treesitter_chunk(content: str, source: str, rel_path: str, max_tokens: int) -> list[Chunk] | None:
    """Attempt to chunk with treesitter-chunker. Returns None if unavailable."""
    try:
        from treesitter_chunker import chunk as ts_chunk
    except ImportError:
        return None

    try:
        ts_chunks = ts_chunk(content, filename=rel_path, max_tokens=max_tokens)
    except Exception:
        return None

    doc_title = Path(rel_path).stem
    chunks = []
    for i, tc in enumerate(ts_chunks):
        text = tc.text if hasattr(tc, "text") else str(tc)
        heading = tc.name if hasattr(tc, "name") else f"chunk_{i}"
        chunks.append(Chunk(
            source=source, path=rel_path, doc_title=doc_title,
            section_heading=heading,
            start_line=tc.start_line if hasattr(tc, "start_line") else 0,
            end_line=tc.end_line if hasattr(tc, "end_line") else 0,
            text=text,
        ))
    return chunks if chunks else None


def chunk_code(source: str, root: Path, rel_path: str, max_tokens: int) -> list[Chunk]:
    """Chunk source code files. Uses treesitter if available, else fixed-size."""
    full_path = root / rel_path
    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not content.strip():
        return []

    ts_result = _try_treesitter_chunk(content, source, rel_path, max_tokens)
    if ts_result is not None:
        return ts_result

    return _chunk_fixed_size(content, source, rel_path, max_tokens)


def _chunk_fixed_size(content: str, source: str, rel_path: str, max_tokens: int) -> list[Chunk]:
    """Fixed-size chunking with overlap for any file type."""
    doc_title = Path(rel_path).stem
    lines = content.splitlines()
    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_lines = 10
    chunks = []

    start = 0
    while start < len(lines):
        end = start
        char_count = 0
        while end < len(lines) and char_count + len(lines[end]) < max_chars:
            char_count += len(lines[end]) + 1
            end += 1
        if end == start:
            end = start + 1

        chunk_text = "\n".join(lines[start:end])
        chunks.append(Chunk(
            source=source, path=rel_path, doc_title=doc_title,
            section_heading=f"lines_{start + 1}_{end}",
            start_line=start + 1, end_line=end,
            text=chunk_text,
        ))
        start = max(start + 1, end - overlap_lines)

    return chunks


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def chunk_file(source: str, root: Path, rel_path: str, max_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS) -> list[Chunk]:
    """Chunk a single file, dispatching by extension."""
    ext = Path(rel_path).suffix.lower()
    if ext in _MD_EXTENSIONS:
        return chunk_markdown(source, root, rel_path, max_tokens)
    elif ext in _CODE_EXTENSIONS:
        return chunk_code(source, root, rel_path, max_tokens)
    else:
        full_path = root / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        if content.strip():
            return _chunk_fixed_size(content, source, rel_path, max_tokens)
        return []


def chunk_source(source_cfg: SourceConfig) -> list[Chunk]:
    root = Path(os.path.expanduser(source_cfg.path))
    if not root.is_dir():
        return []

    extensions = set(source_cfg.file_types)
    all_chunks: list[Chunk] = []

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions:
            continue
        rel = str(file_path.relative_to(root))
        if _should_skip(rel, source_cfg.skip):
            continue
        if source_cfg.min_lines > 0:
            try:
                line_count = sum(1 for _ in file_path.open(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            if line_count < source_cfg.min_lines:
                continue
        all_chunks.extend(chunk_file(source_cfg.name, root, rel, source_cfg.max_chunk_tokens))

    return all_chunks
