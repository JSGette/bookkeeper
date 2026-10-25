"""Multi-format chunker: markdown (heading-based), source code (treesitter), fallback (fixed-size)."""

from __future__ import annotations

import fnmatch
import os
import re
from bisect import bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_MAX_CHUNK_TOKENS = 512
TokenCounter = Callable[[str], int]

# Lazy-loaded tiktoken encoder for accurate token counting
_enc = None

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
    doc_preamble: str = ""
    # Only the embedding prefix is shortened; source text and metadata stay intact.
    _embedding_prefix: str | None = field(default=None, repr=False, compare=False)

    @property
    def embedding_text(self) -> str:
        if self._embedding_prefix is not None:
            return _with_prefix(self._embedding_prefix, self.text)
        prefix = self.doc_title
        if self.doc_preamble:
            prefix += f" | {self.doc_preamble[:200]}"
        if self.section_heading and self.section_heading != self.doc_title:
            prefix += f" — {self.section_heading}"
        return f"{prefix}\n{self.text}"


_DEFAULT_MAX_WHOLE_DOC_TOKENS = 1500


@dataclass
class SourceConfig:
    name: str
    path: str
    index_enabled: bool = True
    skip: list[str] = field(default_factory=list)
    min_lines: int = 0
    generate_index: bool = False
    file_types: list[str] = field(default_factory=lambda: [".md"])
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS
    max_whole_doc_tokens: int = _DEFAULT_MAX_WHOLE_DOC_TOKENS


def _should_skip(rel_path: str, skip_patterns: list[str]) -> bool:
    for pattern in skip_patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def _estimate_tokens(text: str) -> int:
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    return len(_enc.encode(text, disallowed_special=()))


def _with_prefix(prefix: str, text: str) -> str:
    return f"{prefix}\n{text}" if prefix else text


def _fit_end(text: str, start: int, stop: int, fits: Callable[[str], bool], hint: int) -> int:
    """Find a fitting Unicode boundary without decoding partial token byte strings.

    Token counts are not strictly monotonic. The returned candidate is always
    checked, although it need not be the longest possible fitting prefix.
    """
    low = start
    high = min(stop, start + max(1, hint))
    while fits(text[start:high]):
        low = high
        if high == stop:
            return high
        high = min(stop, start + 2 * (high - start))
    while high - low > 1:
        middle = (low + high) // 2
        if fits(text[start:middle]):
            low = middle
        else:
            high = middle
    return low


def _split_spans(
    text: str, max_tokens: int, count_tokens: TokenCounter,
    *, prefix: str = "", embedding_max_tokens: int | None = None,
    overlap_lines: int = 0,
) -> Iterator[tuple[int, int]]:
    """Yield exact source slices, preferring paragraphs and then line boundaries."""
    if max_tokens <= 0 or (embedding_max_tokens is not None and embedding_max_tokens <= 0):
        raise ValueError("Token limits must be positive")

    def fits(part: str) -> bool:
        return count_tokens(part) <= max_tokens and (
            embedding_max_tokens is None
            or count_tokens(_with_prefix(prefix, part)) <= embedding_max_tokens
        )

    start = 0
    hint = min(max_tokens, embedding_max_tokens or max_tokens)
    while start < len(text):
        end = _fit_end(text, start, len(text), fits, hint)
        if end == start:
            raise ValueError("Token limit is too small to preserve a source character")
        if end < len(text):
            window = text[start:end]
            for pattern in (r"\n\n+", r"\n", r"\s+"):
                boundaries = [m.end() for m in re.finditer(pattern, window)
                              if m.end() >= len(window) // 2]
                candidate = next((start + boundary for boundary in reversed(boundaries)
                                  if fits(text[start:start + boundary])), None)
                if candidate is not None:
                    end = candidate
                    break
        yield start, end
        if end == len(text):
            break
        if overlap_lines:
            # Retain at most N complete lines, always making forward progress.
            line_starts = [m.end() + start for m in re.finditer(r"\n", text[start:end])
                           if m.end() + start < end]
            start = line_starts[max(0, len(line_starts) - overlap_lines)] if line_starts else end
        else:
            start = end


def _bounded_prefix(chunk: Chunk, count_tokens: TokenCounter, limit: int | None) -> str | None:
    if limit is None:
        return None
    if limit <= 0:
        raise ValueError("Token limits must be positive")
    prefix = chunk.embedding_text.removesuffix("\n" + chunk.text)
    # Reserve most of the input for evidence. The final combined string is
    # checked independently because tokenizers include special-token overhead.
    budget = limit // 4
    end = _fit_end(prefix, 0, len(prefix), lambda p: count_tokens(p + "\n") <= budget, budget)
    return prefix[:end]


def _section_chunks(
    content: str, source: str, rel_path: str, title: str,
    sections: list[tuple[str, int, int]], max_tokens: int,
    count_tokens: TokenCounter, embedding_max_tokens: int | None,
    *, preamble: str = "", overlap_lines: int = 0,
) -> list[Chunk]:
    line_starts = [0] + [m.end() for m in re.finditer(r"\n", content)]
    chunks = []
    for heading, start, end in sections:
        template = Chunk(source, rel_path, title, heading, 0, 0, "", doc_preamble=preamble)
        prefix = _bounded_prefix(template, count_tokens, embedding_max_tokens)
        section = content[start:end]
        for part_start, part_end in _split_spans(
            section, max_tokens, count_tokens, prefix=prefix or "",
            embedding_max_tokens=embedding_max_tokens, overlap_lines=overlap_lines,
        ):
            first_line = bisect_right(line_starts, start + part_start)
            last_line = bisect_right(line_starts, start + part_end - 1)
            chunks.append(Chunk(
                source, rel_path, title, heading, first_line, last_line,
                section[part_start:part_end], doc_preamble=preamble,
                _embedding_prefix=prefix,
            ))
    return chunks


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
    """Split oversized text without dropping whitespace or Unicode characters."""
    return _split_by_paragraphs(text, max_tokens)


def _split_by_paragraphs(text: str, max_tokens: int) -> list[str]:
    """Split at natural boundaries, with a hard fallback inside long paragraphs."""
    return [text[start:end] for start, end in _split_spans(text, max_tokens, _estimate_tokens)]


def chunk_markdown(
    source: str, root: Path, rel_path: str, max_tokens: int,
    max_whole_doc_tokens: int = _DEFAULT_MAX_WHOLE_DOC_TOKENS,
    *, count_tokens: TokenCounter | None = None, embedding_max_tokens: int | None = None,
) -> list[Chunk]:
    full_path = root / rel_path
    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = content.splitlines(keepends=True)
    if not content.strip():
        return []

    counter = count_tokens or _estimate_tokens
    doc_title = _extract_title(lines) or Path(rel_path).stem

    # The whole-document preference must never override either hard limit.
    if max_whole_doc_tokens > 0 and counter(content) <= min(max_whole_doc_tokens, max_tokens):
        whole = _section_chunks(content, source, rel_path, doc_title, [("", 0, len(content))],
                                max_tokens, counter, embedding_max_tokens)
        if len(whole) == 1:
            return whole

    sections: list[tuple[str, int]] = []
    offset = 0
    fence = ""
    fence_length = 0
    for line in lines:
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if not fence:
                fence, fence_length = run[0], len(run)
            elif run[0] == fence and len(run) >= fence_length and not line[marker.end():].strip():
                fence = ""
            offset += len(line)
            continue
        if not fence and re.match(r"^##\s+", line):
            heading = re.sub(r"^##\s+", "", line).strip()
            sections.append((heading, offset))
        offset += len(line)

    # Extract preamble: text between # title and first ## heading (or end of doc)
    preamble = ""
    if sections and sections[0][1] > 0:
        raw = content[:sections[0][1]]
        preamble = re.sub(r"^#\s+[^\n]*(?:\n|$)", "", raw, count=1, flags=re.MULTILINE).strip()

    if not sections:
        sections = [("", 0)]

    if sections[0][1] > 0:
        sections.insert(0, ("", 0))
    spans = [(heading, start, sections[i + 1][1] if i + 1 < len(sections) else len(content))
             for i, (heading, start) in enumerate(sections)]
    return _section_chunks(content, source, rel_path, doc_title, spans, max_tokens,
                           counter, embedding_max_tokens, preamble=preamble)


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


def chunk_code(
    source: str, root: Path, rel_path: str, max_tokens: int,
    *, count_tokens: TokenCounter | None = None, embedding_max_tokens: int | None = None,
) -> list[Chunk]:
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
        # Third-party chunkers may exceed the cap or omit separators. Recover
        # their boundaries from the source and include every intervening character.
        cursor = 0
        sections = []
        for chunk in ts_result:
            position = content.find(chunk.text, cursor)
            if not chunk.text or position < 0:
                break
            end = position + len(chunk.text)
            sections.append((chunk.section_heading, cursor, end))
            cursor = end
        else:
            if sections:
                heading, start, _ = sections[-1]
                sections[-1] = (heading, start, len(content))
                return _section_chunks(content, source, rel_path, Path(rel_path).stem, sections,
                                       max_tokens, count_tokens or _estimate_tokens, embedding_max_tokens)

    return _chunk_fixed_size(content, source, rel_path, max_tokens,
                            count_tokens=count_tokens, embedding_max_tokens=embedding_max_tokens)


def _chunk_fixed_size(
    content: str, source: str, rel_path: str, max_tokens: int,
    *, count_tokens: TokenCounter | None = None, embedding_max_tokens: int | None = None,
) -> list[Chunk]:
    """Bounded source slices with line overlap, including oversized single lines."""
    chunks = _section_chunks(content, source, rel_path, Path(rel_path).stem,
                             [("", 0, len(content))], max_tokens,
                             count_tokens or _estimate_tokens, embedding_max_tokens, overlap_lines=10)
    for chunk in chunks:
        chunk.section_heading = f"lines_{chunk.start_line}_{chunk.end_line}"
        # A bounded prefix is deliberately fixed before splitting, so adding
        # display-only line metadata cannot exceed the embedding limit.
    return chunks


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def chunk_file(
    source: str, root: Path, rel_path: str,
    max_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
    max_whole_doc_tokens: int = _DEFAULT_MAX_WHOLE_DOC_TOKENS,
    *, count_tokens: TokenCounter | None = None, embedding_max_tokens: int | None = None,
) -> list[Chunk]:
    """Chunk a file; optional counter and full-input limit match its embedding model."""
    ext = Path(rel_path).suffix.lower()
    if ext in _MD_EXTENSIONS:
        return chunk_markdown(source, root, rel_path, max_tokens, max_whole_doc_tokens,
                              count_tokens=count_tokens, embedding_max_tokens=embedding_max_tokens)
    elif ext in _CODE_EXTENSIONS:
        return chunk_code(source, root, rel_path, max_tokens,
                          count_tokens=count_tokens, embedding_max_tokens=embedding_max_tokens)
    else:
        full_path = root / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        if content.strip():
            return _chunk_fixed_size(content, source, rel_path, max_tokens,
                                    count_tokens=count_tokens, embedding_max_tokens=embedding_max_tokens)
        return []


def chunk_source(
    source_cfg: SourceConfig, *, count_tokens: TokenCounter | None = None,
    embedding_max_tokens: int | None = None,
) -> list[Chunk]:
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
        all_chunks.extend(chunk_file(
            source_cfg.name, root, rel, source_cfg.max_chunk_tokens,
            source_cfg.max_whole_doc_tokens,
            count_tokens=count_tokens, embedding_max_tokens=embedding_max_tokens,
        ))

    return all_chunks
