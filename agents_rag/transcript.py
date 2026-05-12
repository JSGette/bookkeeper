"""Transcript distiller: parses Cursor agent transcripts into turn-level chunks with
heuristic extraction and optional Gemma summarization. Runs as a background watcher thread."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .tagger import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    _clean_description,
    _clean_tags,
    tags_from_path,
)
from . import db as db_mod

log = logging.getLogger(__name__)


@dataclass
class Turn:
    index: int
    user_query: str
    assistant_text: str
    files_read: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    byte_offset: int = 0


# ---------------------------------------------------------------------------
# Transcript parser
# ---------------------------------------------------------------------------

_TOOL_READ_RE = re.compile(r"^\[Tool call\]\s+(Read|Glob|Grep)", re.MULTILINE)
_TOOL_EDIT_RE = re.compile(r"^\[Tool call\]\s+(Write|StrReplace|Delete)", re.MULTILINE)
_TOOL_ANY_RE = re.compile(r"^\[Tool call\]\s+(\S+)", re.MULTILINE)
_PATH_RE = re.compile(r"^\s+path:\s+(.+)$", re.MULTILINE)


def parse_turns_jsonl(text: str, start_offset: int = 0) -> list[Turn]:
    """Parse a Cursor agent transcript JSONL file (or tail) into Turn objects.

    Each line is one event ``{"role": "user|assistant", "message": {"content": [...]}}``.
    A new ``Turn`` starts on every ``role:user`` event whose text contains a non-empty
    ``<user_query>...</user_query>`` block (system-injected user events without a real
    query are skipped). Tool attribution mirrors the .txt parser: ``Read|Glob|Grep``
    → ``files_read``, ``Write|StrReplace|Delete`` → ``files_edited``.
    """
    turns: list[Turn] = []
    cur: Turn | None = None
    cur_assistant_parts: list[str] = []
    running_offset = start_offset

    def _flush() -> None:
        if cur is not None:
            cur.assistant_text = "\n".join(cur_assistant_parts)[:2000]
            turns.append(cur)

    for raw_line in text.splitlines(keepends=True):
        line_len = len(raw_line)
        stripped = raw_line.strip()
        if not stripped:
            running_offset += line_len
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            running_offset += line_len
            continue

        role = event.get("role")
        content = (event.get("message") or {}).get("content") or []

        if role == "user":
            user_text = "\n".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
            q = re.search(r"<user_query>\s*(.*?)\s*</user_query>", user_text, re.DOTALL)
            user_query = q.group(1).strip() if q else ""
            if not user_query:
                running_offset += line_len
                continue
            _flush()
            cur = Turn(
                index=len(turns),
                user_query=user_query[:200],
                assistant_text="",
                byte_offset=running_offset,
            )
            cur_assistant_parts = []
        elif role == "assistant" and cur is not None:
            for b in content:
                t = b.get("type")
                if t == "text":
                    txt = (b.get("text") or "").strip()
                    if txt:
                        cur_assistant_parts.append(txt)
                elif t == "tool_use":
                    name = b.get("name") or ""
                    inp = b.get("input") or {}
                    if name and name not in cur.tools_called:
                        cur.tools_called.append(name)
                    p = inp.get("path")
                    if isinstance(p, str) and p:
                        if name in ("Read", "Glob", "Grep") and p not in cur.files_read:
                            cur.files_read.append(p)
                        elif name in ("Write", "StrReplace", "Delete") and p not in cur.files_edited:
                            cur.files_edited.append(p)

        running_offset += line_len

    _flush()
    return turns


def parse_turns(text: str, start_offset: int = 0) -> list[Turn]:
    """Parse a transcript .txt file (or a tail of one) into Turn objects.

    Legacy text-dump format (pre-JSONL Cursor transcripts). Kept as a fallback
    for old projects; current Cursor transcripts are JSONL — see ``parse_turns_jsonl``.
    """
    blocks = re.split(r"^user:\s*$", text, flags=re.MULTILINE)
    turns: list[Turn] = []
    running_offset = start_offset

    for i, block in enumerate(blocks):
        if i == 0 and not block.strip():
            running_offset += len(block) + len("user:\n")
            continue

        query_match = re.search(r"<user_query>\s*(.*?)\s*</user_query>", block, re.DOTALL)
        user_query = query_match.group(1).strip() if query_match else ""
        if not user_query:
            running_offset += len(block) + len("user:\n")
            continue

        assistant_parts = []
        for a_block in re.split(r"^assistant:\s*$", block, flags=re.MULTILINE)[1:]:
            lines = a_block.splitlines()
            for line in lines:
                stripped = line.strip()
                if (stripped and
                    not stripped.startswith("[Tool") and
                    not stripped.startswith("[Thinking]") and
                    not re.match(r"^\s+(path|pattern|command|glob|query|description):", stripped)):
                    assistant_parts.append(stripped)

        files_read = []
        files_edited = []
        tools_called = []

        for m in _TOOL_ANY_RE.finditer(block):
            tools_called.append(m.group(1))

        tool_sections = list(re.finditer(r"^\[Tool call\]\s+(\S+)", block, re.MULTILINE))
        for j, tm in enumerate(tool_sections):
            tool_name = tm.group(1)
            end_pos = tool_sections[j + 1].start() if j + 1 < len(tool_sections) else len(block)
            section = block[tm.end():end_pos]
            path_match = _PATH_RE.search(section)
            if path_match:
                fpath = path_match.group(1).strip()
                if tool_name in ("Read", "Glob", "Grep"):
                    files_read.append(fpath)
                elif tool_name in ("Write", "StrReplace", "Delete"):
                    files_edited.append(fpath)

        turns.append(Turn(
            index=len(turns),
            user_query=user_query[:200],
            assistant_text="\n".join(assistant_parts)[:2000],
            files_read=list(dict.fromkeys(files_read)),
            files_edited=list(dict.fromkeys(files_edited)),
            tools_called=list(dict.fromkeys(tools_called)),
            byte_offset=running_offset,
        ))
        running_offset += len(block) + len("user:\n")

    return turns


# ---------------------------------------------------------------------------
# Summary markdown generation
# ---------------------------------------------------------------------------

def _slug_from_query(query: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower().strip())[:40].strip("-")
    return slug or "untitled"


def _generate_summary_path(output_dir: str, transcript_id: str, first_query: str) -> str:
    slug = _slug_from_query(first_query)
    filename = f"{transcript_id[:8]}_{slug}.md"
    return str(Path(output_dir) / filename)


def _turn_to_markdown(turn: Turn) -> str:
    lines = [f"## Turn {turn.index + 1}: {turn.user_query}"]
    if turn.files_read:
        lines.append(f"- Files read: {', '.join(Path(f).name for f in turn.files_read[:10])}")
    if turn.files_edited:
        lines.append(f"- Files edited: {', '.join(Path(f).name for f in turn.files_edited[:10])}")
    if turn.assistant_text:
        preview = turn.assistant_text[:300]
        if len(turn.assistant_text) > 300:
            preview += "..."
        lines.append(f"- Conclusion: {preview}")
    lines.append("")
    return "\n".join(lines)


def write_summary_file(
    summary_path: str,
    transcript_id: str,
    title: str,
    turns: list[Turn],
    gemma_summary: dict | None = None,
    append: bool = False,
) -> None:
    p = Path(summary_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if append and p.exists():
        existing = p.read_text()
        new_turns = "\n".join(_turn_to_markdown(t) for t in turns)
        p.write_text(existing + "\n" + new_turns)
    else:
        parts = [
            f"# {title}",
            f"",
            f"**Transcript ID**: `{transcript_id}`  ",
            f"**Last updated**: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}  ",
            f"**Turns**: {len(turns)}",
            "",
        ]

        if gemma_summary:
            if gemma_summary.get("summary"):
                parts.extend(["## Summary", "", gemma_summary["summary"], ""])
            if gemma_summary.get("decisions"):
                parts.extend(["## Decisions", ""])
                for d in gemma_summary["decisions"]:
                    parts.append(f"- {d}")
                parts.append("")
            if gemma_summary.get("next_steps"):
                parts.extend(["## Next Steps", ""])
                for s in gemma_summary["next_steps"]:
                    parts.append(f"- {s}")
                parts.append("")

        parts.append("---\n")
        for turn in turns:
            parts.append(_turn_to_markdown(turn))

        p.write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# Gemma summarization (optional, batched)
# ---------------------------------------------------------------------------

_DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "decisions", "next_steps", "tags"],
}


def summarize_with_gemma(turns: list[Turn], raw_tail: str = "") -> dict | None:
    skeleton = "\n".join(
        f"Turn {t.index + 1}: {t.user_query}\n"
        f"  Files: {', '.join(t.files_read[:5] + t.files_edited[:5])}\n"
        f"  Output: {t.assistant_text[:500]}\n"
        for t in turns
    )

    content = skeleton[:50_000]
    if raw_tail:
        content += "\n\n--- Raw transcript tail ---\n" + raw_tail[-30_000:]

    messages = [
        {"role": "system", "content": (
            "You are summarizing an AI coding assistant's work session. "
            "Produce a JSON object with:\n"
            "- summary: 2-3 sentence description of what was accomplished\n"
            "- decisions: list of key technical decisions made (max 5)\n"
            "- next_steps: list of unfinished items or follow-ups (max 5)\n"
            "- tags: 5-10 topic tags for the whole conversation"
        )},
        {"role": "user", "content": content},
    ]

    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "format": _DISTILL_SCHEMA,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {"temperature": 0.1, "num_predict": 2048},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
        data = json.loads(raw)
        return {
            "summary": _clean_description(data.get("summary", "")) or data.get("summary", ""),
            "decisions": data.get("decisions", [])[:5],
            "next_steps": data.get("next_steps", [])[:5],
            "tags": _clean_tags(data.get("tags", [])),
        }
    except Exception as exc:
        log.warning("Gemma summarization failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Session-level chunk builder
# ---------------------------------------------------------------------------

_SUMMARY_EMBEDDING_CAP = 400
_TURN_HEADING_CAP = 120
_SESSION_TAG_CAP = 15


def _aggregate_tags(turns: list[Turn], transcript_tags: list[str]) -> list[str]:
    """Combine transcript-level gemma tags with file-path tags from every turn.

    Deliberately avoids pulling words out of user queries — those pollute BM25
    (tags get doubled in the corpus).
    """
    path_tags: list[str] = []
    for t in turns:
        for f in t.files_read + t.files_edited:
            path_tags.extend(tags_from_path(f))

    seen: set[str] = set()
    deduped: list[str] = []
    for tag in transcript_tags + path_tags:
        tag = tag.lower().strip()
        if tag and tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped[:_SESSION_TAG_CAP]


def _session_chunk_text(turns: list[Turn], session_summary: str) -> str:
    """Body of the single session chunk: summary (if any) plus the list of turn headings."""
    headings = [f"- Turn {t.index + 1}: {t.user_query[:_TURN_HEADING_CAP]}" for t in turns]
    parts: list[str] = []
    if session_summary:
        parts.extend(["## Session Summary", "", session_summary, ""])
    parts.extend(["## Turns", "", *headings])
    return "\n".join(parts)


def _session_embedding_text(title: str, session_summary: str, turns: list[Turn]) -> str:
    """Vector-input text: title + capped summary + turn headings only."""
    headings = "\n".join(
        f"Turn {t.index + 1}: {t.user_query[:_TURN_HEADING_CAP]}" for t in turns
    )
    parts: list[str] = []
    if title:
        parts.append(title)
    if session_summary:
        parts.append(session_summary[:_SUMMARY_EMBEDDING_CAP])
    if headings:
        parts.append(headings)
    return "\n".join(parts)


def _build_session_chunk(
    *,
    title: str,
    summary_path: str,
    output_dir: str,
    all_turns: list[Turn],
    transcript_tags: list[str],
    session_summary_text: str,
    total_turns: int,
) -> dict:
    """Produce the single session chunk (replaces any prior chunk for this path)."""
    text = _session_chunk_text(all_turns, session_summary_text)
    embedding = _session_embedding_text(title, session_summary_text, all_turns)
    tags = _aggregate_tags(all_turns, transcript_tags)
    description = (session_summary_text or title)[:200]

    chunk: dict = {
        "source": "sessions",
        "path": os.path.relpath(summary_path, os.path.expanduser(output_dir)),
        "doc_title": title,
        "section_heading": "",
        "start_line": 0,
        "end_line": 0,
        "tags": tags,
        "description": description,
        "text": text,
        "embedding_text": embedding,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "full_session_path": summary_path,
        "session_turn_count": total_turns,
        "replace_doc": True,
    }
    if session_summary_text:
        chunk["session_summary"] = session_summary_text
    return chunk


def _format_session_summary(gemma_summary: dict) -> str:
    parts: list[str] = []
    if gemma_summary.get("summary"):
        parts.append(gemma_summary["summary"])
    if gemma_summary.get("decisions"):
        parts.append("Decisions:")
        for d in gemma_summary["decisions"]:
            parts.append(f"- {d}")
    if gemma_summary.get("next_steps"):
        parts.append("Next steps:")
        for s in gemma_summary["next_steps"]:
            parts.append(f"- {s}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Incremental distillation orchestrator
# ---------------------------------------------------------------------------

def distill_transcript(
    transcript_path: str,
    conn: sqlite3.Connection,
    output_dir: str,
    gemma_every_n: int = 3,
    use_gemma: bool = True,
) -> list[dict]:
    """Incrementally distill a single transcript file. Returns new chunk metadata dicts."""
    p = Path(transcript_path)
    if not p.exists():
        return []

    state = db_mod.load_transcript_state(conn, transcript_path)
    turns_processed = state.get("turns_processed", 0)
    byte_offset = state.get("byte_offset", 0)
    summary_path = state.get("summary_path", "")
    gemma_summary_turn = state.get("gemma_summary_turn", 0)

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if byte_offset > 0 and byte_offset < len(content):
        new_content = content[byte_offset:]
    elif byte_offset == 0:
        new_content = content
    else:
        return []

    parser = parse_turns_jsonl if p.suffix == ".jsonl" else parse_turns

    new_turns = parser(new_content, byte_offset)
    if not new_turns:
        db_mod.save_transcript_state(conn, transcript_path, {
            **state,
            "byte_offset": len(content),
        })
        return []

    for i, t in enumerate(new_turns):
        t.index = turns_processed + i

    transcript_id = p.stem

    if turns_processed == 0 and new_turns:
        title = new_turns[0].user_query
        if not summary_path:
            summary_path = _generate_summary_path(output_dir, transcript_id, title)
    else:
        title = state.get("title", "Untitled Session")

    # Re-parse the full transcript so the session chunk always reflects every turn,
    # not just the new ones. The chunk is upserted per call.
    all_turns = parser(content)
    for i, t in enumerate(all_turns):
        t.index = i
    total_turns = len(all_turns)

    gemma_summary = None
    transcript_tags: list[str] = state.get("transcript_tags", [])
    session_summary_text: str = state.get("gemma_summary_text", "")

    if use_gemma and (total_turns - gemma_summary_turn) >= gemma_every_n:
        gemma_summary = summarize_with_gemma(all_turns, content[-50_000:])
        if gemma_summary:
            transcript_tags = gemma_summary.get("tags", [])
            gemma_summary_turn = total_turns
            session_summary_text = _format_session_summary(gemma_summary)

    # Markdown summary file (consumed by the personal source + get_full_document).
    # Full rewrite on first pass or whenever gemma refreshes the summary; otherwise
    # append just the new turns to avoid re-emitting the whole file.
    if turns_processed == 0 or gemma_summary:
        write_summary_file(summary_path, transcript_id, title, all_turns, gemma_summary)
    else:
        write_summary_file(summary_path, transcript_id, title, new_turns, append=True)

    session_chunk = _build_session_chunk(
        title=title,
        summary_path=summary_path,
        output_dir=output_dir,
        all_turns=all_turns,
        transcript_tags=transcript_tags,
        session_summary_text=session_summary_text,
        total_turns=total_turns,
    )

    new_byte_offset = byte_offset + len(new_content)
    db_mod.save_transcript_state(conn, transcript_path, {
        "turns_processed": total_turns,
        "byte_offset": new_byte_offset,
        "summary_path": summary_path,
        "gemma_summary_turn": gemma_summary_turn,
        "title": title,
        "transcript_tags": transcript_tags,
        "gemma_summary_text": session_summary_text,
    })

    return [session_chunk]


def distill_all_transcripts(
    transcript_dirs: list[str],
    conn: sqlite3.Connection,
    output_dir: str,
    gemma_every_n: int = 3,
    use_gemma: bool = True,
    flush_fn: Callable[[list[dict]], None] | None = None,
) -> int:
    """Scan all transcript dirs and distill incrementally.

    If flush_fn is provided, it is called with each transcript's new chunks
    immediately after processing (enabling per-file index updates).
    Returns total number of new chunks processed.
    """
    total_new = 0
    output = os.path.expanduser(output_dir)
    os.makedirs(output, exist_ok=True)

    for td in transcript_dirs:
        td_path = Path(os.path.expanduser(td))
        if not td_path.is_dir():
            continue

        # Cursor's current layout: <td>/<uuid>/<uuid>.jsonl plus optional
        # <td>/<uuid>/subagents/<uuid>.jsonl. Subagent transcripts are not
        # citeable, so we exclude them.
        jsonl_files = [f for f in td_path.rglob("*.jsonl") if "subagents" not in f.parts]
        # Legacy text-dump format from older Cursor versions.
        txt_files = list(td_path.glob("*.txt"))

        for transcript_file in sorted(jsonl_files + txt_files):
            state = db_mod.load_transcript_state(conn, str(transcript_file))
            try:
                stat = transcript_file.stat()
                mtime = stat.st_mtime
                fsize = stat.st_size
            except OSError:
                continue
            if state.get("last_mtime") == mtime and fsize <= state.get("byte_offset", 0):
                continue

            new_chunks = distill_transcript(
                str(transcript_file), conn, output,
                gemma_every_n=gemma_every_n, use_gemma=use_gemma,
            )

            if new_chunks and flush_fn:
                flush_fn(new_chunks)
            total_new += len(new_chunks)

            st = db_mod.load_transcript_state(conn, str(transcript_file))
            st["last_mtime"] = mtime
            db_mod.save_transcript_state(conn, str(transcript_file), st)

    return total_new


# ---------------------------------------------------------------------------
# Background watcher thread
# ---------------------------------------------------------------------------

class TranscriptWatcher:
    """Daemon thread that periodically distills transcripts and updates the index."""

    def __init__(
        self,
        transcript_dirs: list[str],
        conn: sqlite3.Connection,
        output_dir: str,
        interval_minutes: float,
        gemma_every_n: int,
        use_gemma: bool,
        flush_fn: Callable[[list[dict]], None],
    ):
        self.transcript_dirs = transcript_dirs
        self.conn = conn
        self.output_dir = output_dir
        self.interval = interval_minutes * 60
        self.gemma_every_n = gemma_every_n
        self.use_gemma = use_gemma
        self.flush_fn = flush_fn
        self._stop = threading.Event()

    def start(self) -> None:
        if self.interval <= 0:
            return
        t = threading.Thread(target=self._run, daemon=True, name="transcript-watcher")
        t.start()
        log.info("Transcript watcher started (interval=%.0fs)", self.interval)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._tick()
            except Exception:
                log.exception("Transcript watcher tick failed")

    def _tick(self) -> None:
        total = distill_all_transcripts(
            self.transcript_dirs, self.conn, self.output_dir,
            gemma_every_n=self.gemma_every_n, use_gemma=self.use_gemma,
            flush_fn=self.flush_fn,
        )
        if total:
            log.info("Transcript watcher: %d total new turn chunks this tick", total)
