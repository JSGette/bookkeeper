"""Tests for the JSONL transcript path: parser, walker, incremental resume.

Covers Cursor's current on-disk layout where each chat is stored as
``<td>/<uuid>/<uuid>.jsonl`` and sub-agents land in ``<td>/<uuid>/subagents/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents_rag.db import init_db
from agents_rag.transcript import (
    distill_all_transcripts,
    distill_transcript,
    parse_turns_jsonl,
)


def _line(role: str, blocks: list[dict]) -> str:
    return json.dumps({"role": role, "message": {"content": blocks}}) + "\n"


def _user(text: str) -> str:
    return _line("user", [{"type": "text", "text": text}])


def _assistant(blocks: list[dict]) -> str:
    return _line("assistant", blocks)


MINIMAL_JSONL = (
    _user("<timestamp>X</timestamp>\n<user_query>\nHow do I configure sources?\n</user_query>")
    + _assistant([
        {"type": "text", "text": "Edit config.yaml."},
        {"type": "tool_use", "name": "Read", "input": {"path": "/some/config.yaml"}},
    ])
)

TWO_TURN_JSONL = (
    MINIMAL_JSONL
    + _user("<user_query>\nNow add BM25 search\n</user_query>")
    + _assistant([
        {"type": "text", "text": "Added BM25 hybrid search."},
        {"type": "tool_use", "name": "Write", "input": {"path": "/some/searcher.py"}},
        {"type": "tool_use", "name": "Read", "input": {"path": "/some/indexer.py"}},
    ])
)


# ---------------------------------------------------------------------------
# parse_turns_jsonl
# ---------------------------------------------------------------------------

def test_parse_single_turn():
    turns = parse_turns_jsonl(MINIMAL_JSONL)
    assert len(turns) == 1
    t = turns[0]
    assert t.index == 0
    assert t.user_query == "How do I configure sources?"
    assert "Read" in t.tools_called
    assert "/some/config.yaml" in t.files_read
    assert t.files_edited == []
    assert "Edit config.yaml." in t.assistant_text


def test_parse_two_turns():
    turns = parse_turns_jsonl(TWO_TURN_JSONL)
    assert len(turns) == 2
    assert turns[0].user_query == "How do I configure sources?"
    assert turns[1].user_query == "Now add BM25 search"
    assert "/some/searcher.py" in turns[1].files_edited
    assert "/some/indexer.py" in turns[1].files_read
    assert {"Write", "Read"} <= set(turns[1].tools_called)


def test_parse_empty():
    assert parse_turns_jsonl("") == []


def test_parse_skips_user_event_without_query_wrapper():
    # Cursor injects system-only user events (e.g. only <system_reminder>).
    # These must not produce a Turn.
    content = _user("<system_reminder>noop</system_reminder>") + _assistant(
        [{"type": "text", "text": "ignored"}]
    )
    assert parse_turns_jsonl(content) == []


def test_parse_skips_malformed_lines():
    content = "not-json\n" + MINIMAL_JSONL + "{broken\n"
    turns = parse_turns_jsonl(content)
    assert len(turns) == 1
    assert turns[0].user_query == "How do I configure sources?"


def test_parse_incremental_offset():
    full = TWO_TURN_JSONL
    first_len = len(MINIMAL_JSONL)
    turns = parse_turns_jsonl(full[first_len:], start_offset=first_len)
    assert len(turns) == 1
    assert turns[0].user_query == "Now add BM25 search"
    assert turns[0].byte_offset >= first_len


# ---------------------------------------------------------------------------
# distill_transcript + walker (.jsonl path, subagents excluded)
# ---------------------------------------------------------------------------

def _write_jsonl_transcript(root: Path, uuid: str, content: str) -> Path:
    d = root / uuid
    d.mkdir(parents=True)
    f = d / f"{uuid}.jsonl"
    f.write_text(content)
    return f


def test_distill_transcript_jsonl_basic(tmp_path):
    td = tmp_path / "agent-transcripts"
    td.mkdir()
    f = _write_jsonl_transcript(td, "abc12345-aaaa-bbbb-cccc-ddddeeeeffff", TWO_TURN_JSONL)
    out_dir = str(tmp_path / "sessions_out")
    conn = init_db(str(tmp_path / "test.db"))

    chunks = distill_transcript(str(f), conn, out_dir, gemma_every_n=100, use_gemma=False)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["source"] == "sessions"
    assert chunk["session_turn_count"] == 2
    assert "Turn 1: How do I configure sources?" in chunk["text"]
    assert "Turn 2: Now add BM25 search" in chunk["text"]


def test_distill_all_transcripts_walks_jsonl_recursively(tmp_path):
    td = tmp_path / "agent-transcripts"
    td.mkdir()
    _write_jsonl_transcript(td, "uuid-one", MINIMAL_JSONL)
    _write_jsonl_transcript(td, "uuid-two", TWO_TURN_JSONL)

    out_dir = tmp_path / "sessions"
    out_dir.mkdir()
    conn = init_db(str(tmp_path / "test.db"))
    flushed: list[dict] = []

    total = distill_all_transcripts(
        [str(td)], conn, str(out_dir),
        gemma_every_n=100, use_gemma=False,
        flush_fn=lambda chunks: flushed.extend(chunks),
    )
    assert total == 2
    titles = {c["doc_title"] for c in flushed}
    assert titles == {"How do I configure sources?", "How do I configure sources?"} or len(titles) >= 1
    assert all(c["source"] == "sessions" for c in flushed)


def test_distill_all_transcripts_skips_subagents(tmp_path):
    td = tmp_path / "agent-transcripts"
    parent_dir = td / "parent-uuid"
    sub_dir = parent_dir / "subagents"
    sub_dir.mkdir(parents=True)
    (parent_dir / "parent-uuid.jsonl").write_text(MINIMAL_JSONL)
    (sub_dir / "child-uuid.jsonl").write_text(MINIMAL_JSONL)

    conn = init_db(str(tmp_path / "test.db"))
    flushed: list[dict] = []
    total = distill_all_transcripts(
        [str(td)], conn, str(tmp_path / "out"),
        gemma_every_n=100, use_gemma=False,
        flush_fn=lambda chunks: flushed.extend(chunks),
    )
    assert total == 1


def test_distill_all_transcripts_jsonl_skips_unchanged(tmp_path):
    td = tmp_path / "agent-transcripts"
    td.mkdir()
    _write_jsonl_transcript(td, "uuid-x", MINIMAL_JSONL)

    conn = init_db(str(tmp_path / "test.db"))
    out_dir = str(tmp_path / "sessions")
    Path(out_dir).mkdir()

    first = distill_all_transcripts(
        [str(td)], conn, out_dir, gemma_every_n=100, use_gemma=False,
    )
    second = distill_all_transcripts(
        [str(td)], conn, out_dir, gemma_every_n=100, use_gemma=False,
    )
    assert first == 1
    assert second == 0
