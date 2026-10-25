"""Extended tests for transcript.py: write_summary_file, distill_all_transcripts, _format_session_summary."""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_rag.db import init_db, load_transcript_state, save_transcript_state
from agents_rag.transcript import (
    Turn,
    _aggregate_tags,
    _format_session_summary,
    _session_embedding_text,
    _turn_to_markdown,
    distill_all_transcripts,
    distill_transcript,
    parse_turns,
    parse_turns_jsonl,
    write_summary_file,
)


# ---------------------------------------------------------------------------
# _turn_to_markdown
# ---------------------------------------------------------------------------


def test_turn_to_markdown_basic():
    turn = Turn(index=0, user_query="How to install?", assistant_text="Run pip install.")
    md = _turn_to_markdown(turn)
    assert "Turn 1" in md
    assert "How to install?" in md
    assert "Run pip install." in md


def test_turn_to_markdown_with_files():
    turn = Turn(
        index=1, user_query="Fix the bug",
        assistant_text="Fixed it.",
        files_read=["/a/b/reader.py"],
        files_edited=["/a/b/writer.py"],
    )
    md = _turn_to_markdown(turn)
    assert "reader.py" in md
    assert "writer.py" in md


def test_turn_to_markdown_truncates_long_text():
    turn = Turn(index=0, user_query="Q", assistant_text="x" * 500)
    md = _turn_to_markdown(turn)
    assert "..." in md


# ---------------------------------------------------------------------------
# _format_session_summary
# ---------------------------------------------------------------------------


def test_format_session_summary_full():
    summary = {
        "summary": "Migrated 5 modules.",
        "decisions": ["Use Bazel", "Keep Make for legacy"],
        "next_steps": ["Test CI", "Update docs"],
    }
    text = _format_session_summary(summary)
    assert "Migrated 5 modules." in text
    assert "Use Bazel" in text
    assert "Test CI" in text


def test_format_session_summary_empty():
    assert _format_session_summary({}) == ""


def test_format_session_summary_partial():
    text = _format_session_summary({"summary": "Did work."})
    assert "Did work." in text


# ---------------------------------------------------------------------------
# write_summary_file
# ---------------------------------------------------------------------------


def test_write_summary_creates_file(tmp_path):
    out = str(tmp_path / "summary.md")
    turns = [Turn(index=0, user_query="Install", assistant_text="Use pip.")]
    write_summary_file(out, "abc123", "Installation Help", turns)
    content = Path(out).read_text()
    assert "# Installation Help" in content
    assert "abc123" in content
    assert "Turn 1" in content


def test_write_summary_with_gemma(tmp_path):
    out = str(tmp_path / "summary.md")
    turns = [Turn(index=0, user_query="Q", assistant_text="A")]
    gemma = {"summary": "Session summary.", "decisions": ["D1"], "next_steps": ["N1"]}
    write_summary_file(out, "id", "Title", turns, gemma_summary=gemma)
    content = Path(out).read_text()
    assert "Session summary." in content
    assert "D1" in content
    assert "N1" in content


def test_write_summary_append_mode(tmp_path):
    out = str(tmp_path / "summary.md")
    Path(out).write_text("# Existing\n\nOld content.\n")
    new_turns = [Turn(index=2, user_query="New Q", assistant_text="New A")]
    write_summary_file(out, "id", "Title", new_turns, append=True)
    content = Path(out).read_text()
    assert "Old content." in content
    assert "New Q" in content


# ---------------------------------------------------------------------------
# distill_all_transcripts
# ---------------------------------------------------------------------------


def _make_transcript(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


def test_distill_all_transcripts_basic(tmp_path):
    t_dir = tmp_path / "transcripts"
    t_dir.mkdir()
    out_dir = tmp_path / "sessions"
    out_dir.mkdir()

    _make_transcript(t_dir / "test1.txt", """\
        user:

        <user_query>
        How to configure?
        </user_query>

        assistant:

        Edit config.yaml.
    """)

    conn = init_db(str(tmp_path / "test.db"))
    flushed = []

    total = distill_all_transcripts(
        [str(t_dir)], conn, str(out_dir),
        gemma_every_n=100, use_gemma=False,
        flush_fn=lambda chunks: flushed.extend(chunks),
    )

    assert total >= 1
    assert len(flushed) >= 1
    assert flushed[0]["source"] == "sessions"


def test_distill_all_transcripts_skips_unchanged(tmp_path):
    t_dir = tmp_path / "transcripts"
    t_dir.mkdir()
    out_dir = tmp_path / "sessions"
    out_dir.mkdir()

    _make_transcript(t_dir / "test1.txt", """\
        user:

        <user_query>
        First question
        </user_query>

        assistant:

        First answer.
    """)

    conn = init_db(str(tmp_path / "test.db"))

    total1 = distill_all_transcripts(
        [str(t_dir)], conn, str(out_dir),
        gemma_every_n=100, use_gemma=False,
    )
    total2 = distill_all_transcripts(
        [str(t_dir)], conn, str(out_dir),
        gemma_every_n=100, use_gemma=False,
    )
    assert total1 >= 1
    assert total2 == 0  # no new content


def test_distill_all_transcripts_nonexistent_dir(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    total = distill_all_transcripts(
        [str(tmp_path / "nope")], conn, str(tmp_path / "out"),
        use_gemma=False,
    )
    assert total == 0


# ---------------------------------------------------------------------------
# Session chunk composition (Proposals 2 + 5)
# ---------------------------------------------------------------------------


def test_session_embedding_text_includes_title_summary_and_headings():
    turns = [
        Turn(index=0, user_query="How do I cache?", assistant_text="A"),
        Turn(index=1, user_query="What eviction policy?", assistant_text="B"),
    ]
    text = _session_embedding_text(
        "Cache design",
        "We picked Redis over Memcached for pub/sub support.",
        turns,
    )
    assert "Cache design" in text
    assert "Redis over Memcached" in text
    assert "Turn 1: How do I cache?" in text
    assert "Turn 2: What eviction policy?" in text


def test_session_embedding_text_without_summary():
    turns = [Turn(index=0, user_query="Q", assistant_text="A")]
    text = _session_embedding_text("Title", "", turns)
    assert "Title" in text
    assert "Turn 1: Q" in text


def test_session_embedding_text_caps_summary_length():
    turns = [Turn(index=0, user_query="Q", assistant_text="A")]
    huge = "x" * 5000
    text = _session_embedding_text("T", huge, turns)
    # Summary portion alone must not exceed the cap.
    assert huge not in text
    assert "x" * 400 in text


def test_aggregate_tags_combines_transcript_and_path_tags():
    turns = [
        Turn(
            index=0, user_query="Q", assistant_text="",
            files_read=["/repo/agents_rag/searcher.py"],
            files_edited=["/repo/agents_rag/indexer.py"],
        ),
    ]
    tags = _aggregate_tags(turns, transcript_tags=["rag", "retrieval"])
    assert "rag" in tags
    assert "retrieval" in tags
    assert "searcher" in tags
    # Must not extract query words.
    assert "q" not in tags


def test_aggregate_tags_dedups_and_caps():
    turns = [
        Turn(
            index=i, user_query="Q", assistant_text="",
            files_read=[f"/a/mod_{i}.py" for _ in range(3)],
        )
        for i in range(20)
    ]
    tags = _aggregate_tags(turns, transcript_tags=["rag"])
    assert len(tags) <= 15
    assert len(set(tags)) == len(tags)


def test_distill_produces_embedding_text_field(tmp_path):
    t_dir = tmp_path / "transcripts"
    t_dir.mkdir()
    out_dir = tmp_path / "sessions"
    out_dir.mkdir()

    _make_transcript(t_dir / "t.txt", """\
        user:

        <user_query>
        What is the cache strategy?
        </user_query>

        assistant:

        We use Redis.
    """)

    conn = init_db(str(tmp_path / "test.db"))
    flushed: list[dict] = []
    distill_all_transcripts(
        [str(t_dir)], conn, str(out_dir),
        gemma_every_n=100, use_gemma=False,
        flush_fn=lambda chunks: flushed.extend(chunks),
    )

    assert flushed
    chunk = flushed[0]
    assert "embedding_text" in chunk
    assert "What is the cache strategy?" in chunk["embedding_text"]


def test_distill_all_transcripts_flush_per_file(tmp_path):
    t_dir = tmp_path / "transcripts"
    t_dir.mkdir()
    out_dir = tmp_path / "sessions"
    out_dir.mkdir()

    for name in ["a.txt", "b.txt"]:
        _make_transcript(t_dir / name, f"""\
            user:

            <user_query>
            Question in {name}
            </user_query>

            assistant:

            Answer for {name}.
        """)

    conn = init_db(str(tmp_path / "test.db"))
    flush_calls = []

    distill_all_transcripts(
        [str(t_dir)], conn, str(out_dir),
        gemma_every_n=100, use_gemma=False,
        flush_fn=lambda chunks: flush_calls.append(len(chunks)),
    )
    # flush_fn should be called once per transcript file
    assert len(flush_calls) == 2


def _event(role: str, text: str) -> bytes:
    return (json.dumps({"role": role, "message": {"content": text}}, ensure_ascii=False) + "\n").encode("utf-8")


@pytest.mark.parametrize("suffix", [".txt", ".jsonl"])
def test_assistant_only_append_refreshes_existing_turn(tmp_path, suffix):
    transcript = tmp_path / ("session" + suffix)
    question = "Which dependency version?"
    answer = "Use z version 3."
    user = (_event("user", f"<user_query>{question}</user_query>") if suffix == ".jsonl"
            else f"user:\n<user_query>{question}</user_query>\n".encode())
    assistant = (_event("assistant", answer) if suffix == ".jsonl"
                 else f"assistant:\n{answer}\n".encode())
    transcript.write_bytes(user)
    conn = init_db(str(tmp_path / "test.db"))
    output = str(tmp_path / "sessions")
    first = distill_transcript(str(transcript), conn, output, use_gemma=False)[0]
    transcript.write_bytes(user + assistant)
    second = distill_transcript(str(transcript), conn, output, use_gemma=False)[0]

    assert second["session_turn_count"] == 1
    assert second["full_content"].encode("utf-8") == user + assistant
    assert second["source_hash"] == hashlib.sha256(user + assistant).hexdigest()
    assert second["source_hash"] != first["source_hash"]
    summary = Path(second["full_session_path"]).read_text(encoding="utf-8")
    assert summary.count("## Turn 1:") == 1
    assert answer in summary
    assert distill_transcript(str(transcript), conn, output, use_gemma=False) == []


def test_full_content_retains_evidence_beyond_summary_caps(tmp_path):
    transcript = tmp_path / "session.jsonl"
    raw = (_event("user", f"<user_query>{'q' * 250}</user_query>")
           + _event("assistant", "answer " * 1000 + "EXACT_LATE_EVIDENCE"))
    transcript.write_bytes(raw)
    conn = init_db(str(tmp_path / "test.db"))
    chunk = distill_transcript(str(transcript), conn, str(tmp_path / "sessions"), use_gemma=False)[0]
    assert chunk["full_content"].encode("utf-8") == raw
    assert "EXACT_LATE_EVIDENCE" in chunk["full_content"]
    assert "EXACT_LATE_EVIDENCE" not in chunk["text"]
    assert chunk["source_path"] == str(transcript.resolve())


def test_partial_json_and_multibyte_tails_do_not_advance_checkpoint(tmp_path):
    transcript = tmp_path / "session.jsonl"
    user = _event("user", "<user_query>配置?</user_query>")
    assistant = _event("assistant", "Réponse: 完了 ✅")
    transcript.write_bytes(user)
    conn = init_db(str(tmp_path / "test.db"))
    output = str(tmp_path / "sessions")
    distill_transcript(str(transcript), conn, output, use_gemma=False)
    state = load_transcript_state(conn, str(transcript))
    assert state["byte_offset"] == len(user)

    # First an incomplete JSON object, then half of a multibyte character.
    split_at = assistant.index("✅".encode("utf-8")) + 1
    for tail in (assistant[:20], assistant[:split_at]):
        transcript.write_bytes(user + tail)
        assert distill_transcript(str(transcript), conn, output, use_gemma=False) == []
        assert load_transcript_state(conn, str(transcript)) == state

    transcript.write_bytes(user + assistant)
    chunk = distill_transcript(str(transcript), conn, output, use_gemma=False)[0]
    assert chunk["full_content"].encode("utf-8") == user + assistant
    assert chunk["transcript_state"]["byte_offset"] == len(user + assistant)
    assert "�" not in chunk["full_content"]


def test_invalid_utf8_does_not_replace_evidence_or_advance_state(tmp_path):
    transcript = tmp_path / "session.jsonl"
    user = _event("user", "<user_query>Question</user_query>")
    transcript.write_bytes(user)
    conn = init_db(str(tmp_path / "test.db"))
    output = str(tmp_path / "sessions")
    distill_transcript(str(transcript), conn, output, use_gemma=False)
    state = load_transcript_state(conn, str(transcript))
    transcript.write_bytes(user + b'\xff\n')
    assert distill_transcript(str(transcript), conn, output, use_gemma=False) == []
    assert load_transcript_state(conn, str(transcript)) == state


@pytest.mark.parametrize("replacement", ["New", "Replaced question"])
def test_replaced_transcript_rebuilds_at_stable_path(tmp_path, replacement):
    transcript = tmp_path / "session.jsonl"
    original = _event("user", "<user_query>Original question</user_query>")
    transcript.write_bytes(original)
    conn = init_db(str(tmp_path / "test.db"))
    output = str(tmp_path / "sessions")
    first = distill_transcript(str(transcript), conn, output, use_gemma=False)[0]
    state = load_transcript_state(conn, str(transcript))
    state.update(gemma_summary_text="Obsolete conclusion", gemma_summary_turn=1)
    save_transcript_state(conn, str(transcript), state)
    raw = _event("user", f"<user_query>{replacement}</user_query>")
    transcript.write_bytes(raw)
    second = distill_transcript(str(transcript), conn, output, use_gemma=False)[0]
    assert second["path"] == first["path"]
    assert second["doc_title"] == replacement
    assert second["full_content"].encode("utf-8") == raw
    assert "Obsolete conclusion" not in second["text"]
    assert second["transcript_state"]["byte_offset"] == len(raw)


def test_scan_detects_rewrite_with_unchanged_size_and_mtime(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_event("user", "<user_query>Old</user_query>"))
    conn = init_db(str(tmp_path / "test.db"))
    output = str(tmp_path / "sessions")
    assert distill_all_transcripts([str(tmp_path)], conn, output, use_gemma=False) == 1
    stat = transcript.stat()
    transcript.write_bytes(_event("user", "<user_query>New</user_query>"))
    os.utime(transcript, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert distill_all_transcripts([str(tmp_path)], conn, output, use_gemma=False) == 1
    assert load_transcript_state(conn, str(transcript))["title"] == "New"


def test_legacy_character_cursor_reprocesses_existing_transcript(tmp_path):
    transcript = tmp_path / "session.jsonl"
    raw = _event("user", "<user_query>配置?</user_query>") + _event("assistant", "Done")
    transcript.write_bytes(raw)
    conn = init_db(str(tmp_path / "test.db"))
    save_transcript_state(conn, str(transcript), {
        "byte_offset": len(raw.decode("utf-8")), "turns_processed": 1,
    })
    chunk = distill_transcript(str(transcript), conn, str(tmp_path / "sessions"), use_gemma=False)[0]
    assert chunk["full_content"].encode("utf-8") == raw
    assert load_transcript_state(conn, str(transcript))["byte_offset"] == len(raw)


def test_flush_failure_leaves_checkpoint_retryable(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_event("user", "<user_query>Question</user_query>"))
    conn = init_db(str(tmp_path / "test.db"))
    output = str(tmp_path / "sessions")

    def failing_flush(chunks):
        assert chunks[0]["transcript_state"]["byte_offset"] == transcript.stat().st_size
        raise RuntimeError("queue full")

    with pytest.raises(RuntimeError, match="queue full"):
        distill_all_transcripts([str(tmp_path)], conn, output, use_gemma=False, flush_fn=failing_flush)
    assert load_transcript_state(conn, str(transcript)) == {}

    # Production callback commits the document and this checkpoint together.
    def durable_flush(chunks):
        chunk = chunks[0]
        save_transcript_state(conn, chunk["transcript_path"], chunk["transcript_state"])

    assert distill_all_transcripts([str(tmp_path)], conn, output, use_gemma=False, flush_fn=durable_flush) == 1
    assert distill_all_transcripts([str(tmp_path)], conn, output, use_gemma=False, flush_fn=durable_flush) == 0


@pytest.mark.parametrize("parser, first, second", [
    (parse_turns_jsonl, _event("user", "<user_query>配置?</user_query>").decode(),
     _event("user", "<user_query>Next</user_query>").decode()),
    (parse_turns, "user:\r\n<user_query>配置?</user_query>\r\n",
     "user:\r\n<user_query>Next</user_query>\r\n"),
])
def test_turn_offsets_are_utf8_bytes(parser, first, second):
    turns = parser(first + second, start_offset=7)
    assert turns[0].byte_offset == 7
    assert turns[1].byte_offset == 7 + len(first.encode("utf-8"))


def test_jsonl_unicode_line_separator_is_not_record_boundary():
    content = (_event("user", "<user_query>Question</user_query>")
               + _event("assistant", "First\u2028Second")).decode("utf-8")
    assert parse_turns_jsonl(content)[0].assistant_text == "First\u2028Second"
