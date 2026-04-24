"""Extended tests for transcript.py: write_summary_file, distill_all_transcripts, _format_session_summary."""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_rag.db import init_db, save_transcript_state
from agents_rag.transcript import (
    Turn,
    _build_turn_embedding_text,
    _format_session_summary,
    _turn_to_markdown,
    distill_all_transcripts,
    distill_transcript,
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
# Turn embedding text composition (Proposal 2)
# ---------------------------------------------------------------------------


def test_build_turn_embedding_text_with_summary():
    turn = Turn(index=0, user_query="How do I cache?", assistant_text="Use Redis.")
    text = _build_turn_embedding_text(
        title="Cache design",
        session_summary="We picked Redis over Memcached for pub/sub support.",
        turn=turn,
        turn_text="## Turn 1: How do I cache?\n- Conclusion: Redis.",
    )
    assert "Cache design" in text
    assert "Redis over Memcached" in text
    assert "How do I cache?" in text
    assert "Turn 1" in text


def test_build_turn_embedding_text_without_summary():
    turn = Turn(index=0, user_query="Q", assistant_text="A")
    text = _build_turn_embedding_text("Title", "", turn, "body")
    assert "Title" in text
    assert "Q" in text
    assert "body" in text


def test_build_turn_embedding_text_caps_summary_length():
    turn = Turn(index=0, user_query="Q", assistant_text="A")
    huge = "x" * 5000
    text = _build_turn_embedding_text("T", huge, turn, "body")
    # Summary portion alone must not exceed the cap
    assert huge not in text
    assert "x" * 400 in text


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
