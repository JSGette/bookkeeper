"""Tests for transcript parsing, turn tagging, and incremental distillation."""

from __future__ import annotations

import textwrap

from agents_rag.transcript import Turn, parse_turns, distill_transcript


# ---------------------------------------------------------------------------
# parse_turns
# ---------------------------------------------------------------------------

MINIMAL_TRANSCRIPT = textwrap.dedent("""\
    user:

    <user_query>
    How do I configure sources?
    </user_query>

    assistant:

    Edit config.yaml and add a sources entry.

    [Tool call] Read
      path: /some/config.yaml
""")

TWO_TURN_TRANSCRIPT = textwrap.dedent("""\
    user:

    <user_query>
    How do I configure sources?
    </user_query>

    assistant:

    Edit config.yaml and add a sources entry.

    [Tool call] Read
      path: /some/config.yaml

    user:

    <user_query>
    Now add BM25 search
    </user_query>

    assistant:

    Added BM25 hybrid search using rank-bm25 library.

    [Tool call] Write
      path: /some/searcher.py
    [Tool call] Read
      path: /some/indexer.py
""")


def test_parse_single_turn():
    turns = parse_turns(MINIMAL_TRANSCRIPT)
    assert len(turns) == 1
    t = turns[0]
    assert t.index == 0
    assert t.user_query == "How do I configure sources?"
    assert "Read" in t.tools_called
    assert "/some/config.yaml" in t.files_read
    assert t.files_edited == []


def test_parse_two_turns():
    turns = parse_turns(TWO_TURN_TRANSCRIPT)
    assert len(turns) == 2

    assert turns[0].user_query == "How do I configure sources?"
    assert turns[1].user_query == "Now add BM25 search"
    assert "/some/searcher.py" in turns[1].files_edited
    assert "/some/indexer.py" in turns[1].files_read
    assert "Write" in turns[1].tools_called
    assert "Read" in turns[1].tools_called


def test_parse_empty():
    assert parse_turns("") == []


def test_parse_no_user_query_tags():
    content = textwrap.dedent("""\
        user:

        Some text without user_query tags.

        assistant:

        Response here.
    """)
    assert parse_turns(content) == []


def test_parse_incremental_offset():
    full = TWO_TURN_TRANSCRIPT
    second_user = full.index("user:", full.index("user:") + 1)
    tail = full[second_user:]
    turns = parse_turns(tail, start_offset=second_user)
    assert len(turns) == 1
    assert turns[0].user_query == "Now add BM25 search"
    assert turns[0].byte_offset >= second_user


# ---------------------------------------------------------------------------
# distill_transcript (incremental, using DB)
# ---------------------------------------------------------------------------

def test_distill_transcript_basic(sample_transcript, tmp_db, tmp_path):
    txt_file = list(sample_transcript.glob("*.txt"))[0]
    output_dir = str(tmp_path / "sessions_out")

    chunks = distill_transcript(
        str(txt_file), tmp_db, output_dir,
        gemma_every_n=100, use_gemma=False,
    )
    # One session chunk regardless of turn count.
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["source"] == "sessions"
    assert chunk["section_heading"] == ""
    assert chunk["session_turn_count"] == 2
    # Chunk body lists every turn heading.
    assert "Turn 1: How do I configure sources?" in chunk["text"]
    assert "Turn 2: Now add BM25 search" in chunk["text"]
    # Embedding text pulls title + headings in for semantic match.
    assert "How do I configure sources?" in chunk["embedding_text"]
    assert "Now add BM25 search" in chunk["embedding_text"]


def test_distill_transcript_incremental_returns_no_chunk_when_no_new_turns(
    sample_transcript, tmp_db, tmp_path,
):
    txt_file = list(sample_transcript.glob("*.txt"))[0]
    output_dir = str(tmp_path / "sessions_out")

    first = distill_transcript(
        str(txt_file), tmp_db, output_dir,
        gemma_every_n=100, use_gemma=False,
    )
    assert len(first) == 1

    second = distill_transcript(
        str(txt_file), tmp_db, output_dir,
        gemma_every_n=100, use_gemma=False,
    )
    assert len(second) == 0


def test_distill_transcript_append_refreshes_full_session_chunk(
    sample_transcript, tmp_db, tmp_path,
):
    txt_file = list(sample_transcript.glob("*.txt"))[0]
    output_dir = str(tmp_path / "sessions_out")

    distill_transcript(
        str(txt_file), tmp_db, output_dir,
        gemma_every_n=100, use_gemma=False,
    )

    with open(txt_file, "a") as f:
        f.write(textwrap.dedent("""
            user:

            <user_query>
            Deploy to production
            </user_query>

            assistant:

            Deployed successfully.
        """))

    new_chunks = distill_transcript(
        str(txt_file), tmp_db, output_dir,
        gemma_every_n=100, use_gemma=False,
    )
    # One chunk, but now spans all three turns.
    assert len(new_chunks) == 1
    chunk = new_chunks[0]
    assert chunk["session_turn_count"] == 3
    assert "Turn 1: How do I configure sources?" in chunk["text"]
    assert "Turn 2: Now add BM25 search" in chunk["text"]
    assert "Turn 3: Deploy to production" in chunk["text"]
