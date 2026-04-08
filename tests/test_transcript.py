"""Tests for transcript parsing, turn tagging, and incremental distillation."""

from __future__ import annotations

import textwrap

from agents_rag.transcript import Turn, parse_turns, tag_turn, distill_transcript


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
    """Parsing from a byte offset should only return turns after that offset."""
    full = TWO_TURN_TRANSCRIPT
    # Find where the second "user:" block starts
    second_user = full.index("user:", full.index("user:") + 1)
    tail = full[second_user:]
    turns = parse_turns(tail, start_offset=second_user)
    assert len(turns) == 1
    assert turns[0].user_query == "Now add BM25 search"
    assert turns[0].byte_offset >= second_user


# ---------------------------------------------------------------------------
# tag_turn
# ---------------------------------------------------------------------------

def test_tag_turn_from_files():
    turn = Turn(
        index=0,
        user_query="Fix the searcher module",
        assistant_text="Done.",
        files_read=["/repo/agents_rag/searcher.py"],
        files_edited=["/repo/agents_rag/indexer.py"],
    )
    tags = tag_turn(turn, transcript_tags=["rag"])
    assert "rag" in tags
    assert "searcher" in tags or "agents-rag" in tags
    # query keyword extraction
    assert "searcher" in tags
    assert "fix" not in tags  # 3 chars, excluded by len(word) > 3 filter


def test_tag_turn_dedup():
    turn = Turn(
        index=0,
        user_query="check the indexer",
        assistant_text="",
        files_read=["/a/indexer.py"],
    )
    tags = tag_turn(turn, transcript_tags=["indexer"])
    # "indexer" should appear only once despite being in both transcript_tags and path
    assert tags.count("indexer") == 1


def test_tag_turn_max_10():
    turn = Turn(
        index=0,
        user_query="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima",
        assistant_text="",
        files_read=[f"/a/{c}.py" for c in "abcdefghij"],
    )
    tags = tag_turn(turn, transcript_tags=["extra1", "extra2"])
    assert len(tags) <= 10


# ---------------------------------------------------------------------------
# distill_transcript (incremental)
# ---------------------------------------------------------------------------

def test_distill_transcript_basic(sample_transcript, tmp_cache, tmp_path):
    """Distill a transcript and verify chunks are produced."""
    txt_file = list(sample_transcript.glob("*.txt"))[0]
    output_dir = str(tmp_path / "sessions_out")

    chunks = distill_transcript(
        str(txt_file), tmp_cache, output_dir,
        gemma_every_n=100, use_gemma=False,
    )
    assert len(chunks) == 2
    assert all(c["source"] == "sessions" for c in chunks)
    assert chunks[0]["description"] == "How do I configure sources?"
    assert chunks[1]["description"] == "Now add BM25 search"


def test_distill_transcript_incremental(sample_transcript, tmp_cache, tmp_path):
    """Second distill of unchanged file produces no new chunks."""
    txt_file = list(sample_transcript.glob("*.txt"))[0]
    output_dir = str(tmp_path / "sessions_out")

    first = distill_transcript(
        str(txt_file), tmp_cache, output_dir,
        gemma_every_n=100, use_gemma=False,
    )
    assert len(first) == 2

    second = distill_transcript(
        str(txt_file), tmp_cache, output_dir,
        gemma_every_n=100, use_gemma=False,
    )
    assert len(second) == 0


def test_distill_transcript_append(sample_transcript, tmp_cache, tmp_path):
    """Appending a new turn to the file produces only the new chunk."""
    txt_file = list(sample_transcript.glob("*.txt"))[0]
    output_dir = str(tmp_path / "sessions_out")

    distill_transcript(
        str(txt_file), tmp_cache, output_dir,
        gemma_every_n=100, use_gemma=False,
    )

    # Append a third turn
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
        str(txt_file), tmp_cache, output_dir,
        gemma_every_n=100, use_gemma=False,
    )
    assert len(new_chunks) == 1
    assert new_chunks[0]["description"] == "Deploy to production"
