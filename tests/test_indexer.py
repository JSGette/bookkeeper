"""Tests for indexer: mtime delta detection, DB roundtrip, reindex."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from agents_rag.db import load_all_chunks, upsert_chunks, upsert_document
from agents_rag.indexer import (
    _mtime_changed_files,
    reindex_source,
)
from tests.conftest import make_source_cfg


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic fake embedder: hash-based so identical texts get identical vectors."""
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % (2**31))
        vecs.append(rng.randn(384).astype(np.float32))
    return np.array(vecs)


# ---------------------------------------------------------------------------
# _mtime_changed_files
# ---------------------------------------------------------------------------

def test_mtime_no_changes():
    mtimes = {"a.md": 1000.0, "b.md": 2000.0}
    assert _mtime_changed_files(mtimes, mtimes) == set()


def test_mtime_modified():
    old = {"a.md": 1000.0, "b.md": 2000.0}
    new = {"a.md": 1000.0, "b.md": 3000.0}
    assert _mtime_changed_files(old, new) == {"b.md"}


def test_mtime_added():
    old = {"a.md": 1000.0}
    new = {"a.md": 1000.0, "c.md": 4000.0}
    assert _mtime_changed_files(old, new) == {"c.md"}


def test_mtime_deleted():
    old = {"a.md": 1000.0, "b.md": 2000.0}
    new = {"a.md": 1000.0}
    assert _mtime_changed_files(old, new) == {"b.md"}


def test_mtime_mixed():
    old = {"a.md": 1.0, "b.md": 2.0, "c.md": 3.0}
    new = {"a.md": 1.0, "b.md": 9.0, "d.md": 4.0}
    changed = _mtime_changed_files(old, new)
    assert changed == {"b.md", "c.md", "d.md"}


# ---------------------------------------------------------------------------
# DB save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_db):
    vecs = np.random.randn(3, 384).astype(np.float32)
    meta = [
        {"source": "test", "path": "a.md", "text": "hello", "section_heading": ""},
        {"source": "test", "path": "a.md", "text": "world", "section_heading": "sec2"},
        {"source": "test", "path": "b.md", "text": "foo", "section_heading": ""},
    ]

    doc_id_a = upsert_document(tmp_db, "test", "a.md", "", "", "hello\nworld")
    upsert_chunks(tmp_db, doc_id_a, meta[:2], vecs[:2])

    doc_id_b = upsert_document(tmp_db, "test", "b.md", "", "", "foo")
    upsert_chunks(tmp_db, doc_id_b, meta[2:], vecs[2:])

    loaded_meta, loaded_vecs = load_all_chunks(tmp_db)
    assert loaded_vecs.shape == vecs.shape
    assert np.allclose(loaded_vecs, vecs, atol=1e-5)
    assert len(loaded_meta) == 3
    assert loaded_meta[0]["path"] == "a.md"
    assert loaded_meta[0]["text"] == "hello"


def test_load_empty_db(tmp_db):
    meta, vecs = load_all_chunks(tmp_db)
    assert len(meta) == 0
    assert vecs.shape == (0, 384)


# ---------------------------------------------------------------------------
# reindex_source (full, no LLM tagging)
# ---------------------------------------------------------------------------

@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_source_full(mock_embed, doc_tree: Path, tmp_db):
    cfg = make_source_cfg(doc_tree, file_types=[".md"])
    meta, vecs, changed = reindex_source(cfg, tmp_db, use_gemma=False)

    assert changed is True
    assert len(meta) > 0
    assert vecs.shape[0] == len(meta)
    assert vecs.shape[1] == 384
    assert all(m["source"] == "test" for m in meta)
    guide_chunks = [m for m in meta if m["path"] == "guide.md"]
    assert len(guide_chunks) >= 2


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_source_no_changes(mock_embed, doc_tree: Path, tmp_db):
    cfg = make_source_cfg(doc_tree, file_types=[".md"])

    meta1, vecs1, changed1 = reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed1 is True

    meta2, vecs2, changed2 = reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed2 is False
    assert len(meta2) == len(meta1)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_source_delta(mock_embed, doc_tree: Path, tmp_db):
    cfg = make_source_cfg(doc_tree, file_types=[".md"])

    meta1, vecs1, _ = reindex_source(cfg, tmp_db, use_gemma=False)
    original_count = len(meta1)

    (doc_tree / "faq.md").write_text("# FAQ\n\n## New Question\n\nNew answer here.\n")

    meta2, vecs2, changed2 = reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed2 is True
    assert vecs2.shape[0] == len(meta2)
    assert abs(len(meta2) - original_count) <= 2
