"""Tests for indexer: mtime delta detection, load/save roundtrip, integrity check."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agents_rag.indexer import (
    _mtime_changed_files,
    embed_texts,
    load_index,
    reindex_source,
    save_index,
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
# save_index / load_index roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_cache: str):
    vecs = np.random.randn(3, 384).astype(np.float32)
    meta = [
        {"source": "test", "path": "a.md", "text": "hello"},
        {"source": "test", "path": "b.md", "text": "world"},
        {"source": "test", "path": "c.md", "text": "foo"},
    ]
    save_index(tmp_cache, vecs, meta)

    # texts.json should exist separately from chunks.json
    idx_dir = Path(tmp_cache)
    assert (idx_dir / "texts.json").exists()
    stored_meta = json.loads((idx_dir / "chunks.json").read_text())
    assert "text" not in stored_meta[0], "chunks.json should not contain text"

    loaded_vecs, loaded_meta = load_index(tmp_cache)
    assert loaded_vecs is not None
    assert loaded_meta is not None
    assert loaded_vecs.shape == vecs.shape
    assert np.allclose(loaded_vecs, vecs, atol=1e-5)
    assert len(loaded_meta) == 3
    assert loaded_meta[0]["path"] == "a.md"
    assert loaded_meta[0]["text"] == "hello"


def test_load_index_without_texts(tmp_cache: str):
    """include_texts=False returns metadata without text field."""
    vecs = np.random.randn(2, 384).astype(np.float32)
    meta = [
        {"source": "test", "path": "a.md", "text": "hello"},
        {"source": "test", "path": "b.md", "text": "world"},
    ]
    save_index(tmp_cache, vecs, meta)
    _, loaded_meta = load_index(tmp_cache, include_texts=False)
    assert loaded_meta is not None
    assert "text" not in loaded_meta[0]


def test_load_missing_index(tmp_cache: str):
    vecs, meta = load_index(tmp_cache)
    assert vecs is None
    assert meta is None


# ---------------------------------------------------------------------------
# Integrity check: mismatched vector/metadata lengths
# ---------------------------------------------------------------------------

def test_load_index_corrupted(tmp_cache: str):
    """Mismatched vector and metadata counts should return (None, None)."""
    vecs = np.random.randn(3, 384).astype(np.float32)
    meta = [
        {"source": "test", "path": "a.md"},
        {"source": "test", "path": "b.md"},
    ]
    texts = ["hello", "world"]
    idx_dir = Path(tmp_cache)
    np.savez_compressed(idx_dir / "index.npz", vectors=vecs)
    (idx_dir / "chunks.json").write_text(json.dumps(meta))
    (idx_dir / "texts.json").write_text(json.dumps(texts))

    loaded_vecs, loaded_meta = load_index(tmp_cache)
    assert loaded_vecs is None
    assert loaded_meta is None


# ---------------------------------------------------------------------------
# reindex_source (full, no LLM tagging)
# ---------------------------------------------------------------------------

@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_source_full(mock_embed, doc_tree: Path, tmp_cache: str):
    """Full reindex of a non-git directory produces metadata and vectors."""
    cfg = make_source_cfg(doc_tree, file_types=[".md"])
    meta, vecs, changed = reindex_source(cfg, tmp_cache, use_gemma=False)

    assert changed is True
    assert len(meta) > 0
    assert vecs.shape[0] == len(meta)
    assert vecs.shape[1] == 384
    assert all(m["source"] == "test" for m in meta)
    guide_chunks = [m for m in meta if m["path"] == "guide.md"]
    assert len(guide_chunks) >= 2


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_source_no_changes(mock_embed, doc_tree: Path, tmp_cache: str):
    """Second reindex with no file changes returns index_changed=False."""
    cfg = make_source_cfg(doc_tree, file_types=[".md"])

    meta1, vecs1, changed1 = reindex_source(cfg, tmp_cache, use_gemma=False)
    assert changed1 is True
    save_index(tmp_cache, vecs1, meta1)

    meta2, vecs2, changed2 = reindex_source(cfg, tmp_cache, use_gemma=False)
    assert changed2 is False
    assert len(meta2) == len(meta1)


@patch("agents_rag.indexer.embed_texts", side_effect=_fake_embed)
def test_reindex_source_delta(mock_embed, doc_tree: Path, tmp_cache: str):
    """Modifying a file triggers delta reindex with embedding reuse."""
    cfg = make_source_cfg(doc_tree, file_types=[".md"])

    meta1, vecs1, _ = reindex_source(cfg, tmp_cache, use_gemma=False)
    save_index(tmp_cache, vecs1, meta1)
    original_count = len(meta1)

    # Modify one file
    (doc_tree / "faq.md").write_text("# FAQ\n\n## New Question\n\nNew answer here.\n")

    meta2, vecs2, changed2 = reindex_source(cfg, tmp_cache, use_gemma=False)
    assert changed2 is True
    assert vecs2.shape[0] == len(meta2)
    assert abs(len(meta2) - original_count) <= 2
