"""Failure and restart regressions for evidence, index state and legacy identities."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agents_rag import db, indexer, tagger
from agents_rag.chunker import SourceConfig
from agents_rag.migration import inventory


@pytest.fixture
def offline_indexer(monkeypatch):
    """Exercise real scan/chunk/persistence paths with only inference stubbed."""
    monkeypatch.setenv("EMBEDDING_MODEL", "offline/test-model")
    monkeypatch.delenv("EMBEDDING_REVISION", raising=False)
    monkeypatch.setattr(indexer, "_model", None)
    calls = SimpleNamespace(
        budget=Mock(return_value=(len, 2048)),
        embed=Mock(side_effect=lambda texts: np.ones((len(texts), 3), dtype=np.float32)),
        tag=Mock(return_value=("offline description", ["offline"])),
        summary=Mock(return_value="offline summary"),
    )
    monkeypatch.setattr(indexer, "embedding_token_budget", calls.budget)
    monkeypatch.setattr(indexer, "embed_texts", calls.embed)
    monkeypatch.setattr(indexer, "tag_file", calls.tag)
    monkeypatch.setattr(indexer, "summarize_document", calls.summary)
    return calls


def _source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    return root, SourceConfig("fixture", str(root), max_whole_doc_tokens=0)


@pytest.mark.parametrize("failure_location", ["root", "subdirectory"])
def test_failed_directory_enumeration_retains_live_evidence_and_checkpoint(
    tmp_path, tmp_db, offline_indexer, monkeypatch, failure_location,
):
    root, cfg = _source(tmp_path)
    (root / "root.md").write_text("root evidence", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "nested.md").write_text("nested evidence", encoding="utf-8")
    indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    before = inventory(tmp_db)
    inaccessible = root if failure_location == "root" else root / "sub"
    real_scandir = indexer.os.scandir

    def denied_scandir(path):
        if Path(path) == inaccessible:
            raise PermissionError("temporary directory access failure")
        return real_scandir(path)

    with monkeypatch.context() as patcher:
        patcher.setattr(indexer.os, "scandir", denied_scandir)
        with pytest.raises(PermissionError, match="temporary directory"):
            indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert inventory(tmp_db) == before
    assert (root / "root.md").exists() and (root / "sub" / "nested.md").exists()
    assert not tmp_db.in_transaction


def test_registered_dimension_cannot_change_when_replacing_its_only_document(tmp_db):
    profile = {"model": "offline/test-model", "revision": "fixed"}
    space = db.register_embedding_space(tmp_db, profile, 3)
    doc_id = db.upsert_document(tmp_db, "fixture", "a.md", "", "", "original evidence")
    original = np.array([[0.25, -0.75, 1.5]], dtype=np.float32)
    db.upsert_chunks(tmp_db, doc_id, [{"text": "original evidence"}], original, embedding_space=space)
    before = inventory(tmp_db)

    with pytest.raises(ValueError, match="registered embedding space"):
        db.upsert_chunks(tmp_db, doc_id, [{"text": "replacement"}], np.ones((1, 4), dtype=np.float32),
                         embedding_space=space)

    assert inventory(tmp_db) == before
    metadata, vectors = db.load_all_chunks(tmp_db, embedding_profile=profile)
    assert metadata[0]["text"] == "original evidence"
    assert metadata[0]["dense_eligible"] is True
    np.testing.assert_array_equal(vectors, original)
    assert not tmp_db.in_transaction


def test_unchanged_cold_restart_does_not_reload_or_reembed_the_model(
    tmp_path, tmp_db, offline_indexer, monkeypatch,
):
    class ResolvedModel:
        tokenizer = SimpleNamespace(name_or_path="offline/tokenizer")
        max_seq_length = 2048

        def __getitem__(self, index):
            return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="resolved-commit")))

    root, cfg = _source(tmp_path)
    (root / "a.md").write_text("unchanged source evidence", encoding="utf-8")
    monkeypatch.setattr(indexer, "_model", ResolvedModel())
    metadata, vectors, changed = indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed is True
    assert db.load_source_state(tmp_db, cfg.name)["embedding_profile"]["revision"] == "resolved-commit"
    before = inventory(tmp_db)

    # A fresh process has the configured identity but no resolved tokenizer/model.
    monkeypatch.setattr(indexer, "_model", None)
    model_loader = Mock(side_effect=AssertionError("clean scan loaded a model"))
    monkeypatch.setattr(indexer, "_get_model", model_loader)
    for mock in vars(offline_indexer).values():
        mock.reset_mock()
    assert indexer.embedding_profile()["revision"] == "unresolved"
    after_metadata, after_vectors, changed = indexer.reindex_source(cfg, tmp_db, use_gemma=False)

    assert changed is False
    assert after_metadata == metadata
    np.testing.assert_array_equal(after_vectors, vectors)
    assert inventory(tmp_db) == before
    model_loader.assert_not_called()
    for mock in vars(offline_indexer).values():
        mock.assert_not_called()


def test_swallowed_chunk_read_failure_retains_evidence_and_allows_retry(
    tmp_path, tmp_db, offline_indexer, monkeypatch,
):
    root, cfg = _source(tmp_path)
    path = root / "a.md"
    path.write_text("original evidence", encoding="utf-8")
    indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    before = inventory(tmp_db)
    path.write_text("replacement evidence", encoding="utf-8")

    # chunk_file's legacy error API returns [] after an OSError.
    with monkeypatch.context() as patcher:
        patcher.setattr(indexer, "chunk_file", lambda *args, **kwargs: [])
        with pytest.raises(ValueError, match="no evidence for nonempty"):
            indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert inventory(tmp_db) == before
    assert not tmp_db.in_transaction

    metadata, _, changed = indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed is True
    assert metadata[0]["text"] == "replacement evidence"
    assert db.get_document_content(tmp_db, cfg.name, "a.md") == "replacement evidence"


def test_legacy_backslash_identity_survives_reindex_and_source_deletion(
    tmp_path, tmp_db, offline_indexer,
):
    root, cfg = _source(tmp_path)
    (root / "sub").mkdir()
    path = root / "sub" / "notes.md"
    path.write_text("current source evidence", encoding="utf-8")
    legacy_path = r"sub\notes.md"
    doc_id = db.upsert_document(tmp_db, cfg.name, legacy_path, "", "", "legacy evidence")
    db.upsert_chunks(tmp_db, doc_id, [{"text": "legacy evidence"}], np.ones((1, 3), dtype=np.float32))

    metadata, _, changed = indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed is True
    assert tmp_db.execute("SELECT id,path,content FROM documents").fetchall() == [(doc_id, legacy_path, "current source evidence")]
    assert {chunk["path"] for chunk in metadata} == {legacy_path}
    assert set(db.load_source_state(tmp_db, cfg.name)["content_hashes"]) == {legacy_path}

    path.unlink()
    metadata, _, changed = indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed is True and metadata == []
    assert tmp_db.execute("SELECT COUNT(*) FROM documents").fetchone() == (0,)
    assert db.load_source_state(tmp_db, cfg.name)["content_hashes"] == {}
    assert {row[0] for row in tmp_db.execute("SELECT content FROM document_revisions")} == {"legacy evidence", "current source evidence"}


def test_nested_transaction_rolls_back_document_chunks_history_and_checkpoint(tmp_db):
    doc_id = db.upsert_document(tmp_db, "fixture", "a.md", "Old title", "", "original evidence")
    db.upsert_chunks(tmp_db, doc_id, [{"text": "original evidence"}], np.ones((1, 3), dtype=np.float32))
    db.save_source_state(tmp_db, "fixture", {"content_hashes": {"a.md": "old"}, "cursor": 1})
    before = inventory(tmp_db)

    with pytest.raises(RuntimeError, match="after checkpoint"):
        with db.transaction(tmp_db):
            replacement = db.upsert_document(tmp_db, "fixture", "a.md", "New title", "", "replacement evidence")
            db.upsert_chunks(tmp_db, replacement, [{"text": "replacement evidence"}], np.zeros((1, 3), dtype=np.float32))
            db.save_source_state(tmp_db, "fixture", {"content_hashes": {"a.md": "new"}, "cursor": 2})
            raise RuntimeError("injected after checkpoint")

    assert inventory(tmp_db) == before
    assert not tmp_db.in_transaction
    # A later independent commit cannot accidentally publish rolled-back writes.
    db.save_deprecated(tmp_db, "unrelated.md", "unrelated write")
    after = inventory(tmp_db)
    for table in ("documents", "chunks", "document_revisions", "evidence_segments", "source_state"):
        assert after[table] == before[table]


def test_indexer_chunk_write_failure_rolls_back_document_and_retains_checkpoint(
    tmp_path, tmp_db, offline_indexer, monkeypatch,
):
    root, cfg = _source(tmp_path)
    path = root / "a.md"
    path.write_text("original evidence", encoding="utf-8")
    indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    before = inventory(tmp_db)
    path.write_text("replacement evidence", encoding="utf-8")
    real_upsert = indexer.upsert_chunks

    def fail_after_chunk_write(*args, **kwargs):
        real_upsert(*args, **kwargs)
        raise RuntimeError("injected chunk persistence failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(indexer, "upsert_chunks", fail_after_chunk_write)
        with pytest.raises(RuntimeError, match="chunk persistence"):
            indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert inventory(tmp_db) == before
    assert not tmp_db.in_transaction

    metadata, _, changed = indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed is True and metadata[0]["text"] == "replacement evidence"


def test_changed_override_reindexes_unchanged_source_bytes(tmp_path, tmp_db, offline_indexer, monkeypatch):
    root, cfg = _source(tmp_path)
    source = root / "a.md"
    source.write_text("source evidence remains unchanged", encoding="utf-8")
    source_bytes = source.read_bytes()
    override = root / ".bookkeeper-meta.yaml"
    override.write_text("files:\n  a.md:\n    description: Old manual description\n    tags: [old-tag]\n", encoding="utf-8")
    monkeypatch.setattr(indexer, "tag_file", tagger.tag_file)
    metadata, _, changed = indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    assert changed is True
    assert metadata[0]["description"] == "Old manual description"
    old_state = db.load_source_state(tmp_db, cfg.name)
    assert indexer.reindex_source(cfg, tmp_db, use_gemma=False)[2] is False

    override.write_text("files:\n  a.md:\n    description: New manual description\n    tags: [new-tag]\n", encoding="utf-8")
    metadata, _, changed = indexer.reindex_source(cfg, tmp_db, use_gemma=False)
    new_state = db.load_source_state(tmp_db, cfg.name)

    assert source.read_bytes() == source_bytes
    assert changed is True
    assert metadata[0]["description"] == "New manual description"
    assert "new-tag" in metadata[0]["tags"] and "old-tag" not in metadata[0]["tags"]
    assert new_state["content_hashes"] == old_state["content_hashes"]
    assert new_state["pipeline"] != old_state["pipeline"]
    assert indexer.reindex_source(cfg, tmp_db, use_gemma=False)[2] is False


def test_changed_tagger_producer_reindexes_unchanged_source_bytes(tmp_path, tmp_db, offline_indexer, monkeypatch):
    root, cfg = _source(tmp_path)
    (root / "a.md").write_text("unchanged source evidence", encoding="utf-8")
    monkeypatch.setattr(tagger, "_tagger_prompt", "Original producer instructions")
    indexer.reindex_source(cfg, tmp_db, use_gemma=True)
    old_state = db.load_source_state(tmp_db, cfg.name)
    assert indexer.reindex_source(cfg, tmp_db, use_gemma=True)[2] is False

    monkeypatch.setattr(tagger, "_tagger_prompt", "Changed producer instructions")
    assert indexer.reindex_source(cfg, tmp_db, use_gemma=True)[2] is True
    new_state = db.load_source_state(tmp_db, cfg.name)
    assert new_state["content_hashes"] == old_state["content_hashes"]
    assert new_state["pipeline"] != old_state["pipeline"]
    assert offline_indexer.tag.call_count == 2
    assert indexer.reindex_source(cfg, tmp_db, use_gemma=True)[2] is False


def test_pending_ingestion_retries_are_deduplicated_before_acceptance(tmp_db):
    chunk = {"source": "fixture", "path": "a.md", "text": "A"}
    first = db.enqueue_ingestion(tmp_db, [chunk])
    # JSON key order is not a different snapshot.
    reordered = {"text": "A", "path": "a.md", "source": "fixture"}
    assert db.enqueue_ingestion(tmp_db, [reordered]) == first
    assert db.pending_ingestion(tmp_db) == [{"id": first, "chunks": [chunk]}]

    db.finish_ingestion(tmp_db, first, error="temporary embedding failure")
    assert db.enqueue_ingestion(tmp_db, [chunk]) == first
    assert db.pending_ingestion(tmp_db) == [{"id": first, "chunks": [chunk]}]
    assert tmp_db.execute("SELECT COUNT(*) FROM ingestion_jobs").fetchone() == (1,)


@pytest.mark.parametrize("transcript", [False, True])
def test_ingestion_a_b_a_is_a_new_accepted_event(tmp_db, transcript):
    identity = {"source": "fixture", "path": "a.md"}
    if transcript:
        identity["transcript_path"] = "legacy/session.jsonl"
    first_a = {**identity, "text": "A"}
    middle_b = {**identity, "text": "B"}

    accepted = []
    for chunk in (first_a, middle_b, first_a):
        job_id = db.enqueue_ingestion(tmp_db, [chunk])
        assert db.enqueue_ingestion(tmp_db, [chunk]) == job_id
        with db.transaction(tmp_db):
            assert db.accept_ingestion(tmp_db, job_id, chunk) is True
            db.finish_ingestion(tmp_db, job_id)
        accepted.append(job_id)

    assert len(set(accepted)) == 3
    assert db.pending_ingestion(tmp_db) == []
    assert tmp_db.execute("SELECT job_id FROM ingestion_heads").fetchone() == (accepted[-1],)
    # A late completion of the original A cannot replace the final accepted A.
    assert db.accept_ingestion(tmp_db, accepted[0], first_a) is False
    assert tmp_db.execute("SELECT job_id FROM ingestion_heads").fetchone() == (accepted[-1],)
