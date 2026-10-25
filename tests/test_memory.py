"""Evidence-backed memory lifecycle, scope and optimistic concurrency contracts."""

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from agents_rag import db


def _evidence(conn, *, scope="docs", path="decision.md", text="Verified source text"):
    doc_id = db.upsert_document(conn, scope, path, "Decision", "", text)
    db.upsert_chunks(conn, doc_id, [{"text": text, "tags": []}], np.array([[1, 0]], dtype=np.float32))
    return conn.execute("SELECT evidence_id FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()[0]


def _save(conn, evidence, **overrides):
    args = {"kind": "decision", "scope": "docs", "content": "Use the documented procedure",
            "evidence_ids": [evidence]}
    return db.save_memory(conn, **{**args, **overrides})


def test_memory_updates_require_current_revision_and_preserve_history(tmp_db):
    evidence = _evidence(tmp_db)
    first = _save(tmp_db, evidence)
    with pytest.raises(ValueError, match="revision conflict"):
        _save(tmp_db, evidence, memory_id=first["id"], content="Unauthorized stale update")
    second = _save(tmp_db, evidence, memory_id=first["id"], expected_revision=first["revision_id"],
                   content="Use the revised procedure")
    with pytest.raises(ValueError, match="revision conflict"):
        _save(tmp_db, evidence, memory_id=first["id"], expected_revision=first["revision_id"],
              content="A different stale edit")
    latest = db.list_memories(tmp_db, "docs")
    assert len(latest) == 1
    assert latest[0]["revision_id"] == second["revision_id"]
    assert latest[0]["previous_revision"] == first["revision_id"]
    history = db.list_memories(tmp_db, "docs", include_history=True)
    assert [r["content"] for r in history] == ["Use the revised procedure", "Use the documented procedure"]
    assert all(r["verification"] == "unverified" for r in history)
    assert db.list_memories(tmp_db, "other") == []


def test_two_concurrent_editors_cannot_both_replace_the_same_revision(tmp_db):
    evidence = _evidence(tmp_db)
    first = _save(tmp_db, evidence)
    barrier = threading.Barrier(2)

    def edit(content):
        barrier.wait(timeout=5)
        try:
            return _save(tmp_db, evidence, memory_id=first["id"],
                         expected_revision=first["revision_id"], content=content)
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(edit, ["Editor one", "Editor two"]))
    assert sum(isinstance(result, dict) for result in outcomes) == 1
    assert sum(isinstance(result, str) and "revision conflict" in result for result in outcomes) == 1
    assert len(db.list_memories(tmp_db, "docs", include_history=True)) == 2


@pytest.mark.parametrize("missing", [False, True])
def test_memory_requires_existing_evidence_in_the_same_scope(tmp_db, missing):
    evidence = "missing-evidence-id" if missing else _evidence(tmp_db, scope="private")
    with pytest.raises(ValueError, match="within the memory scope"):
        _save(tmp_db, evidence)
    assert tmp_db.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0] == 0
    assert tmp_db.execute("SELECT COUNT(*) FROM memory_revisions").fetchone()[0] == 0


@pytest.mark.parametrize("field", ["kind", "scope"])
def test_memory_kind_and_scope_cannot_change(tmp_db, field):
    evidence = _evidence(tmp_db)
    private_evidence = _evidence(tmp_db, scope="private")
    first = _save(tmp_db, evidence)
    overrides = {"kind": "fact"} if field == "kind" else {"scope": "private", "evidence_ids": [private_evidence]}
    with pytest.raises(ValueError, match="kind and scope are immutable"):
        _save(tmp_db, evidence, memory_id=first["id"], expected_revision=first["revision_id"], **overrides)
    assert len(db.list_memories(tmp_db, "docs", include_history=True)) == 1
    assert db.list_memories(tmp_db, "private") == []


def test_identical_memory_update_is_noop_without_generation_change(tmp_db):
    evidence = _evidence(tmp_db)
    first = _save(tmp_db, evidence)
    generation = db.corpus_generation(tmp_db)
    unchanged = _save(tmp_db, evidence, memory_id=first["id"], expected_revision=first["revision_id"])
    assert unchanged["no_op"] is True
    assert unchanged["revision_id"] == first["revision_id"]
    assert db.corpus_generation(tmp_db) == generation
    assert len(db.list_memories(tmp_db, "docs", include_history=True)) == 1


def test_memory_source_currency_tracks_deprecation_revision_and_deletion(tmp_db):
    evidence = _evidence(tmp_db)
    _save(tmp_db, evidence)
    assert db.list_memories(tmp_db, "docs")[0]["sources_current"] is True
    db.save_deprecated(tmp_db, "decision.md", "Superseded source")
    assert db.list_memories(tmp_db, "docs")[0]["sources_current"] is False
    db.remove_deprecated(tmp_db, "decision.md")
    assert db.list_memories(tmp_db, "docs")[0]["sources_current"] is True
    db.upsert_document(tmp_db, "docs", "decision.md", "Decision", "", "Updated source text")
    assert db.list_memories(tmp_db, "docs")[0]["sources_current"] is False
    db.delete_source_documents(tmp_db, "docs")
    remembered = db.list_memories(tmp_db, "docs")[0]
    assert remembered["sources_current"] is False
    assert remembered["verification"] == "unverified"
    assert db.get_evidence(tmp_db, evidence)["text"] == "Verified source text"


@pytest.mark.parametrize("status", ["superseded", "conflicted", "retracted"])
def test_memory_status_changes_are_append_only(tmp_db, status):
    evidence = _evidence(tmp_db)
    first = _save(tmp_db, evidence)
    updated = _save(tmp_db, evidence, memory_id=first["id"], expected_revision=first["revision_id"], status=status)
    assert updated["status"] == status
    latest = db.list_memories(tmp_db, "docs")
    assert len(latest) == 1 and latest[0]["status"] == status
    history = db.list_memories(tmp_db, "docs", include_history=True)
    assert [entry["status"] for entry in history] == [status, "active"]


@pytest.mark.parametrize("valid_from,valid_to", [
    ("2026-01-01T10:00:00+02:00", "2026-01-01T09:00:00Z"),
    ("2026-01-01T08:00:00", "2026-01-01T09:00:00+00:00"),
])
def test_memory_interval_compares_instants_across_timezones(tmp_db, valid_from, valid_to):
    evidence = _evidence(tmp_db)
    _save(tmp_db, evidence, valid_from=valid_from, valid_to=valid_to)
    entry = db.list_memories(tmp_db, "docs")[0]
    assert entry["valid_from"] == valid_from
    assert entry["valid_to"] == valid_to


@pytest.mark.parametrize("valid_from,valid_to", [
    ("2026-01-01T08:00:00Z", "2026-01-01T09:00:00+02:00"),
    ("2026-01-01T09:00:00+01:00", "2026-01-01T08:00:00Z"),
    ("not-a-date", "2026-01-01T09:00:00Z"),
])
def test_memory_rejects_reversed_equal_or_malformed_intervals(tmp_db, valid_from, valid_to):
    evidence = _evidence(tmp_db)
    with pytest.raises(ValueError):
        _save(tmp_db, evidence, valid_from=valid_from, valid_to=valid_to)
    assert db.list_memories(tmp_db, "docs", include_history=True) == []
