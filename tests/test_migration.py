"""Offline migration proofs over historical DDL and synthetic legacy snapshots.

The DDL fixture was extracted from _SCHEMA in the recorded repository commits;
tests require neither Git history, network access, nor an embedding model.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from agents_rag import db
from agents_rag.migration import inventory, migrate


def test_resume_after_cache_reset_before_final_publication(tmp_path, monkeypatch):
    from agents_rag import migration
    source = tmp_path / "old.db"
    destination = tmp_path / "new.db"
    _seed_sqlite(source)
    original = source.read_bytes()
    real_link = migration.os.link
    with monkeypatch.context() as context:
        def fail_publication(*args, **kwargs):
            raise OSError("synthetic interruption before publication")
        context.setattr(migration.os, "link", fail_publication)
        with pytest.raises(OSError, match="synthetic interruption"):
            migrate(source, destination)
    assert not destination.exists()
    assert migration.os.link is real_link
    assert migrate(source, destination, resume=True)["status"] == "validated"
    assert source.read_bytes() == original
    with closing(sqlite3.connect(destination)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM hnsw_cache").fetchone()[0] == 0


SCHEMAS = json.loads((Path(__file__).parent / "fixtures" / "legacy_schemas.json").read_text(encoding="utf-8"))
STAMP = "2026-01-01T00:00:00Z"
TRANSCRIPT_PATH = "C:/legacy/transcripts/session.jsonl"
TRANSCRIPT_HASH = hashlib.sha256(TRANSCRIPT_PATH.encode()).hexdigest()[:16]
VECTOR = np.array([0.25, -0.75, 1.5], dtype=np.float32).tobytes()


def _snapshot_files(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _rows(conn, table, columns=None):
    projection = ",".join(f'"{name}"' for name in columns) if columns else "*"
    return sorted(conn.execute(f'SELECT {projection} FROM "{table}"').fetchall(), key=repr)


def _seed_sqlite(path, ref="62abfcf"):
    fixture = SCHEMAS[ref]
    assert fixture["commit"].startswith(ref)
    assert hashlib.sha256(fixture["schema"].encode()).hexdigest() == fixture["sha256"]
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(fixture["schema"])
        conn.executemany(
            "INSERT INTO documents(id,source,path,title,preamble,content,indexed_at) VALUES(?,?,?,?,?,?,?)",
            [(17, "fixture", "a.md", "Title α", "Preamble", "Raw text α\nSecond line", STAMP),
             (18, "sessions", "session.jsonl", "Historical session", "", "Only a historical summary", STAMP)],
        )
        conn.executemany(
            "INSERT INTO chunks(id,doc_id,section_heading,start_line,end_line,text,tags,description,"
            "doc_preamble,indexed_at,vector,session_summary,full_session_path,session_turn_count) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(29, 17, "Section α", 1, 2, "Raw text α\nSecond line", '["tag"]', "Description", "Preamble", STAMP, VECTOR, None, None, None),
             (30, 18, "Session", 1, 3, "Only a historical summary", "[]", "", "", STAMP, VECTOR, "Summary", TRANSCRIPT_PATH, 3)],
        )
        if "summary" in {row[1] for row in conn.execute("PRAGMA table_info(documents)")}:
            conn.execute("UPDATE documents SET summary='Existing document summary' WHERE id=17")
        conn.execute("INSERT INTO source_state VALUES(?,?,?,?)", ("fixture", "abc123", '{"a.md":1.25}', STAMP))
        conn.execute("INSERT INTO deprecated VALUES('a.md','Original reason')")
        conn.execute("INSERT INTO transcript_state VALUES(?,?)", (TRANSCRIPT_HASH, '{"byte_offset":42,"legacy_flag":true}'))
        conn.execute("INSERT INTO hnsw_cache VALUES(1,?,2,?)", (b"old-derived-index\x00\xff", STAMP))
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='query_log'").fetchone():
            conn.execute(
                "INSERT INTO query_log(id,timestamp,query,source_filter,top_k,result_count,results_json,latency_ms) "
                "VALUES(41,?,'original query α','fixture',5,1,'[{\"path\":\"a.md\"}]',12)", (STAMP,),
            )
            qcols = {row[1] for row in conn.execute("PRAGMA table_info(query_log)")}
            if "query_embedding" in qcols:
                conn.execute("UPDATE query_log SET query_embedding=? WHERE id=41", (VECTOR,))
            if "reranker_used" in qcols:
                conn.execute("UPDATE query_log SET reranker_used=1,rrf_k=60,score_kind='reranker' WHERE id=41")
        # A migration cannot assume it knows all data carried by an older file.
        conn.execute("CREATE TABLE operator_metadata(id INTEGER PRIMARY KEY,payload BLOB,note TEXT)")
        conn.execute("INSERT INTO operator_metadata VALUES(7,?,?)", (b"\x00\xff\x81custom", "Unknown metadata α"))
        conn.commit()
        tables = inventory(conn)
        original = {name: _rows(conn, name) for name in tables}
    return tables, original


def _seed_flat(path):
    path.mkdir()
    vectors = np.eye(3, dtype=np.float32)
    metadata = [
        {"source": "fixture", "path": name, "doc_title": "Legacy title", "text": "metadata " + text,
         "section_heading": text, "indexed_at": STAMP, "tags": ["legacy"],
         "future_field": {"position": i, "note": "retain α"}}
        for i, (name, text) in enumerate([("a.md", "a1"), ("b.md", "b1"), ("a.md", "a2")])
    ]
    np.savez(path / "index.npz", vectors=vectors)
    (path / "chunks.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    texts = ["exact a1 α\n", "exact b1 β\n", "exact a2 γ\n"]
    (path / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    (path / "state.json").write_text(json.dumps({"fixture": {"commit_hash": "abc123", "mtimes": {"a.md": 1.25},
                                                               "indexed_at": STAMP, "unknown_state": "keep"}}), encoding="utf-8")
    (path / "deprecated.json").write_text('{"a.md":"Original reason"}', encoding="utf-8")
    (path / "transcripts").mkdir()
    (path / "transcripts" / f"{TRANSCRIPT_HASH}.state.json").write_text('{"byte_offset":42,"legacy_flag":true}', encoding="utf-8")
    return vectors, metadata, texts


@pytest.mark.parametrize("ref", SCHEMAS)
def test_historical_sqlite_copy_preserves_rows_bytes_ids_and_provenance(tmp_path, ref):
    source = tmp_path / f"{ref}.db"
    tables, original = _seed_sqlite(source, ref)
    source_bytes = source.read_bytes()
    target = tmp_path / "migrated.db"
    report = migrate(source, target)

    assert report["status"] == "validated"
    assert report["cutover"] == "manual"
    assert source.read_bytes() == source_bytes
    with closing(sqlite3.connect(source)) as conn:
        assert inventory(conn) == tables
    with closing(sqlite3.connect(target)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        for table, info in tables.items():
            if table == "hnsw_cache":
                # Derived index is rebuilt; the original artifact remains in source.
                assert _rows(conn, table) == []
            else:
                assert _rows(conn, table, info["columns"]) == original[table]
        revisions = conn.execute(
            "SELECT d.id,d.content,d.completeness,r.content,r.completeness "
            "FROM documents d JOIN document_revisions r ON r.id=d.revision_id ORDER BY d.id").fetchall()
        assert revisions == [
            (17, "Raw text α\nSecond line", "legacy_unknown", "Raw text α\nSecond line", "legacy_unknown"),
            (18, "Only a historical summary", "legacy_summary", "Only a historical summary", "legacy_summary"),
        ]
        evidence = conn.execute(
            "SELECT c.id,c.text,c.vector,e.text,e.revision_id,d.revision_id,e.metadata_json,c.embedding_space_id "
            "FROM chunks c JOIN evidence_segments e ON e.id=c.evidence_id "
            "JOIN documents d ON d.id=c.doc_id ORDER BY c.id").fetchall()
        assert len(evidence) == 2
        for chunk_id, text, vector, evidence_text, revision_id, doc_revision, metadata, space in evidence:
            assert text == evidence_text
            assert vector == VECTOR
            assert revision_id == doc_revision
            assert json.loads(metadata)["legacy_chunk_id"] == chunk_id
            assert space == "legacy_unknown"
        assert db.load_transcript_state(conn, TRANSCRIPT_PATH) == {"byte_offset": 42, "legacy_flag": True}
        stable = inventory(conn)
    # Reopening must not create revisions or change already migrated rows.
    with closing(db.init_db(str(target))) as conn:
        assert inventory(conn) == stable
    assert source.read_bytes() == source_bytes


def test_flat_interleaving_preserves_vector_evidence_pairs_and_raw_artifacts(tmp_path):
    source = tmp_path / "legacy"
    vectors, metadata, texts = _seed_flat(source)
    original = _snapshot_files(source)
    target = tmp_path / "migrated.db"
    migrate(source, target)

    assert _snapshot_files(source) == original
    with closing(sqlite3.connect(target)) as conn:
        mapped = conn.execute(
            "SELECT m.position,d.path,c.text,c.vector,c.evidence_id,m.evidence_id,e.text,e.metadata_json "
            "FROM legacy_chunk_map m JOIN chunks c ON c.id=m.chunk_id "
            "JOIN documents d ON d.id=c.doc_id JOIN evidence_segments e ON e.id=c.evidence_id "
            "ORDER BY m.position").fetchall()
        assert len(mapped) == 3
        for i, path, text, vector, evidence_id, mapped_evidence, evidence_text, evidence_metadata in mapped:
            assert path == metadata[i]["path"]
            assert text == evidence_text == texts[i]
            assert vector == vectors[i].tobytes()
            assert evidence_id == mapped_evidence
            assert json.loads(evidence_metadata)["future_field"] == metadata[i]["future_field"]
        # Grouping reordered DB IDs; the positional map must still recover original order.
        assert conn.execute("SELECT text FROM chunks ORDER BY id").fetchall() == [(texts[0],), (texts[2],), (texts[1],)]
        archived = conn.execute("SELECT path,data,sha256 FROM legacy_artifacts").fetchall()
        assert {path: raw for path, raw, _ in archived} == original
        assert all(hashlib.sha256(raw).hexdigest() == digest for _, raw, digest in archived)
        assert conn.execute("SELECT transcript_hash FROM transcript_state").fetchall() == [(TRANSCRIPT_HASH,)]
        assert db.load_transcript_state(conn, TRANSCRIPT_PATH) == {"byte_offset": 42, "legacy_flag": True}
        assert db.load_source_state(conn, "fixture") == {"commit_hash": "abc123", "mtimes": {"a.md": 1.25},
                                                       "indexed_at": STAMP, "unknown_state": "keep"}
        assert db.load_deprecated(conn) == {"a.md": "Original reason"}
        assert {row[0] for row in conn.execute("SELECT completeness FROM documents")} == {"reconstructed_chunks"}
        assert conn.execute("SELECT COUNT(*) FROM hnsw_cache").fetchone() == (0,)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_interrupted_flat_migration_resumes_without_duplicates_and_matches_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_now", lambda: STAMP)
    source = tmp_path / "legacy"
    _seed_flat(source)
    original = _snapshot_files(source)
    target = tmp_path / "resumed.db"
    with pytest.raises(RuntimeError, match="interruption"):
        migrate(source, target, _interrupt_after=1)
    assert not target.exists()
    stage = Path(str(target) + ".staging")
    assert stage.exists()
    with closing(sqlite3.connect(stage)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone() == (1,)
        assert conn.execute("SELECT position FROM legacy_chunk_map ORDER BY position").fetchall() == [(0,), (2,)]
    with pytest.raises(ValueError, match="resume"):
        migrate(source, target)
    assert migrate(source, target, resume=True)["status"] == "validated"
    assert not stage.exists()
    clean = tmp_path / "clean.db"
    migrate(source, clean)
    with closing(sqlite3.connect(target)) as resumed, closing(sqlite3.connect(clean)) as fresh:
        assert inventory(resumed) == inventory(fresh)
    assert _snapshot_files(source) == original


def test_resume_rejects_changed_source_without_publishing_or_altering_stage(tmp_path):
    source = tmp_path / "legacy"
    _seed_flat(source)
    target = tmp_path / "target.db"
    with pytest.raises(RuntimeError):
        migrate(source, target, _interrupt_after=1)
    stage = Path(str(target) + ".staging")
    before = stage.read_bytes()
    (source / "deprecated.json").write_text('{"a.md":"Changed reason"}', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        migrate(source, target, resume=True)
    assert not target.exists()
    assert stage.read_bytes() == before


def test_resume_rejects_corrupted_archived_artifact(tmp_path):
    source = tmp_path / "legacy"
    _seed_flat(source)
    original = _snapshot_files(source)
    target = tmp_path / "target.db"
    with pytest.raises(RuntimeError):
        migrate(source, target, _interrupt_after=1)
    with closing(sqlite3.connect(str(target) + ".staging")) as conn:
        conn.execute("UPDATE legacy_artifacts SET data=? WHERE path='chunks.json'", (b"corrupted staged artifact",))
        conn.commit()
    with pytest.raises(ValueError, match="(?i)artifact|preserv|mismatch"):
        migrate(source, target, resume=True)
    assert not target.exists()
    assert _snapshot_files(source) == original


@pytest.mark.parametrize("source_kind", ["sqlite", "flat"])
def test_dry_run_creates_no_files_or_directories(tmp_path, source_kind):
    if source_kind == "sqlite":
        source = tmp_path / "source.db"
        _seed_sqlite(source)
    else:
        source = tmp_path / "legacy"
        _seed_flat(source)
    before = _snapshot_files(tmp_path)
    report = migrate(source, tmp_path / "new-directory" / "target.db", dry_run=True)
    assert report["status"] == "dry_run"
    assert report["manifest"]["format"] == source_kind
    assert _snapshot_files(tmp_path) == before
    assert not (tmp_path / "new-directory").exists()


@pytest.mark.parametrize("artifact", ["state.json", "deprecated.json", f"transcripts/{TRANSCRIPT_HASH}.state.json"])
def test_dry_run_validates_optional_state_artifacts(tmp_path, artifact):
    source = tmp_path / "legacy"
    _seed_flat(source)
    (source / artifact).write_text("{invalid JSON", encoding="utf-8")
    before = _snapshot_files(tmp_path)
    target = tmp_path / "target.db"
    with pytest.raises(ValueError):
        migrate(source, target, dry_run=True)
    assert _snapshot_files(tmp_path) == before


@pytest.mark.parametrize("damage", ["missing_pair", "broken_archive", "broken_json", "vector_count", "nonfinite", "texts_count", "late_state_json"])
def test_invalid_flat_inputs_never_publish_target_and_leave_originals_unchanged(tmp_path, damage):
    source = tmp_path / "legacy"
    _seed_flat(source)
    if damage == "missing_pair":
        (source / "index.npz").unlink()
    elif damage == "broken_archive":
        (source / "index.npz").write_bytes(b"not an npz archive")
    elif damage == "broken_json":
        (source / "chunks.json").write_text("{broken", encoding="utf-8")
    elif damage == "vector_count":
        np.savez(source / "index.npz", vectors=np.eye(2, dtype=np.float32))
    elif damage == "nonfinite":
        np.savez(source / "index.npz", vectors=np.full((3, 3), np.nan, dtype=np.float32))
    elif damage == "texts_count":
        (source / "texts.json").write_text('["one text"]', encoding="utf-8")
    else:
        (source / "state.json").write_text("{broken", encoding="utf-8")
    before = _snapshot_files(source)
    target = tmp_path / "target.db"
    with pytest.raises((ValueError, OSError)):
        migrate(source, target)
    assert not target.exists()
    assert _snapshot_files(source) == before


@pytest.mark.parametrize("damage", ["corrupt", "future", "unrecognized", "foreign_key"])
def test_invalid_sqlite_never_publishes_target(tmp_path, damage):
    source = tmp_path / "source.db"
    if damage == "corrupt":
        source.write_bytes(b"not a sqlite database")
    elif damage == "unrecognized":
        with closing(sqlite3.connect(source)) as conn:
            conn.execute("CREATE TABLE other(value TEXT)")
            conn.commit()
    else:
        _seed_sqlite(source)
        with closing(sqlite3.connect(source)) as conn:
            if damage == "future":
                conn.execute(f"PRAGMA user_version={db.SCHEMA_VERSION + 1}")
            else:
                conn.execute("UPDATE chunks SET doc_id=999 WHERE id=29")
            conn.commit()
    original = source.read_bytes()
    target = tmp_path / "target.db"
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        migrate(source, target)
    assert not target.exists()
    assert source.read_bytes() == original


def test_existing_destination_and_same_source_are_never_overwritten(tmp_path):
    source = tmp_path / "source.db"
    _seed_sqlite(source)
    target = tmp_path / "target.db"
    target.write_bytes(b"existing user database")
    original = _snapshot_files(tmp_path)
    with pytest.raises(ValueError, match="refusing overwrite"):
        migrate(source, target)
    with pytest.raises(ValueError, match="original input"):
        migrate(source, source)
    assert _snapshot_files(tmp_path) == original


@pytest.mark.parametrize("suffix", [
    ".staging", ".manifest.json", ".migration.lock",
    ".staging-shm", ".staging-wal", ".staging-journal", "-shm", "-wal", "-journal",
])
def test_migration_sidecars_cannot_overwrite_the_original_database(tmp_path, suffix):
    target = tmp_path / "target.db"
    source = Path(str(target) + suffix)
    _seed_sqlite(source)
    before = _snapshot_files(tmp_path)
    with pytest.raises(ValueError, match="original input"):
        migrate(source, target)
    assert _snapshot_files(tmp_path) == before


@pytest.mark.parametrize("suffix", ["-shm", "-wal", "-journal"])
def test_destination_cannot_replace_a_source_sqlite_sidecar(tmp_path, suffix):
    source = tmp_path / "source.db"
    _seed_sqlite(source)
    target = Path(str(source) + suffix)
    before = _snapshot_files(tmp_path)
    with pytest.raises(ValueError, match="original input"):
        migrate(source, target)
    assert _snapshot_files(tmp_path) == before


@pytest.mark.parametrize("source_kind", ["sqlite", "flat"])
def test_output_hardlink_to_input_is_rejected_before_writes(tmp_path, source_kind):
    if source_kind == "sqlite":
        source = tmp_path / "source.db"
        _seed_sqlite(source)
        original_input = source
    else:
        source = tmp_path / "flat"
        _seed_flat(source)
        original_input = source / "index.npz"
    target = tmp_path / "target.db"
    alias = Path(str(target) + ".staging-shm")
    try:
        alias.hardlink_to(original_input)
    except OSError:
        pytest.skip("Filesystem does not permit hard links")
    before = _snapshot_files(tmp_path)
    with pytest.raises(ValueError, match="original input"):
        migrate(source, target)
    assert _snapshot_files(tmp_path) == before


@pytest.mark.parametrize("blob", [b"bad", b"", np.array([np.nan], dtype=np.float32).tobytes()])
def test_malformed_sqlite_vectors_are_rejected_before_any_output_is_created(tmp_path, blob):
    source = tmp_path / "source.db"
    _seed_sqlite(source)
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("UPDATE chunks SET vector=? WHERE id=29", (blob,))
        conn.commit()
    before = _snapshot_files(tmp_path)
    with pytest.raises(ValueError, match="Invalid vector"):
        migrate(source, tmp_path / "target.db")
    assert _snapshot_files(tmp_path) == before


def test_ambiguous_directory_requires_an_explicit_source(tmp_path):
    source = tmp_path / "legacy"
    _seed_flat(source)
    _seed_sqlite(source / "bookkeeper.db")
    with pytest.raises(ValueError, match="Multiple"):
        migrate(source, tmp_path / "target.db")
    assert not (tmp_path / "target.db").exists()
    assert migrate(source / "bookkeeper.db", tmp_path / "target.db")["status"] == "validated"


@pytest.mark.parametrize("basename", ["bookkeeper.db", "agents-rag.db"])
def test_directory_detects_current_and_pre_rename_database_names(tmp_path, basename):
    source_dir = tmp_path / "legacy"
    source_dir.mkdir()
    source = source_dir / basename
    _seed_sqlite(source)
    report = migrate(source_dir, tmp_path / "target.db")
    assert Path(report["source"]) == source
    assert report["status"] == "validated"


def test_sqlite_backup_includes_committed_wal_rows(tmp_path):
    source = tmp_path / "wal-source.db"
    _seed_sqlite(source)
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("UPDATE documents SET content='Committed WAL evidence α' WHERE id=17")
        writer.commit()
        assert Path(str(source) + "-wal").exists()
        original_main = source.read_bytes()
        original_wal = Path(str(source) + "-wal").read_bytes()
        target = tmp_path / "target.db"
        migrate(source, target)
        assert source.read_bytes() == original_main
        assert Path(str(source) + "-wal").read_bytes() == original_wal
        with closing(sqlite3.connect(target)) as conn:
            assert conn.execute("SELECT content FROM documents WHERE id=17").fetchone() == ("Committed WAL evidence α",)
