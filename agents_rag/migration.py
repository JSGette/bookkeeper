"""Copy-only, resumable migration. Originals are never renamed, deleted or overwritten."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager, closing
from pathlib import Path

import numpy as np

from . import db


@contextmanager
def _read_only(path: Path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        yield conn


def _hash(value):
    def encode(item):
        if isinstance(item, bytes):
            return {"bytes": item.hex()}
        raise TypeError(type(item).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=encode,
                                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def inventory(conn):
    """Hash every user table and column, including unknown legacy metadata."""
    result = {}
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        columns = [r[1] for r in conn.execute(f"PRAGMA table_info({_quote(table)})")]
        rows = conn.execute(f"SELECT * FROM {_quote(table)}").fetchall()
        result[table] = {"columns": columns, "count": len(rows),
                         "hash": _hash(sorted((_hash(row) for row in rows)))}
    return result


def select_source(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"Migration source does not exist: {path}")
    candidates = [path / n for n in ("bookkeeper.db", "agents-rag.db") if (path / n).is_file()]
    flat = (path / "index.npz").is_file() and (path / "chunks.json").is_file()
    if len(candidates) + int(flat) > 1:
        raise ValueError("Multiple databases/indexes found; select the database file or a separate flat-file snapshot explicitly")
    if candidates:
        return candidates[0]
    if flat:
        return path
    raise ValueError("No SQLite database or complete index.npz/chunks.json pair found")


def _flat_inputs(path):
    files = [p for p in path.iterdir() if p.is_file() and p.name in
             {"index.npz", "chunks.json", "texts.json", "state.json", "deprecated.json"}]
    files += list((path / "transcripts").glob("*.state.json"))
    return sorted(files)


def inspect_source(path: str | Path) -> dict:
    source = select_source(path)
    if source.is_file():
        with _read_only(source) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > db.SCHEMA_VERSION:
                raise ValueError(f"Unsupported future schema {version}")
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError(f"Source integrity check failed: {integrity}")
            tables = inventory(conn)
            if not {"documents", "chunks"} <= tables.keys():
                raise ValueError("Not a recognized Bookkeeper SQLite database")
            for chunk_id,blob in conn.execute("SELECT id,vector FROM chunks"):
                if not isinstance(blob,bytes) or not blob or len(blob) % 4 or not np.isfinite(np.frombuffer(blob,dtype=np.float32)).all():
                    raise ValueError(f"Invalid vector BLOB in legacy chunk {chunk_id}; original retained")
        manifest = {"format": "sqlite", "source": str(source), "schema_version": version, "tables": tables}
    else:
        vectors, metadata, _ = _load_flat(source)
        for path in _flat_inputs(source):
            if path.suffix == ".json" and path.name not in {"chunks.json","texts.json"}:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload,dict):
                    raise ValueError(f"Expected object in {path.name}")
                if path.name == "state.json" and not all(isinstance(v,dict) for v in payload.values()):
                    raise ValueError("Source states must be objects")
                if path.name == "deprecated.json" and not all(isinstance(v,str) for v in payload.values()):
                    raise ValueError("Deprecation reasons must be strings")
        manifest = {"format": "flat", "source": str(source), "chunks": len(metadata),
                    "dimension": vectors.shape[1], "files": {
                        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in _flat_inputs(source)}}
    manifest["fingerprint"] = _hash(manifest)
    return manifest


def _load_flat(source):
    with np.load(source / "index.npz", allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    metadata = json.loads((source / "chunks.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, list) or vectors.ndim != 2 or len(vectors) != len(metadata):
        raise ValueError("Flat vector/metadata length or shape mismatch")
    if not np.isfinite(vectors).all() or vectors.shape[1] == 0:
        raise ValueError("Flat vectors must be finite and nonempty")
    original = [dict(m) for m in metadata]
    texts_path = source / "texts.json"
    if texts_path.exists():
        texts = json.loads(texts_path.read_text(encoding="utf-8"))
        if not isinstance(texts, list) or len(texts) != len(metadata) or not all(isinstance(t,str) for t in texts):
            raise ValueError("texts.json length/type mismatch; source was not modified")
        for m, t in zip(metadata, texts):
            m["text"] = t
    for m in metadata:
        if not isinstance(m.get("source"), str) or not isinstance(m.get("path"), str):
            raise ValueError("Every chunk requires source and path strings")
    return vectors, metadata, original


@contextmanager
def _lock(destination):
    """OS lock automatically released after process death; file persists for reuse."""
    lock_path = Path(str(destination) + ".migration.lock")
    with open(lock_path, "a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another migration owns this destination") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_preservation(conn, tables, *, allow_cleared_cache=False):
    for table, expected in tables.items():
        columns = ",".join(_quote(c) for c in expected["columns"])
        rows = conn.execute(f"SELECT {columns} FROM {_quote(table)}").fetchall()
        if table == "hnsw_cache" and allow_cleared_cache and not rows:
            continue  # A prior finalization may have cleared this rebuildable cache.
        actual = _hash(sorted(_hash(row) for row in rows))
        if actual != expected["hash"]:
            raise ValueError(f"Preservation validation failed for {table}")


def _import_flat(conn, source, *, batch_size, interrupt_after=None):
    vectors, metadata, original = _load_flat(source)
    # Preserve source artifacts byte-for-byte, including unknown JSON fields and conflicting text.
    conn.execute("CREATE TABLE IF NOT EXISTS legacy_artifacts(path TEXT PRIMARY KEY,data BLOB NOT NULL,sha256 TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS legacy_chunk_map(position INTEGER PRIMARY KEY,chunk_id INTEGER NOT NULL,evidence_id TEXT NOT NULL)")
    conn.commit()
    with db.transaction(conn):
        for path in _flat_inputs(source):
            raw = path.read_bytes()
            conn.execute("INSERT OR IGNORE INTO legacy_artifacts VALUES (?,?,?)",
                         (str(path.relative_to(source)),raw,hashlib.sha256(raw).hexdigest()))
            archived = conn.execute("SELECT data,sha256 FROM legacy_artifacts WHERE path=?",
                                     (str(path.relative_to(source)),)).fetchone()
            if archived != (raw,hashlib.sha256(raw).hexdigest()):
                raise ValueError(f"Archived artifact integrity failed: {path.name}")
    grouped = {}
    for i,m in enumerate(metadata):
        grouped.setdefault((m["source"],m["path"]),[]).append((i,m))
    completed = 0
    for (src,path),items in grouped.items():
        existing = conn.execute("SELECT position FROM legacy_chunk_map WHERE position=?", (items[0][0],)).fetchone()
        if existing:
            continue
        with db.transaction(conn):
            content = "\n\n".join(m.get("text","") for _,m in items)
            first = items[0][1]
            doc_id = db.upsert_document(conn,src,path,first.get("doc_title",""),first.get("doc_preamble",""),content,
                                        completeness="reconstructed_chunks")
            db.upsert_chunks(conn,doc_id,[m for _,m in items],vectors[[i for i,_ in items]])
            rows = conn.execute("SELECT id,evidence_id FROM chunks WHERE doc_id=? ORDER BY id", (doc_id,)).fetchall()
            conn.executemany("INSERT INTO legacy_chunk_map VALUES (?,?,?)",
                             [(item[0],row[0],row[1]) for item,row in zip(items,rows)])
        completed += 1
        if interrupt_after is not None and completed >= interrupt_after:
            raise RuntimeError("Injected migration interruption")
    with db.transaction(conn):
        state = source / "state.json"
        if state.exists():
            for name,payload in json.loads(state.read_text(encoding="utf-8")).items():
                db.save_source_state(conn,name,payload)
        deprecated = source / "deprecated.json"
        if deprecated.exists():
            for path,reason in json.loads(deprecated.read_text(encoding="utf-8")).items():
                db.save_deprecated(conn,path,reason)
        for path in (source / "transcripts").glob("*.state.json"):
            payload = path.read_text(encoding="utf-8")
            json.loads(payload)
            key = path.name.removesuffix(".state.json")
            if len(key) != 16 or any(c not in "0123456789abcdefABCDEF" for c in key):
                raise ValueError(f"Unknown transcript state filename: {path.name}")
            conn.execute("INSERT OR REPLACE INTO transcript_state VALUES (?,?)", (key,payload))
    rows = conn.execute("SELECT m.position,c.vector,c.text FROM legacy_chunk_map m JOIN chunks c ON c.id=m.chunk_id ORDER BY m.position").fetchall()
    if len(rows) != len(metadata):
        raise ValueError("Missing flat chunk mappings")
    for i,blob,text in rows:
        if blob != vectors[i].tobytes() or text != metadata[i].get("text",""):
            raise ValueError(f"Flat evidence/vector mismatch at {i}")


def _sqlite_paths(path: Path) -> list[Path]:
    """Reserve paths SQLite can create, replace, or remove when opening a database."""
    return [path, *(Path(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal"))]


def migrate(source_path, destination, *, dry_run=False, resume=False, batch_size=100, _interrupt_after=None):
    source = select_source(source_path)
    destination = Path(destination).expanduser().resolve()
    outputs = [*_sqlite_paths(destination), *_sqlite_paths(Path(str(destination) + ".staging")),
               Path(str(destination) + ".manifest.json"), Path(str(destination) + ".migration.lock")]
    inputs = _sqlite_paths(source) if source.is_file() else _flat_inputs(source)
    if any(output.resolve() == path.resolve()
           or (output.exists() and path.exists() and output.samefile(path))
           for output in outputs for path in inputs):
        raise ValueError("Migration output or sidecar would overwrite an original input")
    manifest = inspect_source(source)
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}; refusing overwrite")
    report = {"source": str(source), "destination": str(destination), "manifest": manifest,
              "schema_version": db.SCHEMA_VERSION, "status": "dry_run", "cutover": "manual",
              "requires_stopped_writers": True}
    if dry_run:
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(str(destination) + ".staging")
    manifest_path = Path(str(destination) + ".manifest.json")
    with _lock(destination):
        if stage.exists():
            if not resume or not manifest_path.exists():
                raise ValueError("Staging database exists; use --resume with the original inputs")
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved != manifest:
                raise ValueError("Source manifest changed; use a new destination")
        else:
            if manifest_path.exists():
                if not resume or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                    raise ValueError("Existing manifest is not an authorized resume; choose another destination")
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
            if source.is_file():
                with _read_only(source) as reader, closing(sqlite3.connect(stage)) as target:
                    reader.backup(target)
                # Snapshot consistency covers live WAL; equality prevents a moving input.
                with _read_only(stage) as snapshot:
                    _validate_preservation(snapshot,manifest["tables"])
        conn = db.init_db(str(stage))
        try:
            if source.is_dir():
                _import_flat(conn,source,batch_size=batch_size,interrupt_after=_interrupt_after)
            else:
                _validate_preservation(conn,manifest["tables"],allow_cleared_cache=True)
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or conn.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Migrated database integrity validation failed")
            if inspect_source(source) != manifest:
                raise ValueError("Source changed during migration; keep writers stopped and use a new destination")
            conn.execute("DELETE FROM hnsw_cache")  # Rebuild against final row mapping on first load.
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        # Atomic creation without replacing a destination created concurrently.
        os.link(stage,destination)
        stage.unlink()
    report["status"] = "validated"
    report["rollback"] = "Keep the new database after writes; old-schema export is a separate validated operation"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bookkeeper migrate")
    parser.add_argument("--from", "--source", dest="source", required=True)
    parser.add_argument("--to", "--destination", dest="destination", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = migrate(args.source,args.destination,dry_run=args.dry_run,resume=args.resume)
    except (ValueError,OSError,sqlite3.Error) as exc:
        parser.exit(1,f"Migration failed: {exc}\n")
    print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
