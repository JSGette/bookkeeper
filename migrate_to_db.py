#!/usr/bin/env python3
"""One-time migration: import existing flat-file index into the new SQLite DB."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("migrate")


def migrate(cache_dir: str, source_roots: dict[str, str] | None = None) -> None:
    from agents_rag.db import (
        init_db,
        save_deprecated,
        save_hnsw_cache,
        save_source_state,
        save_transcript_state,
        upsert_chunks,
        upsert_document,
    )

    cache = Path(cache_dir)
    db_path = str(cache / "agents-rag.db")

    if Path(db_path).exists():
        log.warning("Database already exists at %s — aborting to avoid data loss", db_path)
        sys.exit(1)

    conn = init_db(db_path)
    source_roots = source_roots or {}

    # 1. Load existing flat-file index
    vec_path = cache / "index.npz"
    meta_path = cache / "chunks.json"
    texts_path = cache / "texts.json"

    if not vec_path.exists() or not meta_path.exists():
        log.error("No existing index found at %s", cache)
        sys.exit(1)

    vectors = np.load(vec_path)["vectors"]
    metadata = json.loads(meta_path.read_text())

    texts: list[str] = []
    if texts_path.exists():
        texts = json.loads(texts_path.read_text())
        if len(texts) == len(metadata):
            for m, t in zip(metadata, texts):
                m["text"] = t
        else:
            log.warning("texts.json length mismatch, skipping text merge")

    assert len(vectors) == len(metadata), (
        f"Vector/metadata mismatch: {len(vectors)} vs {len(metadata)}"
    )
    log.info("Loaded %d chunks from flat files", len(metadata))

    # 2. Group chunks by (source, path) and upsert documents + chunks
    by_doc: dict[tuple[str, str], list[tuple[dict, int]]] = {}
    for i, m in enumerate(metadata):
        key = (m.get("source", ""), m.get("path", ""))
        by_doc.setdefault(key, []).append((m, i))

    doc_count = 0
    chunk_count = 0
    for (source, path), items in by_doc.items():
        # Try to read original file content for self-contained backup
        content = ""
        root = source_roots.get(source)
        if root:
            full_path = Path(root) / path
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

        if not content:
            # Fallback: concatenate chunk texts
            content = "\n\n".join(m.get("text", "") for m, _ in items)

        title = items[0][0].get("doc_title", "")
        preamble = items[0][0].get("doc_preamble", "")
        doc_id = upsert_document(conn, source, path, title, preamble, content)
        doc_count += 1

        chunk_metas = [m for m, _ in items]
        chunk_vecs_idx = [idx for _, idx in items]
        chunk_vecs = vectors[chunk_vecs_idx]
        upsert_chunks(conn, doc_id, chunk_metas, chunk_vecs)
        chunk_count += len(chunk_metas)

    log.info("Migrated %d documents, %d chunks", doc_count, chunk_count)

    # 3. Migrate source state
    state_path = cache / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            for name, sstate in state.items():
                save_source_state(conn, name, sstate)
            log.info("Migrated source state for %d sources", len(state))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to migrate state.json: %s", e)

    # 4. Migrate deprecated
    deprecated_path = cache / "deprecated.json"
    if deprecated_path.exists():
        try:
            deprecated = json.loads(deprecated_path.read_text())
            for path_key, reason in deprecated.items():
                save_deprecated(conn, path_key, reason)
            log.info("Migrated %d deprecated entries", len(deprecated))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to migrate deprecated.json: %s", e)

    # 5. Migrate transcript state
    transcripts_dir = cache / "transcripts"
    if transcripts_dir.is_dir():
        count = 0
        for state_file in transcripts_dir.glob("*.state.json"):
            try:
                tstate = json.loads(state_file.read_text())
                transcript_path = tstate.get("_original_path", "")
                if not transcript_path:
                    # Reconstruct a synthetic path from the hash
                    transcript_path = f"__migrated__{state_file.stem}"
                save_transcript_state(conn, transcript_path, tstate)
                count += 1
            except (json.JSONDecodeError, OSError):
                continue
        log.info("Migrated %d transcript state files", count)

    # 6. Build and cache HNSW index
    if chunk_count > 0:
        import hnswlib
        import tempfile

        dim = vectors.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=max(len(vectors), 1), ef_construction=200, M=16)
        index.add_items(vectors, list(range(len(vectors))))
        index.set_ef(50)

        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
            tmp_path = f.name
        index.save_index(tmp_path)
        blob = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)
        save_hnsw_cache(conn, blob, len(vectors))
        log.info("Built and cached HNSW index (%d vectors)", len(vectors))

    conn.close()
    log.info("Migration complete: %s", db_path)


def main() -> None:
    import yaml

    config_path = os.environ.get(
        "AGENTS_RAG_CONFIG",
        os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    if not Path(config_path).exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    config = yaml.safe_load(Path(config_path).read_text()) or {}
    cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/agents-rag"))

    source_roots: dict[str, str] = {}
    for s in config.get("sources", []):
        source_roots[s["name"]] = os.path.expanduser(s["path"])

    # Sessions output dir as root for "sessions" source
    sessions_cfg = config.get("sessions", {})
    if sessions_cfg.get("output_dir"):
        source_roots["sessions"] = os.path.expanduser(sessions_cfg["output_dir"])

    migrate(cache_dir, source_roots)


if __name__ == "__main__":
    main()
