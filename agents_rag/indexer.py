"""Embedding indexer with git-based delta reindex for repo sources, mtime for others."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .chunker import Chunk, SourceConfig, chunk_source
from .db import (
    get_document_id,
    load_all_chunks,
    load_source_state,
    save_source_state,
    upsert_chunks,
    upsert_document,
)
from .tagger import load_overrides, tag_file, description_from_content

log = logging.getLogger(__name__)

_model = None
# Serialises the lazy SentenceTransformer init. Without this, concurrent
# embed_texts() callers (HTTP transport spawns one tool-call thread per
# Cursor window) each instantiate their own model, leak loky semaphores,
# and have been observed to segfault.
_model_lock = threading.Lock()
# Serialises model.encode(). PyTorch's MPS backend is not thread-safe;
# concurrent encode() from multiple HTTP worker threads segfaults in
# native code (verified: exit 139 with 8 concurrent workers). One lock
# also keeps CUDA streams sane and prevents oversubscribing the CPU
# inference path. Per-call cost dominates anyway (~30-50ms), so the
# serialisation tax is negligible compared to the search itself.
_encode_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
                _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    with _encode_lock:
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Document-level summarization
# ---------------------------------------------------------------------------

_DOC_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}

# Matches "## Summary\n\n<body>" terminated by next ## heading or EOF.
# Covers save_session, TranscriptWatcher distillation, and any authored doc
# that ships with its own summary.
_EXISTING_SUMMARY_RE = re.compile(
    r"^##\s+Summary\s*\n+(.+?)(?=\n##\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)


def extract_authored_summary(content: str) -> str:
    """Return the body of a '## Summary' section if the doc has one, else ''."""
    m = _EXISTING_SUMMARY_RE.search(content)
    if not m:
        return ""
    body = m.group(1).strip()
    # Reject section markers left in place by empty Summary bodies.
    if not body or body.startswith("##"):
        return ""
    return body[:500]


def summarize_document(content: str, title: str = "", use_gemma: bool = True, path: str = "") -> str:
    """Produce a 2-3 sentence document summary.

    Priority: authored ## Summary section > gemma > first-paragraph fallback.
    Authored summaries (e.g. save_session output, transcript distillation) are
    trusted and passed through verbatim — gemma adds no value and loses fidelity.
    """
    authored = extract_authored_summary(content)
    if authored:
        return authored

    if not use_gemma:
        return description_from_content(content)

    from . import ollama_client

    profile = ollama_client.pick_profile("index_summary", path=path or title)
    excerpt = ollama_client.truncate_for_profile(profile, content)
    messages = [
        {"role": "system", "content": (
            "Summarize this document in 2-3 sentences. Focus on what the document "
            "covers and its key conclusions or recommendations. Be specific — mention "
            "exact tools, concepts, or decisions. Output JSON with a single 'summary' key."
        )},
        {"role": "user", "content": excerpt},
    ]
    data = ollama_client.chat_json(profile, messages, _DOC_SUMMARY_SCHEMA, num_predict=512)
    if data:
        summary = str(data.get("summary", "")).strip()
        if summary:
            return summary
    log.warning("Document summarization failed, using fallback")

    return description_from_content(content)


# ---------------------------------------------------------------------------
# Git-based delta detection
# ---------------------------------------------------------------------------

def _git_head(repo_path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _git_is_ancestor(repo_path: str, ancestor: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
            cwd=repo_path, capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _git_changed_files(
    repo_path: str, old_hash: str, new_hash: str, file_types: list[str],
) -> list[str] | None:
    """Return list of changed files between two commits, or None to signal full reindex."""
    if not _git_is_ancestor(repo_path, old_hash):
        log.info("Cached commit %s is no longer ancestor of HEAD, forcing full reindex", old_hash[:8])
        return None
    try:
        cmd = ["git", "diff", "--name-only", "--diff-filter=ADMR", f"{old_hash}..{new_hash}",
               "--"] + [f"*{ext}" for ext in file_types]
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        return [f for f in result.stdout.strip().splitlines() if f]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mtime-based delta detection (for non-git sources)
# ---------------------------------------------------------------------------

def _collect_mtimes(root: Path, chunks: list[Chunk]) -> dict[str, float]:
    mtimes: dict[str, float] = {}
    for c in chunks:
        if c.path not in mtimes:
            try:
                mtimes[c.path] = (root / c.path).stat().st_mtime
            except OSError:
                pass
    return mtimes


def _mtime_changed_files(old_mtimes: dict[str, float], new_mtimes: dict[str, float]) -> set[str]:
    changed = set()
    for path, mtime in new_mtimes.items():
        if path not in old_mtimes or old_mtimes[path] != mtime:
            changed.add(path)
    for path in old_mtimes:
        if path not in new_mtimes:
            changed.add(path)
    return changed


# ---------------------------------------------------------------------------
# Chunk metadata helpers
# ---------------------------------------------------------------------------

def _chunk_to_meta(c: Chunk) -> dict:
    meta: dict = {
        "source": c.source,
        "path": c.path,
        "doc_title": c.doc_title,
        "section_heading": c.section_heading,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "tags": c.tags,
        "description": c.description,
        "text": c.text,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if c.doc_preamble:
        meta["doc_preamble"] = c.doc_preamble
    return meta


def _persist_chunks_to_db(
    conn: sqlite3.Connection,
    source_cfg: SourceConfig,
    chunks: list[Chunk],
    vectors: np.ndarray,
    metadata: list[dict],
    use_gemma: bool = True,
) -> None:
    """Write chunks and their source documents into the DB."""
    root = Path(os.path.expanduser(source_cfg.path))

    docs_by_path: dict[str, list[tuple[dict, int]]] = {}
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        docs_by_path.setdefault(chunk.path, []).append((meta, i))

    for rel_path, items in docs_by_path.items():
        full_path = root / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        title = items[0][0].get("doc_title", "")
        preamble = items[0][0].get("doc_preamble", "")
        summary = summarize_document(
            content, title=title, use_gemma=use_gemma, path=rel_path,
        )
        doc_id = upsert_document(
            conn, source_cfg.name, rel_path, title, preamble, content, summary=summary,
        )

        chunk_metas = [m for m, _ in items]
        chunk_vecs_idx = [idx for _, idx in items]
        chunk_vecs = vectors[chunk_vecs_idx]
        upsert_chunks(conn, doc_id, chunk_metas, chunk_vecs)


# ---------------------------------------------------------------------------
# Main reindex logic
# ---------------------------------------------------------------------------

def reindex_source(
    source_cfg: SourceConfig,
    conn: sqlite3.Connection,
    use_gemma: bool = True,
) -> tuple[list[dict], np.ndarray, bool]:
    """Reindex a single source. Returns (metadata, vectors, index_changed).

    Uses git delta for git-tracked sources, mtime for others.
    """
    t0 = time.monotonic()
    root = Path(os.path.expanduser(source_cfg.path))
    source_state = load_source_state(conn, source_cfg.name)

    all_chunks = chunk_source(source_cfg)
    log.info("Chunked %s: %d chunks from %d files",
             source_cfg.name, len(all_chunks), len({c.path for c in all_chunks}))

    changed_files: set[str] | None = None
    is_git = (root / ".git").exists()

    if is_git:
        old_hash = source_state.get("commit_hash")
        new_hash = _git_head(str(root))
        if old_hash and new_hash and old_hash != new_hash:
            delta = _git_changed_files(str(root), old_hash, new_hash, source_cfg.file_types)
            if delta is not None:
                changed_files = set(delta)
                log.info("Git delta: %d changed files since %s", len(changed_files), old_hash[:8])
        elif old_hash == new_hash:
            old_mtimes = source_state.get("mtimes", {})
            new_mtimes = _collect_mtimes(root, all_chunks)
            changed_files = _mtime_changed_files(old_mtimes, new_mtimes)
            if not changed_files:
                log.info("No changes detected for %s, skipping reindex", source_cfg.name)
                existing_meta, existing_vecs = load_all_chunks(conn, source=source_cfg.name)
                if existing_meta:
                    return existing_meta, existing_vecs, False
    else:
        old_mtimes = source_state.get("mtimes", {})
        new_mtimes = _collect_mtimes(root, all_chunks)
        changed_files = _mtime_changed_files(old_mtimes, new_mtimes)
        if not changed_files:
            log.info("No mtime changes for %s", source_cfg.name)
            existing_meta, existing_vecs = load_all_chunks(conn, source=source_cfg.name)
            if existing_meta:
                return existing_meta, existing_vecs, False

    # Apply tagging
    overrides = load_overrides(source_cfg.path)

    def _tag_one(chunk: Chunk) -> tuple[str, list[str]]:
        return tag_file(chunk.path, chunk.text, source_cfg.path, str(root), overrides, use_gemma=use_gemma)

    if use_gemma and len(all_chunks) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_tag_one, all_chunks))
        for chunk, (desc, tags) in zip(all_chunks, results):
            chunk.description = desc
            chunk.tags = tags
    else:
        for chunk in all_chunks:
            chunk.description, chunk.tags = _tag_one(chunk)

    # Delta reindex: reuse existing embeddings for unchanged files
    if changed_files is not None and changed_files:
        existing_meta, existing_vecs = load_all_chunks(conn, source=source_cfg.name)
        if existing_meta:
            existing_by_key: dict[tuple, int] = {}
            for i, m in enumerate(existing_meta):
                key = (m["source"], m["path"], m.get("section_heading", ""))
                existing_by_key[key] = i

            new_chunks_to_embed = []
            reuse_indices = []
            for chunk in all_chunks:
                key = (chunk.source, chunk.path, chunk.section_heading)
                if chunk.path not in changed_files and key in existing_by_key:
                    reuse_indices.append(existing_by_key[key])
                else:
                    new_chunks_to_embed.append(chunk)

            if new_chunks_to_embed:
                log.info("Embedding %d new/changed chunks (reusing %d)",
                         len(new_chunks_to_embed), len(reuse_indices))
                new_vecs = embed_texts([c.embedding_text for c in new_chunks_to_embed])
            else:
                new_vecs = np.empty((0, existing_vecs.shape[1] if len(existing_vecs) > 0 else 384))

            parts = []
            if reuse_indices:
                parts.append(existing_vecs[reuse_indices])
            if len(new_vecs) > 0:
                parts.append(new_vecs)

            reused_chunks = [c for c in all_chunks if c.path not in changed_files
                            and (c.source, c.path, c.section_heading) in existing_by_key]
            ordered_chunks = reused_chunks + new_chunks_to_embed
            vectors = np.vstack(parts) if parts else np.empty((0, 384))
            metadata = [_chunk_to_meta(c) for c in ordered_chunks]

            _persist_chunks_to_db(conn, source_cfg, ordered_chunks, vectors, metadata, use_gemma=use_gemma)
            _update_state(conn, source_cfg, root, all_chunks, is_git)
            elapsed = time.monotonic() - t0
            log.info("Delta reindex %s: %d chunks in %.1fs", source_cfg.name, len(metadata), elapsed)
            return metadata, vectors, True

    # Full reindex
    if not all_chunks:
        _update_state(conn, source_cfg, root, all_chunks, is_git)
        return [], np.empty((0, 384)), True

    log.info("Full embedding of %d chunks for %s", len(all_chunks), source_cfg.name)
    vectors = embed_texts([c.embedding_text for c in all_chunks])
    metadata = [_chunk_to_meta(c) for c in all_chunks]

    _persist_chunks_to_db(conn, source_cfg, all_chunks, vectors, metadata, use_gemma=use_gemma)
    _update_state(conn, source_cfg, root, all_chunks, is_git)
    elapsed = time.monotonic() - t0
    log.info("Full reindex %s: %d chunks in %.1fs", source_cfg.name, len(metadata), elapsed)
    return metadata, vectors, True


def _update_state(
    conn: sqlite3.Connection, source_cfg: SourceConfig,
    root: Path, chunks: list[Chunk], is_git: bool,
) -> None:
    source_state: dict = {}
    if is_git:
        head = _git_head(str(root))
        if head:
            source_state["commit_hash"] = head
    source_state["mtimes"] = _collect_mtimes(root, chunks)
    source_state["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_source_state(conn, source_cfg.name, source_state)


def reindex_all(
    sources: list[SourceConfig],
    conn: sqlite3.Connection,
    use_gemma: bool = True,
    source_filter: str | None = None,
) -> tuple[list[dict], np.ndarray, int]:
    """Reindex all (or filtered) sources, merge results. Returns (metadata, vectors, chunks_count)."""
    configured_names = {src.name for src in sources}

    for src in sources:
        if source_filter and src.name != source_filter:
            continue
        reindex_source(src, conn, use_gemma=use_gemma)

    # Load the full merged state from DB
    all_meta, all_vecs = load_all_chunks(conn)
    return all_meta, all_vecs, len(all_meta)
