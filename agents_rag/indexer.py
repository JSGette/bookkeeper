"""Embedding indexer with git-based delta reindex for repo sources, mtime for others."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunker import Chunk, SourceConfig, chunk_source
from .tagger import load_overrides, tag_file

log = logging.getLogger(__name__)

# Lazy-loaded to avoid slow import at module level
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

def _state_path(cache_dir: str) -> Path:
    return Path(cache_dir) / "state.json"


def load_state(cache_dir: str) -> dict:
    p = _state_path(cache_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(cache_dir: str, state: dict) -> None:
    p = _state_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


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
    # Deleted files
    for path in old_mtimes:
        if path not in new_mtimes:
            changed.add(path)
    return changed


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def _index_dir(cache_dir: str) -> Path:
    return Path(cache_dir)


def load_index(cache_dir: str) -> tuple[np.ndarray | None, list[dict] | None]:
    idx_dir = _index_dir(cache_dir)
    vec_path = idx_dir / "index.npz"
    meta_path = idx_dir / "chunks.json"
    if not vec_path.exists() or not meta_path.exists():
        return None, None
    try:
        vectors = np.load(vec_path)["vectors"]
        metadata = json.loads(meta_path.read_text())
        return vectors, metadata
    except Exception:
        return None, None


HNSW_THRESHOLD = 5000


def _save_hnsw(idx_dir: Path, vectors: np.ndarray) -> None:
    """Build and save hnswlib index if the library is available and corpus is large enough."""
    if len(vectors) < HNSW_THRESHOLD:
        return
    try:
        import hnswlib
    except ImportError:
        return
    dim = vectors.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
    index.add_items(vectors, list(range(len(vectors))))
    index.save_index(str(idx_dir / "index.hnsw"))
    log.info("Saved hnswlib index (%d vectors)", len(vectors))


def save_index(cache_dir: str, vectors: np.ndarray, metadata: list[dict]) -> None:
    idx_dir = _index_dir(cache_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(idx_dir / "index.npz", vectors=vectors)
    (idx_dir / "chunks.json").write_text(json.dumps(metadata, indent=2))
    _save_hnsw(idx_dir, vectors)


# ---------------------------------------------------------------------------
# Main reindex logic
# ---------------------------------------------------------------------------

def reindex_source(
    source_cfg: SourceConfig,
    cache_dir: str,
    use_gemma: bool = True,
) -> tuple[list[dict], np.ndarray, bool]:
    """Reindex a single source. Returns (metadata, vectors, index_changed).

    Uses git delta for git-tracked sources, mtime for others.
    """
    t0 = time.monotonic()
    root = Path(os.path.expanduser(source_cfg.path))
    state = load_state(cache_dir)
    source_state = state.get(source_cfg.name, {})

    # Chunk the full source (we need this to know what files exist)
    all_chunks = chunk_source(source_cfg)
    log.info("Chunked %s: %d chunks from %d files",
             source_cfg.name, len(all_chunks), len({c.path for c in all_chunks}))

    # Determine which files changed
    changed_files: set[str] | None = None  # None = full reindex
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
            # Check mtime as a fallback for uncommitted changes
            old_mtimes = source_state.get("mtimes", {})
            new_mtimes = _collect_mtimes(root, all_chunks)
            changed_files = _mtime_changed_files(old_mtimes, new_mtimes)
            if not changed_files:
                log.info("No changes detected for %s, skipping reindex", source_cfg.name)
                existing_vecs, existing_meta = load_index(cache_dir)
                if existing_vecs is not None and existing_meta is not None:
                    src_meta = [m for m in existing_meta if m.get("source") == source_cfg.name]
                    src_vecs_idx = [i for i, m in enumerate(existing_meta) if m.get("source") == source_cfg.name]
                    if src_vecs_idx:
                        return src_meta, existing_vecs[src_vecs_idx], False
    else:
        old_mtimes = source_state.get("mtimes", {})
        new_mtimes = _collect_mtimes(root, all_chunks)
        changed_files = _mtime_changed_files(old_mtimes, new_mtimes)
        if not changed_files:
            log.info("No mtime changes for %s", source_cfg.name)
            existing_vecs, existing_meta = load_index(cache_dir)
            if existing_vecs is not None and existing_meta is not None:
                src_meta = [m for m in existing_meta if m.get("source") == source_cfg.name]
                src_vecs_idx = [i for i, m in enumerate(existing_meta) if m.get("source") == source_cfg.name]
                if src_vecs_idx:
                    return src_meta, existing_vecs[src_vecs_idx], False

    # Apply tagging
    overrides = load_overrides(source_cfg.path)
    for chunk in all_chunks:
        desc, tags = tag_file(
            chunk.path, chunk.text, source_cfg.path, cache_dir, overrides, use_gemma=use_gemma,
        )
        chunk.description = desc
        chunk.tags = tags

    # If delta reindex, reuse existing embeddings for unchanged files
    if changed_files is not None and changed_files:
        existing_vecs, existing_meta = load_index(cache_dir)
        if existing_vecs is not None and existing_meta is not None:
            # Build lookup of existing embeddings by (source, path, section)
            existing_by_key: dict[tuple, int] = {}
            for i, m in enumerate(existing_meta):
                if m.get("source") == source_cfg.name:
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
                new_vecs = np.empty((0, existing_vecs.shape[1]))

            # Rebuild: reused vectors + new vectors
            parts = []
            if reuse_indices:
                parts.append(existing_vecs[reuse_indices])
            if len(new_vecs) > 0:
                parts.append(new_vecs)

            # Reorder metadata to match: reused chunks first, then new
            reused_chunks = [c for c in all_chunks if c.path not in changed_files
                            and (c.source, c.path, c.section_heading) in existing_by_key]
            ordered_chunks = reused_chunks + new_chunks_to_embed
            vectors = np.vstack(parts) if parts else np.empty((0, 384))
            metadata = [_chunk_to_meta(c) for c in ordered_chunks]

            _update_state(state, source_cfg, cache_dir, root, all_chunks, is_git)
            elapsed = time.monotonic() - t0
            log.info("Delta reindex %s: %d chunks in %.1fs", source_cfg.name, len(metadata), elapsed)
            return metadata, vectors, True

    # Full reindex
    if not all_chunks:
        _update_state(state, source_cfg, cache_dir, root, all_chunks, is_git)
        return [], np.empty((0, 384)), True

    log.info("Full embedding of %d chunks for %s", len(all_chunks), source_cfg.name)
    vectors = embed_texts([c.embedding_text for c in all_chunks])
    metadata = [_chunk_to_meta(c) for c in all_chunks]

    _update_state(state, source_cfg, cache_dir, root, all_chunks, is_git)
    elapsed = time.monotonic() - t0
    log.info("Full reindex %s: %d chunks in %.1fs", source_cfg.name, len(metadata), elapsed)
    return metadata, vectors, True


def _chunk_to_meta(c: Chunk) -> dict:
    return {
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


def _update_state(
    state: dict, source_cfg: SourceConfig, cache_dir: str,
    root: Path, chunks: list[Chunk], is_git: bool,
) -> None:
    source_state: dict = {}
    if is_git:
        head = _git_head(str(root))
        if head:
            source_state["commit_hash"] = head
    source_state["mtimes"] = _collect_mtimes(root, chunks)
    source_state["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state[source_cfg.name] = source_state
    save_state(cache_dir, state)


def reindex_all(
    sources: list[SourceConfig],
    cache_dir: str,
    use_gemma: bool = True,
    source_filter: str | None = None,
) -> tuple[list[dict], np.ndarray, int]:
    """Reindex all (or filtered) sources, merge into a single index. Returns (metadata, vectors, chunks_count)."""
    all_meta: list[dict] = []
    all_vecs: list[np.ndarray] = []

    existing_vecs, existing_meta = load_index(cache_dir)
    configured_names = {src.name for src in sources}

    for src in sources:
        if source_filter and src.name != source_filter:
            if existing_meta and existing_vecs is not None:
                for i, m in enumerate(existing_meta):
                    if m.get("source") == src.name:
                        all_meta.append(m)
                        all_vecs.append(existing_vecs[i:i+1])
            continue

        meta, vecs, _ = reindex_source(src, cache_dir, use_gemma=use_gemma)
        all_meta.extend(meta)
        if len(vecs) > 0:
            all_vecs.append(vecs if vecs.ndim == 2 else vecs.reshape(1, -1))

    # Preserve chunks from non-configured sources (e.g. "sessions" from transcript distillation)
    if existing_meta and existing_vecs is not None:
        for i, m in enumerate(existing_meta):
            src_name = m.get("source", "")
            if src_name not in configured_names:
                all_meta.append(m)
                all_vecs.append(existing_vecs[i:i+1])

    if all_vecs:
        merged_vecs = np.vstack(all_vecs)
    else:
        merged_vecs = np.empty((0, 384))

    save_index(cache_dir, merged_vecs, all_meta)
    return all_meta, merged_vecs, len(all_meta)
