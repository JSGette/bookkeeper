"""Embedding indexer with git-based delta reindex for repo sources, mtime for others."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .chunker import Chunk, SourceConfig, chunk_source, chunk_file, _should_skip
from . import db
from .db import (
    get_document_id,
    load_all_chunks,
    load_source_state,
    save_source_state,
    upsert_chunks,
    upsert_document,
)
from .tagger import load_overrides, tag_file, description_from_content, producer_fingerprint

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
                _model = SentenceTransformer(model_name, revision=os.environ.get("EMBEDDING_REVISION"))
    return _model


def embedding_profile() -> dict:
    revision = os.environ.get("EMBEDDING_REVISION")
    if _model is not None and revision is None:
        try:
            revision = _model[0].auto_model.config._commit_hash
        except (AttributeError, TypeError, IndexError):
            pass
    return {"model": os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            "revision": revision or "unresolved", "input_format": "bounded-evidence-v1",
            "tokenizer": getattr(getattr(_model, "tokenizer", None), "name_or_path", None),
            "max_seq_length": getattr(_model, "max_seq_length", None), "normalized": False}


def embedding_config() -> dict:
    """Stable requested identity, separate from lazy resolution of a model commit."""
    return {"model":os.environ.get("EMBEDDING_MODEL","BAAI/bge-small-en-v1.5"),
            "revision":os.environ.get("EMBEDDING_REVISION"),"input_format":"bounded-evidence-v1"}


def embedding_token_budget():
    model = _get_model()
    return (lambda text: len(model.tokenizer.encode(text, add_special_tokens=True))), int(model.max_seq_length)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    count, limit = embedding_token_budget()
    if any(count(text) > limit for text in texts):
        raise ValueError("Embedding input exceeds model token budget; split evidence before embedding")
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
    embedding_space: str | None = None,
    expected_hashes: dict | None = None,
) -> None:
    """Write chunks and their source documents into the DB."""
    root = Path(os.path.expanduser(source_cfg.path))

    docs_by_path: dict[str, list[tuple[dict, int]]] = {}
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        docs_by_path.setdefault(chunk.path, []).append((meta, i))

    for rel_path, items in docs_by_path.items():
        full_path = root / rel_path
        if not full_path.exists() and "\\" in rel_path:
            full_path = root / rel_path.replace("\\", "/")
        try:
            raw = full_path.read_bytes()
            content = raw.decode("utf-8")
            if expected_hashes and hashlib.sha256(raw).hexdigest() != expected_hashes[rel_path]:
                raise ValueError(f"Source changed while indexing: {rel_path}")
        except OSError as exc:
            raise ValueError(f"Cannot read source: {full_path}") from exc
        title = items[0][0].get("doc_title", "")
        preamble = items[0][0].get("doc_preamble", "")
        summary = summarize_document(
            content, title=title, use_gemma=use_gemma, path=rel_path,
        )
        chunk_metas = [m for m, _ in items]
        chunk_vecs = vectors[[idx for _, idx in items]]
        with db.transaction(conn):
            doc_id = upsert_document(conn, source_cfg.name, rel_path, title, preamble, content,
                                     summary=summary, completeness="source")
            upsert_chunks(conn, doc_id, chunk_metas, chunk_vecs, embedding_space=embedding_space)



# ---------------------------------------------------------------------------
# Main reindex logic
# ---------------------------------------------------------------------------

def reindex_source(
    source_cfg: SourceConfig,
    conn: sqlite3.Connection,
    use_gemma: bool = True,
) -> tuple[list[dict], np.ndarray, bool]:
    """Reindex a single source. Returns (metadata, vectors, index_changed).

    Hashes source content and producer settings; processes only changed documents.
    """
    root = Path(os.path.expanduser(source_cfg.path)).resolve()
    if not root.is_dir():
        raise ValueError(f"Source unavailable; retaining existing evidence: {root}")
    state = load_source_state(conn, source_cfg.name)
    with db._write_lock:
        existing_paths = {row[0] for row in conn.execute("SELECT path FROM documents WHERE source=?", (source_cfg.name,))}
    # Retain old path identities, including paths serialized by Windows versions.
    legacy_aliases = {path.replace("\\", "/"): path for path in sorted(existing_paths)}
    files = {}
    physical_paths = {}
    hashes = {}
    mtimes = {}
    def scan_error(error):
        raise error
    # os.walk with onerror is deliberate: pathlib.rglob can hide permission failures.
    paths = [Path(directory) / name for directory,_,names in os.walk(root,onerror=scan_error,followlinks=False)
             for name in names]
    for path in sorted(paths):
        if not path.is_file() or path.suffix.lower() not in source_cfg.file_types:
            continue
        if not path.resolve().is_relative_to(root):
            continue
        relative = path.relative_to(root).as_posix()
        if _should_skip(relative, source_cfg.skip):
            continue
        stable_path = relative if relative in existing_paths else legacy_aliases.get(relative,relative)
        files[stable_path] = path
        physical_paths[stable_path] = relative
        hashes[stable_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        mtimes[stable_path] = path.stat().st_mtime
    # Fingerprint configuration and producers, not timestamps alone. No tokenization/tagging on clean scans.
    overrides = load_overrides(source_cfg.path)
    pipeline = db._digest([vars(source_cfg), embedding_config(), use_gemma, overrides,
                           producer_fingerprint() if use_gemma else None, "index-v2"])
    old_hashes = state.get("content_hashes", {})
    changed = {p for p,h in hashes.items() if old_hashes.get(p) != h}
    removed = existing_paths - set(hashes)
    if state.get("pipeline") != pipeline:
        changed = set(files)
    profile = embedding_profile()
    if profile.get("revision") != "unresolved" and state.get("embedding_profile") != profile:
        changed = set(files)
    if not changed and not removed and state.get("content_hashes") is not None:
        meta, vectors = load_all_chunks(conn, source=source_cfg.name)
        return meta,vectors,False
    count, limit = embedding_token_budget() if changed else (None,None)
    chunks = []
    for path in sorted(changed):
        observed_content = files[path].read_text(encoding="utf-8")
        if source_cfg.min_lines and len(observed_content.splitlines()) < source_cfg.min_lines:
            continue
        pieces = chunk_file(source_cfg.name,root,physical_paths[path],source_cfg.max_chunk_tokens,
                                 max_whole_doc_tokens=source_cfg.max_whole_doc_tokens,
                                 count_tokens=count,embedding_max_tokens=limit)
        if observed_content.strip() and not pieces:
            raise ValueError(f"Chunking produced no evidence for nonempty source: {path}")
        for piece in pieces:
            piece.path = path
        chunks.extend(pieces)
    # Tag once per changed document and reuse the result for its segments.
    tagged = {}
    for chunk in chunks:
        if chunk.path not in tagged:
            text = files[chunk.path].read_text(encoding="utf-8")
            tagged[chunk.path] = tag_file(chunk.path,text,source_cfg.path,str(root),overrides,use_gemma=use_gemma)
        chunk.description,chunk.tags = tagged[chunk.path]
    metadata = [_chunk_to_meta(c) for c in chunks]
    if chunks:
        vectors = embed_texts([c.embedding_text for c in chunks])
        space = db.register_embedding_space(conn,embedding_profile(),vectors.shape[1])
        _persist_chunks_to_db(conn,source_cfg,chunks,vectors,metadata,use_gemma=use_gemma,
                              embedding_space=space,expected_hashes=hashes)
    # Empty/deleted documents lose only their live projection; immutable revisions survive.
    with db.transaction(conn):
        represented = {c.path for c in chunks}
        for path in removed | (changed - represented):
            conn.execute("DELETE FROM documents WHERE source=? AND path=?", (source_cfg.name,path))
        save_source_state(conn,source_cfg.name,{
            "content_hashes": hashes, "mtimes": mtimes,
            "pipeline": pipeline, "embedding_profile": embedding_profile(),
            "commit_hash": _git_head(str(root)) if (root / ".git").exists() else None,
            "indexed_at": db._now(),
        })
    meta,vectors = load_all_chunks(conn,source=source_cfg.name)
    return meta,vectors,True


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
        if not src.index_enabled:
            log.info("Skipping source %s: indexing disabled by config", src.name)
            continue
        reindex_source(src, conn, use_gemma=use_gemma)

    # Load the full merged state from DB
    all_meta, all_vecs = load_all_chunks(conn)
    return all_meta, all_vecs, len(all_meta)
