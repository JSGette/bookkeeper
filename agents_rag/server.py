"""MCP stdio server exposing knowledge base tools with background transcript watcher."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from fastmcp import FastMCP

from .chunker import SourceConfig
from .index_generator import update_index_file
from .indexer import embed_texts, load_index, reindex_all, save_index
from .searcher import Searcher, load_deprecated, save_deprecated
from .tagger import configure as configure_tagger
from .transcript import TranscriptWatcher, distill_all_transcripts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
]


def _load_config() -> dict:
    config_path = os.environ.get("AGENTS_RAG_CONFIG")
    search = [config_path] if config_path else _DEFAULT_CONFIG_PATHS
    for p in search:
        if p and Path(p).exists():
            log.info("Loading config from %s", p)
            return yaml.safe_load(Path(p).read_text()) or {}
    log.warning("No config found, using defaults")
    return {}


def _parse_sources(config: dict) -> list[SourceConfig]:
    sources = []
    for s in config.get("sources", []):
        sources.append(SourceConfig(
            name=s["name"],
            path=os.path.expanduser(s["path"]),
            skip=s.get("skip", []),
            min_lines=s.get("min_lines", 0),
            generate_index=s.get("generate_index", False),
            file_types=s.get("file_types", [".md"]),
            max_chunk_tokens=s.get("max_chunk_tokens", 512),
        ))
    return sources


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

config = _load_config()
sources = _parse_sources(config)
cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/agents-rag"))
use_gemma = config.get("use_gemma", True)
configure_tagger(config)

sessions_config = config.get("sessions", {})
transcript_dirs = [os.path.expanduser(d) for d in sessions_config.get("transcript_dirs", [])]
sessions_output_dir = os.path.expanduser(sessions_config.get("output_dir", "~/.cache/agents-rag/sessions"))
gemma_every_n = sessions_config.get("gemma_every_n_turns", 3)
watch_interval = sessions_config.get("watch_interval_minutes", 5)

searcher = Searcher(cache_dir)
index_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Index queue — serialises all embed-merge-save-reload work on a single thread
# ---------------------------------------------------------------------------

_index_q: queue.Queue[list[dict]] = queue.Queue()


def _index_worker() -> None:
    """Drain the queue in batches: embed new chunks, merge, save, reload."""
    while True:
        batch = _index_q.get()
        if batch is None:
            break
        # Drain anything else that arrived while we were waiting
        while not _index_q.empty():
            try:
                more = _index_q.get_nowait()
                if more is not None:
                    batch.extend(more)
            except queue.Empty:
                break
        try:
            with index_lock:
                existing_vecs, existing_meta = load_index(cache_dir)
                # Dedup: drop chunks already present in the index
                if existing_meta:
                    existing_keys = {
                        (m["source"], m["path"], m.get("section_heading", ""))
                        for m in existing_meta
                    }
                    batch = [
                        c for c in batch
                        if (c["source"], c["path"], c.get("section_heading", "")) not in existing_keys
                    ]
                if not batch:
                    log.info("Index queue: all chunks already present, skipping")
                    continue
                texts = [c["text"] for c in batch]
                new_vecs = embed_texts(texts)
                if existing_vecs is not None and existing_meta is not None:
                    merged_vecs = np.vstack([existing_vecs, new_vecs])
                    merged_meta = existing_meta + batch
                else:
                    merged_vecs = new_vecs
                    merged_meta = batch
                save_index(cache_dir, merged_vecs, merged_meta)
                searcher.reload()
            log.info("Index queue: flushed %d chunks", len(batch))
        except Exception:
            log.exception("Index queue: flush failed")


_index_thread = threading.Thread(target=_index_worker, daemon=True, name="index-queue")
_index_thread.start()

mcp = FastMCP(name="agents-rag")

# Start background watcher if configured
if transcript_dirs and watch_interval > 0:
    _watcher = TranscriptWatcher(
        transcript_dirs=transcript_dirs,
        cache_dir=cache_dir,
        output_dir=sessions_output_dir,
        interval_minutes=watch_interval,
        gemma_every_n=gemma_every_n,
        use_gemma=use_gemma,
        searcher=searcher,
        index_lock=index_lock,
    )
    _watcher.start()

_USAGE_GUIDE = """\
# agents-rag Usage Guide

Query results are prior notes and hypotheses, not ground truth — they may be outdated.
Follow this protocol when consuming results:

1. Call `query_knowledge_base` with a short natural-language query describing the task.
2. Use source to filter by source name, or omit to search all sources.
3. Prefer recently indexed docs — check the `indexed_at` field; older docs are more likely stale.
4. If a retrieved doc references source files, configs, or code paths, read those sources \
too and add them to your working context — the doc is a pointer, the code is the truth.
5. If a retrieved doc makes a factual claim about the codebase (e.g., "X blocks Y", \
"module Z has a cycle"), verify it against the current source code before repeating it.
6. If a result has "deprecated": true, do not use it as actionable context — treat it \
purely as historical background. Mention it only if the user asks about past decisions \
or evolution of the codebase. Never repeat its claims as current facts.
7. If during verification (step 5) a doc's claims are contradicted by the current source \
code, call `deprecate_doc` with the doc path and a brief reason explaining what changed. \
This keeps the knowledge base accurate for future queries.
8. At the end of a meaningful work session (implementation, investigation, debugging), \
call `save_session` with a structured summary: title, what was done, key decisions, \
files touched, and next steps. This lets the next chat resume without re-discovering context.
9. Previous chat sessions are automatically indexed in the background. When starting new \
work, `query_knowledge_base` will surface prior session notes alongside documentation.
10. Some results include `text_preview` instead of full `text` when the chunk is large. \
If the preview looks relevant, use the `path`, `start_line`, and `end_line` fields to \
read the original file directly (e.g., via the Read tool). For session chunks where the \
path is a generated summary, the preview is usually sufficient context.
"""


@mcp.resource("agents-rag://usage-guide")
def usage_guide() -> str:
    """Agent usage guidelines for consuming knowledge base results."""
    return _USAGE_GUIDE


@mcp.tool
def query_knowledge_base(query: str, top_k: int = 5, source: str | None = None) -> str:
    """Search the knowledge base for documentation relevant to a query.

    Args:
        query: Natural language query describing what you're looking for.
        top_k: Number of results to return (default 5).
        source: Filter by source name (as defined in config.yaml), or omit for both.
    """
    if not searcher._metadata:
        searcher.load()
    if not searcher._metadata:
        return json.dumps({"error": "Index is empty. Run reindex first."})

    results = searcher.query(query, top_k=top_k, source=source)
    return json.dumps(results, indent=2)


_reindex_status: dict = {"running": False, "last_result": None}


def _reindex_worker(source_filter: str | None) -> None:
    """Background worker for reindex."""
    t0 = time.monotonic()
    try:
        with index_lock:
            meta, vecs, count = reindex_all(
                sources, cache_dir, use_gemma=use_gemma, source_filter=source_filter,
            )
            searcher.reload()

        index_updated = False
        for src in sources:
            if src.generate_index and (source_filter is None or source_filter == src.name):
                src_meta = [m for m in meta if m.get("source") == src.name]
                if update_index_file(src.path, src_meta):
                    index_updated = True

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _reindex_status["last_result"] = {
            "status": "completed",
            "chunks_indexed": count,
            "elapsed_ms": elapsed_ms,
            "index_updated": index_updated,
        }
        log.info("Reindex completed: %d chunks in %dms", count, elapsed_ms)
    except Exception as exc:
        _reindex_status["last_result"] = {"status": "failed", "error": str(exc)}
        log.exception("Reindex failed")
    finally:
        _reindex_status["running"] = False


@mcp.tool
def reindex(source: str | None = None) -> str:
    """Reindex documentation sources. Supports incremental git-based delta reindex.

    Runs in the background — returns immediately. Call again to check status.

    Args:
        source: Reindex a specific source by name, or omit to reindex all.
    """
    if _reindex_status["running"]:
        return json.dumps({"status": "already_running"})

    _reindex_status["running"] = True
    _reindex_status["last_result"] = None
    t = threading.Thread(target=_reindex_worker, args=(source,), daemon=True, name="reindex")
    t.start()
    return json.dumps({"status": "started", "source": source or "all"})


def _flush_transcript_chunks(chunks: list[dict]) -> None:
    """Enqueue transcript chunks for async embedding and index merge."""
    _index_q.put(chunks)


@mcp.tool
def distill_transcripts() -> str:
    """Scan configured transcript directories and index new chat sessions.

    Runs automatically in the background every few minutes, but can be called
    manually to force an immediate scan. Each transcript is flushed to the index
    immediately after processing.
    """
    t0 = time.monotonic()
    total = distill_all_transcripts(
        transcript_dirs, cache_dir, sessions_output_dir,
        gemma_every_n=gemma_every_n, use_gemma=use_gemma,
        flush_fn=_flush_transcript_chunks,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return json.dumps({
        "new_turns_indexed": total,
        "elapsed_ms": elapsed_ms,
    })


@mcp.tool
def list_sources() -> str:
    """List all configured documentation sources and their stats."""
    if not searcher._metadata:
        searcher.load()

    result = []
    for src in sources:
        file_count = len({
            m["path"] for m in (searcher._metadata or [])
            if m.get("source") == src.name
        })
        chunk_count = sum(
            1 for m in (searcher._metadata or [])
            if m.get("source") == src.name
        )
        result.append({
            "name": src.name,
            "path": src.path,
            "file_count": file_count,
            "chunk_count": chunk_count,
        })

    # Include session stats
    session_chunks = sum(1 for m in (searcher._metadata or []) if m.get("source") == "sessions")
    if session_chunks > 0:
        result.append({
            "name": "sessions",
            "path": sessions_output_dir,
            "file_count": len({m["path"] for m in (searcher._metadata or []) if m.get("source") == "sessions"}),
            "chunk_count": session_chunks,
        })

    return json.dumps(result, indent=2)


@mcp.tool
def save_session(
    title: str,
    summary: str,
    relevant_files: list[str] | None = None,
    decisions: list[str] | None = None,
    next_steps: list[str] | None = None,
) -> str:
    """Persist a structured summary of the current work session for future retrieval.

    Call this at the end of a meaningful work session (implementation, investigation,
    debugging) so the next chat can resume without re-discovering context.

    Args:
        title: Short name for the task (e.g. "Add BM25 hybrid search").
        summary: 1-2 sentence description of what was accomplished.
        relevant_files: List of files central to the work.
        decisions: Key technical decisions made and their rationale.
        next_steps: What remains to be done.
    """
    t0 = time.monotonic()
    output_dir = sessions_output_dir
    os.makedirs(output_dir, exist_ok=True)

    slug = title.lower()
    slug = __import__("re").sub(r"[^a-z0-9]+", "-", slug)[:40].strip("-") or "session"
    date_str = time.strftime("%Y%m%d", time.gmtime())
    filename = f"{date_str}_{slug}.md"
    filepath = os.path.join(output_dir, filename)

    parts = [
        f"# {title}",
        "",
        f"**Saved**: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "## Summary",
        "",
        summary,
        "",
    ]

    if decisions:
        parts.extend(["## Decisions", ""])
        for d in decisions:
            parts.append(f"- {d}")
        parts.append("")

    if relevant_files:
        parts.extend(["## Relevant Files", ""])
        for f in relevant_files:
            parts.append(f"- `{f}`")
        parts.append("")

    if next_steps:
        parts.extend(["## Next Steps", ""])
        for s in next_steps:
            parts.append(f"- {s}")
        parts.append("")

    Path(filepath).write_text("\n".join(parts))

    # Enqueue for async indexing
    _index_q.put([{
        "source": "sessions",
        "path": filename,
        "doc_title": title,
        "section_heading": "",
        "start_line": 0,
        "end_line": 0,
        "tags": [],
        "description": summary[:200],
        "text": "\n".join(parts),
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }])

    return json.dumps({
        "status": "queued",
        "path": filepath,
    })


@mcp.tool
def deprecate_doc(path: str, reason: str = "", undeprecate: bool = False) -> str:
    """Mark a document as outdated so it is flagged as historical context in future queries.

    Args:
        path: Document path as it appears in query results (e.g. "datadog-agent/plans/go-module-bazel-migration-plan.md").
        reason: Why the doc is outdated (e.g. "dependency cycles were resolved in PR #1234").
        undeprecate: Set to true to remove the deprecation flag and restore the doc to normal status.
    """
    deprecated = load_deprecated(cache_dir)
    if undeprecate:
        removed = deprecated.pop(path, None)
        save_deprecated(cache_dir, deprecated)
        searcher._deprecated = deprecated
        if removed is not None:
            return json.dumps({"status": "restored", "path": path})
        return json.dumps({"status": "not_found", "path": path, "message": "Doc was not deprecated."})

    if not reason:
        return json.dumps({"error": "A reason is required when deprecating a doc."})

    deprecated[path] = reason
    save_deprecated(cache_dir, deprecated)
    searcher._deprecated = deprecated
    return json.dumps({"status": "deprecated", "path": path, "reason": reason})


def main():
    mcp.run()


if __name__ == "__main__":
    main()
