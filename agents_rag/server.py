"""MCP stdio server exposing knowledge base tools with background transcript watcher."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from fastmcp import FastMCP

from . import db
from .chunker import SourceConfig
from .index_generator import update_index_file
from .indexer import embed_texts, reindex_all
from .searcher import Searcher
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
            max_whole_doc_tokens=s.get("max_whole_doc_tokens", 1500),
        ))
    return sources


_USAGE_GUIDE = """\
# agents-rag Usage Guide

Query results are prior notes and hypotheses, not ground truth — they may be outdated.
Results contain **summaries**, not raw text. Follow this protocol:

1. Call `query_knowledge_base` with a short natural-language query describing the task.
2. Use `source` to filter by source name, or omit to search all sources.
3. Prefer recently indexed docs — check `indexed_at`; older docs are more likely stale.
4. **Document results** contain a `summary` field (a 2-3 sentence document-level overview) \
and a `matched_sections` list showing which sections triggered the match. Use the summary \
to decide relevance.
5. **Session results** contain a `session_summary` field (distilled overview with decisions \
and next steps). Multiple matching turns from the same session are collapsed into one \
result; `matched_turns` shows which turns matched.
6. If the summary is sufficient for your task, use it directly. If you need exact code, \
specific details, or full context, call `get_full_document` with the `path` and `source` \
fields from the result. Use `matched_sections` to know which part of the document to \
focus on.
7. If a result references source files, configs, or code paths, read those sources too — \
the doc is a pointer, the code is the truth.
8. If a result makes a factual claim about the codebase (e.g., "X blocks Y"), verify it \
against the current source code before repeating it.
9. If a result has `"deprecated": true`, do not use it as actionable context — treat it \
purely as historical background.
10. If during verification (step 8) a doc's claims are contradicted by the current source \
code, call `deprecate_doc` with the doc path and a brief reason explaining what changed.
11. At the end of a meaningful work session, call `save_session` with a structured summary: \
title, what was done, key decisions, relevant files, and next steps.
12. Previous chat sessions are automatically indexed in the background. When starting new \
work, `query_knowledge_base` will surface prior session notes alongside documentation.
"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config_override: dict | None = None) -> FastMCP:
    """Build and return the FastMCP app with all tools registered.

    Args:
        config_override: If provided, used instead of loading config from disk.
                         Useful for testing.
    """
    config = config_override if config_override is not None else _load_config()
    sources = _parse_sources(config)
    cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/agents-rag"))
    db_path = os.path.join(cache_dir, "agents-rag.db")
    use_gemma = config.get("use_gemma", True)
    configure_tagger(config)

    sessions_config = config.get("sessions", {})
    transcript_dirs = [os.path.expanduser(d) for d in sessions_config.get("transcript_dirs", [])]
    sessions_output_dir = os.path.expanduser(sessions_config.get("output_dir", "~/.cache/agents-rag/sessions"))
    gemma_every_n = sessions_config.get("gemma_every_n_turns", 3)
    watch_interval = sessions_config.get("watch_interval_minutes", 5)

    search_config = config.get("search", {})
    score_threshold_z = float(search_config.get("score_threshold_z", 0.8))
    min_results = int(search_config.get("min_results", 2))

    conn = db.init_db(db_path)
    searcher = Searcher(conn)
    index_lock = threading.Lock()

    # Index queue — serialises all embed-merge-save-reload work on a single thread
    index_q: queue.Queue[list[dict] | None] = queue.Queue()

    def _index_worker() -> None:
        """Drain the queue in batches: embed new chunks, merge into DB, reload searcher."""
        while True:
            batch = index_q.get()
            if batch is None:
                break
            while not index_q.empty():
                try:
                    more = index_q.get_nowait()
                    if more is not None:
                        batch.extend(more)
                except queue.Empty:
                    break
            try:
                with index_lock:
                    # Deduplicate against existing chunks
                    existing_meta, _ = db.load_all_chunks(conn)
                    existing_keys = {
                        (m["source"], m["path"], m.get("section_heading", ""))
                        for m in existing_meta
                    } if existing_meta else set()

                    batch = [
                        c for c in batch
                        if (c["source"], c["path"], c.get("section_heading", "")) not in existing_keys
                    ]
                    if not batch:
                        log.info("Index queue: all chunks already present, skipping")
                        continue

                    texts = [c["text"] for c in batch]
                    new_vecs = embed_texts(texts)

                    # Group by (source, path) and upsert
                    by_doc: dict[tuple[str, str], list[tuple[dict, int]]] = {}
                    for i, c in enumerate(batch):
                        key = (c["source"], c["path"])
                        by_doc.setdefault(key, []).append((c, i))

                    for (src, path), items in by_doc.items():
                        doc_id = db.get_document_id(conn, src, path)
                        if doc_id is None:
                            content = items[0][0].get("text", "")
                            title = items[0][0].get("doc_title", "")
                            doc_id = db.upsert_document(conn, src, path, title, "", content)
                        chunk_metas = [c for c, _ in items]
                        chunk_indices = [idx for _, idx in items]
                        chunk_vecs = new_vecs[chunk_indices]
                        # Append to existing chunks (don't delete old ones)
                        for j, cm in enumerate(chunk_metas):
                            vec_blob = chunk_vecs[j].astype(np.float32).tobytes()
                            conn.execute(
                                """INSERT INTO chunks
                                   (doc_id, section_heading, start_line, end_line, text, tags,
                                    description, doc_preamble, indexed_at, vector,
                                    session_summary, full_session_path, session_turn_count)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    doc_id,
                                    cm.get("section_heading", ""),
                                    cm.get("start_line", 0),
                                    cm.get("end_line", 0),
                                    cm.get("text", ""),
                                    json.dumps(cm.get("tags", [])),
                                    cm.get("description", ""),
                                    cm.get("doc_preamble", ""),
                                    cm.get("indexed_at", ""),
                                    vec_blob,
                                    cm.get("session_summary"),
                                    cm.get("full_session_path"),
                                    cm.get("session_turn_count"),
                                ),
                            )
                        conn.commit()

                    searcher.reload()
                log.info("Index queue: flushed %d chunks", len(batch))
            except Exception:
                log.exception("Index queue: flush failed")

    index_thread = threading.Thread(target=_index_worker, daemon=True, name="index-queue")
    index_thread.start()

    def _flush_transcript_chunks(chunks: list[dict]) -> None:
        index_q.put(chunks)

    mcp = FastMCP(name="agents-rag")

    if transcript_dirs and watch_interval > 0:
        watcher = TranscriptWatcher(
            transcript_dirs=transcript_dirs,
            conn=conn,
            output_dir=sessions_output_dir,
            interval_minutes=watch_interval,
            gemma_every_n=gemma_every_n,
            use_gemma=use_gemma,
            flush_fn=_flush_transcript_chunks,
        )
        watcher.start()

    # ------------------------------------------------------------------
    # MCP resource + tools (closures over app state)
    # ------------------------------------------------------------------

    @mcp.resource("agents-rag://usage-guide")
    def usage_guide() -> str:
        """Agent usage guidelines for consuming knowledge base results."""
        return _USAGE_GUIDE

    @mcp.tool
    def query_knowledge_base(query: str, max_results: int = 5, source: str | None = None) -> str:
        """Search the knowledge base for documentation relevant to a query.

        Results contain document-level summaries, not raw text. Call
        `get_full_document` with the result's `path` and `source` to read
        the complete content when the summary isn't enough.

        Args:
            query: Natural language query describing what you're looking for.
            max_results: Number of results to return (default 5).
            source: Filter by source name (as defined in config.yaml), or omit for both.
        """
        if not searcher._metadata:
            searcher.load()
        if not searcher._metadata:
            return json.dumps({"error": "Index is empty. Run reindex first."})

        t0 = time.monotonic()
        results = searcher.query(
            query, top_k=max_results, source=source,
            score_threshold_z=score_threshold_z, min_results=min_results,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        result_json = json.dumps(results, indent=2)
        try:
            db.log_query(conn, query, source, max_results, len(results), result_json, latency_ms)
        except Exception:
            log.warning("Failed to log query to audit table", exc_info=True)

        return result_json

    _reindex_status: dict = {"running": False, "last_result": None}

    def _reindex_worker(source_filter: str | None) -> None:
        t0 = time.monotonic()
        try:
            with index_lock:
                meta, vecs, count = reindex_all(
                    sources, conn, use_gemma=use_gemma, source_filter=source_filter,
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

    @mcp.tool
    def distill_transcripts() -> str:
        """Scan configured transcript directories and index new chat sessions.

        Runs automatically in the background every few minutes, but can be called
        manually to force an immediate scan. Each transcript is flushed to the index
        immediately after processing.
        """
        t0 = time.monotonic()
        total = distill_all_transcripts(
            transcript_dirs, conn, sessions_output_dir,
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

        session_chunks = sum(1 for m in (searcher._metadata or []) if m.get("source") == "sessions")
        if session_chunks > 0:
            result.append({
                "name": "sessions",
                "path": sessions_output_dir,
                "file_count": len({m["path"] for m in (searcher._metadata or []) if m.get("source") == "sessions"}),
                "chunk_count": session_chunks,
            })

        return json.dumps(result, indent=2)

    def _save_session_sync(
        title: str,
        summary: str,
        relevant_files: list[str] | None,
        decisions: list[str] | None,
        next_steps: list[str] | None,
    ) -> str:
        output_dir = sessions_output_dir
        os.makedirs(output_dir, exist_ok=True)

        slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:40].strip("-") or "session"
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

        content = "\n".join(parts)
        Path(filepath).write_text(content)

        index_q.put([{
            "source": "sessions",
            "path": filename,
            "doc_title": title,
            "section_heading": "",
            "start_line": 0,
            "end_line": 0,
            "tags": [],
            "description": summary[:200],
            "text": content,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }])

        return json.dumps({
            "status": "queued",
            "path": filepath,
        })

    @mcp.tool
    async def save_session(
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
        return await asyncio.to_thread(
            _save_session_sync, title, summary, relevant_files, decisions, next_steps
        )

    @mcp.tool
    def get_full_document(path: str, source: str | None = None) -> str:
        """Read the full content of a document from the knowledge base. Use when a
        query result chunk is relevant and you need the complete document.

        Args:
            path: The path field from a query result.
            source: The source field from the query result. If omitted, searches all sources.
        """
        # Try DB — search specific source or all sources
        if source:
            content = db.get_document_content(conn, source, path)
            if content:
                return content
        else:
            row = conn.execute(
                "SELECT content FROM documents WHERE path=? LIMIT 1", (path,),
            ).fetchone()
            if row:
                return row[0]

        # Fall back to filesystem for session files
        p = Path(path)
        if not p.is_absolute():
            p = Path(sessions_output_dir) / path
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

        return json.dumps({"error": f"Document not found: {path}"})

    @mcp.tool
    def deprecate_doc(path: str, reason: str = "", undeprecate: bool = False) -> str:
        """Mark a document as outdated so it is flagged as historical context in future queries.

        Args:
            path: Document path as it appears in query results (e.g. "datadog-agent/plans/go-module-bazel-migration-plan.md").
            reason: Why the doc is outdated (e.g. "dependency cycles were resolved in PR #1234").
            undeprecate: Set to true to remove the deprecation flag and restore the doc to normal status.
        """
        if undeprecate:
            removed = db.remove_deprecated(conn, path)
            searcher._deprecated = db.load_deprecated(conn)
            if removed:
                return json.dumps({"status": "restored", "path": path})
            return json.dumps({"status": "not_found", "path": path, "message": "Doc was not deprecated."})

        if not reason:
            return json.dumps({"error": "A reason is required when deprecating a doc."})

        db.save_deprecated(conn, path, reason)
        searcher._deprecated = db.load_deprecated(conn)
        return json.dumps({"status": "deprecated", "path": path, "reason": reason})

    @mcp.tool
    def query_audit_log(limit: int = 20) -> str:
        """Return recent knowledge base queries and their results for auditing.

        Args:
            limit: Maximum number of entries to return (default 20, most recent first).
        """
        entries = db.get_recent_queries(conn, limit=limit)
        return json.dumps(entries, indent=2)

    return mcp


def main():
    mcp = create_app()
    mcp.run()


if __name__ == "__main__":
    main()
