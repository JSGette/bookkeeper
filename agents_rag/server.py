"""MCP stdio server exposing knowledge base tools with background transcript watcher."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from fastmcp import FastMCP

from . import db, reranker
from .config import load_config, parse_sources, validate_config
from .distill import distill_blob as _distill_blob, prepare_cloud_context as _prepare_cloud_context
from .index_generator import update_index_file
from .indexer import embed_texts, reindex_all
from . import ollama_client
from .searcher import Searcher
from .tagger import configure as configure_tagger
from .transcript import TranscriptWatcher, distill_all_transcripts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _weak_match_hint(meta: dict) -> str:
    """Hint string returned to the agent when the top result is low-confidence
    or there are no results. Phrased to discourage paraphrase loops."""
    if meta["confidence"] == "none":
        return (
            "No results passed the relevance threshold. The corpus likely does "
            "not cover this topic — do NOT keep rephrasing. Read source files "
            "directly, or call `status` to verify the index is loaded."
        )
    return (
        "Top result confidence is low. The retriever could not find a strong "
        "match. Avoid rephrasing the same query — open the highest-scoring "
        "result with `get_full_document` to confirm relevance, or read source "
        "files directly. Repeated paraphrases of an unanswerable query show up "
        "in the audit log and waste both our time."
    )


def _ts_to_epoch(timestamp: str) -> float:
    """Parse an ISO-8601 'YYYY-MM-DDTHH:MM:SSZ' timestamp into epoch seconds."""
    try:
        return time.mktime(time.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError):
        return time.time()


def _load_usage_guide() -> str:
    path = Path(__file__).resolve().parent.parent / "docs" / "knowledge-base-protocol.md"
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("Failed to read usage guide at %s: %s", path, exc)
    log.warning("Usage guide missing at %s", path)
    return "# bookkeeper\n\nUsage guide file not found.\n"


_USAGE_GUIDE = _load_usage_guide()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config_override: dict | None = None) -> FastMCP:
    """Build and return the FastMCP app with all tools registered.

    Args:
        config_override: If provided, used instead of loading config from disk.
                         Useful for testing.
    """
    config = config_override if config_override is not None else load_config()
    errors = validate_config(config)
    if errors:
        for e in errors:
            log.error("Config error: %s", e)
    sources = parse_sources(config)
    cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/bookkeeper"))
    db_path = os.path.join(cache_dir, "bookkeeper.db")
    use_gemma = config.get("use_gemma", True)
    ollama_client.configure(config)
    configure_tagger(config)
    runtime_distill = config.get("runtime_distill", {})
    runtime_distill_enabled = bool(runtime_distill.get("enabled", True))
    max_distill_docs = int(runtime_distill.get("max_distill_docs", 3))

    sessions_config = config.get("sessions", {})
    transcript_dirs = [os.path.expanduser(d) for d in sessions_config.get("transcript_dirs", [])]
    sessions_output_dir = os.path.expanduser(sessions_config.get("output_dir", "~/.cache/bookkeeper/sessions"))
    gemma_every_n = sessions_config.get("gemma_every_n_turns", 3)
    watch_interval = sessions_config.get("watch_interval_minutes", 5)

    search_config = config.get("search", {})
    score_threshold_z = float(search_config.get("score_threshold_z", 0.8))
    min_results = int(search_config.get("min_results", 2))
    rrf_k = int(search_config.get("rrf_k", 10))
    dedupe_threshold = float(search_config.get("dedupe_similarity_threshold", 0.88))
    dedupe_window_seconds = int(search_config.get("dedupe_window_seconds", 120))

    reranker_config = config.get("reranker", {})
    reranker_enabled = bool(reranker_config.get("enabled", True))
    reranker_model = reranker_config.get("model", "BAAI/bge-reranker-base")
    reranker_pool_multiplier = int(reranker_config.get("pool_multiplier", 6))

    conn = db.init_db(db_path)
    read_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    read_conn.execute("PRAGMA busy_timeout=30000")
    read_conn.execute("PRAGMA foreign_keys=ON")
    try:
        read_conn.execute("PRAGMA query_only=ON")
    except Exception:
        # Older SQLite builds may not support query_only; reads still work.
        pass
    _read_lock = threading.Lock()

    # Checkpoint+truncate the WAL on clean shutdown so SIGTERMs from
    # launchd/systemd don't leave 100MB+ -wal files behind. kill -9 still
    # skips this (atexit doesn't run), but normal `daemon stop` will.
    import atexit
    def _checkpoint_and_close() -> None:
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            read_conn.close()
        except Exception:
            pass
    atexit.register(_checkpoint_and_close)

    searcher = Searcher(
        conn,
        reranker_enabled=reranker_enabled,
        reranker_model=reranker_model,
        reranker_pool_multiplier=reranker_pool_multiplier,
        rrf_k=rrf_k,
    )

    # Warm the cross-encoder in the background so the first user query doesn't
    # eat the ~60s cold-start latency we observed in the audit log.
    if reranker_enabled:
        threading.Thread(
            target=reranker.warm, args=(reranker_model,),
            daemon=True, name="reranker-warmup",
        ).start()

    # Index queue — serialises all embed-merge-save-reload work on a single thread
    index_q: queue.Queue[list[dict] | None] = queue.Queue()

    def _index_worker() -> None:
        """Drain the queue in batches: embed new chunks, merge into DB, reload searcher.

        Session chunks (source == "sessions") are upserted per document — there is
        exactly one chunk per session, and it must replace prior ones on every tick.
        Non-session chunks retain append-with-dedup semantics.
        """
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
                session_chunks = [c for c in batch if c.get("source") == "sessions"]
                other_chunks = [c for c in batch if c.get("source") != "sessions"]

                if other_chunks:
                    existing_meta, _ = db.load_all_chunks(conn)
                    existing_keys = {
                        (m["source"], m["path"], m.get("section_heading", ""))
                        for m in existing_meta
                    } if existing_meta else set()
                    other_chunks = [
                        c for c in other_chunks
                        if (c["source"], c["path"], c.get("section_heading", "")) not in existing_keys
                    ]

                effective = session_chunks + other_chunks
                if not effective:
                    log.info("Index queue: nothing to flush")
                    continue

                texts = [c.get("embedding_text") or c["text"] for c in effective]
                new_vecs = embed_texts(texts)

                by_doc: dict[tuple[str, str], list[tuple[dict, int]]] = {}
                for i, c in enumerate(effective):
                    key = (c["source"], c["path"])
                    by_doc.setdefault(key, []).append((c, i))

                for (src, path), items in by_doc.items():
                    chunk_metas = [c for c, _ in items]
                    chunk_indices = [idx for _, idx in items]
                    chunk_vecs = new_vecs[chunk_indices]
                    title = items[0][0].get("doc_title", "")

                    if src == "sessions":
                        summary = items[0][0].get("session_summary", "")
                        content = items[0][0].get("text", "")
                        doc_id = db.upsert_document(conn, src, path, title, "", content, summary=summary)
                        db.upsert_chunks(conn, doc_id, chunk_metas, chunk_vecs)
                    else:
                        doc_id = db.get_document_id(conn, src, path)
                        if doc_id is None:
                            content = items[0][0].get("text", "")
                            doc_id = db.upsert_document(conn, src, path, title, "", content)
                        db.append_chunks(conn, doc_id, chunk_metas, chunk_vecs)

                searcher.reload()
                log.info("Index queue: flushed %d chunks (%d session, %d other)",
                         len(effective), len(session_chunks), len(other_chunks))
            except Exception:
                log.exception("Index queue: flush failed")

    index_thread = threading.Thread(target=_index_worker, daemon=True, name="index-queue")
    index_thread.start()

    def _flush_transcript_chunks(chunks: list[dict]) -> None:
        index_q.put(chunks)

    # Eager load: in HTTP transport mode multiple Cursor windows race the
    # lazy first-call `searcher.load()` and serialise on the SQLite mutex,
    # which can pin queries past Cursor's MCP timeout. Pay the ~2s build
    # cost once at startup instead.
    try:
        searcher.load()
    except Exception:
        log.exception("Searcher eager-load failed; falling back to lazy load")

    mcp = FastMCP(name="bookkeeper")

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

    @mcp.resource("bookkeeper://usage-guide")
    def usage_guide() -> str:
        """Agent usage guidelines for consuming knowledge base results."""
        return _USAGE_GUIDE

    @mcp.tool
    def query_knowledge_base(query: str, max_results: int = 5, source: str | None = None) -> str:
        """Search the knowledge base for documentation relevant to a query.

        Returns a JSON envelope: {"results": [...], "meta": {...}}. Each result
        carries a `confidence` band ("high"|"medium"|"low"). The `meta` block
        reports `scoring` ("rerank"|"rrf"|"cached"), overall `confidence`,
        `weak_match` flag, and a `hint` string when the search is weak or the
        query is a near-duplicate of a recent one — read the hint and act on it
        instead of rephrasing.

        Results contain document-level summaries, not raw text. Treat the
        `summary` / `session_summary` as a relevance hint only — call
        `get_full_document` with the result's `path` and `source` before
        quoting facts, naming functions, citing decisions, continuing prior
        work, or writing code based on a result. See the
        `bookkeeper://usage-guide` resource for the full protocol.

        Args:
            query: Natural language query describing what you're looking for.
            max_results: Number of results to return (default 5).
            source: Filter by source name (as defined in config.yaml), or omit for both.
        """
        if not searcher._metadata:
            searcher.load()
        if not searcher._metadata:
            return json.dumps({
                "results": [],
                "meta": {"error": "Index is empty. Run reindex first."},
            })

        t0 = time.monotonic()
        query_vec = searcher.embed_query(query)
        query_blob = query_vec.astype(np.float32).tobytes()

        cached = _check_recent_duplicate(query_vec, source)
        if cached is not None:
            envelope = _build_cached_envelope(query, cached)
            payload = json.dumps(envelope, indent=2)
            latency_ms = int((time.monotonic() - t0) * 1000)
            try:
                db.log_query(
                    conn, query, source, max_results,
                    len(envelope["results"]), payload, latency_ms,
                    reranker_used=cached["reranker_used"],
                    rrf_k=cached["rrf_k"],
                    score_kind="cached",
                    query_embedding=query_blob,
                )
            except Exception:
                log.warning("Failed to log cached query to audit table", exc_info=True)
            return payload

        results, query_meta = searcher.query_with_meta(
            query, top_k=max_results, source=source,
            score_threshold_z=score_threshold_z, min_results=min_results,
            query_embedding=query_vec,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        envelope = {
            "results": results,
            "meta": {
                "scoring": query_meta["score_kind"],
                "rrf_k": query_meta["rrf_k"],
                "reranker_used": query_meta["reranker_used"],
                "confidence": query_meta["confidence"],
                "weak_match": query_meta["weak_match"],
                "hint": _weak_match_hint(query_meta) if query_meta["weak_match"] else None,
            },
        }
        payload = json.dumps(envelope, indent=2)
        try:
            db.log_query(
                conn, query, source, max_results, len(results), payload, latency_ms,
                reranker_used=query_meta["reranker_used"],
                rrf_k=query_meta["rrf_k"],
                score_kind=query_meta["score_kind"],
                query_embedding=query_blob,
            )
        except Exception:
            log.warning("Failed to log query to audit table", exc_info=True)

        return payload

    def _check_recent_duplicate(query_vec: np.ndarray, source_filter: str | None) -> dict | None:
        """Return the most similar recent query above the dedupe threshold, or None."""
        try:
            with _read_lock:
                recent = db.get_recent_query_embeddings(
                    read_conn, source_filter,
                    since_seconds=dedupe_window_seconds, limit=5,
                )
        except Exception:
            log.warning("Failed to read recent query embeddings", exc_info=True)
            return None
        if not recent:
            return None

        q = query_vec.astype(np.float32)
        q_norm = float(np.linalg.norm(q)) or 1.0
        best: tuple[float, dict] | None = None
        for row in recent:
            blob = row.get("query_embedding")
            if not blob:
                continue
            try:
                v = np.frombuffer(blob, dtype=np.float32)
            except ValueError:
                continue
            if v.shape != q.shape:
                continue
            v_norm = float(np.linalg.norm(v)) or 1.0
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            if sim >= dedupe_threshold and (best is None or sim > best[0]):
                best = (sim, {**row, "similarity": round(sim, 4)})
        return best[1] if best else None

    def _build_cached_envelope(current_query: str, cached: dict) -> dict:
        """Reconstruct a response envelope from a prior logged query."""
        try:
            prior_payload = json.loads(cached["results_json"])
        except (json.JSONDecodeError, TypeError):
            prior_payload = []
        # Audit log can contain either legacy list-shaped payloads or the new
        # envelope shape; normalise so cached returns look like the new shape.
        if isinstance(prior_payload, dict) and "results" in prior_payload:
            prior_results = prior_payload.get("results", [])
        elif isinstance(prior_payload, list):
            prior_results = prior_payload
        else:
            prior_results = []

        seconds_ago = max(0, int(time.time() - _ts_to_epoch(cached["timestamp"])))
        hint = (
            f"This query is a near-duplicate (cosine={cached['similarity']}) of "
            f"a query made {seconds_ago}s ago: \"{cached['query']}\" "
            f"(audit id {cached['id']}). Returning prior results unchanged. "
            "Stop rephrasing — call `get_full_document` on the results you care "
            "about, or read source files directly. If the corpus genuinely "
            "lacks coverage, no amount of rewording will fix it."
        )
        return {
            "results": prior_results,
            "meta": {
                "scoring": "cached",
                "rrf_k": cached["rrf_k"],
                "reranker_used": cached["reranker_used"],
                "confidence": "high" if prior_results else "none",
                "weak_match": not prior_results,
                "repeated_query": True,
                "previous_query_id": cached["id"],
                "previous_query": cached["query"],
                "similarity": cached["similarity"],
                "hint": hint,
            },
        }

    _reindex_status: dict = {"running": False, "last_result": None}
    _reindex_lock = threading.Lock()

    def _reindex_worker(source_filter: str | None) -> None:
        t0 = time.monotonic()
        try:
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
            with _reindex_lock:
                _reindex_status["last_result"] = {
                    "status": "completed",
                    "chunks_indexed": count,
                    "elapsed_ms": elapsed_ms,
                    "index_updated": index_updated,
                }
            log.info("Reindex completed: %d chunks in %dms", count, elapsed_ms)
        except Exception as exc:
            with _reindex_lock:
                _reindex_status["last_result"] = {"status": "failed", "error": str(exc)}
            log.exception("Reindex failed")
        finally:
            with _reindex_lock:
                _reindex_status["running"] = False

    @mcp.tool
    def reindex(source: str | None = None) -> str:
        """Reindex documentation sources. Supports incremental git-based delta reindex.

        Runs in the background — returns immediately. Call again to check status.

        Args:
            source: Reindex a specific source by name, or omit to reindex all.
        """
        with _reindex_lock:
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
        def _db_lookup() -> str | None:
            with _read_lock:
                if source:
                    content = db.get_document_content(read_conn, source, path)
                    if content:
                        return content
                else:
                    row = read_conn.execute(
                        "SELECT content FROM documents WHERE path=? LIMIT 1", (path,),
                    ).fetchone()
                    if row:
                        return row[0]
                return None

        # Try DB — search specific source or all sources
        content = _db_lookup()
        if content:
            return content

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
        with _read_lock:
            entries = db.get_recent_queries(read_conn, limit=limit)
        return json.dumps(entries, indent=2)

    @mcp.tool
    def status() -> str:
        """Report server health: index stats, Ollama reachability, reindex state, watcher status."""
        info: dict = {
            "index_loaded": searcher._metadata is not None and len(searcher._metadata) > 0,
            "chunk_count": searcher.chunk_count,
            "source_stats": searcher.source_stats(),
            "sources_configured": [s.name for s in sources],
            "transcript_dirs": len(transcript_dirs),
        }
        with _reindex_lock:
            info["reindex_running"] = _reindex_status["running"]
            info["reindex_last_result"] = _reindex_status["last_result"]

        ollama_st = ollama_client.status()
        info["ollama"] = ollama_st
        info["runtime_distill_enabled"] = runtime_distill_enabled

        return json.dumps(info, indent=2)

    @mcp.tool
    def distill_blob(
        text: str,
        intent: str = "default",
        path: str = "",
        model_profile: str = "",
    ) -> str:
        """Compress large text locally (Gemma or Qwen3-Coder) before cloud context.

        Args:
            text: Raw content (tool stdout, logs, file excerpt).
            intent: default | tool_output | code
            path: Optional file path hint for profile routing.
            model_profile: Force 'fast' or 'code'; empty = auto.
        """
        if not runtime_distill_enabled:
            return json.dumps({"error": "runtime_distill disabled in config", "digest": text[:2000]})
        profile = model_profile.strip() or None
        result = _distill_blob(text, intent, path=path, model_profile=profile)
        return json.dumps(result, indent=2)

    @mcp.tool
    def prepare_cloud_context(
        query: str,
        max_output_tokens: int = 4000,
        source: str = "",
        include_full_docs: bool = False,
        user_prompt: str = "",
        tool_outputs_json: str = "",
    ) -> str:
        """Retrieve KB hits and distill into a packet for Claude (less token waste).

        Args:
            query: Search query for prior work.
            max_output_tokens: Cap per local distill call.
            source: Optional source filter (personal, repo, sessions).
            include_full_docs: If true, fetch and distill top code/doc hits locally.
            user_prompt: Current user task (optional).
            tool_outputs_json: JSON list of {name, stdout, stderr} from shell tools.
        """
        if not runtime_distill_enabled:
            return json.dumps({"error": "runtime_distill disabled", "markdown": ""})

        if not searcher._metadata:
            searcher.load()
        src_filter = source.strip() or None
        results = searcher.query(
            query, top_k=5, source=src_filter,
            score_threshold_z=score_threshold_z, min_results=min_results,
        ) if searcher._metadata else []

        tool_outputs = []
        if tool_outputs_json.strip():
            try:
                tool_outputs = json.loads(tool_outputs_json)
            except json.JSONDecodeError:
                tool_outputs = []

        def _get_doc(src: str, path: str) -> str:
            return get_full_document(path, src)

        packet = _prepare_cloud_context(
            query,
            results,
            max_output_tokens=max_output_tokens,
            include_full_docs=include_full_docs,
            get_document_fn=_get_doc,
            tool_outputs=tool_outputs if isinstance(tool_outputs, list) else None,
            user_prompt=user_prompt or None,
            max_distill_docs=max_distill_docs,
        )
        return json.dumps(packet, indent=2)

    @mcp.tool
    def gc(source: str | None = None) -> str:
        """Remove index entries for files that no longer exist on disk.

        Args:
            source: Garbage-collect a specific source, or omit for all.
        """
        total_removed = 0
        for src in sources:
            if source and src.name != source:
                continue
            root = Path(os.path.expanduser(src.path))
            if not root.is_dir():
                continue
            extensions = set(src.file_types)
            live_paths = set()
            for fp in root.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in extensions:
                    live_paths.add(str(fp.relative_to(root)))
            removed = db.gc_source(conn, src.name, live_paths)
            total_removed += removed
        if total_removed:
            searcher.reload()
        return json.dumps({"removed_documents": total_removed})

    @mcp.tool
    def get_context_block(query: str, max_tokens: int = 2000) -> str:
        """Return a pre-formatted context block suitable for inclusion in a subagent prompt.

        Unlike query_knowledge_base which returns JSON, this returns concise text
        with file paths, summaries, and key pointers -- optimized for pasting into
        a Task tool prompt.

        Args:
            query: Natural language query.
            max_tokens: Approximate token budget for the block.
        """
        if not searcher._metadata:
            searcher.load()
        if not searcher._metadata:
            return "No knowledge base index available."

        results = searcher.query(
            query, top_k=5, source=None,
            score_threshold_z=score_threshold_z, min_results=min_results,
        )
        if not results:
            return "No relevant context found."

        lines = ["Context from prior work (verify against current code):"]
        token_count = 10
        for r in results:
            src = r.get("source", "")
            path = r.get("path", "")
            summary = r.get("session_summary") or r.get("summary") or ""
            sections = ", ".join(r.get("matched_sections", []))
            entry = f"\n- [{src}] {path}"
            if sections:
                entry += f" (sections: {sections})"
            if summary:
                entry += f"\n  {summary[:300]}"
            est = len(entry.split()) * 1.3
            if token_count + est > max_tokens:
                break
            lines.append(entry)
            token_count += est

        return "\n".join(lines)

    return mcp


def _run_stdio() -> None:
    create_app().run()


def _run_http(host: str, port: int) -> None:
    """Run as a long-lived HTTP MCP server that all Cursor windows share."""
    log.info("Starting bookkeeper HTTP server on %s:%d", host, port)
    create_app().run(transport="http", host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Dispatches between stdio (default), HTTP, and daemon admin."""
    import argparse
    from . import daemon as _daemon

    parser = argparse.ArgumentParser(prog="bookkeeper")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Run the MCP server (default if no subcommand)")
    serve.add_argument("--http", action="store_true", help="HTTP transport (shared across Cursor windows)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    dctl = sub.add_parser("daemon", help="Manage the shared HTTP daemon")
    dctl.add_argument("action", choices=["start", "stop", "restart", "reload", "status", "logs", "install"])
    dctl.add_argument("--port", type=int, default=None,
                      help="Override the persisted port (install/start)")
    dctl.add_argument("--host", default=None,
                      help="Bind host (install/start). Use 0.0.0.0 to expose on a "
                           "trusted network (Tailscale, VPN). Default: 127.0.0.1")
    dctl.add_argument("--follow", "-f", action="store_true", help="Tail logs (logs action)")

    args = parser.parse_args(argv)

    if args.cmd is None or args.cmd == "serve":
        if getattr(args, "http", False):
            _run_http(args.host, args.port)
        else:
            _run_stdio()
        return 0

    if args.cmd == "daemon":
        return _daemon.dispatch(
            args.action, port=args.port, host=args.host, follow=args.follow,
        )

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
