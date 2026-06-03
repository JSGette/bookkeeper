# bookkeeper

MCP server for semantic search over local documentation, source code, and chat session history. Indexes configurable directory trees, embeds chunks with a local model, and exposes search via MCP stdio tools. Designed for Cursor — the agent calls `query_knowledge_base` instead of scanning files manually.

## Architecture

```
Documents/Code → Chunker → Tagger → Embedder (BGE) → Vector index
                     ↓                                        ↑
              Doc Summarizer (Gemma) → documents.summary      │
                                                              │
Cursor agent → MCP tool call → Hybrid search (BM25+vector+RRF) → Summaries + pointers returned
                                       │
                                   Audit log (query_log table)

Transcripts → Background Watcher → Turn Parser → Gemma Summarizer → Session index
```

### Components

| Module | Purpose |
|--------|---------|
| `chunker.py` | Multi-format chunking: markdown at `##` headings, source code via treesitter (optional) or fixed-size, with a configurable token ceiling |
| `tagger.py` | Three-layer tagging: path-based → LLM via Ollama (cached) → manual `.bookkeeper-meta.yaml` overrides. Prompt and examples are configurable |
| `indexer.py` | Embeds chunks with `BAAI/bge-small-en-v1.5`, generates document-level summaries via Gemma, persists to SQLite. Git-based delta reindex with mtime fallback |
| `searcher.py` | Hybrid BM25 + vector cosine search fused via RRF. Returns document summaries + pointers instead of raw chunks. Configurable score-gap filtering, freshness boost, and optional HNSW for large corpora |
| `transcript.py` | Parses Cursor agent transcripts into turn-level chunks with heuristic extraction and optional Gemma summarization. Background watcher flushes each transcript to the index immediately |
| `index_generator.py` | Regenerates `INDEX.md` for sources with `generate_index: true` |
| `db.py` | SQLite storage backend: documents (with summaries), chunks, vectors, state, and query audit log |
| `server.py` | FastMCP stdio server exposing the tools below, plus background transcript watcher and query audit logging |

### MCP Tools

| Tool | Description |
|------|-------------|
| `query_knowledge_base(query, max_results, source)` | Hybrid semantic + keyword search. Returns `{results, meta}` with per-result `confidence` bands; `meta.weak_match` and `meta.repeated_query` carry a `hint` string the agent should obey instead of rephrasing |
| `get_full_document(path, source)` | Retrieve the full content of a document when the summary isn't sufficient |
| `reindex(source)` | Rebuild index (incremental via git diff). Session chunks are preserved |
| `list_sources()` | Show configured sources and chunk counts |
| `deprecate_doc(path, reason, undeprecate)` | Mark/unmark a doc as outdated |
| `save_session(title, summary, ...)` | Persist a structured session summary for future retrieval |
| `distill_transcripts()` | Force immediate scan and indexing of transcript directories |
| `distill_blob(text, intent, ...)` | Locally compress logs/code/text (Gemma or Qwen3-Coder) before cloud context |
| `prepare_cloud_context(query, ...)` | Search KB + optional full-doc distill → JSON packet + markdown for Claude |
| `query_audit_log(limit)` | Return recent queries and their results for auditing |

## Setup

```bash
pip install -e .

# Optional extras:
pip install -e ".[code]"   # treesitter-chunker for AST-aware code chunking
pip install -e ".[all]"    # includes treesitter-chunker
```

Install the shared daemon and register it in `~/.cursor/mcp.json`:

```bash
./install.sh           # default: one shared HTTP daemon for all Cursor windows
./install.sh --stdio   # legacy: per-Cursor-window subprocess
```

In daemon mode (recommended), `install.sh` runs:
- `bookkeeper daemon install` to write the platform unit (launchd on macOS, systemd-user on Linux).
- `bookkeeper daemon start` to launch on the first free port at/after 8765.
- Rewrites `~/.cursor/mcp.json` to a URL entry so every Cursor window shares one process:

```json
{
  "mcpServers": {
    "bookkeeper": { "url": "http://127.0.0.1:8765/mcp" }
  }
}
```

Manage the daemon with `bookkeeper daemon {start|stop|restart|status|logs}`. Status prints
the port, liveness, and log paths. Logs live under `~/.cache/bookkeeper/logs/`.

`install.sh` also writes **`.cursor/rules/bookkeeper.mdc`** (`alwaysApply: true`) into
this repo and every `sources[].path` from `config.yaml` (e.g. `datadog-agent`). Cursor
only loads rules from the **open workspace** — not from `~/.cursor/rules/`. After
install, reload the window; the rule appears under **Settings → Rules → Project Rules**
when that folder is open. For all repos without per-project files, paste
`docs/knowledge-base-protocol.md` into **User Rules**.

## Configuration

Config lookup order:

1. `AGENTS_RAG_CONFIG` environment variable
2. `config.yaml` next to `pyproject.toml`

```yaml
sources:
  - name: notes
    path: ~/my-notes
    file_types: [.md]           # default if omitted
    skip: [INDEX.md]
    generate_index: true

  - name: repo
    path: ~/my-project
    index_enabled: false          # optional (default true); disable for always-live repos
    file_types: [.md, .go, .py, .bzl]
    skip:
      - "vendor/*"
      - ".git/*"
    min_lines: 3
    max_chunk_tokens: 512       # per-chunk ceiling (default 512)

sessions:
  transcript_dirs:
    - ~/.cursor/projects/my-project/agent-transcripts
  output_dir: ~/my-notes/sessions
  gemma_every_n_turns: 3        # re-summarize with Gemma every N new turns
  watch_interval_minutes: 5     # background scan interval (0 to disable)

search:
  score_threshold_z: 0.8            # z-score multiplier for result filtering (default 0.8)
  min_results: 2                    # minimum results returned per query (default 2)
  dedupe_similarity_threshold: 0.88 # cosine threshold above which a recent query is treated as a duplicate
  dedupe_window_seconds: 120        # how far back to look when checking for duplicate queries

model: BAAI/bge-small-en-v1.5
cache_dir: ~/.cache/bookkeeper
use_gemma: true

# Optional: override the LLM tagger prompt or few-shot examples
# tagger_prompt: "You are a documentation indexer for ..."
# tagger_examples: ~/my-notes/tagger-examples.jsonl
```

### Source handling

- Each configured source is indexed independently into one shared vector index
- Set `index_enabled: false` on a source to keep it configured but exclude it from indexing
- `sessions` is a special source created by the transcript watcher / `save_session` tool
- `reindex()` rebuilds configured sources but preserves session chunks
- Queries can filter by source name or search across all sources

## Agent Usage Guidelines

Canonical protocol: [`docs/knowledge-base-protocol.md`](docs/knowledge-base-protocol.md)
(MCP resource `bookkeeper://usage-guide`; Cursor rule via `./install.sh`).

High level: query RAG when prior context is likely relevant, then triage with
`query_knowledge_base` → `prepare_cloud_context` before large cloud turns (use the
`markdown` field) → `distill_blob` on large tool output →
`get_full_document` only for exact quotes or verification → verify against current code →
`save_session` when done.

## Session Memory

A background thread scans `transcript_dirs` every N minutes, parsing new turns and flushing them to the index immediately per-transcript (no waiting for the full batch).

Each turn becomes a searchable chunk with heuristic tags from file paths and query keywords. Every N turns, Gemma generates a structured summary with decisions, next steps, and broad topic tags.

`save_session` provides a proactive complement — call it to persist structured context without waiting for background processing.

### Example Cursor rule

```
---
description: Session memory protocol for bookkeeper
alwaysApply: true
---

At the end of each implementation or investigation session, call `save_session` with:
- title: short name for the task
- summary: 1-2 sentence description of what was done
- relevant_files: list of files that were central to the work
- decisions: key choices made and their rationale
- next_steps: what remains to be done

Before starting work on a topic, call `query_knowledge_base` to check for prior session
notes. If results reference a previous session, use those file lists and decisions as
starting context instead of re-scanning the codebase.
```

## Hybrid Search

Two retrieval paths fused via Reciprocal Rank Fusion (RRF, default k=10):

1. **BM25** — keyword matching via `rank-bm25`
2. **Vector cosine** — semantic matching via `BAAI/bge-small-en-v1.5` (configurable via `EMBEDDING_MODEL` env var)

Post-fusion: score-gap filtering (configurable z-score threshold, default 0.8σ) removes noise while guaranteeing at least `min_results` (default 2). Freshness boost (time-decay over ~1 week) promotes recent docs. `hnswlib` is used for ANN-backed vector retrieval.

## LLM Tagging

Optional. Requires Ollama with any model. By default, bookkeeper uses profile-based models (`OLLAMA_MODEL_FAST` / `OLLAMA_MODEL_CODE`, default `qwen3-coder:30b` for both). Legacy single-model fallback via `OLLAMA_MODEL` is still supported.

- Tag cache is **content-aware** — file changes invalidate automatically
- System prompt and few-shot examples are configurable via `tagger_prompt` / `tagger_examples`
- Graceful fallback to path-based tags if Ollama is unavailable

## Incremental Reindex

- **Git-tracked**: stores HEAD hash, uses `git diff` for changed files only
- **Non-git**: mtime comparison
- Delta reindex reuses existing embeddings for unchanged chunks

## Manual Overrides

`.bookkeeper-meta.yaml` in any directory:

```yaml
files:
  some-doc.md:
    description: "Architecture overview of the frobnicator module"
    tags: [architecture, frobnicator, design]
```

Priority: manual override > LLM > path-based.

## Web UI

A read-only browser for the knowledge base index. Client-side keyword and tag filtering over all indexed chunks.

```bash
bookkeeper-web            # http://127.0.0.1:8808
bookkeeper-web --port 9000
```

Endpoints:

| Route | Description |
|-------|-------------|
| `GET /` | Single-page HTML app |
| `GET /api/chunks` | All chunk metadata (text replaced with a 200-char preview) |
| `GET /api/stats` | Aggregate counts: total chunks, per-source, per-tag, deprecated |
| `GET /api/queries?limit=50` | Recent query audit log (query text, results, latency) |

The UI loads all metadata client-side and filters by keyword (path, title, heading, description, tags) and/or selected tags. No semantic search — use the MCP `query_knowledge_base` tool for that.

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastmcp` | MCP server framework |
| `sentence-transformers` | Local embedding model |
| `numpy` | Vector math |
| `rank-bm25` | BM25 keyword search |
| `httpx` | Ollama HTTP API |
| `pyyaml` | Config parsing |
| `tiktoken` | Token counting |

**Optional:** `treesitter-chunker` (code chunking).
**Runtime:** Ollama with an LLM (for semantic tagging, document summarization, and transcript distillation).
