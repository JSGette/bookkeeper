# agents-rag

MCP server for semantic search over local documentation, source code, and chat session history. Indexes configurable directory trees, embeds chunks with a local model, and exposes search via MCP stdio tools. Designed for Cursor — the agent calls `query_knowledge_base` instead of scanning files manually.

## Architecture

```
Documents/Code → Multi-format Chunker → Tagger (path + LLM + overrides) → Embedder (MiniLM) → Vector index
                                                                                                    ↑
Cursor agent → MCP tool call → Hybrid search (BM25 + vector + RRF) → Top-K chunks returned ────────┘

Transcripts → Background Watcher → Turn-level Parser → Gemma Summarizer → Session index ───────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| `chunker.py` | Multi-format chunking: markdown at `##` headings, source code via treesitter (optional) or fixed-size, with a configurable token ceiling |
| `tagger.py` | Three-layer tagging: path-based → LLM via Ollama (cached) → manual `.agents-rag-meta.yaml` overrides. Prompt and examples are configurable |
| `indexer.py` | Embeds chunks with `all-MiniLM-L6-v2`, persists to cache dir. Git-based delta reindex, mtime fallback. Preserves session chunks across reindex. Optional HNSW for large corpora |
| `searcher.py` | Hybrid BM25 + vector cosine search fused via RRF. Score-gap filtering and freshness boost |
| `transcript.py` | Parses Cursor agent transcripts into turn-level chunks with heuristic extraction and optional Gemma summarization. Background watcher flushes each transcript to the index immediately |
| `index_generator.py` | Regenerates `INDEX.md` for sources with `generate_index: true` |
| `server.py` | FastMCP stdio server exposing the tools below, plus background transcript watcher |

### MCP Tools

| Tool | Description |
|------|-------------|
| `query_knowledge_base(query, top_k, source)` | Hybrid semantic + keyword search across indexed docs and sessions |
| `reindex(source)` | Rebuild index (incremental via git diff). Session chunks are preserved |
| `list_sources()` | Show configured sources and chunk counts |
| `deprecate_doc(path, reason, undeprecate)` | Mark/unmark a doc as outdated |
| `save_session(title, summary, ...)` | Persist a structured session summary for future retrieval |
| `distill_transcripts()` | Force immediate scan and indexing of transcript directories |

## Setup

```bash
pip install -e .

# Optional extras:
pip install -e ".[code]"   # treesitter-chunker for AST-aware code chunking
pip install -e ".[perf]"   # hnswlib for ANN search on large corpora
pip install -e ".[all]"    # both
```

Register in `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agents-rag": {
      "command": "agents-rag"
    }
  }
}
```

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

model: all-MiniLM-L6-v2
cache_dir: ~/.cache/agents-rag
use_gemma: true

# Optional: override the LLM tagger prompt or few-shot examples
# tagger_prompt: "You are a documentation indexer for ..."
# tagger_examples: ~/my-notes/tagger-examples.jsonl
```

### Source handling

- Each configured source is indexed independently into one shared vector index
- `sessions` is a special source created by the transcript watcher / `save_session` tool
- `reindex()` rebuilds configured sources but preserves session chunks
- Queries can filter by source name or search across all sources

## Agent Usage Guidelines

Query results are **prior notes and hypotheses, not ground truth** — they may be outdated.

1. Call `query_knowledge_base` with a short natural-language query.
2. Use `source` to filter by name, or omit to search all.
3. Prefer recently indexed docs — check `indexed_at`.
4. If a doc **references source files**, read those too — the doc is a pointer, the code is the truth.
5. **Verify factual claims** against current source code before repeating them.
6. `"deprecated": true` means historical context only.
7. If a doc is **contradicted by current code**, call `deprecate_doc`.
8. At the end of a work session, call `save_session` with a structured summary.

## Session Memory

A background thread scans `transcript_dirs` every N minutes, parsing new turns and flushing them to the index immediately per-transcript (no waiting for the full batch).

Each turn becomes a searchable chunk with heuristic tags from file paths and query keywords. Every N turns, Gemma generates a structured summary with decisions, next steps, and broad topic tags.

`save_session` provides a proactive complement — call it to persist structured context without waiting for background processing.

### Example Cursor rule

```
---
description: Session memory protocol for agents-rag
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

Two retrieval paths fused via Reciprocal Rank Fusion (RRF, k=60):

1. **BM25** — keyword matching via `rank-bm25`
2. **Vector cosine** — semantic matching via `all-MiniLM-L6-v2`

Post-fusion: score-gap filtering (mean + 1.5σ threshold) removes noise. Freshness boost (time-decay over ~1 week) promotes recent docs. For corpora >5000 chunks, optional `hnswlib` provides sub-ms ANN search.

## LLM Tagging

Optional. Requires Ollama with any model (configure via `OLLAMA_MODEL` env var, default `gemma4:e4b`).

- Tag cache is **content-aware** — file changes invalidate automatically
- System prompt and few-shot examples are configurable via `tagger_prompt` / `tagger_examples`
- Graceful fallback to path-based tags if Ollama is unavailable

## Incremental Reindex

- **Git-tracked**: stores HEAD hash, uses `git diff` for changed files only
- **Non-git**: mtime comparison
- Delta reindex reuses existing embeddings for unchanged chunks

## Manual Overrides

`.agents-rag-meta.yaml` in any directory:

```yaml
files:
  some-doc.md:
    description: "Architecture overview of the frobnicator module"
    tags: [architecture, frobnicator, design]
```

Priority: manual override > LLM > path-based.

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

**Optional:** `treesitter-chunker` (code chunking), `hnswlib` (ANN search).
**Runtime:** Ollama with an LLM (for semantic tagging and transcript summarization).
