# Experimental: runtime distillation (Gemma + Qwen3-Coder 30B)

Branch: `experimental`

## Models

| Profile | Default model | Use |
|---------|---------------|-----|
| `fast` | `gemma4:e4b` | Tagger, prose summaries, light transcripts, short blobs |
| `code` | `qwen3-coder:30b` | Code paths, logs, `prepare_cloud_context`, code-heavy sessions |

No 70B profile in this branch.

## MCP tools

- **`distill_blob`** — compress arbitrary text locally.
- **`prepare_cloud_context`** — search + optional full-doc distill → JSON with `markdown` field.

## Cursor usage

Before a large cloud turn:

```
prepare_cloud_context(query="...", include_full_docs=true, user_prompt="...")
```

Paste the `markdown` field into the prompt. Use `distill_blob` on huge shell output before reading it in the orchestrator.

Restart the RAG daemon after pulling this branch so MCP picks up new tools.
