# Knowledge Base Protocol

Query results are prior notes and hypotheses, not ground truth — they may be outdated.

## Query when relevant

Use `query_knowledge_base` when prior notes are likely helpful:

- migration / prior-session / "where did we leave off"
- architecture planning, investigations, or broad refactors
- requests that reference earlier decisions, docs, or previously debugged failures

Skip it for clearly local tasks where prior context is unlikely to help (small edits, direct bug fix in open file, formatting-only changes).

## MCP response shape

`query_knowledge_base` returns `{"results": [...], "meta": {...}}`. Each result has a
`confidence` band ("high"|"medium"|"low"). Check `meta` first:

- `scoring`: "rerank" | "rrf" | "cached"
- `confidence`: overall band (top result's band)
- `weak_match`: true when the top result is low-confidence or empty
- `repeated_query`: true when this query was deduped against a recent near-duplicate
- `hint`: read and obey when `weak_match` or `repeated_query` is set — do not rephrase
  the same query in a loop

## Workflow

1. If prior context is likely useful, call `query_knowledge_base` with a short
   natural-language query. Use `source` to filter by source name, or omit for all
   sources. Prefer fresh `indexed_at`.
2. If `meta.weak_match` or `meta.repeated_query` is set, follow `meta.hint`. Do not
   keep rephrasing — read source files directly or accept missing coverage.
3. **Triage by summary only.** Use `summary` (docs) / `session_summary` (sessions)
   to decide which hits are relevant. Do not quote or act on a summary alone.
4. **Pre-cloud context (planning, multi-file work, or large prompts).** Call
   `prepare_cloud_context` with your task query and `user_prompt`. For code work set
   `include_full_docs=true`. Use the returned `markdown` field as prior context for
   Claude (or other cloud models). Do **not** paste raw `get_full_document` output into
   cloud prompts — local Gemma/Qwen distillation is cheaper and enough for planning.
5. **Large tool output.** After shell/tools return more than ~4k characters, call
   `distill_blob` (`intent=tool_output` for logs, `intent=code` for build/test output)
   before the orchestrator reads the full stdout.
6. **Exact detail only.** Call `get_full_document` when you need a verbatim quote, line
   number, or to verify a specific claim — then jump to `matched_sections` first.
7. **Verify against current code.** If a result references source files, read those too.
   The doc is a pointer; the code is the truth.
8. **Save context at the end.** After meaningful work, call `save_session` with title,
   summary, decisions, `relevant_files`, next_steps.

## When the summary alone is enough

- Deciding whether a hit is relevant (relevance gate).
- Picking which paths to open with the Read tool.

## When you MUST call `get_full_document`

- Citing a specific fact, function name, config value, line number, PR/Jira ID, or
  decision rationale to the user.
- Summarising prior decisions or next steps from a session — `session_summary` is still
  a distillation; fetch the full session before treating bullets as authoritative.
- Continuing work from a prior session (need full Decisions / Next Steps / Relevant Files).
- Whenever you are about to write or modify code based on a retrieved doc.
- Whenever the user asks "what did we decide about X" or "where did we leave Y".

## Local distillation tools

| Tool | When |
|------|------|
| `prepare_cloud_context` | Before a large cloud turn: search + optional local distill → `markdown` packet |
| `distill_blob` | Compress one blob (log, file excerpt) locally |
| `get_context_block` | Subagents: lightweight summary pointers only (no LLM distill) |
| `get_full_document` | Exact quotes / verification only |

Cloud models **orchestrate and decide**. Local models **compress** retrieved or tool
output. They do not replace verification against current source code.

## Invariants

- Results with `"deprecated": true` are historical only.
- If a doc is contradicted by current code, call `deprecate_doc` with a brief reason.
- Prior chat sessions are auto-indexed; treat them like docs.

## Notes

- Subagents (Task tool): `get_context_block` or `prepare_cloud_context` in the
  subagent prompt under "Prior context from knowledge base:".
- Call `status` if tools fail or Ollama profiles are unloaded.
- If MCP is unavailable, fall back to `~/REPOS/AGENTS/INDEX.md`.
