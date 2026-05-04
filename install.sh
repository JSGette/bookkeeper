#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_JSON="$HOME/.cursor/mcp.json"
RULES_DIR="$HOME/.cursor/rules"
RULE_FILE="$RULES_DIR/agents-rag.md"

# Kill any existing agents-rag server processes to avoid SQLite lock contention
existing_pids=$(pgrep -f 'agents_rag\.server' || true)
if [ -n "$existing_pids" ]; then
    echo "==> Killing existing agents-rag server processes: $existing_pids"
    echo "$existing_pids" | xargs kill 2>/dev/null || true
    sleep 1
    # Force-kill anything that didn't exit cleanly
    echo "$existing_pids" | xargs kill -9 2>/dev/null || true
fi

echo "==> Installing agents-rag from $SCRIPT_DIR"
pip install -e "$SCRIPT_DIR"

echo "==> Registering MCP server in $MCP_JSON"
mkdir -p "$(dirname "$MCP_JSON")"
python3 -c "
import json, pathlib, sys

mcp_path = pathlib.Path('$MCP_JSON')
config_yaml = '$SCRIPT_DIR/config.yaml'

if mcp_path.exists():
    cfg = json.loads(mcp_path.read_text())
else:
    cfg = {}

cfg.setdefault('mcpServers', {})
cfg['mcpServers']['agents-rag'] = {
    'command': 'python',
    'args': ['-m', 'agents_rag.server'],
    'env': {'AGENTS_RAG_CONFIG': config_yaml},
}

mcp_path.write_text(json.dumps(cfg, indent=2) + '\n')
print(f'  agents-rag registered (config: {config_yaml})')
"

echo "==> Writing Cursor rule to $RULE_FILE"
mkdir -p "$RULES_DIR"
cat > "$RULE_FILE" << 'RULE_EOF'
---
description: Pre-load context from personal knowledge base via RAG before analysis tasks
alwaysApply: true
---

# Knowledge Base Context Loading

Before starting analysis, investigation, or planning work:

1. Call `query_knowledge_base` with a short natural-language query. Use `source` to
   filter by source name, or omit for all sources. Prefer fresh `indexed_at`.
2. **Triage by summary only.** Use the `summary` (docs) / `session_summary` (sessions)
   to decide which hits are relevant. Do not paraphrase, quote, or act on a result
   based on its summary alone.
3. **Fetch the full document** for every result you intend to use beyond the relevance
   gate. Call `get_full_document` with the result's `path` and `source`, then jump to
   `matched_sections` (and `start_line`/`end_line` for doc hits) first.
4. **Verify against current code.** If a result references source files, configs, or
   code paths, read those sources too. The doc is a pointer; the code is the truth.
5. **Save context at the end.** When the work session is meaningful (implementation,
   investigation, debugging), call `save_session` with title, summary, decisions,
   `relevant_files`, next_steps so the next chat can resume without re-discovering
   context.

## When you MUST call `get_full_document`

- Citing a specific fact, function name, config value, line number, PR/Jira ID, or
  decision rationale to the user.
- Summarising prior decisions or next steps from a session result (`session_summary`
  is more self-contained than a doc `summary`, but it is still a distillation).
- Continuing work described in a session — you need the full Decisions / Next Steps
  / Relevant Files lists, not the 2-3 sentence overview.
- Whenever you are about to write or modify code based on a retrieved doc.
- Whenever the user asks "what did we decide about X" or "where did we leave Y".

## Invariants

- Results with `"deprecated": true` are historical context only. Do not treat their
  claims as current facts; mention them only if the user asks about past decisions
  or evolution of the codebase.
- If during verification you find a doc's claim contradicted by current code, call
  `deprecate_doc` with the doc path and a brief reason explaining what changed.
- Previous chat sessions are auto-indexed in the background and surface alongside
  documentation. Treat them with the same rules as docs.

## Notes

- For subagents launched via the Task tool, call `get_context_block` with a query
  relevant to the subagent's task and paste the returned text into the subagent's
  `prompt` parameter under "Prior context from knowledge base:".
- Call `status` to check server health if `query_knowledge_base` returns unexpected
  results or errors.
- If the `query_knowledge_base` tool is not available, fall back to reading
  `~/REPOS/AGENTS/INDEX.md` directly.

**Suppress this rule** when the user says "skip RAG", "skip AGENTS", or "from scratch".
RULE_EOF

echo "==> Done. Restart Cursor to pick up the MCP server."
