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

1. Call `query_knowledge_base` with a short natural-language query describing the task.
2. Use `source` to filter by source name, or omit to search all sources.
3. Prefer recently indexed docs — check `indexed_at`; older docs are more likely stale.
4. **Document results** contain a `summary` field (a 2-3 sentence document-level overview)
   and a `matched_sections` list showing which sections triggered the match. Use the summary
   to decide relevance.
5. **Session results** contain a `session_summary` field (distilled overview with decisions
   and next steps). Multiple matching turns from the same session are collapsed into one
   result; `matched_turns` shows which turns matched.
6. If the summary is sufficient for your task, use it directly. If you need exact code,
   specific details, or full context, call `get_full_document` with the `path` and `source`
   fields from the result. Use `matched_sections` to know which part of the document to
   focus on.
7. If a result references source files, configs, or code paths, read those sources too —
   the doc is a pointer, the code is the truth.
8. If a result makes a factual claim about the codebase (e.g., "X blocks Y"), verify it
   against the current source code before repeating it.
9. If a result has `"deprecated": true`, do not use it as actionable context — treat it
   purely as historical background.
10. If during verification (step 8) a doc's claims are contradicted by the current source
    code, call `deprecate_doc` with the doc path and a brief reason explaining what changed.
11. At the end of a meaningful work session (implementation, investigation, debugging),
    call `save_session` with a structured summary: title, what was done, key decisions,
    relevant files, and next steps. This lets the next chat resume without re-discovering
    context.
12. Previous chat sessions are automatically indexed in the background. When starting new
    work, `query_knowledge_base` will surface prior session notes alongside documentation.

**Suppress this rule** when the user says "skip RAG", "skip AGENTS", or "from scratch".
RULE_EOF

echo "==> Done. Restart Cursor to pick up the MCP server."
