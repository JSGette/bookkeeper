#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_JSON="$HOME/.cursor/mcp.json"
RULES_DIR="$HOME/.cursor/rules"
RULE_FILE="$RULES_DIR/agents-rag.md"
CONFIG_YAML="$SCRIPT_DIR/config.yaml"

# Flags. Default is shared-daemon mode; --stdio keeps the legacy per-window spawn.
MODE="daemon"
for arg in "$@"; do
    case "$arg" in
        --stdio) MODE="stdio" ;;
        --daemon) MODE="daemon" ;;
    esac
done

# Kill any existing per-window stdio processes so they don't fight the daemon
# for the SQLite WAL or hold stale code on the next Cursor restart.
existing_pids=$(pgrep -f 'agents_rag\.server' || true)
if [ -n "$existing_pids" ]; then
    echo "==> Killing existing agents-rag processes: $existing_pids"
    echo "$existing_pids" | xargs kill 2>/dev/null || true
    sleep 1
    echo "$existing_pids" | xargs kill -9 2>/dev/null || true
fi

echo "==> Installing agents-rag from $SCRIPT_DIR"
pip install -e "$SCRIPT_DIR"

if [ "$MODE" = "daemon" ]; then
    echo "==> Installing shared daemon (one process across all Cursor windows)"
    # Install + start the daemon; the helper picks a free port at/after 8765
    # and persists it to ~/.cache/agents-rag/daemon.port.
    AGENTS_RAG_CONFIG="$CONFIG_YAML" agents-rag daemon install >/dev/null
    AGENTS_RAG_CONFIG="$CONFIG_YAML" agents-rag daemon start
    PORT=$(cat "$HOME/.cache/agents-rag/daemon.port")

    echo "==> Registering MCP server (HTTP url) in $MCP_JSON"
    mkdir -p "$(dirname "$MCP_JSON")"
    MCP_JSON="$MCP_JSON" PORT="$PORT" python3 - <<'PYEOF'
import json, os, pathlib
mcp_path = pathlib.Path(os.environ['MCP_JSON'])
port = int(os.environ['PORT'])
cfg = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
cfg.setdefault('mcpServers', {})
cfg['mcpServers']['agents-rag'] = {'url': f'http://127.0.0.1:{port}/mcp'}
mcp_path.write_text(json.dumps(cfg, indent=2) + '\n')
print(f"  agents-rag registered as http://127.0.0.1:{port}/mcp")
PYEOF
else
    echo "==> Registering MCP server (stdio) in $MCP_JSON"
    mkdir -p "$(dirname "$MCP_JSON")"
    MCP_JSON="$MCP_JSON" CONFIG_YAML="$CONFIG_YAML" python3 - <<'PYEOF'
import json, os, pathlib
mcp_path = pathlib.Path(os.environ['MCP_JSON'])
cfg = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
cfg.setdefault('mcpServers', {})
cfg['mcpServers']['agents-rag'] = {
    'command': 'python',
    'args': ['-m', 'agents_rag.server'],
    'env': {'AGENTS_RAG_CONFIG': os.environ['CONFIG_YAML']},
}
mcp_path.write_text(json.dumps(cfg, indent=2) + '\n')
print(f"  agents-rag registered (stdio, config: {os.environ['CONFIG_YAML']})")
PYEOF
fi

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

echo "==> Done. Restart Cursor windows to pick up the MCP server."
if [ "$MODE" = "daemon" ]; then
    echo "    Daemon controls: agents-rag daemon {start|stop|restart|status|logs}"
fi
