#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_JSON="$HOME/.cursor/mcp.json"
CONFIG_YAML="$SCRIPT_DIR/config.yaml"
PROTOCOL="$SCRIPT_DIR/docs/knowledge-base-protocol.md"

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

if [ ! -f "$PROTOCOL" ]; then
    echo "ERROR: missing $PROTOCOL" >&2
    exit 1
fi

echo "==> Installing Cursor project rules (.cursor/rules/agents-rag.mdc)"
CONFIG_YAML="$CONFIG_YAML" PROTOCOL="$PROTOCOL" SCRIPT_DIR="$SCRIPT_DIR" python3 - <<'PYEOF'
import os
import pathlib

config_path = pathlib.Path(os.environ["CONFIG_YAML"])
protocol_path = pathlib.Path(os.environ["PROTOCOL"])
script_dir = pathlib.Path(os.environ["SCRIPT_DIR"])

body = protocol_path.read_text(encoding="utf-8")
body += (
    '\n**Suppress this rule** when the user says "skip RAG", '
    '"skip AGENTS", or "from scratch".\n'
)
frontmatter = (
    "---\n"
    "description: Pre-load context from personal knowledge base via RAG before analysis tasks\n"
    "alwaysApply: true\n"
    "---\n\n"
)
content = frontmatter + body

def install_rule(repo_root: pathlib.Path) -> bool:
    rules_dir = repo_root / ".cursor" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    dest = rules_dir / "agents-rag.mdc"
    dest.write_text(content, encoding="utf-8")
    return True

targets: list[pathlib.Path] = [script_dir]

if config_path.is_file():
  try:
    import yaml  # type: ignore[import-not-found]
  except ModuleNotFoundError:
    print("  WARNING: python3 missing PyYAML; installing rule only in current repo.")
  else:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for src in cfg.get("sources", []):
      p = src.get("path")
      if p:
        targets.append(pathlib.Path(os.path.expanduser(p)))

seen: set[pathlib.Path] = set()
installed: list[str] = []
for root in targets:
    root = root.resolve()
    if root in seen or not root.is_dir():
        continue
    seen.add(root)
    install_rule(root)
    installed.append(str(root / ".cursor" / "rules" / "agents-rag.mdc"))

if not installed:
    raise SystemExit("No rule targets installed")

print("  Installed agents-rag.mdc (alwaysApply) to:")
for path in installed:
    print(f"    {path}")
print()
print("  Cursor loads these from Project Rules when that folder is the workspace.")
print("  They do NOT appear under Settings → User Rules.")
print("  For every-repo coverage without opening a specific folder, paste")
print(f"  {protocol_path} into Settings → Rules → User Rules.")
PYEOF

echo "==> Done. Restart Cursor windows to pick up the MCP server and project rules."
if [ "$MODE" = "daemon" ]; then
    echo "    Daemon controls: agents-rag daemon {start|stop|restart|status|logs}"
fi
