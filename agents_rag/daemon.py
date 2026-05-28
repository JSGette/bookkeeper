"""Daemon management for the shared HTTP MCP server.

One process per machine instead of one per Cursor window — shared HNSW index,
shared reranker, shared dedupe window. Backed by launchd on macOS and a
user-scope systemd unit on Linux.

The module is intentionally thin: the heavy lifting is delegated to launchctl /
systemctl and the daemon process itself is `bookkeeper serve --http`.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

CACHE_DIR = Path(os.path.expanduser("~/.cache/bookkeeper"))
PORT_FILE = CACHE_DIR / "daemon.port"
HOST_FILE = CACHE_DIR / "daemon.host"
LOG_DIR = CACHE_DIR / "logs"
STDOUT_LOG = LOG_DIR / "daemon.out.log"
STDERR_LOG = LOG_DIR / "daemon.err.log"

LABEL = "com.user.bookkeeper"
LAUNCHD_PLIST = Path(os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist"))
SYSTEMD_DIR = Path(os.path.expanduser("~/.config/systemd/user"))
SYSTEMD_UNIT = SYSTEMD_DIR / "bookkeeper.service"

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"
PORT_SCAN_RANGE = 20


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def _platform() -> str:
    """Return 'darwin' | 'linux' or raise — Windows isn't supported (yet)."""
    sysname = platform.system().lower()
    if sysname not in ("darwin", "linux"):
        raise RuntimeError(f"Unsupported platform for daemon mode: {sysname}")
    return sysname


# ---------------------------------------------------------------------------
# Port handling
# ---------------------------------------------------------------------------

def find_free_port(start: int = DEFAULT_PORT, scan: int = PORT_SCAN_RANGE) -> int:
    """Return the first free TCP port at or after `start`."""
    for candidate in range(start, start + scan):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    raise RuntimeError(
        f"No free port found in [{start}, {start + scan}). "
        "Stop a process or pass --port explicitly.",
    )


def read_port() -> int | None:
    """Return the persisted daemon port, or None when no daemon has been installed."""
    if not PORT_FILE.exists():
        return None
    try:
        return int(PORT_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def write_port(port: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PORT_FILE.write_text(f"{port}\n")


def read_host() -> str:
    """Persisted bind host (e.g. `127.0.0.1` or `0.0.0.0`). Defaults to loopback."""
    if not HOST_FILE.exists():
        return DEFAULT_HOST
    try:
        host = HOST_FILE.read_text().strip()
        return host or DEFAULT_HOST
    except OSError:
        return DEFAULT_HOST


def write_host(host: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HOST_FILE.write_text(f"{host}\n")


# ---------------------------------------------------------------------------
# Liveness probe
# ---------------------------------------------------------------------------

def is_running(port: int | None = None, timeout: float = 0.5) -> bool:
    """Cheap TCP probe — true when something accepts on the daemon port."""
    port = port if port is not None else read_port()
    if port is None:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# launchd / systemd render + install
# ---------------------------------------------------------------------------

_PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>-m</string>
    <string>agents_rag.server</string>
    <string>serve</string>
    <string>--http</string>
    <string>--host</string>
    <string>{host}</string>
    <string>--port</string>
    <string>{port}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>AGENTS_RAG_CONFIG</key><string>{config}</string>
    <key>PATH</key><string>{path}</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{stdout}</string>
  <key>StandardErrorPath</key><string>{stderr}</string>
  <key>WorkingDirectory</key><string>{home}</string>
</dict>
</plist>
"""

_SYSTEMD_TEMPLATE = """\
[Unit]
Description=bookkeeper shared MCP daemon
After=network.target

[Service]
Type=simple
ExecStart={python} -m agents_rag.server serve --http --host {host} --port {port}
Environment=AGENTS_RAG_CONFIG={config}
Environment=PATH={path}
WorkingDirectory={home}
Restart=on-failure
RestartSec=5
StandardOutput=append:{stdout}
StandardError=append:{stderr}

[Install]
WantedBy=default.target
"""


def _render_plist(host: str, port: int, config_path: str) -> str:
    return _PLIST_TEMPLATE.format(
        label=LABEL,
        python=sys.executable,
        host=host,
        port=port,
        config=config_path,
        path=os.environ.get("PATH", "/usr/bin:/bin"),
        stdout=str(STDOUT_LOG),
        stderr=str(STDERR_LOG),
        home=os.path.expanduser("~"),
    )


def _render_systemd(host: str, port: int, config_path: str) -> str:
    return _SYSTEMD_TEMPLATE.format(
        python=sys.executable,
        host=host,
        port=port,
        config=config_path,
        path=os.environ.get("PATH", "/usr/bin:/bin"),
        stdout=str(STDOUT_LOG),
        stderr=str(STDERR_LOG),
        home=os.path.expanduser("~"),
    )


def install(
    port: int | None,
    config_path: str | None = None,
    host: str | None = None,
) -> tuple[str, int]:
    """Write the platform unit file and persist host+port. Returns (host, port).

    Pass `host="0.0.0.0"` to expose the daemon on the LAN — recommended only
    behind Tailscale or another auth layer, since the MCP transport itself
    has no built-in authentication.
    """
    sysname = _platform()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = config_path or os.environ.get("AGENTS_RAG_CONFIG", "")
    chosen_host = host if host is not None else read_host()
    chosen = port if port is not None else (read_port() or find_free_port())
    # If the persisted port is already taken by an UNRELATED service, hop.
    if port is None and read_port() == chosen and not is_running(chosen) and not _port_free(chosen):
        chosen = find_free_port(chosen + 1)

    if sysname == "darwin":
        LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
        LAUNCHD_PLIST.write_text(_render_plist(chosen_host, chosen, cfg))
    else:
        SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
        SYSTEMD_UNIT.write_text(_render_systemd(chosen_host, chosen, cfg))
        _run(["systemctl", "--user", "daemon-reload"], check=False)

    write_port(chosen)
    write_host(chosen_host)
    return chosen_host, chosen


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# launchctl / systemctl shells
# ---------------------------------------------------------------------------

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _launchctl(action: str) -> None:
    if action == "start":
        _run(["launchctl", "load", "-w", str(LAUNCHD_PLIST)], check=False)
    elif action == "stop":
        _run(["launchctl", "unload", str(LAUNCHD_PLIST)], check=False)
    elif action == "reload":
        _run(["launchctl", "unload", str(LAUNCHD_PLIST)], check=False)
        _run(["launchctl", "load", "-w", str(LAUNCHD_PLIST)], check=False)


def _systemctl(action: str) -> None:
    mapping = {"start": "start", "stop": "stop", "reload": "restart"}
    _run(["systemctl", "--user", mapping[action], "bookkeeper.service"], check=False)


def _ctl(action: str) -> None:
    if _platform() == "darwin":
        _launchctl(action)
    else:
        _systemctl(action)


# ---------------------------------------------------------------------------
# Public actions
# ---------------------------------------------------------------------------

def start() -> tuple[str, int]:
    if not _unit_exists():
        host, port = install(None)
    else:
        host = read_host()
        port = read_port() or DEFAULT_PORT
    _ctl("start")
    _wait_until_ready(port)
    return host, port


def stop() -> None:
    _ctl("stop")


def restart() -> tuple[str, int]:
    host = read_host()
    port = read_port() or DEFAULT_PORT
    _ctl("reload")
    _wait_until_ready(port)
    return host, port


def reload() -> tuple[str, int]:
    # Same as restart for now — Cursor doesn't keep open SSE streams that
    # would suffer from a full bounce, and FastMCP doesn't expose SIGHUP.
    return restart()


def status() -> dict:
    port = read_port()
    host = read_host()
    return {
        "platform": _platform(),
        "host": host,
        "port": port,
        "running": is_running(port),
        "url": f"http://{host}:{port}/mcp" if port else None,
        "unit_path": str(LAUNCHD_PLIST if _platform() == "darwin" else SYSTEMD_UNIT),
        "stdout_log": str(STDOUT_LOG),
        "stderr_log": str(STDERR_LOG),
    }


def logs(follow: bool = False) -> int:
    """Tail the daemon's stdout/stderr logs. Returns the exit code of `tail`."""
    args = ["tail", "-n", "200"]
    if follow:
        args.append("-f")
    args.extend([str(STDOUT_LOG), str(STDERR_LOG)])
    return subprocess.run(args).returncode


def _unit_exists() -> bool:
    return (LAUNCHD_PLIST if _platform() == "darwin" else SYSTEMD_UNIT).exists()


def _wait_until_ready(port: int, timeout: float = 10.0) -> None:
    """Block until the daemon answers on its port. Useful right after start."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running(port):
            return
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# CLI dispatcher (called from server.main)
# ---------------------------------------------------------------------------

def dispatch(
    action: str,
    port: int | None = None,
    host: str | None = None,
    follow: bool = False,
) -> int:
    if action == "install":
        chosen_host, chosen = install(port, host=host)
        print(f"Installed daemon on http://{chosen_host}:{chosen}/mcp. "
              f"Run `bookkeeper daemon start` to launch.")
        return 0
    if action == "start":
        # `daemon start --host X` re-installs first so the unit file picks up
        # the new bind address; otherwise we'd silently launch the old config.
        if host is not None and host != read_host():
            install(None, host=host)
        chosen_host, chosen = start()
        print(f"Daemon running on http://{chosen_host}:{chosen}/mcp")
        return 0
    if action == "stop":
        stop()
        print("Daemon stopped.")
        return 0
    if action in ("restart", "reload"):
        chosen_host, chosen = reload() if action == "reload" else restart()
        print(f"Daemon {action}ed on http://{chosen_host}:{chosen}/mcp")
        return 0
    if action == "status":
        import json
        print(json.dumps(status(), indent=2))
        return 0
    if action == "logs":
        return logs(follow=follow)
    print(f"Unknown daemon action: {action}", file=sys.stderr)
    return 2
