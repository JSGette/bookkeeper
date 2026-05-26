"""Tests for agents_rag.daemon: port selection, plist/unit rendering, dispatch."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from unittest.mock import patch

import pytest

from agents_rag import daemon


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Re-root CACHE_DIR / LOG_DIR / unit paths into tmp_path so tests don't
    touch ~/.cache or ~/Library."""
    cache = tmp_path / "cache"
    logs = cache / "logs"
    plist = tmp_path / "LaunchAgents" / "com.user.agents-rag.plist"
    sys_dir = tmp_path / "systemd" / "user"
    unit = sys_dir / "agents-rag.service"

    monkeypatch.setattr(daemon, "CACHE_DIR", cache)
    monkeypatch.setattr(daemon, "PORT_FILE", cache / "daemon.port")
    monkeypatch.setattr(daemon, "HOST_FILE", cache / "daemon.host")
    monkeypatch.setattr(daemon, "LOG_DIR", logs)
    monkeypatch.setattr(daemon, "STDOUT_LOG", logs / "daemon.out.log")
    monkeypatch.setattr(daemon, "STDERR_LOG", logs / "daemon.err.log")
    monkeypatch.setattr(daemon, "LAUNCHD_PLIST", plist)
    monkeypatch.setattr(daemon, "SYSTEMD_DIR", sys_dir)
    monkeypatch.setattr(daemon, "SYSTEMD_UNIT", unit)
    return tmp_path


def test_find_free_port_returns_open_socket():
    port = daemon.find_free_port(start=40000, scan=20)
    assert 40000 <= port < 40020
    # Sanity: the returned port must actually be free at call time.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


def test_find_free_port_skips_busy_ports():
    """When the start port is taken, the scanner must hop forward."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        busy_port = taken.getsockname()[1]
        port = daemon.find_free_port(start=busy_port, scan=20)
    assert port != busy_port


def test_port_persistence_roundtrip(isolated_paths):
    assert daemon.read_port() is None
    daemon.write_port(9999)
    assert daemon.read_port() == 9999


def test_is_running_returns_false_when_no_daemon(isolated_paths):
    assert daemon.is_running() is False
    daemon.write_port(1)  # unlikely to have anything on port 1
    assert daemon.is_running() is False


def test_is_running_detects_open_socket(isolated_paths):
    """A listener on the persisted port is the daemon liveness signal."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        daemon.write_port(port)
        assert daemon.is_running(timeout=0.5) is True


def test_install_writes_launchd_plist_on_macos(isolated_paths, monkeypatch):
    monkeypatch.setattr(daemon, "_platform", lambda: "darwin")
    host, chosen = daemon.install(port=12345, config_path="/tmp/cfg.yaml")
    assert (host, chosen) == ("127.0.0.1", 12345)
    assert daemon.LAUNCHD_PLIST.exists()
    content = daemon.LAUNCHD_PLIST.read_text()
    assert "com.user.agents-rag" in content
    assert "<string>12345</string>" in content
    assert "<string>127.0.0.1</string>" in content
    assert "/tmp/cfg.yaml" in content
    assert "agents_rag.server" in content
    assert daemon.read_port() == 12345
    assert daemon.read_host() == "127.0.0.1"


def test_install_writes_systemd_unit_on_linux(isolated_paths, monkeypatch):
    monkeypatch.setattr(daemon, "_platform", lambda: "linux")
    monkeypatch.setattr(daemon, "_run", lambda cmd, check=True: subprocess.CompletedProcess(cmd, 0))
    host, chosen = daemon.install(port=12346, config_path="/tmp/cfg.yaml")
    assert (host, chosen) == ("127.0.0.1", 12346)
    assert daemon.SYSTEMD_UNIT.exists()
    content = daemon.SYSTEMD_UNIT.read_text()
    assert "ExecStart=" in content
    assert "--host 127.0.0.1 --port 12346" in content
    assert "AGENTS_RAG_CONFIG=/tmp/cfg.yaml" in content


def test_install_auto_picks_port_when_none(isolated_paths, monkeypatch):
    monkeypatch.setattr(daemon, "_platform", lambda: "darwin")
    monkeypatch.setattr(daemon, "find_free_port", lambda start=8765, scan=20: 8765)
    host, chosen = daemon.install(port=None, config_path="/tmp/cfg.yaml")
    assert (host, chosen) == ("127.0.0.1", 8765)


def test_install_reuses_persisted_port_when_free(isolated_paths, monkeypatch):
    """If a port has already been claimed by a prior install, keep using it
    instead of hopping — Cursor configs out there reference that port."""
    monkeypatch.setattr(daemon, "_platform", lambda: "darwin")
    monkeypatch.setattr(daemon, "_port_free", lambda port: True)
    daemon.write_port(8770)
    host, chosen = daemon.install(port=None, config_path="/tmp/cfg.yaml")
    assert (host, chosen) == ("127.0.0.1", 8770)


def test_install_with_explicit_host_writes_bind_address(isolated_paths, monkeypatch):
    """--host 0.0.0.0 must end up in both the plist and daemon.host file so
    `daemon status` can report the right URL and Cursor on another machine
    knows where to connect."""
    monkeypatch.setattr(daemon, "_platform", lambda: "darwin")
    host, chosen = daemon.install(port=12345, host="0.0.0.0", config_path="/tmp/cfg.yaml")
    assert (host, chosen) == ("0.0.0.0", 12345)
    assert daemon.read_host() == "0.0.0.0"
    content = daemon.LAUNCHD_PLIST.read_text()
    assert "<string>0.0.0.0</string>" in content


def test_dispatch_status_prints_json(isolated_paths, monkeypatch, capsys):
    monkeypatch.setattr(daemon, "_platform", lambda: "darwin")
    daemon.write_port(9000)
    rc = daemon.dispatch("status")
    assert rc == 0
    out = capsys.readouterr().out
    assert '"port": 9000' in out
    assert '"running"' in out


def test_dispatch_stop_invokes_ctl(isolated_paths, monkeypatch):
    monkeypatch.setattr(daemon, "_platform", lambda: "darwin")
    calls: list[list[str]] = []
    monkeypatch.setattr(daemon, "_run",
                        lambda cmd, check=True: (calls.append(cmd), subprocess.CompletedProcess(cmd, 0))[1])
    daemon.dispatch("stop")
    assert any("launchctl" in c[0] and "unload" in c for c in calls)


def test_dispatch_install_creates_unit_and_persists_port(isolated_paths, monkeypatch):
    monkeypatch.setattr(daemon, "_platform", lambda: "darwin")
    rc = daemon.dispatch("install", port=11111)
    assert rc == 0
    assert daemon.LAUNCHD_PLIST.exists()
    assert daemon.read_port() == 11111


def test_unsupported_platform_raises(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Unsupported platform"):
        daemon._platform()


@pytest.mark.skipif(sys.platform == "win32", reason="HTTP daemon not supported on Windows")
def test_server_main_starts_http_transport(monkeypatch):
    """`agents-rag serve --http` must wire through to FastMCP's HTTP transport
    with the host/port the user requested. We mock create_app so the test
    doesn't actually bind a socket."""
    from agents_rag import server

    captured: dict = {}

    class FakeApp:
        def run(self, transport=None, host=None, port=None):
            captured["transport"] = transport
            captured["host"] = host
            captured["port"] = port

    monkeypatch.setattr(server, "create_app", lambda: FakeApp())
    server.main(["serve", "--http", "--host", "127.0.0.1", "--port", "0"])
    assert captured == {"transport": "http", "host": "127.0.0.1", "port": 0}


def test_server_main_defaults_to_stdio(monkeypatch):
    from agents_rag import server

    called = {"stdio": False}

    class FakeApp:
        def run(self, transport=None, host=None, port=None):
            called["stdio"] = transport is None

    monkeypatch.setattr(server, "create_app", lambda: FakeApp())
    server.main([])
    assert called["stdio"] is True
