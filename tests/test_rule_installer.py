"""Tests for project-rule installer helper."""

from __future__ import annotations

from pathlib import Path

from agents_rag.rule_installer import install_project_rules


def test_install_project_rules_handles_missing_yaml_module(tmp_path, monkeypatch):
    protocol = tmp_path / "knowledge-base-protocol.md"
    protocol.write_text("# Protocol\n")
    config = tmp_path / "config.yaml"
    config.write_text("sources:\n  - name: other\n    path: ~/missing\n")
    script_dir = tmp_path / "repo"
    script_dir.mkdir()

    def _raise_for_yaml(name: str):
        if name == "yaml":
            raise ModuleNotFoundError("yaml not installed")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("agents_rag.rule_installer.importlib.import_module", _raise_for_yaml)
    installed = install_project_rules(config, protocol, script_dir)

    assert installed == [script_dir / ".cursor" / "rules" / "bookkeeper.mdc"]
    assert installed[0].is_file()


def test_install_project_rules_installs_sources_from_config(tmp_path):
    protocol = tmp_path / "knowledge-base-protocol.md"
    protocol.write_text("# Protocol\n")
    docs_repo = tmp_path / "docs-repo"
    docs_repo.mkdir()
    script_dir = tmp_path / "repo"
    script_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        "sources:\n"
        "  - name: docs\n"
        f"    path: {docs_repo}\n",
    )

    installed = install_project_rules(config, protocol, script_dir)
    installed_set = {p.resolve() for p in installed}

    assert (script_dir / ".cursor" / "rules" / "bookkeeper.mdc").resolve() in installed_set
    assert (docs_repo / ".cursor" / "rules" / "bookkeeper.mdc").resolve() in installed_set
