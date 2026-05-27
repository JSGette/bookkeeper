"""Ollama chat client with fast (Gemma) and code (Qwen3-Coder) profiles."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Serialise generative calls (MPS / single-GPU Ollama is not safely concurrent).
_chat_lock = threading.Lock()

_CODE_EXTENSIONS = frozenset({
    ".go", ".py", ".bzl", ".bazel", ".rs", ".ts", ".tsx", ".js", ".jsx",
    ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".swift",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".json",
})


def _parse_keep_alive(raw: str) -> int | str:
    if raw.lstrip("-").isdigit():
        return int(raw)
    return raw


@dataclass(frozen=True)
class ModelProfile:
    name: str
    max_input_chars: int
    timeout_s: float
    keep_alive: int | str
    num_predict: int
    num_ctx: int | None = None


_DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "fast": ModelProfile(
        name=os.environ.get("OLLAMA_MODEL_FAST", "gemma4:e4b"),
        max_input_chars=30_000,
        timeout_s=float(os.environ.get("OLLAMA_TIMEOUT", "120")),
        keep_alive=_parse_keep_alive(os.environ.get("OLLAMA_KEEP_ALIVE", "5m")),
        num_predict=2048,
    ),
    "code": ModelProfile(
        name=os.environ.get("OLLAMA_MODEL_CODE", "qwen3-coder:30b"),
        max_input_chars=80_000,
        timeout_s=float(os.environ.get("OLLAMA_TIMEOUT_CODE", "300")),
        keep_alive=_parse_keep_alive(os.environ.get("OLLAMA_KEEP_ALIVE_CODE", "30m")),
        num_predict=4096,
        num_ctx=32_768,
    ),
}

_configured = False
_profiles: dict[str, ModelProfile] = dict(_DEFAULT_PROFILES)
_base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def configure(config: dict) -> None:
    """Apply ollama section from config.yaml."""
    global _configured, _profiles, _base_url
    ollama = config.get("ollama", {})
    _base_url = ollama.get("url", _base_url)

    models_cfg = ollama.get("models", {})
    if not models_cfg:
        # Legacy: single OLLAMA_MODEL env / use_gemma only
        legacy = os.environ.get("OLLAMA_MODEL")
        if legacy:
            _profiles = dict(_DEFAULT_PROFILES)
            _profiles["fast"] = ModelProfile(
                name=legacy,
                max_input_chars=_DEFAULT_PROFILES["fast"].max_input_chars,
                timeout_s=_DEFAULT_PROFILES["fast"].timeout_s,
                keep_alive=_DEFAULT_PROFILES["fast"].keep_alive,
                num_predict=_DEFAULT_PROFILES["fast"].num_predict,
            )
        _configured = True
        return

    profiles: dict[str, ModelProfile] = dict(_DEFAULT_PROFILES)
    for key, spec in models_cfg.items():
        if not isinstance(spec, dict):
            continue
        base = _DEFAULT_PROFILES.get(key, _DEFAULT_PROFILES["fast"])
        opts = spec.get("options") or {}
        profiles[key] = ModelProfile(
            name=spec.get("name", base.name),
            max_input_chars=int(spec.get("max_input_chars", base.max_input_chars)),
            timeout_s=float(spec.get("timeout_s", base.timeout_s)),
            keep_alive=_parse_keep_alive(str(spec.get("keep_alive", base.keep_alive))),
            num_predict=int(spec.get("num_predict", base.num_predict)),
            num_ctx=int(opts["num_ctx"]) if opts.get("num_ctx") else base.num_ctx,
        )
    _profiles = profiles
    _configured = True


def get_profile(name: str) -> ModelProfile:
    if name not in _profiles:
        raise KeyError(f"Unknown model profile {name!r}; have {list(_profiles)}")
    return _profiles[name]


def is_code_path(path: str) -> bool:
    from pathlib import Path
    return Path(path).suffix.lower() in _CODE_EXTENSIONS


def pick_profile(
    task: str,
    *,
    path: str = "",
    source: str = "",
    char_count: int = 0,
    code_heavy: bool = False,
) -> str:
    """Return profile name ('fast' | 'code') for a workload."""
    if task in ("tagger", "redaction", "index_summary", "index_summary_prose", "transcript"):
        if task == "transcript" and code_heavy:
            return "code"
        if task in ("index_summary", "index_summary_prose") and path and is_code_path(path):
            return "code"
        return "fast"
    if task in ("runtime", "runtime_distill", "tool_digest", "code_digest", "prepare_cloud_context"):
        if char_count > 0 and char_count < 8000 and not code_heavy and not is_code_path(path):
            return "fast"
        return "code"
    return "fast"


def chat_json(
    profile: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    num_predict: int | None = None,
) -> dict[str, Any] | None:
    """Call Ollama /api/chat with JSON schema format; return parsed dict or None."""
    try:
        prof = get_profile(profile)
    except KeyError:
        fallback = _profiles.get("fast", _DEFAULT_PROFILES["fast"])
        log.warning(
            "Unknown profile %s; falling back to fast profile model=%s",
            profile,
            fallback.name,
        )
        prof = fallback
    with _chat_lock:
        try:
            options: dict[str, Any] = {
                "temperature": 0.1,
                "num_predict": num_predict or prof.num_predict,
            }
            if prof.num_ctx:
                options["num_ctx"] = prof.num_ctx
            resp = httpx.post(
                f"{_base_url}/api/chat",
                json={
                    "model": prof.name,
                    "messages": messages,
                    "stream": False,
                    "format": schema,
                    "keep_alive": prof.keep_alive,
                    "options": options,
                },
                timeout=prof.timeout_s,
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
            return json.loads(raw)
        except Exception as exc:
            log.warning("Ollama chat_json profile=%s model=%s failed: %s", profile, prof.name, exc)
            return None


def truncate_for_profile(profile: str, text: str) -> str:
    try:
        return text[: get_profile(profile).max_input_chars]
    except KeyError:
        fallback = _profiles.get("fast", _DEFAULT_PROFILES["fast"])
        log.warning(
            "Unknown profile %s in truncate_for_profile; using fast limit=%d",
            profile,
            fallback.max_input_chars,
        )
        return text[: fallback.max_input_chars]


def status() -> dict[str, Any]:
    """Reachability + which configured models are pulled."""
    info: dict[str, Any] = {"url": _base_url, "profiles": {}}
    try:
        resp = httpx.get(f"{_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        info["reachable"] = True
        for key, prof in _profiles.items():
            info["profiles"][key] = {
                "model": prof.name,
                "loaded": any(prof.name in n or n.startswith(prof.name.split(":")[0]) for n in names),
            }
    except Exception as exc:
        info["reachable"] = False
        info["error"] = str(exc)
    return info
