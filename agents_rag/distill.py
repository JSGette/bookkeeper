"""Runtime distillation via local Ollama (Gemma / Qwen3-Coder)."""

from __future__ import annotations

import json
import logging
from typing import Any

from .ollama_client import chat_json, pick_profile, truncate_for_profile

log = logging.getLogger(__name__)

_BLOB_SCHEMAS: dict[str, dict[str, Any]] = {
    "default": {
        "type": "object",
        "properties": {
            "digest": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["digest", "key_points"],
    },
    "tool_output": {
        "type": "object",
        "properties": {
            "digest": {"type": "string"},
            "errors": {"type": "array", "items": {"type": "string"}},
            "suggested_next": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["digest", "errors", "suggested_next"],
    },
    "code": {
        "type": "object",
        "properties": {
            "digest": {"type": "string"},
            "symbols": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["digest", "symbols", "open_questions"],
    },
}


def distill_blob(
    text: str,
    intent: str = "default",
    *,
    path: str = "",
    model_profile: str | None = None,
    max_output_tokens: int = 1500,
) -> dict[str, Any]:
    """Compress arbitrary text for cloud orchestrator context."""
    if not text.strip():
        return {"digest": "", "key_points": [], "profile": "none"}

    schema_key = intent if intent in _BLOB_SCHEMAS else "default"
    if intent in ("tool_output", "tool_output_digest", "log"):
        schema_key = "tool_output"
    elif intent in ("code", "code_digest"):
        schema_key = "code"

    profile = model_profile or pick_profile(
        "runtime",
        path=path,
        char_count=len(text),
        code_heavy=schema_key == "code",
    )
    clipped = truncate_for_profile(profile, text)
    schema = _BLOB_SCHEMAS[schema_key]

    system = (
        "Compress the user content for a senior engineer who will plan next steps. "
        "Do not solve the task. Be specific (paths, symbols, line refs). "
        "Output JSON matching the schema only."
    )
    if schema_key == "tool_output":
        system += " Group failures; quote at most 3 critical lines verbatim."
    elif schema_key == "code":
        system += " List real symbol names from the text; do not invent APIs."

    data = chat_json(
        profile,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": clipped},
        ],
        schema,
        num_predict=max_output_tokens,
    )
    if not data:
        return {
            "digest": clipped[:2000],
            "key_points": [],
            "profile": profile,
            "fallback": True,
        }
    data["profile"] = profile
    data["fallback"] = False
    return data


def render_cloud_packet(packet: dict[str, Any]) -> str:
    """Markdown view of a CloudContextPacket for pasting into prompts."""
    lines = ["## Distilled context (local RAG; verify against repo)"]
    meta = packet.get("meta", {})
    if meta.get("profile"):
        lines.append(f"_Profile: {meta['profile']}, ~{meta.get('output_tokens_est', '?')} tokens_")
    for item in packet.get("prior_work", []):
        lines.append(f"\n### [{item.get('source', '')}] {item.get('path', '')}")
        if item.get("relevance"):
            lines.append(f"_relevance: {item['relevance']}_")
        lines.append(item.get("distilled", ""))
    for ptr in packet.get("code_pointers", []):
        lines.append(f"\n- `{ptr.get('path', '')}`: {ptr.get('why', '')}")
        for sym in ptr.get("symbols", [])[:8]:
            lines.append(f"  - {sym}")
    for td in packet.get("tool_digest", []):
        lines.append(f"\n### Tool: {td.get('tool', 'shell')}")
        lines.append(td.get("distilled", ""))
    for q in packet.get("open_questions", []):
        lines.append(f"\n- **Open:** {q}")
    return "\n".join(lines)


def prepare_cloud_context(
    query: str,
    search_results: list[dict[str, Any]],
    *,
    max_output_tokens: int = 4000,
    include_full_docs: bool = False,
    get_document_fn=None,
    tool_outputs: list[dict[str, str]] | None = None,
    user_prompt: str | None = None,
    max_distill_docs: int = 3,
) -> dict[str, Any]:
    """Build a structured packet from search hits (+ optional full-doc distill)."""
    packet: dict[str, Any] = {
        "query": query,
        "meta": {"profile": "fast", "distilled_docs": 0},
        "prior_work": [],
        "code_pointers": [],
        "tool_digest": [],
        "open_questions": [],
    }

    distilled = 0
    for r in search_results[:max_distill_docs + 2]:
        src = r.get("source", "")
        path = r.get("path", "")
        summary = r.get("session_summary") or r.get("summary") or ""
        entry: dict[str, Any] = {
            "source": src,
            "path": path,
            "relevance": r.get("confidence", "medium"),
            "distilled": summary[:500] if summary else "",
        }

        if include_full_docs and get_document_fn and distilled < max_distill_docs:
            try:
                full = get_document_fn(src, path)
            except Exception as exc:
                log.warning("get_document failed for %s/%s: %s", src, path, exc)
                full = ""
            if full and len(full) > 1500:
                profile = pick_profile("runtime_distill", path=path, source=src, char_count=len(full))
                blob = distill_blob(
                    full,
                    intent="code" if profile == "code" else "default",
                    path=path,
                    model_profile=profile,
                    max_output_tokens=max_output_tokens,
                )
                entry["distilled"] = blob.get("digest", summary)
                packet["meta"]["profile"] = blob.get("profile", profile)
                distilled += 1
                symbols = blob.get("symbols") or []
                if symbols:
                    packet["code_pointers"].append({
                        "path": path,
                        "symbols": symbols[:12],
                        "why": query[:120],
                    })
        packet["prior_work"].append(entry)

    packet["meta"]["distilled_docs"] = distilled

    if tool_outputs:
        for t in tool_outputs[:5]:
            stdout = t.get("stdout", "") or t.get("output", "")
            stderr = t.get("stderr", "")
            combined = f"{stdout}\n{stderr}".strip()
            if not combined:
                continue
            blob = distill_blob(
                combined,
                intent="tool_output",
                path=t.get("path", ""),
                max_output_tokens=max_output_tokens,
            )
            packet["tool_digest"].append({
                "tool": t.get("name", "shell"),
                "distilled": blob.get("digest", combined[:1500]),
                "errors": blob.get("errors", [])[:5],
            })

    if user_prompt and len(user_prompt) > 500:
        blob = distill_blob(
            user_prompt[:20_000],
            intent="default",
            max_output_tokens=max_output_tokens,
        )
        packet["open_questions"].append(
            "User task (distilled): " + blob.get("digest", user_prompt[:300])
        )

    packet["markdown"] = render_cloud_packet(packet)
    return packet
