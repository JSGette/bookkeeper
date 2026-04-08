"""Three-layer tagger: path-based (instant) → LLM via Ollama (cached) → manual overrides."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import httpx
import yaml

log = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))

_STRIP_PREFIXES = {"docs", "public", "src"}

# Module-level tagger config, set via configure()
_tagger_prompt: str | None = None
_tagger_examples_path: str | None = None


def configure(config: dict) -> None:
    """Apply tagger settings from the top-level config dict."""
    global _tagger_prompt, _tagger_examples_path
    _tagger_prompt = config.get("tagger_prompt")
    _tagger_examples_path = config.get("tagger_examples")


# ---------------------------------------------------------------------------
# Layer 1: path-based tags
# ---------------------------------------------------------------------------

def tags_from_path(rel_path: str) -> list[str]:
    parts = Path(rel_path).parts
    tags = []
    for p in parts[:-1]:
        token = p.lower().replace("_", "-")
        if token not in _STRIP_PREFIXES and len(token) > 1:
            tags.append(token)
    stem = Path(rel_path).stem.lower().replace("_", "-")
    if stem not in ("readme", "index") and len(stem) > 1:
        tags.append(stem)
    return list(dict.fromkeys(tags))


def description_from_content(text: str) -> str:
    """Extract first non-empty, non-heading paragraph as a description."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|") or stripped.startswith("```"):
            continue
        if stripped.startswith("---"):
            continue
        if stripped.startswith("-") or stripped.startswith("*"):
            continue
        return stripped[:200]
    return ""


# ---------------------------------------------------------------------------
# Layer 2: LLM via Ollama (cached per file content)
# ---------------------------------------------------------------------------

def _tag_cache_path(cache_dir: str, rel_path: str, text: str) -> Path:
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    key = hashlib.sha256(f"{rel_path}:{content_hash}".encode()).hexdigest()[:16]
    return Path(cache_dir) / "tags" / f"{key}.json"


def _load_cached_tags(cache_dir: str, rel_path: str, text: str) -> dict | None:
    p = _tag_cache_path(cache_dir, rel_path, text)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_cached_tags(cache_dir: str, rel_path: str, text: str, data: dict) -> None:
    p = _tag_cache_path(cache_dir, rel_path, text)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


_DEFAULT_SYSTEM_PROMPT = (
    "You are a technical documentation indexer. Your job is to write a one-sentence "
    "description and 5-10 tags for each document.\n\n"
    "Rules for the description:\n"
    "- Describe WHY someone would read this document, not what it contains.\n"
    "- Be specific: mention exact tools, thresholds, versions, or root causes.\n"
    "- NEVER start with 'This document', 'This page', 'A comprehensive', or 'An overview'.\n"
    "- Max 200 characters.\n\n"
    "Rules for tags:\n"
    "- Use lowercase, hyphenated compound terms where meaningful (e.g. build-system, "
    "quality-gate).\n"
    "- Include domain-specific terms, not generic filler (e.g. data, report, changes, "
    "documentation, table).\n"
    "- 5-10 tags.\n\n"
    "Output a JSON object with keys 'description' and 'tags'."
)


def _get_system_prompt() -> str:
    return _tagger_prompt or _DEFAULT_SYSTEM_PROMPT


_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "tags"],
}

_FILLER_TAGS = frozenset({
    "data", "report", "changes", "documentation", "table", "document",
    "summary", "overview", "information", "content", "details", "page",
    "text", "file", "section", "list", "item", "update", "status",
})

_DESCRIPTION_STRIP_RE = re.compile(
    r"^(This (document|page|plan|file|section|table|diagram)|"
    r"A (comprehensive|detailed|complete|brief|short|high-level)|"
    r"An (overview|analysis|examination)|"
    r"Learn how to|Find the|Guidance on|Understand (how|the|why)|"
    r"Procedure for|Step-by-step guide)\s+",
    re.IGNORECASE,
)

_GARBAGE_PATTERNS = re.compile(
    r"(no specific|not enough context|no technical details|cannot determine)",
    re.IGNORECASE,
)


def _load_few_shot_examples() -> list[dict]:
    """Load few-shot examples from configured path or bundled default."""
    if _tagger_examples_path:
        examples_path = Path(os.path.expanduser(_tagger_examples_path))
    else:
        examples_path = Path(__file__).parent / "examples.jsonl"
    if not examples_path.exists():
        return []
    examples = []
    for line in examples_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def _build_messages(text: str) -> list[dict]:
    """Build the chat messages array with system prompt, few-shot examples, and the document."""
    messages = [{"role": "system", "content": _get_system_prompt()}]

    for ex in _load_few_shot_examples():
        messages.append({"role": "user", "content": ex["excerpt"]})
        messages.append({"role": "assistant", "content": json.dumps({
            "description": ex["description"],
            "tags": ex["tags"],
        })})

    messages.append({"role": "user", "content": text[:100_000]})
    return messages


def _clean_description(desc: str) -> str:
    """Strip generic openers and detect garbage output."""
    desc = desc.strip()
    if _GARBAGE_PATTERNS.search(desc) or len(desc) < 20:
        return ""
    desc = _DESCRIPTION_STRIP_RE.sub("", desc)
    if desc and desc[0].islower():
        desc = desc[0].upper() + desc[1:]
    return desc[:200]


def _clean_tags(tags: list) -> list[str]:
    """Filter filler tags and normalize."""
    cleaned = []
    seen: set[str] = set()
    for t in tags:
        t = str(t).lower().strip()
        if not t or len(t) < 2 or t in _FILLER_TAGS or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    return cleaned[:10]


def generate_tags_gemma(text: str, rel_path: str, cache_dir: str, force: bool = False) -> dict | None:
    """Call LLM via Ollama to generate semantic tags and description.

    Returns {"description": str, "tags": [str]} or None on failure.
    """
    if not force:
        cached = _load_cached_tags(cache_dir, rel_path, text)
        if cached is not None:
            return cached

    messages = _build_messages(text)

    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "format": _FORMAT_SCHEMA,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 64,
                    "num_predict": 1024,
                    "repeat_penalty": 1.15,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
    except Exception as exc:
        log.warning("Ollama unavailable (%s), falling back to path-based tags for %s", exc, rel_path)
        return None

    raw = re.sub(r"^```json?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        data = json.loads(raw)
        result = {
            "description": _clean_description(str(data.get("description", ""))),
            "tags": _clean_tags(data.get("tags", [])),
        }
        _save_cached_tags(cache_dir, rel_path, text, result)
        return result
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("Failed to parse Gemma response for %s: %s — raw: %s", rel_path, exc, raw[:200])
        return None


# ---------------------------------------------------------------------------
# Layer 3: manual overrides from .agents-rag-meta.yaml
# ---------------------------------------------------------------------------

def load_overrides(source_root: str) -> dict[str, dict]:
    """Load all .agents-rag-meta.yaml files under a source root.

    Returns {relative_path: {"description": str, "tags": [str]}}.
    """
    root = Path(os.path.expanduser(source_root))
    overrides: dict[str, dict] = {}
    for meta_path in root.rglob(".agents-rag-meta.yaml"):
        try:
            data = yaml.safe_load(meta_path.read_text()) or {}
        except (yaml.YAMLError, OSError):
            continue
        meta_dir = meta_path.parent.relative_to(root)
        for filename, entry in data.get("files", {}).items():
            rel = str(meta_dir / filename) if str(meta_dir) != "." else filename
            overrides[rel] = {
                "description": str(entry.get("description", "")),
                "tags": [str(t) for t in entry.get("tags", [])],
            }
    return overrides


# ---------------------------------------------------------------------------
# Combined tagger
# ---------------------------------------------------------------------------

def tag_file(
    rel_path: str,
    text: str,
    source_root: str,
    cache_dir: str,
    overrides: dict[str, dict],
    use_gemma: bool = True,
) -> tuple[str, list[str]]:
    """Return (description, tags) for a file, merging all three layers.

    Priority: manual override > LLM > path-based / first-paragraph.
    Tags are unioned across all layers.
    """
    path_tags = tags_from_path(rel_path)
    fallback_desc = description_from_content(text)

    gemma_desc = ""
    gemma_tags: list[str] = []
    if use_gemma:
        result = generate_tags_gemma(text, rel_path, cache_dir)
        if result:
            gemma_desc = result.get("description", "")
            gemma_tags = result.get("tags", [])

    override = overrides.get(rel_path, {})
    override_desc = override.get("description", "")
    override_tags = override.get("tags", [])

    # Gemma garbage detection returns "" — fall through to fallback_desc
    description = override_desc or gemma_desc or fallback_desc

    all_tags = override_tags + gemma_tags + path_tags
    seen: set[str] = set()
    tags: list[str] = []
    for t in all_tags:
        t_lower = t.lower().strip()
        if t_lower and t_lower not in seen:
            seen.add(t_lower)
            tags.append(t_lower)

    return description, tags[:10]
