"""Tests for distill helpers (no Ollama required)."""

import json

from agents_rag.distill import distill_blob, prepare_cloud_context, render_cloud_packet


def test_distill_blob_empty():
    out = distill_blob("")
    assert out["digest"] == ""
    assert out["profile"] == "none"


def test_render_cloud_packet():
    packet = {
        "meta": {"profile": "fast"},
        "prior_work": [{"source": "personal", "path": "a.md", "distilled": "Did X"}],
        "code_pointers": [],
        "tool_digest": [],
        "open_questions": [],
    }
    md = render_cloud_packet(packet)
    assert "a.md" in md
    assert "Did X" in md


def test_prepare_cloud_context_propagates_max_output_tokens(monkeypatch):
    calls = []

    def fake_distill_blob(text, intent="default", *, path="", model_profile=None, max_output_tokens=1500):
        calls.append({
            "intent": intent,
            "path": path,
            "max_output_tokens": max_output_tokens,
            "text": text,
        })
        return {"digest": f"digest-{intent}", "profile": model_profile or "fast"}

    monkeypatch.setattr("agents_rag.distill.distill_blob", fake_distill_blob)

    packet = prepare_cloud_context(
        "query",
        [{"source": "repo", "path": "x.py", "summary": "short"}],
        max_output_tokens=321,
        include_full_docs=True,
        get_document_fn=lambda _src, _path: "x" * 2001,
        tool_outputs=[{"name": "shell", "stdout": "stderr-like output", "stderr": "", "path": "build.log"}],
        user_prompt="u" * 700,
        max_distill_docs=1,
    )

    assert packet["meta"]["distilled_docs"] == 1
    assert len(calls) == 3
    assert all(c["max_output_tokens"] == 321 for c in calls)
