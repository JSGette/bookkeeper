"""Tests for distill helpers (no Ollama required)."""

import json

from agents_rag.distill import distill_blob, render_cloud_packet


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
