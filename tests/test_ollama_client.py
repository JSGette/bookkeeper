"""Tests for ollama_client profile routing."""

from agents_rag.ollama_client import chat_json, configure, is_code_path, pick_profile


def test_is_code_path():
    assert is_code_path("pkg/foo.go")
    assert not is_code_path("notes/readme.md")


def test_pick_profile_tagger():
    assert pick_profile("tagger") == "fast"


def test_pick_profile_code_index():
    assert pick_profile("index_summary", path="a/b/c.py") == "code"
    assert pick_profile("index_summary", path="doc.md") == "fast"


def test_pick_profile_runtime():
    assert pick_profile("runtime", char_count=1000, code_heavy=False) == "fast"
    assert pick_profile("runtime", char_count=50_000, code_heavy=True) == "code"


def test_configure_profiles():
    configure({
        "ollama": {
            "models": {
                "fast": {"name": "gemma-test"},
                "code": {"name": "qwen-test"},
            },
        },
    })
    from agents_rag.ollama_client import get_profile
    assert get_profile("fast").name == "gemma-test"
    assert get_profile("code").name == "qwen-test"


def test_configure_profiles_preserves_missing_defaults():
    configure({"ollama": {"models": {"fast": {"name": "gemma-only"}}}})
    from agents_rag.ollama_client import get_profile

    assert get_profile("fast").name == "gemma-only"
    # Missing profile keeps a default fallback instead of raising.
    assert get_profile("code").name


def test_chat_json_unknown_profile_falls_back_to_fast(monkeypatch):
    configure({"ollama": {"models": {"fast": {"name": "gemma-fallback"}}}})

    seen = {}

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "{\"ok\": true}"}}

    def fake_post(_url, json, timeout):
        seen["model"] = json["model"]
        seen["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr("agents_rag.ollama_client.httpx.post", fake_post)
    out = chat_json("missing-profile", [{"role": "user", "content": "hi"}], {"type": "object"})

    assert out == {"ok": True}
    assert seen["model"] == "gemma-fallback"
