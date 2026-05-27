"""Tests for ollama_client profile routing."""

from agents_rag.ollama_client import configure, is_code_path, pick_profile


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
