"""Offline evaluation checks using synthetic, explicitly labeled temporary data."""

import hashlib
import json
import sqlite3

import numpy as np
import pytest
import tiktoken

import benchmark
from agents_rag import db
from agents_rag.distill import serialize_packet


@pytest.fixture
def evaluation_db(tmp_path):
    path = tmp_path / "evaluation.db"
    conn = db.init_db(str(path))
    document = db.upsert_document(conn, "repo", "a.md", "Title", "", "marker-one\nmarker-two")
    db.upsert_chunks(conn, document, [{"text": "marker-one", "start_line": 1, "end_line": 1}],
                     np.array([[1.0, 0.0]], dtype=np.float32))
    conn.close()
    return path


class StubSearcher:
    init_calls = []
    query_calls = []

    def __init__(self, conn, **kwargs):
        self.init_calls.append(kwargs)
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        self._dense_eligible = np.array([True])

    def load(self):
        return True

    def query_with_meta(self, query, **kwargs):
        self.query_calls.append((query, kwargs))
        if query == "retrieval":
            results = [{"source": "repo", "path": "a.md", "confidence": "medium",
                        "summary": "An intentionally lossy summary", "evidence_spans": [
                            {"evidence_id": "e-1", "revision_id": "r-1", "text": "marker-one", "start_line": 1, "end_line": 1},
                            {"evidence_id": "e-2", "revision_id": "r-1", "text": "marker-two " + "✅" * 100,
                             "start_line": 2, "end_line": 2},
                        ]}]
        elif query == "summary":
            results = [{"source": "repo", "path": "a.md", "summary": "marker-one marker-two"}]
        else:
            results = []
        return results, {"weak_match": not results, "score_kind": "rrf", "reranker_used": False}


def test_wire_budget_distinguishes_retrieved_packed_and_oracle_support(evaluation_db, monkeypatch):
    StubSearcher.init_calls.clear()
    StubSearcher.query_calls.clear()
    monkeypatch.setattr(benchmark, "Searcher", StubSearcher)
    cases = [
        {"query": "retrieval", "expected_paths": ["a.md"], "required_evidence": ["marker-one", "marker-two"], "source": "repo"},
        {"query": "summary", "expected_paths": ["a.md"], "required_evidence": ["marker-one", "marker-two"]},
        {"query": "miss", "expected_paths": ["a.md"], "required_evidence": ["marker-one"]},
        {"query": "unanswerable", "expected_paths": [], "unanswerable": True},
    ]
    before = hashlib.sha256(evaluation_db.read_bytes()).hexdigest()
    with benchmark.readonly_database(evaluation_db) as conn:
        report = benchmark.evaluate(conn, cases, {}, budgets=[512, 2000])
    assert hashlib.sha256(evaluation_db.read_bytes()).hexdigest() == before
    assert len(StubSearcher.query_calls) == len(cases)  # Retrieval reused across budgets.
    encoder = tiktoken.get_encoding("cl100k_base")
    for row in report["cases"]:
        tokens = len(encoder.encode(serialize_packet(row["packet"]), disallowed_special=()))
        assert tokens == row["packet_tokens"] == row["packet"]["meta"]["token_count"]
        assert tokens <= row["budget"]
    retrieved = [row for row in report["cases"] if row["query"] == "retrieval"]
    assert retrieved[0]["retrieval_path_recall"] == 1
    assert retrieved[0]["retrieved_evidence_coverage"] == 1
    assert retrieved[0]["oracle_source_evidence_coverage"] == 1
    assert retrieved[0]["packed_evidence_coverage"] == 0.5
    assert retrieved[1]["packed_evidence_coverage"] == 1
    summarized = next(row for row in report["cases"] if row["query"] == "summary")
    assert summarized["retrieved_evidence_coverage"] == summarized["packed_evidence_coverage"] == 0
    assert summarized["oracle_source_evidence_coverage"] == 1
    missed = next(row for row in report["cases"] if row["query"] == "miss")
    assert missed["retrieval_path_recall"] == missed["retrieved_evidence_coverage"] == 0
    assert missed["oracle_source_evidence_coverage"] == 1
    assert report["budgets"]["512"]["unanswerable_abstention_proxy_rate"] == 1
    assert "not entailment or answer correctness" in report["definitions"]["evidence_coverage"]
    assert report["budgets"]["512"]["latency_ms"]["total"]["p95"] >= 0
    assert report["corpus"]["counts"]["documents"] == 1


def test_configured_mode_passes_production_settings_and_known_profile(evaluation_db, monkeypatch):
    StubSearcher.init_calls.clear()
    StubSearcher.query_calls.clear()
    monkeypatch.setattr(benchmark, "Searcher", StubSearcher)
    profile = {"model": "cached-model", "revision": "known-sha", "input_format": "bounded-evidence-v1"}
    monkeypatch.setattr(benchmark, "_configured_embedding_profile", lambda conn: profile)
    config = {"reranker": {"enabled": True, "model": "cached-reranker", "pool_multiplier": 9},
              "search": {"rrf_k": 17, "score_threshold_z": 0.3, "min_results": 4}}
    with benchmark.readonly_database(evaluation_db) as conn:
        report = benchmark.evaluate(conn, [{"query": "missing", "expected_paths": []}], config,
                                    lexical_only=False, top_k=8, budgets=[512])
    assert StubSearcher.init_calls[-1] == {
        "reranker_enabled": True, "reranker_model": "cached-reranker", "reranker_pool_multiplier": 9,
        "rrf_k": 17, "expected_embedding_profile": profile,
    }
    assert StubSearcher.query_calls[-1][1] == {"top_k": 8, "source": None, "score_threshold_z": 0.3, "min_results": 4}
    assert report["settings"]["mode"] == "configured_local_models"
    assert report["fingerprints"]["embedding_profile"] == benchmark._digest(profile)


def test_real_lexical_search_never_calls_models_or_writes_database(evaluation_db, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Lexical evaluation must not invoke any model")

    monkeypatch.setattr("agents_rag.searcher.embed_texts", forbidden)
    monkeypatch.setattr("agents_rag.searcher._rerank_fn", forbidden)
    monkeypatch.setattr("agents_rag.distill.distill_blob", forbidden)
    with benchmark.readonly_database(evaluation_db) as conn:
        fingerprint = benchmark._corpus_fingerprint(conn)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("UPDATE corpus_state SET generation=999")
        report = benchmark.evaluate(conn, [{"query": "marker-one", "expected_paths": ["a.md"],
                                            "required_evidence": ["marker-one"]}],
                                    {"reranker": {"enabled": True}}, budgets=[1000])
        assert benchmark._corpus_fingerprint(conn) == fingerprint
        assert conn.execute("SELECT COUNT(*) FROM hnsw_cache").fetchone()[0] == 0
    assert report["cases"][0]["retrieved_evidence_coverage"] == 1
    assert report["settings"]["searcher"]["reranker_enabled"] is False


def test_cli_writes_report_from_explicit_inputs(evaluation_db, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "Searcher", StubSearcher)
    cases, config, output = tmp_path / "cases.jsonl", tmp_path / "config.yaml", tmp_path / "report.json"
    cases.write_text(json.dumps({"query": "missing", "expected_paths": [], "unanswerable": True}) + "\n", encoding="utf-8")
    config.write_text("reranker:\n  enabled: true\n", encoding="utf-8")
    assert benchmark.main(["--db", str(evaluation_db), "--cases", str(cases), "--config", str(config),
                           "--budgets", "512", "1000", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["settings"]["mode"] == "lexical_only"
    assert set(report["budgets"]) == {"512", "1000"}


def test_readonly_open_rejects_legacy_schema_without_migrating(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE legacy(value TEXT)")
    conn.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Migrate a separate copy"):
        with benchmark.readonly_database(path):
            pytest.fail("Unsupported schema must not be evaluated")
    assert path.read_bytes() == before


@pytest.mark.parametrize("case", [
    {"query": "missing labels"},
    {"query": "wrong type", "expected_paths": "a.md"},
    {"query": "contradiction", "expected_paths": ["a.md"], "unanswerable": True},
    {"query": "bad markers", "expected_paths": [], "required_evidence": [1]},
])
def test_case_validation_rejects_ambiguous_labels(tmp_path, case):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(ValueError, match="Case line 1"):
        benchmark.load_cases(path)


def test_impossible_budget_is_reported_without_truncating_query(evaluation_db, monkeypatch):
    monkeypatch.setattr(benchmark, "Searcher", StubSearcher)
    query = "Exact user instruction " * 500
    with benchmark.readonly_database(evaluation_db) as conn:
        report = benchmark.evaluate(conn, [{"query": query, "expected_paths": []}], {}, budgets=[100])
    assert report["cases"][0]["query"] == query
    assert "packet_error" in report["cases"][0]
    assert report["budgets"]["100"]["budget_errors"] == 1


def test_configured_mode_refuses_unknown_embedding_profile(evaluation_db, monkeypatch):
    monkeypatch.setattr("agents_rag.indexer.embedding_token_budget", lambda: pytest.fail("Unknown spaces must not load a model"))
    with benchmark.readonly_database(evaluation_db) as conn:
        with pytest.raises(ValueError, match="No known embedding revision"):
            benchmark.evaluate(conn, [{"query": "query", "expected_paths": []}], {}, lexical_only=False)


def test_output_cannot_overwrite_database_sidecar(evaluation_db, tmp_path):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--db", str(evaluation_db), "--cases", str(tmp_path / "unused.jsonl"),
                        "--output", str(evaluation_db) + "-wal"])
    assert exc.value.code == 2
