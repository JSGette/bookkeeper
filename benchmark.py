"""Read-only retrieval/packet evaluation against user-labeled JSONL cases.

Measures retrieval and substring support, not generated answer quality.
Local model inference is opt-in; the default is lexical retrieval only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

import tiktoken
import yaml

from agents_rag import db
from agents_rag.distill import prepare_cloud_context, serialize_packet
from agents_rag.searcher import Searcher


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_cases(path: str | Path) -> list[dict]:
    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Case line {number}: invalid JSON") from exc
        if not isinstance(case, dict) or not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"Case line {number}: query must be a nonempty string")
        for key in ("expected_paths", "required_evidence"):
            value = case.get(key, [] if key == "required_evidence" else None)
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                raise ValueError(f"Case line {number}: {key} must be a list of nonempty strings")
            case[key] = list(dict.fromkeys(value))
        if not isinstance(case.get("unanswerable", False), bool):
            raise ValueError(f"Case line {number}: unanswerable must be boolean")
        if case.get("unanswerable") and (case["expected_paths"] or case["required_evidence"]):
            raise ValueError(f"Case line {number}: unanswerable cases cannot also specify supporting evidence")
        for key in ("source", "category"):
            if key in case and not isinstance(case[key], str):
                raise ValueError(f"Case line {number}: {key} must be a string")
        case.setdefault("category", "uncategorized")
        cases.append(case)
    if not cases:
        raise ValueError("The case file contains no labeled cases")
    return cases


@contextmanager
def readonly_database(path: str | Path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Database does not exist: {resolved}")
    conn = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != db.SCHEMA_VERSION:
            raise ValueError(f"Evaluation requires schema {db.SCHEMA_VERSION}; found {version}. Migrate a separate copy first.")
        yield conn
    finally:
        conn.close()


def _corpus_fingerprint(conn) -> dict:
    """Hash the logical snapshot, including WAL-visible data, without writing it."""
    digest, counts = hashlib.sha256(), {}
    for table in ("documents", "chunks", "document_revisions", "evidence_segments", "embedding_spaces", "deprecated"):
        count = 0
        digest.update(table.encode())
        for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY 1'):
            values = [hashlib.sha256(value).hexdigest() if isinstance(value, bytes) else value for value in row]
            digest.update(json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode())
            count += 1
        counts[table] = count
    return {"sha256": digest.hexdigest(), "counts": counts, "schema_version": db.SCHEMA_VERSION,
            "generation": db.corpus_generation(conn)}


@contextmanager
def _local_model_settings(config: dict):
    """Configured-model mode may use cached models, never download missing ones."""
    overrides = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")}
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _configured_embedding_profile(conn) -> dict:
    from agents_rag import indexer
    known = [json.loads(row[0]) for row in conn.execute("SELECT metadata_json FROM embedding_spaces")]
    profile = indexer.embedding_profile()
    if not any(item.get("model") == profile["model"] and item.get("revision") not in (None, "unresolved") for item in known):
        raise ValueError("No known embedding revision for the configured model; use --lexical-only or rebuild a separate index")
    indexer.embedding_token_budget()
    profile = indexer.embedding_profile()
    if profile not in known:
        raise ValueError("Configured embedding profile does not match this database; use --lexical-only")
    return profile


def _settings(config: dict, lexical_only: bool, profile: dict, top_k: int) -> dict:
    reranker, search = config.get("reranker", {}), config.get("search", {})
    return {
        "mode": "lexical_only" if lexical_only else "configured_local_models",
        "configured_embedding_model": os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "top_k": top_k, "score_threshold_z": float(search.get("score_threshold_z", 0.8)),
        "min_results": int(search.get("min_results", 2)),
        "searcher": {
            "reranker_enabled": False if lexical_only else bool(reranker.get("enabled", True)),
            "reranker_model": reranker.get("model", "BAAI/bge-reranker-base"),
            "reranker_pool_multiplier": int(reranker.get("pool_multiplier", 6)),
            "rrf_k": int(search.get("rrf_k", 10)), "expected_embedding_profile": profile,
        },
    }


def _coverage(required: list[str], texts: list[str]) -> float | None:
    return sum(any(marker in text for text in texts) for marker in required) / len(required) if required else None


def _raw_texts(results: list[dict]) -> list[str]:
    texts = []
    for result in results:
        for span in result.get("evidence_spans") or [result]:
            if "summary" in str(span.get("completeness", "")) or span.get("derivation") == "legacy_summary":
                continue
            text = next((span[key] for key in ("evidence_text", "evidence", "text", "content")
                         if isinstance(span.get(key), str) and span[key]), "")
            if text:
                texts.append(text)
    return texts


def _oracle_texts(conn, case: dict) -> list[str]:
    """Current labeled source documents, independent of what retrieval found."""
    if not case["expected_paths"]:
        return []
    placeholders = ",".join("?" for _ in case["expected_paths"])
    sql = f"SELECT content,completeness FROM documents WHERE path IN ({placeholders})"
    params = list(case["expected_paths"])
    if case.get("source"):
        sql += " AND source=?"
        params.append(case["source"])
    return [content for content, completeness in conn.execute(sql, params)
            if content and "summary" not in (completeness or "")]


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summarize(rows: list[dict]) -> dict:
    def mean(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return statistics.mean(values) if values else None
    abstained = [row for row in rows if row["abstention_proxy"]]
    unanswerable = [row for row in rows if row["unanswerable"]]
    return {
        "cases": len(rows), "retrieval_path_recall": mean("retrieval_path_recall"),
        "retrieved_evidence_coverage": mean("retrieved_evidence_coverage"),
        "packed_evidence_coverage": mean("packed_evidence_coverage"),
        "oracle_source_evidence_coverage": mean("oracle_source_evidence_coverage"),
        "mean_packet_tokens": mean("packet_tokens"),
        "max_packet_tokens": max((row["packet_tokens"] for row in rows if row["packet_tokens"] is not None), default=None),
        "budget_errors": sum("packet_error" in row for row in rows),
        "abstention_proxy_precision": sum(row["unanswerable"] for row in abstained) / len(abstained) if abstained else None,
        "unanswerable_abstention_proxy_rate": sum(row["abstention_proxy"] for row in unanswerable) / len(unanswerable) if unanswerable else None,
        "latency_ms": {stage: {"p50": _percentile([row[stage + "_ms"] for row in rows], 0.5),
                                "p95": _percentile([row[stage + "_ms"] for row in rows], 0.95)}
                       for stage in ("retrieval", "packing", "total")},
    }


def evaluate(conn, cases: list[dict], config: dict, *, budgets=(1000, 2000, 4000),
             lexical_only: bool = True, top_k: int = 5) -> dict:
    setup_started = time.perf_counter()
    if top_k < 1 or not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("top_k and budgets must be positive")
    profile = {"evaluation_mode": "lexical_only"} if lexical_only else _configured_embedding_profile(conn)
    settings = _settings(config, lexical_only, profile, top_k)
    searcher = Searcher(conn, **settings["searcher"])
    searcher.load()
    if lexical_only:
        searcher._dense_eligible[:] = False
        searcher._hnsw = None
    encoder = tiktoken.get_encoding("cl100k_base")
    setup_ms = (time.perf_counter() - setup_started) * 1000
    records = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        results, search_meta = searcher.query_with_meta(
            case["query"], top_k=top_k, source=case.get("source") or None,
            score_threshold_z=settings["score_threshold_z"], min_results=settings["min_results"],
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        retrieved_paths = {result.get("path", "").replace("\\", "/") for result in results}
        expected_paths = {path.replace("\\", "/") for path in case["expected_paths"]}
        required = case.get("required_evidence", [])
        baseline = {
            "case": index, "query": case["query"], "category": case.get("category", "uncategorized"),
            "unanswerable": case.get("unanswerable", False), "expected_paths": sorted(expected_paths),
            "retrieved_paths": sorted(retrieved_paths), "search_meta": search_meta,
            "retrieval_path_recall": len(expected_paths & retrieved_paths) / len(expected_paths) if expected_paths else None,
            "retrieved_evidence_coverage": _coverage(required, _raw_texts(results)),
            "oracle_source_evidence_coverage": _coverage(required, _oracle_texts(conn, case)),
            "retrieval_ms": retrieval_ms,
        }
        for budget in dict.fromkeys(budgets):
            started = time.perf_counter()
            row = {**baseline, "budget": budget}
            try:
                packet = prepare_cloud_context(case["query"], results, max_context_tokens=budget)
                wire_tokens = len(encoder.encode(serialize_packet(packet), disallowed_special=()))
                if wire_tokens > budget or wire_tokens != packet["meta"]["token_count"]:
                    raise AssertionError("Serialized context violated its declared budget")
                row.update(packet=packet, packet_tokens=wire_tokens,
                           packed_evidence_coverage=_coverage(required, _raw_texts(packet["prior_work"])),
                           abstention_proxy=not packet["prior_work"] or bool(search_meta.get("weak_match", True)))
            except ValueError as exc:
                row.update(packet_error=str(exc), packet_tokens=None, packed_evidence_coverage=0.0 if required else None,
                           abstention_proxy=True)
            row["packing_ms"] = (time.perf_counter() - started) * 1000
            row["total_ms"] = retrieval_ms + row["packing_ms"]
            records.append(row)
    corpus = _corpus_fingerprint(conn)
    versions = {}
    for package in ("tiktoken", "sentence-transformers", "rank-bm25", "hnswlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    from agents_rag import reranker
    reranker_revision = "unresolved"
    if not lexical_only and reranker._model_name_loaded == settings["searcher"]["reranker_model"]:
        reranker_config = getattr(getattr(reranker._model, "model", None), "config", None)
        reranker_revision = getattr(reranker_config, "_commit_hash", None) or "unresolved"
    model_identity = {"embedding": profile, "reranker": {
        "model": settings["searcher"]["reranker_model"], "revision": reranker_revision,
        "enabled": settings["searcher"]["reranker_enabled"],
    }}
    return {
        "evaluation_version": 1, "settings": settings, "corpus": corpus, "dependencies": versions,
        "setup_ms": setup_ms, "model_identity": model_identity,
        "fingerprints": {"settings": _digest(settings), "cases": _digest(cases), "corpus": corpus["sha256"],
                         "embedding_profile": _digest(profile), "models": _digest(model_identity), "config": _digest(config)},
        "definitions": {
            "tokens": "Full serialize_packet response, cl100k_base; approximate for other provider tokenizers.",
            "evidence_coverage": "Case-sensitive required substrings within raw source spans; excludes summaries; not entailment or answer correctness.",
            "oracle_source_evidence_coverage": "Required substrings in current labeled documents, regardless of retrieval; not a historical-revision oracle.",
            "abstention_proxy": "No packed evidence OR Searcher weak_match. Heuristic, uncalibrated; lexical-only often marks relevant hits weak.",
            "latency": "One retrieval per case reused across budgets; packing measured separately. Index/embedding setup reported separately; first reranker call may be cold. No generative/cloud LLM calls.",
        },
        "budgets": {str(budget): _summarize([row for row in records if row["budget"] == budget]) for budget in dict.fromkeys(budgets)},
        "categories": {category: {str(budget): _summarize([row for row in records if row["category"] == category and row["budget"] == budget])
                                  for budget in dict.fromkeys(budgets)} for category in sorted({row["category"] for row in records})},
        "cases": records,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Explicit schema-v1 database; opened read-only")
    parser.add_argument("--cases", required=True, help="JSONL: query, expected_paths, optional required_evidence/source/unanswerable/category")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--budgets", nargs="+", type=int, default=[1000, 2000, 4000])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--lexical-only", action="store_true", help="Default: no model inference")
    mode.add_argument("--configured-models", action="store_true", help="Opt in to configured cached local embeddings/reranking; no downloads")
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).expanduser().resolve()
        database_path = Path(args.db).expanduser().resolve()
        protected = {Path(value).expanduser().resolve() for value in (args.db, args.cases, args.config)}
        protected.update(Path(str(database_path) + suffix) for suffix in ("-wal", "-shm", "-journal"))
        if output in protected or (output.exists() and any(path.exists() and output.samefile(path) for path in protected)):
            raise ValueError("Output must not overwrite the database, cases or configuration")
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a mapping")
        cases = load_cases(args.cases)
        with readonly_database(args.db) as conn, _local_model_settings(config):
            report = evaluate(conn, cases, config, budgets=args.budgets,
                              lexical_only=not args.configured_models, top_k=args.top_k)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    except (ValueError, OSError, sqlite3.DatabaseError) as exc:
        parser.error(str(exc))
    print(f"Evaluated {len(cases)} labeled cases in {report['settings']['mode']} mode; report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
