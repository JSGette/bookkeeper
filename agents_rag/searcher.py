"""Hybrid search: BM25 keyword + vector cosine via hnswlib, fused via Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import re
import sqlite3
import tempfile
import threading
from pathlib import Path

import hnswlib
import numpy as np
from rank_bm25 import BM25Okapi

from . import db
from .indexer import embed_texts
from .reranker import rerank as _rerank_fn

log = logging.getLogger(__name__)

# Lower K means rank-0 hits outweigh later ranks more steeply. With the
# cross-encoder reranker owning final precision, RRF's job is just to feed a
# good candidate pool; a smaller K gives that pool a clearer head.
RRF_K = 10
_RERANK_PASSAGE_CHARS = 2000

# Reranker bands remain heuristics, not calibrated probabilities. RRF is
# ranking-only: its magnitude changes with k even when relevance does not.
_RERANK_HIGH = 0.6
_RERANK_LOW = 0.4


def _confidence_for(
    score: float, score_kind: str, *, cosine: float | None = None,
    lexical_support: bool = False,
) -> str:
    if score_kind == "rerank":
        if score >= _RERANK_HIGH:
            return "high"
        if score >= _RERANK_LOW:
            return "medium"
        return "low"
    if cosine is not None and cosine >= 0.5 and lexical_support:
        return "medium"
    return "low"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\-]+", text.lower())


class Searcher:
    def __init__(
        self,
        conn: sqlite3.Connection,
        reranker_enabled: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
        reranker_pool_multiplier: int = 6,
        rrf_k: int = RRF_K,
        expected_embedding_profile: dict | None = None,
    ):
        self.conn = conn
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict] | None = None
        self._bm25: BM25Okapi | None = None
        self._hnsw: hnswlib.Index | None = None
        self._deprecated: dict[str, str] = {}
        self._reranker_enabled = reranker_enabled
        self._reranker_model = reranker_model
        self._reranker_pool_multiplier = max(1, reranker_pool_multiplier)
        self._rrf_k = max(1, rrf_k)
        self._embedding_profile = expected_embedding_profile
        self._normalized_vectors: np.ndarray | None = None
        self._dense_eligible = np.array([], dtype=bool)
        self._index_fingerprint: str | None = None
        self.loaded_generation: int | None = None
        # Query and load must see the same metadata, vectors and index generation.
        self._load_lock = threading.RLock()

    def _build_bm25(self) -> None:
        if not self._metadata:
            self._bm25 = None
            return
        # Title, heading, and tags are mixed into the tokenized body once — not
        # doubled. Doubling was boosting generic auto-derived titles (slash
        # commands, first-query-as-title session names) so every query sharing a
        # word with the title pinned that doc to the top of BM25.
        corpus = []
        for m in self._metadata:
            tokens = _tokenize(m.get("text", ""))
            title_tokens = _tokenize(m.get("doc_title", ""))
            heading_tokens = _tokenize(m.get("section_heading", ""))
            tag_tokens = _tokenize(" ".join(m.get("tags", [])))
            tokens = title_tokens + heading_tokens + tag_tokens + tokens
            corpus.append(tokens)
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None

    def _build_hnsw(self) -> None:
        """Build hnswlib index from vectors. Try loading cached blob first."""
        self._hnsw = None
        if (self._vectors is None or self._vectors.size == 0
                or not np.all(self._dense_eligible)):
            return

        dim = self._vectors.shape[1]
        n = len(self._vectors)

        cached = db.load_hnsw_cache(self.conn, fingerprint=self._index_fingerprint)
        if cached is not None:
            blob, cached_count = cached
            if cached_count == n:
                tmp_path = None
                try:
                    index = hnswlib.Index(space="cosine", dim=dim)
                    with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
                        f.write(blob)
                        tmp_path = f.name
                    index.load_index(tmp_path, max_elements=n)
                    index.set_ef(50)
                    self._hnsw = index
                    log.info("Loaded cached hnswlib index (%d vectors)", n)
                    return
                except Exception as exc:
                    log.warning("Failed to load cached HNSW, rebuilding: %s", exc)
                finally:
                    if tmp_path:
                        Path(tmp_path).unlink(missing_ok=True)

        # Build fresh
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=max(n, 1), ef_construction=200, M=16)
        index.add_items(self._vectors, list(range(n)))
        index.set_ef(50)
        self._hnsw = index
        log.info("Built hnswlib index (%d vectors)", n)

        self._cache_hnsw()

    def _cache_hnsw(self) -> None:
        """Serialize current HNSW index and store in DB."""
        if self._hnsw is None or self._vectors is None:
            return
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
                tmp_path = f.name
            self._hnsw.save_index(tmp_path)
            blob = Path(tmp_path).read_bytes()
            db.save_hnsw_cache(self.conn, blob, len(self._vectors), fingerprint=self._index_fingerprint)
        except Exception as exc:
            log.warning("Failed to cache HNSW index: %s", exc)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def load(self) -> bool:
        with self._load_lock:
            return self._load_locked()

    def _load_locked(self) -> bool:
        with db.transaction(self.conn):
            meta, vecs = db.load_all_chunks(self.conn, embedding_profile=self._embedding_profile)
            deprecated = db.load_deprecated(self.conn)
            generation = db.corpus_generation(self.conn)
        self._vectors = vecs
        self._metadata = meta
        self._deprecated = deprecated
        self.loaded_generation = generation
        self._dense_eligible = np.array([m.get("dense_eligible", True) for m in meta], dtype=bool)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self._normalized_vectors = vecs / np.where(norms == 0, 1, norms)
        self._index_fingerprint = db.index_fingerprint(meta, vecs)
        if not meta:
            log.warning("No chunks in database")
            self._bm25 = None
            self._hnsw = None
            return False
        self._build_bm25()
        self._build_hnsw()
        log.info("Loaded index: %d chunks, %d deprecated docs", len(meta), len(self._deprecated))
        return True

    def reload(self) -> bool:
        return self.load()

    @property
    def chunk_count(self) -> int:
        with self._load_lock:
            return len(self._metadata) if self._metadata else 0

    def source_stats(self) -> dict[str, int]:
        with self._load_lock:
            counts: dict[str, int] = {}
            for m in self._metadata or []:
                src = m.get("source", "unknown")
                counts[src] = counts.get(src, 0) + 1
            return counts

    def embed_query(self, query_text: str) -> np.ndarray:
        """Public hook so callers (e.g. the dedupe pre-check) can reuse the
        embedding instead of paying for it twice per query."""
        return embed_texts([query_text])[0]

    def _vector_search(
        self,
        query_text: str,
        n: int,
        source: str | None,
        query_vec: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        """Return [(index, rank)] from vector similarity. Uses HNSW when no source filter."""
        if (self._vectors is None or self._metadata is None or self._vectors.size == 0
                or not np.any(self._dense_eligible) or n <= 0):
            return []

        if query_vec is None:
            query_vec = embed_texts([query_text])[0]
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if (query_vec.shape != (self._vectors.shape[1],) or not np.all(np.isfinite(query_vec))
                or not np.linalg.norm(query_vec)):
            return []

        if self._hnsw is not None and source is None:
            labels, _distances = self._hnsw.knn_query(
                query_vec.reshape(1, -1), k=min(n, len(self._vectors)),
            )
            return [(int(labels[0][i]), i) for i in range(len(labels[0]))]

        # Brute-force fallback for source-filtered queries
        query_norm = query_vec / (np.linalg.norm(query_vec) or 1)
        scores = self._normalized_vectors @ query_norm
        scores = np.where(self._dense_eligible, scores, -np.inf)

        if source:
            mask = np.array([m.get("source") == source for m in self._metadata])
            scores = np.where(mask, scores, -np.inf)

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(idx), rank) for rank, idx in enumerate(top_indices) if scores[idx] >= 0]

    def _bm25_search(self, query_text: str, n: int, source: str | None) -> list[tuple[int, int]]:
        if self._bm25 is None or self._metadata is None or n <= 0:
            return []

        tokens = _tokenize(query_text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # Common terms can have nonpositive BM25 IDF in tiny corpora. Token
        # overlap still supplies lexical evidence; score sign is not relevance.
        eligible = np.array([any(t in freq for t in tokens) for freq in self._bm25.doc_freqs])
        if source:
            mask = np.array([m.get("source") == source for m in self._metadata])
            eligible &= mask
        scores = np.where(eligible, scores, -np.inf)

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(idx), rank) for rank, idx in enumerate(top_indices) if eligible[idx]]

    def _rrf_fuse(
        self,
        vector_results: list[tuple[int, int]],
        bm25_results: list[tuple[int, int]],
        top_k: int,
    ) -> tuple[list[int], dict[int, float]]:
        rrf_scores: dict[int, float] = {}
        k = self._rrf_k
        for idx, rank in vector_results:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
        for idx, rank in bm25_results:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
        return sorted_indices[:top_k], rrf_scores

    def _passage_for_rerank(self, idx: int) -> str:
        m = self._metadata[idx] if self._metadata else {}
        parts = [
            m.get("doc_title", ""),
            m.get("section_heading", ""),
            m.get("text", ""),
        ]
        return "\n".join(p for p in parts if p)[:_RERANK_PASSAGE_CHARS]

    def _rerank_pool(
        self,
        query_text: str,
        fused_indices: list[int],
        rrf_scores: dict[int, float],
    ) -> tuple[list[int], dict[int, float], bool]:
        """Rescore fused candidates with the cross-encoder. Falls back to RRF on failure.

        The trailing bool tells callers whether the cross-encoder actually ran —
        needed so audit logs can distinguish reranker-scored rows from silent
        RRF fallbacks that happen to land in the same numeric range.
        """
        if not self._reranker_enabled or not fused_indices or self._metadata is None:
            return fused_indices, rrf_scores, False

        passages = [self._passage_for_rerank(i) for i in fused_indices]
        scores = _rerank_fn(query_text, passages, model_name=self._reranker_model)
        if scores is None or len(scores) != len(fused_indices):
            return fused_indices, rrf_scores, False

        new_scores = {idx: float(scores[i]) for i, idx in enumerate(fused_indices)}
        ordered = sorted(fused_indices, key=lambda i: new_scores[i], reverse=True)
        return ordered, new_scores, True

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        source: str | None = None,
        score_threshold_z: float = 0.8,
        min_results: int = 2,
    ) -> list[dict]:
        results, _meta = self.query_with_meta(
            query_text,
            top_k=top_k,
            source=source,
            score_threshold_z=score_threshold_z,
            min_results=min_results,
        )
        return results

    def query_with_meta(
        self,
        query_text: str,
        top_k: int = 5,
        source: str | None = None,
        score_threshold_z: float = 0.8,
        min_results: int = 2,
        query_embedding: np.ndarray | None = None,
    ) -> tuple[list[dict], dict]:
        """Same as query() but also returns scoring-path metadata for auditing.

        meta keys: reranker_used (bool), rrf_k (int), score_kind ("rerank"|"rrf"),
        confidence ("high"|"medium"|"low"|"none"), weak_match (bool).
        """
        with self._load_lock:
            return self._query_locked(
                query_text, top_k, source, score_threshold_z, min_results, query_embedding,
            )

    def _query_locked(
        self, query_text: str, top_k: int, source: str | None,
        score_threshold_z: float, min_results: int,
        query_embedding: np.ndarray | None,
    ) -> tuple[list[dict], dict]:
        query_meta = {
            "reranker_used": False,
            "rrf_k": self._rrf_k,
            "score_kind": "rrf",
            "confidence": "none",
            "weak_match": True,
            "generation": self.loaded_generation,
            "confidence_calibrated": False,
        }
        if self._vectors is None or not self._metadata or top_k <= 0:
            return [], query_meta

        if query_embedding is None and self._vectors.size and np.any(self._dense_eligible):
            query_embedding = self.embed_query(query_text)
        if query_embedding is not None:
            query_embedding = np.asarray(query_embedding, dtype=np.float32)
        rerank_pool = top_k * self._reranker_pool_multiplier if self._reranker_enabled else top_k
        candidate_n = max(top_k * 3, rerank_pool)
        vector_results = self._vector_search(query_text, candidate_n, source, query_vec=query_embedding)
        bm25_results = self._bm25_search(query_text, candidate_n, source)
        fused_indices, scores = self._rrf_fuse(vector_results, bm25_results, rerank_pool)

        fused_indices, scores, did_rerank = self._rerank_pool(query_text, fused_indices, scores)
        if did_rerank:
            query_meta["reranker_used"] = True
            query_meta["score_kind"] = "rerank"
        fused_indices = fused_indices[:top_k]

        if len(fused_indices) > 1:
            scores_arr = np.array([scores[i] for i in fused_indices])
            threshold = scores_arr.mean() + score_threshold_z * scores_arr.std()
            filtered = [i for i in fused_indices if scores[i] >= threshold]
            if len(filtered) < min(min_results, len(fused_indices)):
                filtered = sorted(fused_indices, key=lambda i: scores[i], reverse=True)[
                    :min(min_results, len(fused_indices))
                ]
            fused_indices = filtered

        rrf_scores = scores

        results = []
        session_groups: dict[str, dict] = {}
        query_tokens = set(_tokenize(query_text))

        for idx in fused_indices:
            meta = self._metadata[idx]
            text = meta.get("text", "")
            doc_path = meta.get("path", "")
            src = meta.get("source", "")
            span = {
                "evidence_id": meta.get("evidence_id"),
                "revision_id": meta.get("revision_id"),
                "completeness": meta.get("completeness", "unknown"),
                "text": text,
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
            }
            cosine = None
            if (query_embedding is not None and self._dense_eligible[idx]
                    and query_embedding.shape == (self._vectors.shape[1],)
                    and np.all(np.isfinite(query_embedding))):
                query_norm = np.linalg.norm(query_embedding)
                if query_norm:
                    cosine = float(self._normalized_vectors[idx] @ (query_embedding / query_norm))
            lexical_support = bool(query_tokens.intersection(_tokenize(
                " ".join([text, meta.get("doc_title", ""), meta.get("section_heading", "")]),
            )))
            common = {
                "evidence_id": span["evidence_id"],
                "revision_id": span["revision_id"],
                "completeness": span["completeness"],
                "evidence": text,
                "evidence_spans": [span],
                "confidence": _confidence_for(
                    scores[idx], query_meta["score_kind"], cosine=cosine,
                    lexical_support=lexical_support,
                ),
                "cosine_similarity": cosine,
                "lexical_support": lexical_support,
                "trust": "untrusted_source",
            }
            if doc_path in self._deprecated:
                common["deprecated"] = True
                common["deprecation_notice"] = (
                    "HISTORICAL CONTEXT ONLY — this doc has been marked as outdated: "
                    f"{self._deprecated[doc_path]}. Do not treat claims in this doc as current facts."
                )

            if src == "sessions":
                turn_label = meta.get("section_heading", "")
                if doc_path in session_groups:
                    group = session_groups[doc_path]
                    if turn_label:
                        group.setdefault("matched_turns", []).append(turn_label)
                    group["score"] = max(group["score"], round(rrf_scores[idx], 4))
                    group["evidence_spans"].append(span)
                    group["evidence"] += "\n\n" + text
                    continue

                session_groups[doc_path] = {
                    **common,
                    "source": "sessions",
                    "path": doc_path,
                    "score": round(rrf_scores[idx], 4),
                    "tags": meta.get("tags", []),
                    "description": meta.get("doc_title", ""),
                    "indexed_at": meta.get("indexed_at", ""),
                    "full_session_path": meta.get("full_session_path", ""),
                    "session_turn_count": meta.get("session_turn_count", 0),
                }
                if turn_label:
                    session_groups[doc_path]["matched_turns"] = [turn_label]
                summary = meta.get("session_summary", "")
                if summary:
                    session_groups[doc_path]["session_summary"] = summary
                else:
                    session_groups[doc_path]["text"] = text
                continue

            result: dict = {
                **common,
                "source": src,
                "path": doc_path,
                "score": round(rrf_scores[idx], 4),
                "tags": meta.get("tags", []),
                "description": meta.get("description", ""),
                "indexed_at": meta.get("indexed_at", ""),
                "matched_sections": [meta.get("section_heading", "")],
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
            }

            doc_summary = meta.get("doc_summary", "")
            if doc_summary:
                result["summary"] = doc_summary
            else:
                doc_context = meta.get("doc_preamble", "")
                result["summary"] = doc_context or text[:500]

            results.append(result)

        results = self._group_document_results(results)

        for group in session_groups.values():
            results.append(group)
        results.sort(key=lambda r: r.get("score", 0), reverse=True)

        if results:
            top_band = results[0]["confidence"]
            query_meta["confidence"] = top_band
            query_meta["weak_match"] = top_band == "low"
        # Empty results: weak_match=True / confidence="none" already set above.

        return results, query_meta

    def _group_document_results(self, results: list[dict]) -> list[dict]:
        """Merge multiple results from the same file into one result with combined sections."""
        groups: dict[tuple[str, str], list[dict]] = {}

        for r in results:
            key = (r.get("source", ""), r.get("path", ""))
            groups.setdefault(key, []).append(r)

        merged: list[dict] = []
        for (_src, _path), items in groups.items():
            if len(items) == 1:
                merged.append(items[0])
                continue

            best = max(items, key=lambda r: r.get("score", 0))
            all_sections = []
            for r in items:
                for s in r.get("matched_sections", []):
                    if s and s not in all_sections:
                        all_sections.append(s)

            first_line = min((r.get("start_line") or 0) for r in items) or 0
            last_line = max((r.get("end_line") or 0) for r in items) or 0

            merged_result: dict = {
                **best,
                "matched_sections": all_sections,
                "start_line": first_line,
                "end_line": last_line,
                "merged_chunks": len(items),
                "evidence": "\n\n".join(r["evidence"] for r in items if "evidence" in r),
                "evidence_spans": [span for r in items for span in r.get("evidence_spans", [])],
            }
            dep = next((r.get("deprecation_notice") for r in items if r.get("deprecated")), None)
            if dep:
                merged_result["deprecated"] = True
                merged_result["deprecation_notice"] = dep
            merged.append(merged_result)

        return merged
