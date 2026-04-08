"""Hybrid search: BM25 keyword + vector cosine, fused via Reciprocal Rank Fusion."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .indexer import embed_texts, load_index

log = logging.getLogger(__name__)

MAX_INLINE_LINES = 50
RRF_K = 60  # RRF constant — 60 is the standard default


def _deprecated_path(cache_dir: str) -> Path:
    return Path(cache_dir) / "deprecated.json"


def load_deprecated(cache_dir: str) -> dict[str, str]:
    """Load deprecated docs. Returns {path: reason}."""
    p = _deprecated_path(cache_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_deprecated(cache_dir: str, deprecated: dict[str, str]) -> None:
    p = _deprecated_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(deprecated, indent=2))


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"[a-z0-9_\-]+", text.lower())


class Searcher:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict] | None = None
        self._bm25: BM25Okapi | None = None
        self._hnsw = None  # hnswlib.Index or None
        self._deprecated: dict[str, str] = {}

    def _build_bm25(self) -> None:
        """Build BM25 index from loaded metadata texts."""
        if not self._metadata:
            self._bm25 = None
            return
        corpus = [_tokenize(m.get("text", "")) for m in self._metadata]
        self._bm25 = BM25Okapi(corpus)

    def _load_hnsw(self) -> None:
        """Load hnswlib index if available and corpus is large enough."""
        self._hnsw = None
        if self._vectors is None or len(self._vectors) < 5000:
            return
        hnsw_path = Path(self.cache_dir) / "index.hnsw"
        if not hnsw_path.exists():
            return
        try:
            import hnswlib
            dim = self._vectors.shape[1]
            index = hnswlib.Index(space="cosine", dim=dim)
            index.load_index(str(hnsw_path), max_elements=len(self._vectors))
            index.set_ef(50)
            self._hnsw = index
            log.info("Loaded hnswlib index (%d vectors)", len(self._vectors))
        except ImportError:
            pass
        except Exception as exc:
            log.warning("Failed to load hnswlib index: %s", exc)

    def load(self) -> bool:
        vecs, meta = load_index(self.cache_dir)
        if vecs is None or meta is None:
            log.warning("No index found at %s", self.cache_dir)
            return False
        self._vectors = vecs
        self._metadata = meta
        self._deprecated = load_deprecated(self.cache_dir)
        self._build_bm25()
        self._load_hnsw()
        log.info("Loaded index: %d chunks, %d deprecated docs", len(meta), len(self._deprecated))
        return True

    def reload(self) -> bool:
        return self.load()

    @property
    def chunk_count(self) -> int:
        return len(self._metadata) if self._metadata else 0

    def source_stats(self) -> dict[str, int]:
        if not self._metadata:
            return {}
        counts: dict[str, int] = {}
        for m in self._metadata:
            src = m.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def _vector_search(self, query_text: str, n: int, source: str | None) -> list[tuple[int, int]]:
        """Return [(index, rank)] from vector cosine similarity. Uses HNSW if available."""
        if self._vectors is None or self._metadata is None:
            return []

        query_vec = embed_texts([query_text])[0]

        # HNSW path: fast ANN for large corpora (no source filtering at ANN level)
        if self._hnsw is not None and source is None:
            labels, distances = self._hnsw.knn_query(query_vec.reshape(1, -1), k=min(n, len(self._vectors)))
            return [(int(labels[0][i]), i) for i in range(len(labels[0]))]

        # Brute-force path
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = self._vectors / norms
        query_norm = query_vec / (np.linalg.norm(query_vec) or 1)

        scores = normed @ query_norm

        if source:
            mask = np.array([m.get("source") == source for m in self._metadata])
            scores = np.where(mask, scores, -1.0)

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(idx), rank) for rank, idx in enumerate(top_indices) if scores[idx] >= 0]

    def _bm25_search(self, query_text: str, n: int, source: str | None) -> list[tuple[int, int]]:
        """Return [(index, rank)] from BM25 keyword search."""
        if self._bm25 is None or self._metadata is None:
            return []

        tokens = _tokenize(query_text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        if source:
            mask = np.array([m.get("source") == source for m in self._metadata])
            scores = np.where(mask, scores, -1.0)

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(idx), rank) for rank, idx in enumerate(top_indices) if scores[idx] > 0]

    def _rrf_fuse(
        self,
        vector_results: list[tuple[int, int]],
        bm25_results: list[tuple[int, int]],
        top_k: int,
    ) -> tuple[list[int], dict[int, float]]:
        """Reciprocal Rank Fusion: combine two ranked lists into one."""
        rrf_scores: dict[int, float] = {}
        for idx, rank in vector_results:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for idx, rank in bm25_results:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
        return sorted_indices[:top_k], rrf_scores

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[dict]:
        if self._vectors is None or self._metadata is None or len(self._metadata) == 0:
            return []

        candidate_n = top_k * 3
        vector_results = self._vector_search(query_text, candidate_n, source)
        bm25_results = self._bm25_search(query_text, candidate_n, source)
        fused_indices, rrf_scores = self._rrf_fuse(vector_results, bm25_results, top_k)

        # Freshness boost: recent docs get a small additive bump that decays over ~1 week
        now = time.time()
        for idx in fused_indices:
            indexed_at = self._metadata[idx].get("indexed_at", "")
            if indexed_at:
                try:
                    ts = time.mktime(time.strptime(indexed_at, "%Y-%m-%dT%H:%M:%SZ"))
                    age_hours = max(0, (now - ts) / 3600)
                    rrf_scores[idx] += 0.005 * math.exp(-age_hours / (24 * 7))
                except (ValueError, OverflowError):
                    pass

        # Score-gap filtering: drop results below mean + 1.5σ
        if len(fused_indices) > 1:
            scores_arr = np.array([rrf_scores[i] for i in fused_indices])
            threshold = scores_arr.mean() + 1.5 * scores_arr.std()
            fused_indices = [i for i in fused_indices if rrf_scores[i] >= threshold]
            if not fused_indices:
                # Keep at least the top result
                fused_indices = [max(rrf_scores, key=rrf_scores.get)]

        results = []
        for idx in fused_indices:
            meta = self._metadata[idx]
            text = meta.get("text", "")
            line_count = text.count("\n") + 1

            doc_path = meta.get("path", "")
            result = {
                "source": meta.get("source", ""),
                "path": doc_path,
                "section": meta.get("section_heading", ""),
                "score": round(rrf_scores[idx], 4),
                "tags": meta.get("tags", []),
                "description": meta.get("description", ""),
                "indexed_at": meta.get("indexed_at", ""),
            }

            deprecation_reason = self._deprecated.get(doc_path)
            if deprecation_reason:
                result["deprecated"] = True
                result["deprecation_notice"] = (
                    f"HISTORICAL CONTEXT ONLY — this doc has been marked as outdated: "
                    f"{deprecation_reason}. Do not treat claims in this doc as current facts."
                )

            if line_count <= MAX_INLINE_LINES:
                result["text"] = text
            else:
                result["text_preview"] = text[:500] + "..."
                result["start_line"] = meta.get("start_line", 0)
                result["end_line"] = meta.get("end_line", 0)

            results.append(result)

        return results
