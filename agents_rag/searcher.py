"""Hybrid search: BM25 keyword + vector cosine via hnswlib, fused via Reciprocal Rank Fusion."""

from __future__ import annotations

import io
import logging
import math
import re
import sqlite3
import tempfile
import time
from pathlib import Path

import hnswlib
import numpy as np
from rank_bm25 import BM25Okapi

from . import db
from .indexer import embed_texts

log = logging.getLogger(__name__)

MAX_INLINE_LINES = 50
MAX_GROUP_TOKENS = 2048
RRF_K = 60

_CHARS_PER_TOKEN = 4


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\-]+", text.lower())


class Searcher:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict] | None = None
        self._bm25: BM25Okapi | None = None
        self._hnsw: hnswlib.Index | None = None
        self._deprecated: dict[str, str] = {}

    def _build_bm25(self) -> None:
        if not self._metadata:
            self._bm25 = None
            return
        corpus = [_tokenize(m.get("text", "")) for m in self._metadata]
        self._bm25 = BM25Okapi(corpus)

    def _build_hnsw(self) -> None:
        """Build hnswlib index from vectors. Try loading cached blob first."""
        self._hnsw = None
        if self._vectors is None or len(self._vectors) == 0:
            return

        dim = self._vectors.shape[1]
        n = len(self._vectors)

        cached = db.load_hnsw_cache(self.conn)
        if cached is not None:
            blob, cached_count = cached
            if cached_count == n:
                try:
                    index = hnswlib.Index(space="cosine", dim=dim)
                    with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
                        f.write(blob)
                        tmp_path = f.name
                    index.load_index(tmp_path, max_elements=n)
                    Path(tmp_path).unlink(missing_ok=True)
                    index.set_ef(50)
                    self._hnsw = index
                    log.info("Loaded cached hnswlib index (%d vectors)", n)
                    return
                except Exception as exc:
                    log.warning("Failed to load cached HNSW, rebuilding: %s", exc)
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
        try:
            with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
                tmp_path = f.name
            self._hnsw.save_index(tmp_path)
            blob = Path(tmp_path).read_bytes()
            Path(tmp_path).unlink(missing_ok=True)
            db.save_hnsw_cache(self.conn, blob, len(self._vectors))
        except Exception as exc:
            log.warning("Failed to cache HNSW index: %s", exc)

    def load(self) -> bool:
        meta, vecs = db.load_all_chunks(self.conn)
        if not meta:
            log.warning("No chunks in database")
            self._vectors = np.empty((0, 384), dtype=np.float32)
            self._metadata = []
            self._deprecated = db.load_deprecated(self.conn)
            return False
        self._vectors = vecs
        self._metadata = meta
        self._deprecated = db.load_deprecated(self.conn)
        self._build_bm25()
        self._build_hnsw()
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
        """Return [(index, rank)] from vector similarity. Uses HNSW when no source filter."""
        if self._vectors is None or self._metadata is None or len(self._vectors) == 0:
            return []

        query_vec = embed_texts([query_text])[0]

        if self._hnsw is not None and source is None:
            labels, _distances = self._hnsw.knn_query(
                query_vec.reshape(1, -1), k=min(n, len(self._vectors)),
            )
            return [(int(labels[0][i]), i) for i in range(len(labels[0]))]

        # Brute-force fallback for source-filtered queries
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

        if len(fused_indices) > 1:
            scores_arr = np.array([rrf_scores[i] for i in fused_indices])
            threshold = scores_arr.mean() + 1.5 * scores_arr.std()
            fused_indices = [i for i in fused_indices if rrf_scores[i] >= threshold]
            if not fused_indices:
                fused_indices = [max(rrf_scores, key=rrf_scores.get)]

        results = []
        session_groups: dict[str, dict] = {}

        for idx in fused_indices:
            meta = self._metadata[idx]
            text = meta.get("text", "")
            doc_path = meta.get("path", "")
            src = meta.get("source", "")

            if src == "sessions":
                turn_label = meta.get("section_heading", "")
                if doc_path in session_groups:
                    group = session_groups[doc_path]
                    group["matched_turns"].append(turn_label)
                    group["score"] = max(group["score"], round(rrf_scores[idx], 4))
                    continue

                session_groups[doc_path] = {
                    "source": "sessions",
                    "path": doc_path,
                    "score": round(rrf_scores[idx], 4),
                    "tags": meta.get("tags", []),
                    "description": meta.get("doc_title", ""),
                    "indexed_at": meta.get("indexed_at", ""),
                    "matched_turns": [turn_label],
                    "full_session_path": meta.get("full_session_path", ""),
                    "session_turn_count": meta.get("session_turn_count", 0),
                }
                summary = meta.get("session_summary", "")
                if summary:
                    session_groups[doc_path]["session_summary"] = summary
                else:
                    session_groups[doc_path]["text"] = text
                continue

            line_count = text.count("\n") + 1
            result: dict = {
                "source": src,
                "path": doc_path,
                "section": meta.get("section_heading", ""),
                "score": round(rrf_scores[idx], 4),
                "tags": meta.get("tags", []),
                "description": meta.get("description", ""),
                "indexed_at": meta.get("indexed_at", ""),
            }

            doc_context = meta.get("doc_preamble", "")
            if doc_context:
                result["doc_context"] = doc_context

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

        results = self._group_document_results(results)

        for group in session_groups.values():
            results.append(group)
        results.sort(key=lambda r: r.get("score", 0), reverse=True)

        return results

    def _group_document_results(self, results: list[dict]) -> list[dict]:
        """Merge multiple results from the same file into one contiguous block."""
        groups: dict[tuple[str, str], list[dict]] = {}

        for r in results:
            key = (r.get("source", ""), r.get("path", ""))
            groups.setdefault(key, []).append(r)

        merged: list[dict] = []
        for (src, path), items in groups.items():
            if len(items) == 1:
                merged.append(items[0])
                continue

            # Read full doc content from DB
            content = db.get_document_content(self.conn, src, path)
            if not content:
                merged.extend(items)
                continue

            items.sort(key=lambda r: r.get("start_line", 0))
            first_line = min(r.get("start_line", 0) for r in items if r.get("start_line"))
            last_line = max(r.get("end_line", 0) for r in items if r.get("end_line"))

            if not first_line or not last_line:
                merged.extend(items)
                continue

            all_lines = content.splitlines()
            slice_text = "\n".join(all_lines[first_line - 1 : last_line])
            est_tokens = len(slice_text) // _CHARS_PER_TOKEN
            if est_tokens > MAX_GROUP_TOKENS:
                merged.extend(items)
                continue

            best = max(items, key=lambda r: r.get("score", 0))
            merged_result: dict = {
                "source": src,
                "path": path,
                "section": ", ".join(
                    r["section"] for r in items if r.get("section")
                ),
                "score": best["score"],
                "tags": best.get("tags", []),
                "description": best.get("description", ""),
                "indexed_at": best.get("indexed_at", ""),
                "start_line": first_line,
                "end_line": last_line,
                "text": slice_text,
                "merged_chunks": len(items),
            }
            if best.get("doc_context"):
                merged_result["doc_context"] = best["doc_context"]
            dep = best.get("deprecation_notice")
            if dep:
                merged_result["deprecated"] = True
                merged_result["deprecation_notice"] = dep
            merged.append(merged_result)

        return merged
