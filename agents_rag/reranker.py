"""Cross-encoder reranker.

Sits after RRF fusion: given a query and a pool of candidate passages, produces
a relevance score per passage with proper dynamic range (unlike RRF, which
collapses everything into 1/(K+rank) slivers). We use these scores to reorder
the top pool and to support a meaningful z-score threshold downstream.

Model is loaded lazily on first use; download happens on first query (~280MB
for BAAI/bge-reranker-base). If the load or predict call fails, `rerank`
returns None and callers fall back to the RRF ordering.
"""

from __future__ import annotations

import inspect
import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

_model = None
_model_name_loaded: str | None = None
_model_load_failed = False
_model_lock = threading.Lock()


def _get_reranker(model_name: str):
    global _model, _model_name_loaded, _model_load_failed
    if _model is not None and _model_name_loaded == model_name:
        return _model
    if _model_load_failed:
        return None

    with _model_lock:
        if _model is not None and _model_name_loaded == model_name:
            return _model
        if _model_load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder
            from torch import nn

            log.info("Loading cross-encoder reranker: %s", model_name)
            # Select the explicit activation on both supported API generations.
            # Otherwise saved model defaults can return either logits or scores,
            # and applying a second sigmoid makes weak matches look relevant.
            activation_parameter = (
                "activation_fn"
                if "activation_fn" in inspect.signature(CrossEncoder).parameters
                else "default_activation_function"
            )
            _model = CrossEncoder(model_name, **{activation_parameter: nn.Sigmoid()})
            _model_name_loaded = model_name
            return _model
        except Exception as exc:
            # Treat any failure (no network, bad model name, missing lib) as
            # permanent for this process so we don't pay the timeout repeatedly.
            log.warning("Failed to load cross-encoder %s: %s", model_name, exc)
            _model_load_failed = True
            return None


def rerank(
    query: str,
    passages: list[str],
    model_name: str = "BAAI/bge-reranker-base",
) -> np.ndarray | None:
    """Return sigmoid-normalized relevance scores, or None if unavailable.

    Scores are in [0, 1]; they are not calibrated relevance probabilities.
    """
    if not passages:
        return np.array([], dtype=np.float32)
    model = _get_reranker(model_name)
    if model is None:
        return None
    try:
        pairs = [(query, p) for p in passages]
        predictions = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        scores = np.asarray(predictions, dtype=np.float32)
        if scores.shape != (len(passages),):
            raise ValueError("Expected one relevance score per passage")
        if not np.all(np.isfinite(scores) & (scores >= 0.0) & (scores <= 1.0)):
            raise ValueError("Expected finite relevance scores in [0, 1]")
        return scores
    except Exception as exc:
        log.warning("Reranker predict failed: %s", exc)
        return None


def warm(model_name: str = "BAAI/bge-reranker-base") -> bool:
    """Force-load the reranker model. Safe to call from a startup thread.

    Returns True if the model is ready (or already was), False if loading failed.
    A no-op when called repeatedly with the same model.
    """
    return _get_reranker(model_name) is not None


def reset_for_testing() -> None:
    """Clear singleton state. Test-only helper."""
    global _model, _model_name_loaded, _model_load_failed
    with _model_lock:
        _model = None
        _model_name_loaded = None
        _model_load_failed = False
