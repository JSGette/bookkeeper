"""Unit tests for the cross-encoder reranker wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents_rag import reranker


@pytest.fixture(autouse=True)
def _reset_reranker():
    reranker.reset_for_testing()
    yield
    reranker.reset_for_testing()


def test_rerank_empty_passages_returns_empty_array():
    scores = reranker.rerank("query", [])
    assert isinstance(scores, np.ndarray)
    assert scores.size == 0


def test_rerank_returns_none_when_model_load_fails():
    with patch.object(reranker, "_get_reranker", return_value=None):
        assert reranker.rerank("query", ["a", "b"]) is None


def test_rerank_scores_are_sigmoid_of_raw_logits():
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([0.0, 2.0, -2.0], dtype=np.float32)

    with patch.object(reranker, "_get_reranker", return_value=fake_model):
        scores = reranker.rerank("q", ["p1", "p2", "p3"])

    assert scores is not None
    assert len(scores) == 3
    assert scores[0] == pytest.approx(0.5, abs=1e-4)
    assert scores[1] > scores[0] > scores[2]
    assert 0.0 <= scores.min() and scores.max() <= 1.0


def test_rerank_catches_predict_failure():
    fake_model = MagicMock()
    fake_model.predict.side_effect = RuntimeError("boom")

    with patch.object(reranker, "_get_reranker", return_value=fake_model):
        assert reranker.rerank("q", ["p"]) is None


def test_warm_returns_true_when_model_loads():
    fake_model = MagicMock()
    with patch.object(reranker, "_get_reranker", return_value=fake_model) as mock_get:
        assert reranker.warm("any-model") is True
    mock_get.assert_called_once_with("any-model")


def test_warm_returns_false_when_model_load_fails():
    with patch.object(reranker, "_get_reranker", return_value=None):
        assert reranker.warm("any-model") is False


def test_get_reranker_caches_on_failure():
    """Once load fails, subsequent calls short-circuit without retrying."""
    call_count = {"n": 0}

    def _raising_import(*_args, **_kwargs):
        call_count["n"] += 1
        raise ImportError("no sentence-transformers")

    with patch("sentence_transformers.CrossEncoder", side_effect=_raising_import):
        assert reranker._get_reranker("any-model") is None
        assert reranker._get_reranker("any-model") is None

    # Only one attempt — flag sticks.
    assert call_count["n"] == 1
