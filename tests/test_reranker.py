"""Unit tests for the cross-encoder reranker wrapper."""

from __future__ import annotations

from types import SimpleNamespace
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


def test_rerank_preserves_already_normalized_model_scores():
    fake_model = MagicMock()
    expected = np.array([0.001, 0.1, 0.5, 0.9], dtype=np.float32)
    fake_model.predict.return_value = expected

    with patch.object(reranker, "_get_reranker", return_value=fake_model):
        scores = reranker.rerank("q", ["p1", "p2", "p3", "p4"])

    assert scores is not None
    np.testing.assert_array_equal(scores, expected)
    fake_model.predict.assert_called_once_with(
        [("q", "p1"), ("q", "p2"), ("q", "p3"), ("q", "p4")],
        show_progress_bar=False,
        convert_to_numpy=True,
    )


@pytest.mark.parametrize("legacy_api", [False, True])
def test_cross_encoder_uses_exactly_one_sigmoid_for_logits(legacy_api):
    """Model stubs apply the supplied activation as CrossEncoder.predict does."""
    import torch

    logits = torch.tensor([-10000.0, -8.0, 0.0, 8.0, 10000.0])

    class FakeEncoder:
        def predict(self, pairs, *, show_progress_bar, convert_to_numpy):
            assert len(pairs) == len(logits)
            assert show_progress_bar is False
            assert convert_to_numpy is True
            return self.activation(logits).numpy()

    class CurrentEncoder(FakeEncoder):
        def __init__(self, name, *, activation_fn=None):
            assert name == "offline-model"
            assert isinstance(activation_fn, torch.nn.Sigmoid)
            self.activation = activation_fn

    class LegacyEncoder(FakeEncoder):
        def __init__(self, name, *, default_activation_function=None):
            assert name == "offline-model"
            assert isinstance(default_activation_function, torch.nn.Sigmoid)
            self.activation = default_activation_function

    encoder = LegacyEncoder if legacy_api else CurrentEncoder
    with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=encoder)}):
        with np.errstate(over="raise", invalid="raise"):
            scores = reranker.rerank("q", ["p"] * len(logits), model_name="offline-model")

    assert scores is not None
    assert scores.dtype == np.float32
    np.testing.assert_allclose(
        scores, [0.0, 0.0003353501, 0.5, 0.9996646499, 1.0], rtol=1e-6,
    )
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("predictions", [
    [np.nan, 0.5],
    [np.inf, 0.5],
    [-0.1, 0.5],
    [1.1, 0.5],
    [[0.1, 0.9], [0.2, 0.8]],
    [0.5],
])
def test_rerank_falls_back_for_invalid_model_output(predictions):
    fake_model = MagicMock()
    fake_model.predict.return_value = predictions
    with patch.object(reranker, "_get_reranker", return_value=fake_model):
        assert reranker.rerank("q", ["p1", "p2"]) is None


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

    with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=_raising_import)}):
        assert reranker._get_reranker("any-model") is None
        assert reranker._get_reranker("any-model") is None

    # Only one attempt — flag sticks.
    assert call_count["n"] == 1
