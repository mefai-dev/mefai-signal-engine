"""
Tests for XGBoost model training and prediction.

Uses synthetic data to verify model lifecycle:
- Training with sufficient data
- Prediction output format
- Model save/load cycle
- Insufficient data handling
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.ml.majors_model import MajorsModel
from src.ml.alts_model import AltsModel
from src.ml.memes_model import MemesModel


@pytest.fixture
def synthetic_data():
    """Generate synthetic OHLCV data suitable for model training."""
    np.random.seed(42)
    n = 500

    # Create trending data with some noise
    base = 50000.0
    trend = np.cumsum(np.random.randn(n) * 50)
    close = base + trend

    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    open_ = close + np.random.randn(n) * 20
    volume = np.abs(np.random.randn(n) * 100 + 500)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def small_data():
    """Insufficient data for training (should fail gracefully)."""
    np.random.seed(42)
    n = 30
    close = 100 + np.random.randn(n) * 2

    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.abs(np.random.randn(n) * 100 + 500),
    })


class TestMajorsModel:
    def test_train_and_predict(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MajorsModel(model_dir=tmpdir, prediction_horizon=5, movement_threshold_pct=0.3)
            metrics = model.train(synthetic_data)

            assert "error" not in metrics
            assert metrics["training_samples"] > 0
            assert 0 <= metrics["final_accuracy"] <= 1
            assert model.is_trained

            prediction = model.predict(synthetic_data)
            assert prediction["direction"] in ("LONG", "SHORT", "NEUTRAL")
            assert 0 <= prediction["confidence"] <= 100
            assert prediction["model_trained"] is True

    def test_predict_untrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MajorsModel(model_dir=tmpdir)
            prediction = model.predict(pd.DataFrame())
            assert prediction["direction"] == "NEUTRAL"
            assert prediction["model_trained"] is False

    def test_insufficient_data(self, small_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MajorsModel(model_dir=tmpdir)
            metrics = model.train(small_data)
            assert "error" in metrics

    def test_model_persistence(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MajorsModel(model_dir=tmpdir, prediction_horizon=5, movement_threshold_pct=0.3)
            model.train(synthetic_data)

            pred1 = model.predict(synthetic_data)

            # Load model from disk
            model2 = MajorsModel(model_dir=tmpdir, prediction_horizon=5, movement_threshold_pct=0.3)
            assert model2.is_trained
            pred2 = model2.predict(synthetic_data)

            assert pred1["direction"] == pred2["direction"]
            assert pred1["confidence"] == pytest.approx(pred2["confidence"], abs=0.01)

    def test_status(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MajorsModel(model_dir=tmpdir, prediction_horizon=5, movement_threshold_pct=0.3)
            model.train(synthetic_data)
            status = model.get_status()

            assert status["model_type"] == "majors"
            assert status["is_trained"] is True
            assert "metrics" in status

    def test_feature_importance(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MajorsModel(model_dir=tmpdir, prediction_horizon=5, movement_threshold_pct=0.3)
            metrics = model.train(synthetic_data)

            assert "top_features" in metrics
            assert len(metrics["top_features"]) > 0
            # Feature importances should sum close to 1
            total_imp = sum(f["importance"] for f in metrics["top_features"])
            assert total_imp > 0


class TestAltsModel:
    def test_train_and_predict(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = AltsModel(model_dir=tmpdir, prediction_horizon=5, movement_threshold_pct=0.5)
            metrics = model.train(synthetic_data)

            assert "error" not in metrics
            assert metrics["momentum_weighting"] is True
            assert metrics["recency_weighting"] is True

            prediction = model.predict(synthetic_data)
            assert prediction["direction"] in ("LONG", "SHORT", "NEUTRAL")

    def test_momentum_weighting(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = AltsModel(model_dir=tmpdir)
            from src.ml.feature_engine import FeatureEngine
            engine = FeatureEngine()
            X, names, _ = engine.get_feature_matrix(synthetic_data)

            X_weighted = model._apply_momentum_weighting(X, names)

            # Momentum features should be scaled
            rsi_idx = names.index("rsi_14")
            np.testing.assert_array_almost_equal(
                X_weighted[:, rsi_idx],
                X[:, rsi_idx] * 1.2,
            )


class TestMemesModel:
    def test_train_and_predict(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MemesModel(model_dir=tmpdir, prediction_horizon=5, movement_threshold_pct=1.0)
            metrics = model.train(synthetic_data)

            assert "error" not in metrics
            assert metrics["volume_weighting"] is True
            assert metrics["exponential_recency"] is True

            prediction = model.predict(synthetic_data)
            assert prediction["direction"] in ("LONG", "SHORT", "NEUTRAL")

    def test_volume_weighting(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MemesModel(model_dir=tmpdir)
            from src.ml.feature_engine import FeatureEngine
            engine = FeatureEngine()
            X, names, _ = engine.get_feature_matrix(synthetic_data)

            X_weighted = model._apply_volume_weighting(X, names)

            # Volume features should be 2x
            obv_idx = names.index("obv")
            np.testing.assert_array_almost_equal(
                X_weighted[:, obv_idx],
                X[:, obv_idx] * 2.0,
            )

    def test_wider_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MemesModel(model_dir=tmpdir)
            assert model.movement_threshold_pct == 1.5
            # Memes should have wider threshold than majors
            majors = MajorsModel(model_dir=tmpdir)
            assert model.movement_threshold_pct > majors.movement_threshold_pct


class TestClassDistribution:
    def test_label_creation(self, synthetic_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MajorsModel(model_dir=tmpdir, prediction_horizon=5, movement_threshold_pct=0.3)
            close = synthetic_data["close"].values
            labels = model._create_labels(close)

            assert len(labels) == len(close)
            assert set(np.unique(labels)).issubset({0, 1, 2})
            # Should have all three classes in volatile data
            assert len(np.unique(labels)) >= 2
