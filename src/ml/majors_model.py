"""
Majors Model - XGBoost classifier for BTC, ETH, BNB.

Conservative hyperparameters tuned for high-cap, lower-volatility assets.
Uses standard technical features with emphasis on trend-following indicators.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from src.ml.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

# Symbols this model handles
MAJORS_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# Default hyperparameters - conservative for stable assets
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


class MajorsModel:
    """
    XGBoost ensemble model for major cryptocurrency pairs.

    Classifies price movement into 3 classes:
    - 0: DOWN (price drops > threshold within prediction horizon)
    - 1: NEUTRAL (price stays within threshold)
    - 2: UP (price rises > threshold within prediction horizon)
    """

    def __init__(
        self,
        model_dir: str = "models",
        params: Optional[Dict] = None,
        prediction_horizon: int = 12,
        movement_threshold_pct: float = 0.5,
    ):
        self.model_dir = model_dir
        self.params = params or DEFAULT_PARAMS.copy()
        self.prediction_horizon = prediction_horizon
        self.movement_threshold_pct = movement_threshold_pct

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_engine = FeatureEngine()
        self.last_train_time: Optional[datetime] = None
        self.train_metrics: Dict = {}
        self.is_trained = False

        os.makedirs(model_dir, exist_ok=True)
        self._try_load()

    def _model_path(self) -> str:
        return os.path.join(self.model_dir, "majors_xgb.joblib")

    def _try_load(self):
        """Attempt to load a previously saved model."""
        path = self._model_path()
        if os.path.exists(path):
            try:
                saved = joblib.load(path)
                self.model = saved["model"]
                self.last_train_time = saved.get("train_time")
                self.train_metrics = saved.get("metrics", {})
                self.is_trained = True
                logger.info("Loaded majors model from %s (trained %s)", path, self.last_train_time)
            except Exception as e:
                logger.warning("Failed to load majors model: %s", e)

    def _create_labels(self, close: np.ndarray) -> np.ndarray:
        """
        Create classification labels based on future price movement.

        For each candle, look ahead `prediction_horizon` candles and
        classify the movement:
        - 0: price dropped more than threshold
        - 1: price stayed within threshold (neutral)
        - 2: price rose more than threshold
        """
        n = len(close)
        labels = np.full(n, 1, dtype=np.int32)  # Default neutral

        for i in range(n - self.prediction_horizon):
            future_close = close[i + self.prediction_horizon]
            pct_change = (future_close - close[i]) / close[i] * 100

            if pct_change > self.movement_threshold_pct:
                labels[i] = 2  # UP
            elif pct_change < -self.movement_threshold_pct:
                labels[i] = 0  # DOWN
            else:
                labels[i] = 1  # NEUTRAL

        return labels

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the model on OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            Dictionary of training metrics.
        """
        logger.info("Training majors model on %d candles...", len(df))

        # Compute features
        X, feature_names, valid_mask = self.feature_engine.get_feature_matrix(df)

        # Create labels (only for valid feature rows)
        close_valid = df["close"].values[valid_mask]
        y = self._create_labels(close_valid)

        # Remove last `prediction_horizon` rows (labels are unknown)
        train_end = len(X) - self.prediction_horizon
        if train_end < 100:
            logger.warning("Not enough data to train majors model (%d samples)", train_end)
            return {"error": "insufficient_data", "samples": train_end}

        X_train = X[:train_end]
        y_train = y[:train_end]

        # Time-series cross validation for evaluation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]

            temp_model = xgb.XGBClassifier(**self.params)
            temp_model.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                verbose=False,
            )
            preds = temp_model.predict(X_v)
            cv_scores.append(accuracy_score(y_v, preds))

        # Train final model on all data
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, verbose=False)

        # Evaluate on last fold
        final_preds = self.model.predict(X_train[-len(X_train) // 4:])
        final_true = y_train[-len(X_train) // 4:]

        self.train_metrics = {
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "final_accuracy": float(accuracy_score(final_true, final_preds)),
            "final_precision": float(precision_score(final_true, final_preds, average="weighted", zero_division=0)),
            "final_recall": float(recall_score(final_true, final_preds, average="weighted", zero_division=0)),
            "final_f1": float(f1_score(final_true, final_preds, average="weighted", zero_division=0)),
            "training_samples": len(X_train),
            "feature_count": len(feature_names),
            "class_distribution": {
                "down": int(np.sum(y_train == 0)),
                "neutral": int(np.sum(y_train == 1)),
                "up": int(np.sum(y_train == 2)),
            },
        }

        # Feature importance
        importance = self.model.feature_importances_
        top_features = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        self.train_metrics["top_features"] = [
            {"name": name, "importance": float(imp)}
            for name, imp in top_features
        ]

        self.last_train_time = datetime.utcnow()
        self.is_trained = True

        # Save model
        self._save()

        logger.info(
            "Majors model trained: accuracy=%.3f, f1=%.3f, samples=%d",
            self.train_metrics["final_accuracy"],
            self.train_metrics["final_f1"],
            len(X_train),
        )

        return self.train_metrics

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Generate prediction from current market data.

        Args:
            df: Recent OHLCV DataFrame (needs at least 200 candles for SMA200).

        Returns:
            Dictionary with prediction class, probabilities, and confidence.
        """
        if not self.is_trained or self.model is None:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "probabilities": {"down": 0.33, "neutral": 0.34, "up": 0.33},
                "model_trained": False,
            }

        X, _, valid_mask = self.feature_engine.get_feature_matrix(df)
        if len(X) == 0:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "probabilities": {"down": 0.33, "neutral": 0.34, "up": 0.33},
                "model_trained": True,
                "error": "no_valid_features",
            }

        # Use only the latest feature row
        X_latest = X[-1:].reshape(1, -1)

        proba = self.model.predict_proba(X_latest)[0]
        pred_class = int(np.argmax(proba))

        direction_map = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
        direction = direction_map[pred_class]

        # Confidence is the margin between top prediction and second best
        sorted_proba = sorted(proba, reverse=True)
        confidence = (sorted_proba[0] - sorted_proba[1]) * 100

        return {
            "direction": direction,
            "confidence": float(np.clip(confidence, 0, 100)),
            "probabilities": {
                "down": float(proba[0]),
                "neutral": float(proba[1]),
                "up": float(proba[2]),
            },
            "predicted_class": pred_class,
            "model_trained": True,
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
        }

    def _save(self):
        """Persist model and metadata to disk."""
        path = self._model_path()
        joblib.dump({
            "model": self.model,
            "train_time": self.last_train_time,
            "metrics": self.train_metrics,
        }, path)
        logger.info("Saved majors model to %s", path)

    def get_status(self) -> Dict:
        """Return model status information."""
        return {
            "model_type": "majors",
            "symbols": MAJORS_SYMBOLS,
            "is_trained": self.is_trained,
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
            "metrics": self.train_metrics,
            "params": {k: v for k, v in self.params.items() if k != "verbosity"},
        }
