"""
Alts Model - XGBoost classifier for mid-cap altcoins.

Momentum-heavy hyperparameters tuned for SOL, AVAX, LINK, DOT, ADA, MATIC.
Higher learning rate and deeper trees to capture faster-moving patterns.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from src.ml.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

ALTS_SYMBOLS = ["SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT", "MATICUSDT"]

# Momentum-heavy parameters: deeper trees, faster learning, less regularization
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.08,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "min_child_weight": 3,
    "gamma": 0.05,
    "reg_alpha": 0.05,
    "reg_lambda": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


class AltsModel:
    """
    XGBoost model for mid-cap altcoins.

    Optimized for momentum-driven markets with:
    - Deeper trees (max_depth=7) to capture complex patterns
    - Higher learning rate (0.08) for faster adaptation
    - Less regularization to capture volatile moves
    - Wider movement threshold (0.8%) reflecting higher volatility
    """

    def __init__(
        self,
        model_dir: str = "models",
        params: Optional[Dict] = None,
        prediction_horizon: int = 12,
        movement_threshold_pct: float = 0.8,
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
        return os.path.join(self.model_dir, "alts_xgb.joblib")

    def _try_load(self):
        path = self._model_path()
        if os.path.exists(path):
            try:
                saved = joblib.load(path)
                self.model = saved["model"]
                self.last_train_time = saved.get("train_time")
                self.train_metrics = saved.get("metrics", {})
                self.is_trained = True
                logger.info("Loaded alts model from %s", path)
            except Exception as e:
                logger.warning("Failed to load alts model: %s", e)

    def _create_labels(self, close: np.ndarray) -> np.ndarray:
        n = len(close)
        labels = np.full(n, 1, dtype=np.int32)

        for i in range(n - self.prediction_horizon):
            future_close = close[i + self.prediction_horizon]
            pct_change = (future_close - close[i]) / close[i] * 100

            if pct_change > self.movement_threshold_pct:
                labels[i] = 2
            elif pct_change < -self.movement_threshold_pct:
                labels[i] = 0

        return labels

    def _apply_momentum_weighting(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        """
        Apply momentum emphasis by scaling momentum-related features.

        For alts, momentum indicators (RSI, MACD, EMA crossovers) get
        a slight boost in their feature values to amplify their signal.
        This is a feature engineering technique, not weight manipulation.
        """
        X_weighted = X.copy()
        momentum_features = {"rsi_14", "rsi_7", "macd_line", "macd_histogram", "ema_9_21_cross", "volume_ratio"}

        for i, name in enumerate(feature_names):
            if name in momentum_features:
                # Enhance signal by 20% - makes momentum features slightly more prominent
                X_weighted[:, i] = X_weighted[:, i] * 1.2

        return X_weighted

    def train(self, df: pd.DataFrame) -> Dict:
        logger.info("Training alts model on %d candles...", len(df))

        X, feature_names, valid_mask = self.feature_engine.get_feature_matrix(df)
        X = self._apply_momentum_weighting(X, feature_names)

        close_valid = df["close"].values[valid_mask]
        y = self._create_labels(close_valid)

        train_end = len(X) - self.prediction_horizon
        if train_end < 100:
            logger.warning("Not enough data for alts model (%d samples)", train_end)
            return {"error": "insufficient_data", "samples": train_end}

        X_train = X[:train_end]
        y_train = y[:train_end]

        # Compute sample weights: recent data weighted more heavily
        # Linear weight from 0.5 (oldest) to 1.5 (newest)
        sample_weights = np.linspace(0.5, 1.5, len(X_train))

        # Time-series cross validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
            w_t = sample_weights[train_idx]

            temp_model = xgb.XGBClassifier(**self.params)
            temp_model.fit(
                X_t, y_t,
                sample_weight=w_t,
                eval_set=[(X_v, y_v)],
                verbose=False,
            )
            preds = temp_model.predict(X_v)
            cv_scores.append(accuracy_score(y_v, preds))

        # Final model
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

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
            "momentum_weighting": True,
            "recency_weighting": True,
            "class_distribution": {
                "down": int(np.sum(y_train == 0)),
                "neutral": int(np.sum(y_train == 1)),
                "up": int(np.sum(y_train == 2)),
            },
        }

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
        self._save()

        logger.info(
            "Alts model trained: accuracy=%.3f, f1=%.3f, samples=%d",
            self.train_metrics["final_accuracy"],
            self.train_metrics["final_f1"],
            len(X_train),
        )
        return self.train_metrics

    def predict(self, df: pd.DataFrame) -> Dict:
        if not self.is_trained or self.model is None:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "probabilities": {"down": 0.33, "neutral": 0.34, "up": 0.33},
                "model_trained": False,
            }

        X, feature_names, valid_mask = self.feature_engine.get_feature_matrix(df)
        if len(X) == 0:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "probabilities": {"down": 0.33, "neutral": 0.34, "up": 0.33},
                "model_trained": True,
                "error": "no_valid_features",
            }

        X = self._apply_momentum_weighting(X, feature_names)
        X_latest = X[-1:].reshape(1, -1)

        proba = self.model.predict_proba(X_latest)[0]
        pred_class = int(np.argmax(proba))

        direction_map = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
        sorted_proba = sorted(proba, reverse=True)
        confidence = (sorted_proba[0] - sorted_proba[1]) * 100

        return {
            "direction": direction_map[pred_class],
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
        path = self._model_path()
        joblib.dump({
            "model": self.model,
            "train_time": self.last_train_time,
            "metrics": self.train_metrics,
        }, path)
        logger.info("Saved alts model to %s", path)

    def get_status(self) -> Dict:
        return {
            "model_type": "alts",
            "symbols": ALTS_SYMBOLS,
            "is_trained": self.is_trained,
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
            "metrics": self.train_metrics,
            "params": {k: v for k, v in self.params.items() if k != "verbosity"},
        }
