"""
Memes Model - XGBoost classifier for meme coins.

Volume and sentiment-weighted hyperparameters for PEPE, DOGE, SHIB, FLOKI, BONK.
Deepest trees, highest learning rate, least regularization.
Meme coins are driven by volume spikes and social sentiment more than fundamentals.
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

MEME_SYMBOLS = ["1000PEPEUSDT", "DOGEUSDT", "1000SHIBUSDT", "1000FLOKIUSDT", "1000BONKUSDT"]

# Aggressive parameters for volatile meme coins
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 2,
    "gamma": 0.02,
    "reg_alpha": 0.02,
    "reg_lambda": 0.5,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


class MemesModel:
    """
    XGBoost model for meme coins.

    Key differences from Majors/Alts:
    - Volume features weighted 2x (meme coins are volume-driven)
    - Wider movement threshold (1.5%) reflecting extreme volatility
    - Deepest trees (max_depth=8) to capture non-linear pump/dump patterns
    - Highest learning rate (0.1) for rapid adaptation
    - Exponential recency weighting (meme coin patterns decay fast)
    """

    def __init__(
        self,
        model_dir: str = "models",
        params: Optional[Dict] = None,
        prediction_horizon: int = 12,
        movement_threshold_pct: float = 1.5,
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
        return os.path.join(self.model_dir, "memes_xgb.joblib")

    def _try_load(self):
        path = self._model_path()
        if os.path.exists(path):
            try:
                saved = joblib.load(path)
                self.model = saved["model"]
                self.last_train_time = saved.get("train_time")
                self.train_metrics = saved.get("metrics", {})
                self.is_trained = True
                logger.info("Loaded memes model from %s", path)
            except Exception as e:
                logger.warning("Failed to load memes model: %s", e)

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

    def _apply_volume_weighting(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        """
        Emphasize volume-related features for meme coins.

        Volume spikes are the primary driver of meme coin price action.
        OBV, volume_ratio, and VWAP get 2x weight.
        """
        X_weighted = X.copy()
        volume_features = {"obv", "volume_ratio", "vwap", "high_low_range"}

        for i, name in enumerate(feature_names):
            if name in volume_features:
                X_weighted[:, i] = X_weighted[:, i] * 2.0

        return X_weighted

    def train(self, df: pd.DataFrame) -> Dict:
        logger.info("Training memes model on %d candles...", len(df))

        X, feature_names, valid_mask = self.feature_engine.get_feature_matrix(df)
        X = self._apply_volume_weighting(X, feature_names)

        close_valid = df["close"].values[valid_mask]
        y = self._create_labels(close_valid)

        train_end = len(X) - self.prediction_horizon
        if train_end < 100:
            logger.warning("Not enough data for memes model (%d samples)", train_end)
            return {"error": "insufficient_data", "samples": train_end}

        X_train = X[:train_end]
        y_train = y[:train_end]

        # Exponential recency weighting: meme patterns are short-lived
        # Weight ranges from 0.3 (oldest) to 2.0 (newest) on exponential curve
        t = np.linspace(0, 3, len(X_train))
        sample_weights = 0.3 + 1.7 * (np.exp(t) - 1) / (np.exp(3) - 1)

        # Cross validation
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
            "volume_weighting": True,
            "exponential_recency": True,
            "movement_threshold_pct": self.movement_threshold_pct,
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
            "Memes model trained: accuracy=%.3f, f1=%.3f, samples=%d",
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

        X = self._apply_volume_weighting(X, feature_names)
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
        logger.info("Saved memes model to %s", path)

    def get_status(self) -> Dict:
        return {
            "model_type": "memes",
            "symbols": MEME_SYMBOLS,
            "is_trained": self.is_trained,
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
            "metrics": self.train_metrics,
            "params": {k: v for k, v in self.params.items() if k != "verbosity"},
        }
