"""
Mefai Signal Engine - Meta-Ensemble

Combines predictions from Transformer, HMM regime detector, RL position sizer,
and XGBoost using dynamic weighting, Platt scaling for confidence calibration,
and disagreement detection.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

logger = logging.getLogger(__name__)


@dataclass
class EnsembleSignal:
    direction: str
    confidence: float
    position_size: float
    stop_loss: float
    take_profit: float
    risk_metrics: Dict
    model_contributions: Dict
    disagreement_level: float
    calibrated_confidence: float


class PlattScaler:
    """
    Platt scaling for probability calibration.

    Fits a logistic regression on model scores vs actual outcomes
    to produce calibrated probabilities.
    P(y=1|f) = 1 / (1 + exp(A*f + B))
    """

    def __init__(self):
        self.A = -1.0
        self.B = 0.0
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit Platt scaling parameters A and B.

        Args:
            scores: Raw model scores/confidences
            labels: Binary outcomes (1 = correct prediction, 0 = incorrect)
        """
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        n = len(labels)

        if n < 5 or n_pos < 2 or n_neg < 2:
            logger.warning("Not enough data for Platt scaling, using defaults")
            return

        # Target probabilities (Bayesian prior smoothing)
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1.0 / (n_neg + 2)
        targets = np.where(labels == 1, t_pos, t_neg)

        def objective(params):
            a, b = params
            p = sigmoid(a * scores + b)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            nll = -np.sum(targets * np.log(p) + (1 - targets) * np.log(1 - p))
            return nll

        result = minimize(objective, x0=[-1.0, 0.0], method="Nelder-Mead")
        self.A, self.B = result.x
        self.is_fitted = True
        logger.info(f"Platt scaling fitted: A={self.A:.4f}, B={self.B:.4f}")

    def calibrate(self, score: float) -> float:
        """Apply Platt scaling to a raw score."""
        return float(sigmoid(self.A * score + self.B))


class MetaEnsemble:
    """
    Meta-ensemble that combines all model predictions.

    Components:
        - Transformer price predictor (direction + magnitude)
        - HMM regime detector (regime context)
        - RL position sizer (position sizing)
        - XGBoost signal (direction + confidence)

    Combination method:
        - Dynamic weights based on exponential moving average of recent accuracy
        - Platt scaling for confidence calibration
        - Disagreement detection for risk reduction
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        ema_alpha: float = 0.1,
        disagreement_threshold: float = 0.4,
    ):
        if model_names is None:
            self.model_names = ["transformer", "xgboost", "hmm", "rl"]
        else:
            self.model_names = model_names

        self.n_models = len(self.model_names)
        self.weights = np.ones(self.n_models) / self.n_models
        self.ema_alpha = ema_alpha
        self.disagreement_threshold = disagreement_threshold

        self.accuracy_ema = np.ones(self.n_models) * 0.5
        self.platt_scaler = PlattScaler()
        self.prediction_history: List[Dict] = []
        self.actual_history: List[float] = []

    def update_weights(self, model_accuracies: Dict[str, float]):
        """
        Update model weights based on recent accuracy using EMA.

        Args:
            model_accuracies: Dict of model_name -> accuracy (0 to 1)
        """
        for i, name in enumerate(self.model_names):
            if name in model_accuracies:
                acc = model_accuracies[name]
                self.accuracy_ema[i] = (
                    self.ema_alpha * acc + (1 - self.ema_alpha) * self.accuracy_ema[i]
                )

        # Convert EMA accuracies to weights (softmax-like normalization)
        # Subtract min to avoid numerical issues
        adjusted = self.accuracy_ema - self.accuracy_ema.min() + 0.1
        self.weights = adjusted / adjusted.sum()

        logger.info(
            f"Updated weights: "
            + ", ".join(f"{n}={w:.3f}" for n, w in zip(self.model_names, self.weights))
        )

    def combine(
        self,
        predictions: Dict[str, Dict],
        current_price: float,
        atr: float,
    ) -> EnsembleSignal:
        """
        Combine predictions from all models into a single signal.

        Args:
            predictions: Dict of model predictions, each containing:
                - direction: "LONG", "SHORT", or "NEUTRAL" (for directional models)
                - confidence: float 0-1
                - position_size: float 0-1 (for RL model)
                - regime: str (for HMM model)
                - predicted_return: float (for transformer/xgboost)
            current_price: Current market price
            atr: Average True Range for stop/target calculation

        Returns:
            EnsembleSignal with combined decision
        """
        direction_scores = []
        confidences = []
        model_contributions = {}

        for i, name in enumerate(self.model_names):
            pred = predictions.get(name, {})
            weight = self.weights[i]

            direction = pred.get("direction", "NEUTRAL")
            confidence = pred.get("confidence", 0.5)

            # Convert direction to numeric: LONG=+1, SHORT=-1, NEUTRAL=0
            if direction == "LONG":
                dir_score = 1.0
            elif direction == "SHORT":
                dir_score = -1.0
            else:
                dir_score = 0.0

            weighted_score = dir_score * confidence * weight
            direction_scores.append(weighted_score)
            confidences.append(confidence * weight)

            model_contributions[name] = {
                "direction": direction,
                "confidence": confidence,
                "weight": float(weight),
                "contribution": float(weighted_score),
            }

        # Aggregate direction
        total_score = sum(direction_scores)
        total_weight = sum(self.weights)

        if total_score > 0.1:
            final_direction = "LONG"
        elif total_score < -0.1:
            final_direction = "SHORT"
        else:
            final_direction = "NEUTRAL"

        # Raw confidence = magnitude of aggregate score normalized
        raw_confidence = min(1.0, abs(total_score) / max(total_weight * 0.5, 1e-8))

        # Calibrate confidence
        if self.platt_scaler.is_fitted:
            calibrated_confidence = self.platt_scaler.calibrate(raw_confidence)
        else:
            calibrated_confidence = raw_confidence

        # Disagreement detection
        disagreement = self._compute_disagreement(predictions)

        if disagreement > self.disagreement_threshold:
            calibrated_confidence *= max(0.3, 1.0 - disagreement)
            logger.info(
                f"High model disagreement ({disagreement:.2f}), "
                f"reducing confidence to {calibrated_confidence:.2f}"
            )

        # Position sizing - prefer RL model output if available
        rl_pred = predictions.get("rl", {})
        if "position_size" in rl_pred:
            base_position = rl_pred["position_size"]
        else:
            base_position = calibrated_confidence

        # Apply regime adjustment if HMM provides it
        hmm_pred = predictions.get("hmm", {})
        regime_multiplier = hmm_pred.get("position_multiplier", 1.0)
        final_position = min(1.0, base_position * regime_multiplier)

        # Stop loss and take profit based on ATR
        if final_direction == "LONG":
            stop_loss = current_price - 2.0 * atr
            take_profit = current_price + 3.0 * atr
        elif final_direction == "SHORT":
            stop_loss = current_price + 2.0 * atr
            take_profit = current_price - 3.0 * atr
        else:
            stop_loss = current_price - 1.5 * atr
            take_profit = current_price + 1.5 * atr

        # Risk metrics
        risk_metrics = {
            "raw_confidence": float(raw_confidence),
            "disagreement": float(disagreement),
            "regime": hmm_pred.get("regime", "unknown"),
            "regime_multiplier": float(regime_multiplier),
            "n_models_agree": self._count_agreement(predictions, final_direction),
            "atr": float(atr),
            "risk_reward_ratio": float(abs(take_profit - current_price) / max(abs(stop_loss - current_price), 1e-8)),
        }

        return EnsembleSignal(
            direction=final_direction,
            confidence=float(raw_confidence),
            position_size=float(final_position),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            risk_metrics=risk_metrics,
            model_contributions=model_contributions,
            disagreement_level=float(disagreement),
            calibrated_confidence=float(calibrated_confidence),
        )

    def _compute_disagreement(self, predictions: Dict[str, Dict]) -> float:
        """
        Compute disagreement level among models.

        Returns a value from 0 (full agreement) to 1 (maximum disagreement).
        Uses variance of direction scores weighted by confidence.
        """
        scores = []
        for name in self.model_names:
            pred = predictions.get(name, {})
            direction = pred.get("direction", "NEUTRAL")
            confidence = pred.get("confidence", 0.5)

            if direction == "LONG":
                scores.append(confidence)
            elif direction == "SHORT":
                scores.append(-confidence)
            else:
                scores.append(0.0)

        if len(scores) < 2:
            return 0.0

        scores = np.array(scores)
        # Normalize disagreement: std of [-1, 1] range, max std is 1.0
        return float(np.std(scores))

    def _count_agreement(self, predictions: Dict[str, Dict], direction: str) -> int:
        """Count how many models agree with the final direction."""
        count = 0
        for name in self.model_names:
            pred = predictions.get(name, {})
            if pred.get("direction") == direction:
                count += 1
        return count

    def record_outcome(self, prediction: Dict, actual_return: float):
        """
        Record a prediction outcome for calibration and weight updates.

        Args:
            prediction: The prediction dict that was made
            actual_return: The actual return that occurred
        """
        self.prediction_history.append(prediction)
        self.actual_history.append(actual_return)

        # Update model accuracies based on direction correctness
        if len(self.actual_history) >= 20:
            recent_preds = self.prediction_history[-50:]
            recent_actuals = self.actual_history[-50:]

            model_accuracies = {}
            for name in self.model_names:
                correct = 0
                total = 0
                for pred, actual in zip(recent_preds, recent_actuals):
                    model_pred = pred.get(name, {})
                    model_dir = model_pred.get("direction", "NEUTRAL")

                    if model_dir == "NEUTRAL":
                        continue

                    total += 1
                    if (model_dir == "LONG" and actual > 0) or (
                        model_dir == "SHORT" and actual < 0
                    ):
                        correct += 1

                if total > 0:
                    model_accuracies[name] = correct / total

            if model_accuracies:
                self.update_weights(model_accuracies)

        # Recalibrate Platt scaler periodically
        if len(self.actual_history) >= 50 and len(self.actual_history) % 25 == 0:
            self._recalibrate()

    def _recalibrate(self):
        """Recalibrate Platt scaler on accumulated history."""
        if len(self.prediction_history) < 50:
            return

        scores = []
        labels = []

        for pred, actual in zip(self.prediction_history[-200:], self.actual_history[-200:]):
            # Get the ensemble confidence from the prediction
            conf = 0.0
            direction = "NEUTRAL"
            for name in self.model_names:
                model_pred = pred.get(name, {})
                d = model_pred.get("direction", "NEUTRAL")
                c = model_pred.get("confidence", 0.5)
                if d != "NEUTRAL":
                    conf = max(conf, c)
                    direction = d

            scores.append(conf)
            correct = (direction == "LONG" and actual > 0) or (
                direction == "SHORT" and actual < 0
            )
            labels.append(1 if correct else 0)

        self.platt_scaler.fit(np.array(scores), np.array(labels))

    def get_model_weights(self) -> Dict[str, float]:
        """Return current model weights."""
        return {name: float(w) for name, w in zip(self.model_names, self.weights)}

    def get_model_ema_accuracy(self) -> Dict[str, float]:
        """Return current EMA accuracy per model."""
        return {name: float(a) for name, a in zip(self.model_names, self.accuracy_ema)}

    def summary(self) -> Dict:
        """Return ensemble state summary."""
        return {
            "model_weights": self.get_model_weights(),
            "ema_accuracy": self.get_model_ema_accuracy(),
            "platt_calibrated": self.platt_scaler.is_fitted,
            "platt_params": {"A": self.platt_scaler.A, "B": self.platt_scaler.B},
            "n_recorded_outcomes": len(self.actual_history),
            "disagreement_threshold": self.disagreement_threshold,
        }
