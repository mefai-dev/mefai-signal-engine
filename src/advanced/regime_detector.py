"""
Mefai Signal Engine - Market Regime Detector

Hidden Markov Model for detecting market regimes (bull trend, bear trend,
sideways, high volatility) using returns, volatility, volume ratio, and RSI
as observable features.
"""

import logging
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    BULL_TREND = 0
    BEAR_TREND = 1
    SIDEWAYS = 2
    HIGH_VOLATILITY = 3


REGIME_LABELS = {
    MarketRegime.BULL_TREND: "Bull Trend",
    MarketRegime.BEAR_TREND: "Bear Trend",
    MarketRegime.SIDEWAYS: "Sideways",
    MarketRegime.HIGH_VOLATILITY: "High Volatility",
}

REGIME_POSITION_MULTIPLIER = {
    MarketRegime.BULL_TREND: 1.0,
    MarketRegime.BEAR_TREND: 0.7,
    MarketRegime.SIDEWAYS: 0.5,
    MarketRegime.HIGH_VOLATILITY: 0.3,
}


@dataclass
class RegimeResult:
    current_regime: MarketRegime
    regime_label: str
    confidence: float
    regime_probabilities: Dict[str, float]
    transition_matrix: np.ndarray
    position_multiplier: float
    regime_durations: Dict[str, float]
    regime_history: List[int]


@dataclass
class RegimeStats:
    avg_duration: float = 0.0
    count: int = 0
    total_candles: int = 0
    avg_return: float = 0.0
    avg_volatility: float = 0.0


class RegimeDetector:
    """
    Hidden Markov Model regime detector with 4 market states.

    Features used as observations:
        - Log returns
        - Rolling volatility (20 period)
        - Volume ratio (current / rolling 20 mean)
        - RSI (14 period)
    """

    def __init__(self, n_regimes: int = 4, n_iter: int = 200, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            tol=1e-4,
        )
        self.is_fitted = False
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.regime_mapping: Dict[int, MarketRegime] = {}

    @staticmethod
    def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute Relative Strength Index."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi = np.full(len(closes), 50.0)

        if len(gains) < period:
            return rsi

        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def extract_features(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Extract HMM observation features from price and volume data.

        Args:
            closes: Close prices array
            volumes: Volume array

        Returns:
            Feature matrix of shape (n_valid, 4)
        """
        n = len(closes)
        if n < 21:
            raise ValueError(f"Need at least 21 candles, got {n}")

        log_returns = np.diff(np.log(closes + 1e-10))

        rolling_vol = np.full(n, np.nan)
        for i in range(19, n):
            rolling_vol[i] = np.std(log_returns[max(0, i - 19) : i + 1])

        vol_mean = np.full(n, np.nan)
        for i in range(19, n):
            vol_mean[i] = np.mean(volumes[max(0, i - 19) : i + 1])
        volume_ratio = np.where(vol_mean > 0, volumes / vol_mean, 1.0)

        rsi = self.compute_rsi(closes, period=14)

        start_idx = 20
        features = np.column_stack(
            [
                log_returns[start_idx - 1 :],
                rolling_vol[start_idx:],
                volume_ratio[start_idx:],
                rsi[start_idx:],
            ]
        )

        valid_mask = ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
        features = features[valid_mask]

        return features

    def _assign_regimes(self, features: np.ndarray, states: np.ndarray):
        """
        Map HMM hidden states to market regimes based on feature statistics.

        Uses mean return and volatility per state to classify:
            - High return + low vol = BULL_TREND
            - Low return + low vol = BEAR_TREND
            - Near-zero return + low vol = SIDEWAYS
            - High vol = HIGH_VOLATILITY
        """
        state_stats = {}
        for state in range(self.n_regimes):
            mask = states == state
            if mask.sum() == 0:
                state_stats[state] = (0.0, 0.0)
                continue
            mean_ret = features[mask, 0].mean()
            mean_vol = features[mask, 1].mean()
            state_stats[state] = (mean_ret, mean_vol)

        vol_values = [v[1] for v in state_stats.values()]
        vol_median = np.median(vol_values)

        high_vol_state = max(state_stats, key=lambda s: state_stats[s][1])

        remaining = [s for s in range(self.n_regimes) if s != high_vol_state]
        remaining.sort(key=lambda s: state_stats[s][0], reverse=True)

        self.regime_mapping = {}
        self.regime_mapping[high_vol_state] = MarketRegime.HIGH_VOLATILITY

        if len(remaining) >= 3:
            self.regime_mapping[remaining[0]] = MarketRegime.BULL_TREND
            self.regime_mapping[remaining[-1]] = MarketRegime.BEAR_TREND
            for s in remaining[1:-1]:
                self.regime_mapping[s] = MarketRegime.SIDEWAYS
        elif len(remaining) == 2:
            self.regime_mapping[remaining[0]] = MarketRegime.BULL_TREND
            self.regime_mapping[remaining[1]] = MarketRegime.BEAR_TREND
        elif len(remaining) == 1:
            self.regime_mapping[remaining[0]] = MarketRegime.SIDEWAYS

    def fit(self, closes: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Fit the HMM on historical price and volume data.

        Args:
            closes: Close prices, shape (n_candles,)
            volumes: Volumes, shape (n_candles,)

        Returns:
            Fit statistics dict
        """
        features = self.extract_features(closes, volumes)

        if len(features) < 50:
            raise ValueError(f"Need at least 50 valid feature rows, got {len(features)}")

        self.feature_means = features.mean(axis=0)
        self.feature_stds = features.std(axis=0) + 1e-8
        features_norm = (features - self.feature_means) / self.feature_stds

        self.model.fit(features_norm)
        states = self.model.predict(features_norm)
        self._assign_regimes(features, states)
        self.is_fitted = True

        score = self.model.score(features_norm)
        logger.info(f"HMM fitted on {len(features)} samples, log-likelihood={score:.2f}")

        return {
            "n_samples": len(features),
            "log_likelihood": score,
            "aic": -2 * score + 2 * self._count_params(),
            "bic": -2 * score + self._count_params() * np.log(len(features)),
            "regime_mapping": {str(k): v.name for k, v in self.regime_mapping.items()},
        }

    def _count_params(self) -> int:
        n = self.n_regimes
        n_features = 4
        start_probs = n - 1
        trans_probs = n * (n - 1)
        means = n * n_features
        covs = n * n_features * (n_features + 1) // 2
        return start_probs + trans_probs + means + covs

    def detect(self, closes: np.ndarray, volumes: np.ndarray) -> RegimeResult:
        """
        Detect current market regime.

        Args:
            closes: Recent close prices (at least 21 candles)
            volumes: Recent volumes

        Returns:
            RegimeResult with current regime, confidence, and statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before detection")

        features = self.extract_features(closes, volumes)
        features_norm = (features - self.feature_means) / self.feature_stds

        state_probs = self.model.predict_proba(features_norm)
        states = self.model.predict(features_norm)

        current_state = states[-1]
        current_probs = state_probs[-1]
        current_regime = self.regime_mapping.get(current_state, MarketRegime.SIDEWAYS)
        confidence = float(current_probs[current_state])

        regime_probs = {}
        for hmm_state, regime in self.regime_mapping.items():
            regime_probs[REGIME_LABELS[regime]] = float(current_probs[hmm_state])

        transition_matrix = self.model.transmat_

        regime_history = [int(self.regime_mapping.get(s, MarketRegime.SIDEWAYS)) for s in states]
        durations = self._compute_regime_durations(regime_history)

        return RegimeResult(
            current_regime=current_regime,
            regime_label=REGIME_LABELS[current_regime],
            confidence=confidence,
            regime_probabilities=regime_probs,
            transition_matrix=transition_matrix,
            position_multiplier=REGIME_POSITION_MULTIPLIER[current_regime],
            regime_durations=durations,
            regime_history=regime_history,
        )

    def _compute_regime_durations(self, regime_history: List[int]) -> Dict[str, float]:
        """Calculate average duration (in candles) for each regime."""
        if not regime_history:
            return {}

        durations: Dict[int, List[int]] = {r: [] for r in MarketRegime}
        current_regime = regime_history[0]
        current_duration = 1

        for i in range(1, len(regime_history)):
            if regime_history[i] == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = regime_history[i]
                current_duration = 1
        durations[current_regime].append(current_duration)

        result = {}
        for regime in MarketRegime:
            d_list = durations[regime]
            label = REGIME_LABELS[regime]
            if d_list:
                result[label] = round(np.mean(d_list), 1)
            else:
                result[label] = 0.0

        return result

    def get_signal_adjustment(
        self, closes: np.ndarray, volumes: np.ndarray
    ) -> Dict:
        """
        Get signal adjustment recommendations based on current regime.

        Returns:
            Dict with position_multiplier, preferred_direction, and notes
        """
        result = self.detect(closes, volumes)

        adjustment = {
            "regime": result.regime_label,
            "confidence": result.confidence,
            "position_multiplier": result.position_multiplier,
            "preferred_direction": None,
            "notes": [],
        }

        if result.current_regime == MarketRegime.BULL_TREND:
            adjustment["preferred_direction"] = "LONG"
            adjustment["notes"].append("Favor long entries, tighter stops on shorts")
        elif result.current_regime == MarketRegime.BEAR_TREND:
            adjustment["preferred_direction"] = "SHORT"
            adjustment["notes"].append("Favor short entries, tighter stops on longs")
        elif result.current_regime == MarketRegime.SIDEWAYS:
            adjustment["preferred_direction"] = "NEUTRAL"
            adjustment["notes"].append("Range-bound - use mean reversion strategies")
        elif result.current_regime == MarketRegime.HIGH_VOLATILITY:
            adjustment["preferred_direction"] = "NEUTRAL"
            adjustment["notes"].append("High volatility - reduce position sizes, widen stops")

        trans = result.transition_matrix
        current_state_idx = None
        for hmm_state, regime in self.regime_mapping.items():
            if regime == result.current_regime:
                current_state_idx = hmm_state
                break

        if current_state_idx is not None:
            stay_prob = trans[current_state_idx, current_state_idx]
            adjustment["regime_persistence"] = float(stay_prob)
            if stay_prob < 0.7:
                adjustment["notes"].append(
                    f"Regime may change soon (persistence={stay_prob:.1%})"
                )

        return adjustment
