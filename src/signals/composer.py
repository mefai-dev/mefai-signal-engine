"""
Signal Composer - Multi-layer signal composition engine.

Combines five scoring layers into a final trading signal:
1. Technical Analysis (30-45%)
2. Correlation Analysis (10-15%)
3. On-Chain Metrics (15-25%)
4. Sentiment Scoring (10-15%)
5. XGBoost ML Predictions (35%)

The weights are adaptive - they shift based on market conditions:
- Trending market: technical + ML get more weight
- Range-bound: correlation + on-chain get more weight
- High-volatility: sentiment weight increases (news-driven)
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.signals.technical import TechnicalAnalyzer
from src.signals.correlation import CorrelationAnalyzer
from src.signals.onchain import OnChainAnalyzer
from src.signals.sentiment import SentimentAnalyzer
from src.ml.trainer import Trainer

logger = logging.getLogger(__name__)


class SignalComposer:
    """
    Multi-layer signal composition engine.

    Produces final trading signals by combining five independent
    analysis layers, each scoring from -100 to +100.

    Output includes:
    - Direction: LONG / SHORT / NEUTRAL
    - Confidence: 0-100%
    - Entry, stop loss, take profit levels
    - Risk/reward ratio
    - Full layer breakdown for transparency
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        trainer: Optional[Trainer] = None,
    ):
        self.config = config or {}

        # Default weights (sum to 1.0)
        weights = self.config.get("signal_weights", {})
        self.base_weights = {
            "technical": weights.get("technical", 0.30),
            "correlation": weights.get("correlation", 0.10),
            "onchain": weights.get("onchain", 0.15),
            "sentiment": weights.get("sentiment", 0.10),
            "ml_prediction": weights.get("ml_prediction", 0.35),
        }

        thresholds = self.config.get("signal_thresholds", {})
        self.thresholds = {
            "strong_long": thresholds.get("strong_long", 60),
            "long": thresholds.get("long", 30),
            "neutral_upper": thresholds.get("neutral_upper", 15),
            "neutral_lower": thresholds.get("neutral_lower", -15),
            "short": thresholds.get("short", -30),
            "strong_short": thresholds.get("strong_short", -60),
        }

        risk_config = self.config.get("risk", {})
        self.atr_stop_multiplier = risk_config.get("atr_stop_multiplier", 1.5)
        self.atr_tp_multiplier = risk_config.get("atr_tp_multiplier", 3.0)
        self.default_sl_pct = risk_config.get("default_stop_loss_pct", 1.5)
        self.default_tp_pct = risk_config.get("default_take_profit_pct", 3.0)

        # Initialize analysis layers
        self.technical = TechnicalAnalyzer()
        self.correlation = CorrelationAnalyzer()
        self.onchain = OnChainAnalyzer(
            base_url=self.config.get("binance", {}).get("base_url", "https://fapi.binance.com")
        )
        self.sentiment = SentimentAnalyzer(
            feeds=self.config.get("sentiment", {}).get("feeds"),
            max_age_hours=self.config.get("sentiment", {}).get("article_max_age_hours", 24),
        )
        self.trainer = trainer

        # Signal history for performance tracking
        self._signal_history: List[Dict] = []

    async def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> Dict:
        """
        Generate a complete trading signal for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            df: Recent OHLCV DataFrame (200+ candles recommended)

        Returns:
            Complete signal with direction, confidence, levels, and layer breakdown.
        """
        current_price = float(df["close"].iloc[-1])

        # Update price cache for on-chain analyzer
        if len(df) >= 2:
            prev_close = float(df["close"].iloc[-2])
            price_change = (current_price - prev_close) / prev_close * 100
            self.onchain.update_price(symbol, current_price, price_change)

        # Run all analysis layers
        # Technical (synchronous)
        tech_result = self.technical.score(df)
        tech_score = tech_result["score"]

        # Correlation (synchronous)
        corr_result = self.correlation.score(symbol, df)
        corr_score = corr_result["score"]

        # On-chain (async - API calls)
        try:
            onchain_result = await self.onchain.score(symbol)
            onchain_score = onchain_result["score"]
        except Exception as e:
            logger.warning("On-chain scoring failed for %s: %s", symbol, e)
            onchain_result = {"score": 0.0, "error": str(e)}
            onchain_score = 0.0

        # Sentiment (synchronous - uses cached data)
        sentiment_result = self.sentiment.score(symbol)
        sentiment_score = sentiment_result["score"]

        # ML prediction (synchronous)
        ml_result = {"direction": "NEUTRAL", "confidence": 0.0, "model_trained": False}
        ml_score = 0.0
        if self.trainer:
            model = self.trainer.get_model_for_symbol(symbol)
            ml_result = model.predict(df)
            # Convert ML direction to score
            if ml_result["direction"] == "LONG":
                ml_score = ml_result["confidence"]
            elif ml_result["direction"] == "SHORT":
                ml_score = -ml_result["confidence"]

        # Adaptive weight adjustment based on market conditions
        weights = self._adapt_weights(tech_result, df)

        # Compute composite score
        composite_score = (
            tech_score * weights["technical"]
            + corr_score * weights["correlation"]
            + onchain_score * weights["onchain"]
            + sentiment_score * weights["sentiment"]
            + ml_score * weights["ml_prediction"]
        )

        # Determine direction and confidence
        direction = self._classify_direction(composite_score)
        confidence = self._calculate_confidence(
            composite_score, tech_score, corr_score,
            onchain_score, sentiment_score, ml_score
        )

        # Calculate entry, SL, TP levels
        atr = self._get_current_atr(df)
        entry, sl, tp, rr_ratio = self._calculate_levels(
            current_price, direction, atr
        )

        signal = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "confidence": float(confidence),
            "composite_score": float(np.clip(composite_score, -100, 100)),
            "entry_price": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "risk_reward_ratio": rr_ratio,
            "current_price": current_price,
            "layers": {
                "technical": {
                    "score": float(tech_score),
                    "weight": float(weights["technical"]),
                    "details": tech_result,
                },
                "correlation": {
                    "score": float(corr_score),
                    "weight": float(weights["correlation"]),
                    "details": corr_result,
                },
                "onchain": {
                    "score": float(onchain_score),
                    "weight": float(weights["onchain"]),
                    "details": onchain_result,
                },
                "sentiment": {
                    "score": float(sentiment_score),
                    "weight": float(weights["sentiment"]),
                    "details": sentiment_result,
                },
                "ml_prediction": {
                    "score": float(ml_score),
                    "weight": float(weights["ml_prediction"]),
                    "details": ml_result,
                },
            },
            "weights_used": weights,
        }

        # Store in history
        self._signal_history.append({
            "symbol": symbol,
            "timestamp": signal["timestamp"],
            "direction": direction,
            "confidence": confidence,
            "entry_price": entry,
            "composite_score": composite_score,
        })

        # Keep only last 1000 signals in memory
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-1000:]

        return signal

    def _adapt_weights(self, tech_result: Dict, df: pd.DataFrame) -> Dict:
        """
        Dynamically adjust layer weights based on market conditions.

        - Strong trend (high ADX): increase technical + ML weight
        - High volatility (wide BB): increase sentiment weight
        - Low volatility (BB squeeze): increase on-chain weight
        """
        weights = self.base_weights.copy()

        # Check ADX for trend strength
        adx = tech_result.get("dimensions", {}).get("trend", {}).get("details", {}).get("adx", 20)

        if adx > 40:
            # Strong trend: boost technical and ML
            weights["technical"] += 0.05
            weights["ml_prediction"] += 0.05
            weights["correlation"] -= 0.05
            weights["sentiment"] -= 0.05
        elif adx < 20:
            # Weak trend (range-bound): boost correlation and on-chain
            weights["correlation"] += 0.05
            weights["onchain"] += 0.05
            weights["technical"] -= 0.05
            weights["ml_prediction"] -= 0.05

        # Check for volatility squeeze
        bb_squeeze = (
            tech_result.get("dimensions", {})
            .get("volatility", {})
            .get("details", {})
            .get("bb_squeeze", False)
        )
        if bb_squeeze:
            weights["onchain"] += 0.03
            weights["sentiment"] += 0.02
            weights["technical"] -= 0.05

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _classify_direction(self, score: float) -> str:
        """Classify composite score into direction."""
        if score >= self.thresholds["strong_long"]:
            return "STRONG_LONG"
        elif score >= self.thresholds["long"]:
            return "LONG"
        elif score <= self.thresholds["strong_short"]:
            return "STRONG_SHORT"
        elif score <= self.thresholds["short"]:
            return "SHORT"
        else:
            return "NEUTRAL"

    @staticmethod
    def _calculate_confidence(
        composite: float, tech: float, corr: float,
        onchain: float, sentiment: float, ml: float
    ) -> float:
        """
        Calculate signal confidence based on layer agreement.

        Higher confidence when layers agree on direction.
        Lower confidence when layers conflict.
        """
        scores = [tech, corr, onchain, sentiment, ml]

        # Count how many layers agree on direction
        positive = sum(1 for s in scores if s > 10)
        negative = sum(1 for s in scores if s < -10)

        # Agreement ratio: higher when all layers point same way
        agreement = max(positive, negative) / max(len(scores), 1)

        # Base confidence from composite strength
        base_confidence = min(abs(composite), 100)

        # Boost by agreement, penalize by disagreement
        if positive > 0 and negative > 0:
            # Mixed signals: reduce confidence
            conflict_penalty = min(positive, negative) / max(len(scores), 1)
            confidence = base_confidence * (1.0 - conflict_penalty * 0.3)
        else:
            # Full agreement: boost confidence
            confidence = base_confidence * (0.8 + agreement * 0.2)

        return float(np.clip(confidence, 0, 100))

    def _calculate_levels(
        self,
        current_price: float,
        direction: str,
        atr: Optional[float],
    ) -> Tuple[float, float, float, float]:
        """
        Calculate entry, stop loss, and take profit levels.

        Uses ATR for dynamic levels when available,
        falls back to fixed percentage otherwise.
        """
        entry = current_price

        if atr and atr > 0:
            if direction in ("LONG", "STRONG_LONG"):
                sl = entry - atr * self.atr_stop_multiplier
                tp = entry + atr * self.atr_tp_multiplier
            elif direction in ("SHORT", "STRONG_SHORT"):
                sl = entry + atr * self.atr_stop_multiplier
                tp = entry - atr * self.atr_tp_multiplier
            else:
                # Neutral: no levels
                return entry, 0.0, 0.0, 0.0
        else:
            if direction in ("LONG", "STRONG_LONG"):
                sl = entry * (1 - self.default_sl_pct / 100)
                tp = entry * (1 + self.default_tp_pct / 100)
            elif direction in ("SHORT", "STRONG_SHORT"):
                sl = entry * (1 + self.default_sl_pct / 100)
                tp = entry * (1 - self.default_tp_pct / 100)
            else:
                return entry, 0.0, 0.0, 0.0

        # Risk/reward ratio
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr_ratio = reward / risk if risk > 0 else 0.0

        return (
            round(entry, 8),
            round(sl, 8),
            round(tp, 8),
            round(rr_ratio, 2),
        )

    def _get_current_atr(self, df: pd.DataFrame) -> Optional[float]:
        """Extract current ATR from dataframe."""
        featured = self.technical.feature_engine.compute_all_features(df)
        atr = featured["atr_14"].iloc[-1]
        return float(atr) if not np.isnan(atr) else None

    def get_signal_history(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Return recent signal history, optionally filtered by symbol."""
        history = self._signal_history
        if symbol:
            history = [s for s in history if s["symbol"] == symbol]
        return history[-limit:]
