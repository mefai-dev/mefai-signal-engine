"""
Technical Analysis Scoring Layer.

Evaluates price action across four dimensions:
- Trend: EMA/SMA alignment, ADX strength, directional bias
- Momentum: RSI, MACD, rate of change
- Volatility: Bollinger Band position, ATR expansion/contraction
- Volume: OBV trend, volume ratio, VWAP position

Each dimension produces a sub-score from -100 to +100.
Final technical score is the weighted average.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.ml.feature_engine import FeatureEngine


class TechnicalAnalyzer:
    """
    Multi-dimensional technical analysis scoring engine.

    Scoring methodology:
    - Each indicator contributes a normalized score (-100 to +100)
    - Indicators are grouped into dimensions
    - Dimension scores are weighted and combined
    - Final output: composite score + breakdown per dimension
    """

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.dimension_weights = {
            "trend": 0.30,
            "momentum": 0.30,
            "volatility": 0.20,
            "volume": 0.20,
        }

    def score(self, df: pd.DataFrame) -> Dict:
        """
        Generate technical analysis score from OHLCV data.

        Args:
            df: DataFrame with at least 200 rows of OHLCV data.

        Returns:
            Dict with composite score, dimension breakdown, and key levels.
        """
        featured = self.feature_engine.compute_all_features(df)

        # Get the latest row with valid features
        latest = featured.iloc[-1]
        close = latest["close"]

        trend_score, trend_details = self._score_trend(featured, latest)
        momentum_score, momentum_details = self._score_momentum(featured, latest)
        volatility_score, volatility_details = self._score_volatility(featured, latest)
        volume_score, volume_details = self._score_volume(featured, latest)

        composite = (
            trend_score * self.dimension_weights["trend"]
            + momentum_score * self.dimension_weights["momentum"]
            + volatility_score * self.dimension_weights["volatility"]
            + volume_score * self.dimension_weights["volume"]
        )

        # Detect support and resistance levels
        support, resistance = self._find_support_resistance(df)

        return {
            "score": float(np.clip(composite, -100, 100)),
            "dimensions": {
                "trend": {"score": float(trend_score), "details": trend_details},
                "momentum": {"score": float(momentum_score), "details": momentum_details},
                "volatility": {"score": float(volatility_score), "details": volatility_details},
                "volume": {"score": float(volume_score), "details": volume_details},
            },
            "key_levels": {
                "support": float(support) if support else None,
                "resistance": float(resistance) if resistance else None,
                "ema_9": float(latest["ema_9"]) if not np.isnan(latest["ema_9"]) else None,
                "ema_21": float(latest["ema_21"]) if not np.isnan(latest["ema_21"]) else None,
                "sma_50": float(latest["sma_50"]) if not np.isnan(latest["sma_50"]) else None,
                "sma_200": float(latest["sma_200"]) if not np.isnan(latest["sma_200"]) else None,
                "bb_upper": float(latest["bb_upper"]) if not np.isnan(latest["bb_upper"]) else None,
                "bb_lower": float(latest["bb_lower"]) if not np.isnan(latest["bb_lower"]) else None,
                "vwap": float(latest["vwap"]) if not np.isnan(latest["vwap"]) else None,
            },
            "current_price": float(close),
        }

    def _score_trend(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[float, Dict]:
        """
        Score trend strength and direction.

        Components:
        - EMA 9/21 alignment: +/- based on cross direction
        - SMA 50/200 golden/death cross: strong directional signal
        - ADX: trend strength (>25 = strong trend)
        - Price vs moving averages: bullish if above, bearish if below
        """
        score = 0.0
        details = {}
        close = latest["close"]

        # EMA 9/21 crossover (-30 to +30)
        ema_9 = latest.get("ema_9", np.nan)
        ema_21 = latest.get("ema_21", np.nan)
        if not np.isnan(ema_9) and not np.isnan(ema_21) and ema_21 != 0:
            ema_cross = (ema_9 - ema_21) / ema_21 * 100
            ema_score = np.clip(ema_cross * 15, -30, 30)
            score += ema_score
            details["ema_9_21_cross"] = "bullish" if ema_cross > 0 else "bearish"
            details["ema_cross_pct"] = float(ema_cross)

        # SMA 50/200 golden/death cross (-25 to +25)
        sma_50 = latest.get("sma_50", np.nan)
        sma_200 = latest.get("sma_200", np.nan)
        if not np.isnan(sma_50) and not np.isnan(sma_200) and sma_200 != 0:
            sma_cross = (sma_50 - sma_200) / sma_200 * 100
            sma_score = np.clip(sma_cross * 5, -25, 25)
            score += sma_score
            details["sma_50_200_cross"] = "golden" if sma_cross > 0 else "death"

        # ADX trend strength (0 to +20, direction from DI comparison)
        adx = latest.get("adx_14", np.nan)
        if not np.isnan(adx):
            if adx > 25:
                # Strong trend - amplify existing direction signals
                trend_amplifier = min((adx - 25) / 50, 1.0)
                direction = 1 if score > 0 else -1 if score < 0 else 0
                score += direction * trend_amplifier * 20
                details["adx"] = float(adx)
                details["trend_strength"] = "strong" if adx > 40 else "moderate"
            else:
                details["adx"] = float(adx)
                details["trend_strength"] = "weak"

        # Price position relative to MAs (-25 to +25)
        ma_above_count = 0
        ma_total = 0
        for ma_name in ["ema_9", "ema_21", "sma_50", "sma_200"]:
            ma_val = latest.get(ma_name, np.nan)
            if not np.isnan(ma_val) and ma_val > 0:
                ma_total += 1
                if close > ma_val:
                    ma_above_count += 1

        if ma_total > 0:
            ma_ratio = (ma_above_count / ma_total - 0.5) * 2  # -1 to +1
            score += ma_ratio * 25
            details["price_vs_ma"] = f"above {ma_above_count}/{ma_total}"

        return float(np.clip(score, -100, 100)), details

    def _score_momentum(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[float, Dict]:
        """
        Score momentum indicators.

        Components:
        - RSI(14): overbought/oversold detection
        - RSI(7): short-term momentum shift
        - MACD: signal line cross and histogram direction
        - MACD Histogram trend: acceleration/deceleration
        """
        score = 0.0
        details = {}

        # RSI(14) scoring (-35 to +35)
        rsi_14 = latest.get("rsi_14", np.nan)
        if not np.isnan(rsi_14):
            # RSI between 30-70 is neutral. Outside = momentum signal
            if rsi_14 > 70:
                # Overbought - but in strong trends this is bullish momentum
                score += min((rsi_14 - 50) * 0.7, 35)
                details["rsi_14_zone"] = "overbought"
            elif rsi_14 < 30:
                # Oversold - bearish momentum
                score -= min((50 - rsi_14) * 0.7, 35)
                details["rsi_14_zone"] = "oversold"
            else:
                # Mid range: slight directional bias
                rsi_bias = (rsi_14 - 50) * 0.5
                score += rsi_bias
                details["rsi_14_zone"] = "neutral"
            details["rsi_14"] = float(rsi_14)

        # RSI(7) short-term momentum (-15 to +15)
        rsi_7 = latest.get("rsi_7", np.nan)
        if not np.isnan(rsi_7):
            rsi7_bias = (rsi_7 - 50) * 0.3
            score += np.clip(rsi7_bias, -15, 15)
            details["rsi_7"] = float(rsi_7)

        # MACD line vs signal (-30 to +30)
        macd_line = latest.get("macd_line", np.nan)
        macd_signal = latest.get("macd_signal", np.nan)
        macd_hist = latest.get("macd_histogram", np.nan)

        if not np.isnan(macd_line) and not np.isnan(macd_signal):
            # MACD cross direction
            if macd_line > macd_signal:
                score += 15
                details["macd_cross"] = "bullish"
            else:
                score -= 15
                details["macd_cross"] = "bearish"

            # MACD line position (above/below zero)
            if macd_line > 0:
                score += 10
            else:
                score -= 10
            details["macd_line"] = float(macd_line)

        # MACD histogram momentum (-20 to +20)
        if not np.isnan(macd_hist):
            # Check histogram trend (last 3 values)
            hist_values = df["macd_histogram"].dropna().tail(3).values
            if len(hist_values) >= 3:
                if hist_values[-1] > hist_values[-2] > hist_values[-3]:
                    score += 10  # Accelerating bullish
                    details["macd_hist_trend"] = "accelerating_bullish"
                elif hist_values[-1] < hist_values[-2] < hist_values[-3]:
                    score -= 10  # Accelerating bearish
                    details["macd_hist_trend"] = "accelerating_bearish"
                elif hist_values[-1] > hist_values[-2]:
                    score += 5
                    details["macd_hist_trend"] = "turning_bullish"
                else:
                    score -= 5
                    details["macd_hist_trend"] = "turning_bearish"

        return float(np.clip(score, -100, 100)), details

    def _score_volatility(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[float, Dict]:
        """
        Score volatility conditions.

        Components:
        - Bollinger Band position: where price sits in the bands
        - BB width: squeeze detection (contraction = pending breakout)
        - ATR trend: expanding or contracting volatility
        """
        score = 0.0
        details = {}
        close = latest["close"]

        bb_upper = latest.get("bb_upper", np.nan)
        bb_lower = latest.get("bb_lower", np.nan)
        bb_middle = latest.get("bb_middle", np.nan)

        if not np.isnan(bb_upper) and not np.isnan(bb_lower) and not np.isnan(bb_middle):
            bb_width = bb_upper - bb_lower
            if bb_width > 0:
                # Position within bands: 0 = lower band, 1 = upper band
                bb_position = (close - bb_lower) / bb_width
                details["bb_position"] = float(bb_position)

                # Score based on position
                if bb_position > 0.8:
                    # Near upper band - bullish momentum but extended
                    score += 20
                    details["bb_zone"] = "upper_band"
                elif bb_position < 0.2:
                    # Near lower band - bearish momentum but potential reversal
                    score -= 20
                    details["bb_zone"] = "lower_band"
                else:
                    # Mid-band: slight directional bias
                    score += (bb_position - 0.5) * 40
                    details["bb_zone"] = "mid_band"

                # BB width relative to price (squeeze detection)
                bb_width_pct = bb_width / close * 100
                details["bb_width_pct"] = float(bb_width_pct)

                # Check for Bollinger squeeze (width below recent average)
                bb_widths = (df["bb_upper"] - df["bb_lower"]).dropna().tail(50)
                if len(bb_widths) > 10:
                    avg_width = bb_widths.mean()
                    if bb_width < avg_width * 0.6:
                        details["bb_squeeze"] = True
                        # Squeeze amplifies direction from other signals
                        score *= 1.3
                    else:
                        details["bb_squeeze"] = False

        # ATR trend (-30 to +30)
        atr = latest.get("atr_14", np.nan)
        if not np.isnan(atr):
            atr_values = df["atr_14"].dropna().tail(20).values
            if len(atr_values) >= 5:
                atr_sma = np.mean(atr_values[-5:])
                atr_sma_prev = np.mean(atr_values[:5])
                if atr_sma_prev > 0:
                    atr_change = (atr_sma - atr_sma_prev) / atr_sma_prev * 100
                    details["atr_trend"] = "expanding" if atr_change > 0 else "contracting"
                    details["atr_change_pct"] = float(atr_change)

                    # Expanding volatility amplifies trend direction
                    if atr_change > 10:
                        direction = 1 if score > 0 else -1
                        score += direction * min(atr_change * 0.5, 20)

            details["atr_14"] = float(atr)

        return float(np.clip(score, -100, 100)), details

    def _score_volume(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[float, Dict]:
        """
        Score volume indicators.

        Components:
        - OBV trend: accumulation vs distribution
        - Volume ratio: current vs average (spike detection)
        - VWAP position: institutional fair value
        - Taker buy ratio: buy vs sell pressure
        """
        score = 0.0
        details = {}
        close = latest["close"]

        # OBV trend (-30 to +30)
        obv_values = df["obv"].dropna().tail(20).values if "obv" in df else np.array([])
        if len(obv_values) >= 10:
            obv_sma_short = np.mean(obv_values[-5:])
            obv_sma_long = np.mean(obv_values[-10:])
            if obv_sma_long != 0:
                obv_trend = (obv_sma_short - obv_sma_long) / abs(obv_sma_long) * 100
                score += np.clip(obv_trend * 3, -30, 30)
                details["obv_trend"] = "accumulation" if obv_trend > 0 else "distribution"

        # Volume ratio (-25 to +25)
        vol_ratio = latest.get("volume_ratio", np.nan)
        if not np.isnan(vol_ratio):
            details["volume_ratio"] = float(vol_ratio)

            # High volume confirms trend, low volume weakens it
            if vol_ratio > 2.0:
                direction = 1 if score > 0 else -1 if score < 0 else 0
                score += direction * 25
                details["volume_signal"] = "high_volume_confirmation"
            elif vol_ratio > 1.5:
                direction = 1 if score > 0 else -1 if score < 0 else 0
                score += direction * 15
                details["volume_signal"] = "above_average"
            elif vol_ratio < 0.5:
                score *= 0.7  # Low volume weakens any signal
                details["volume_signal"] = "low_volume_warning"
            else:
                details["volume_signal"] = "normal"

        # VWAP position (-25 to +25)
        vwap = latest.get("vwap", np.nan)
        if not np.isnan(vwap) and vwap > 0:
            vwap_dist = (close - vwap) / vwap * 100
            score += np.clip(vwap_dist * 10, -25, 25)
            details["vwap_position"] = "above" if close > vwap else "below"
            details["vwap_distance_pct"] = float(vwap_dist)

        # Taker buy ratio (if available)
        if "taker_buy_volume" in df.columns and "volume" in df.columns:
            recent_taker = df["taker_buy_volume"].tail(10).sum()
            recent_vol = df["volume"].tail(10).sum()
            if recent_vol > 0:
                buy_ratio = recent_taker / recent_vol
                details["taker_buy_ratio"] = float(buy_ratio)
                # Above 0.5 = more buying than selling
                buy_pressure = (buy_ratio - 0.5) * 40
                score += np.clip(buy_pressure, -20, 20)

        return float(np.clip(score, -100, 100)), details

    def _find_support_resistance(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> Tuple[float, float]:
        """
        Identify nearest support and resistance levels using swing highs/lows.

        Uses a simple pivot point detection: a swing high is a local maximum
        with lower highs on both sides, and vice versa for swing lows.
        """
        data = df.tail(lookback)
        if len(data) < 10:
            return None, None

        high = data["high"].values
        low = data["low"].values
        close_current = data["close"].iloc[-1]

        swing_highs = []
        swing_lows = []
        window = 5

        for i in range(window, len(high) - window):
            # Swing high: higher than surrounding candles
            if high[i] == max(high[i - window: i + window + 1]):
                swing_highs.append(high[i])

            # Swing low: lower than surrounding candles
            if low[i] == min(low[i - window: i + window + 1]):
                swing_lows.append(low[i])

        # Find nearest support (highest swing low below current price)
        supports = [s for s in swing_lows if s < close_current]
        support = max(supports) if supports else None

        # Find nearest resistance (lowest swing high above current price)
        resistances = [r for r in swing_highs if r > close_current]
        resistance = min(resistances) if resistances else None

        return support, resistance
