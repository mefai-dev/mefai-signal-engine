"""
Feature Engine - Technical indicator computation for ML models.

Computes 15+ features from raw OHLCV data using pure numpy.
No TA-Lib dependency - all calculations implemented from scratch.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class OrderBlock:
    """Detected order block zone."""
    price_high: float
    price_low: float
    block_type: str  # "bullish" or "bearish"
    strength: float  # 0-1, based on volume and wick ratio
    candle_index: int


@dataclass
class FairValueGap:
    """Detected fair value gap (imbalance zone)."""
    high: float
    low: float
    gap_type: str  # "bullish" or "bearish"
    size_pct: float
    candle_index: int


class FeatureEngine:
    """
    Computes all technical features from OHLCV data.

    Features computed:
    1.  RSI(14)
    2.  RSI(7)
    3.  MACD Line
    4.  MACD Signal
    5.  MACD Histogram
    6.  Bollinger Upper Band
    7.  Bollinger Middle Band
    8.  Bollinger Lower Band
    9.  ATR(14)
    10. ADX(14)
    11. EMA(9)
    12. EMA(21)
    13. SMA(50)
    14. SMA(200)
    15. OBV (On-Balance Volume)
    16. VWAP
    17. Order Block Score
    18. Fair Value Gap Score
    """

    def __init__(self):
        self._feature_names = [
            "rsi_14", "rsi_7", "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "atr_14", "adx_14",
            "ema_9", "ema_21", "sma_50", "sma_200", "obv", "vwap",
            "order_block_score", "fvg_score",
            "close_to_bb_upper", "close_to_bb_lower",
            "ema_9_21_cross", "sma_50_200_cross",
            "volume_ratio", "price_change_pct", "high_low_range"
        ]

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from OHLCV dataframe.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                and a datetime index or 'timestamp' column.

        Returns:
            DataFrame with all computed features appended.
        """
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)

        result = df.copy()

        # Trend indicators
        result["ema_9"] = self._ema(close, 9)
        result["ema_21"] = self._ema(close, 21)
        result["sma_50"] = self._sma(close, 50)
        result["sma_200"] = self._sma(close, 200)

        # Momentum indicators
        result["rsi_14"] = self._rsi(close, 14)
        result["rsi_7"] = self._rsi(close, 7)

        macd_line, macd_signal, macd_hist = self._macd(close)
        result["macd_line"] = macd_line
        result["macd_signal"] = macd_signal
        result["macd_histogram"] = macd_hist

        # Volatility indicators
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(close, 20, 2.0)
        result["bb_upper"] = bb_upper
        result["bb_middle"] = bb_middle
        result["bb_lower"] = bb_lower

        result["atr_14"] = self._atr(high, low, close, 14)

        # Directional movement
        result["adx_14"] = self._adx(high, low, close, 14)

        # Volume indicators
        result["obv"] = self._obv(close, volume)
        result["vwap"] = self._vwap(high, low, close, volume)

        # Smart money concepts
        ob_scores = self._order_block_score(open_, high, low, close, volume)
        result["order_block_score"] = ob_scores

        fvg_scores = self._fvg_score(open_, high, low, close)
        result["fvg_score"] = fvg_scores

        # Derived features
        bb_range = np.where(
            (bb_upper - bb_lower) > 0,
            bb_upper - bb_lower,
            1.0
        )
        result["close_to_bb_upper"] = (bb_upper - close) / bb_range
        result["close_to_bb_lower"] = (close - bb_lower) / bb_range

        ema_9 = result["ema_9"].values
        ema_21 = result["ema_21"].values
        result["ema_9_21_cross"] = np.where(
            ema_21 != 0,
            (ema_9 - ema_21) / np.abs(ema_21) * 100,
            0.0
        )

        sma_50 = result["sma_50"].values
        sma_200 = result["sma_200"].values
        result["sma_50_200_cross"] = np.where(
            sma_200 != 0,
            (sma_50 - sma_200) / np.abs(sma_200) * 100,
            0.0
        )

        # Volume ratio (current vs 20-period average)
        vol_sma = self._sma(volume, 20)
        result["volume_ratio"] = np.where(vol_sma > 0, volume / vol_sma, 1.0)

        # Price change percentage
        result["price_change_pct"] = np.where(
            open_ != 0,
            (close - open_) / open_ * 100,
            0.0
        )

        # High-low range as percentage of close
        result["high_low_range"] = np.where(
            close != 0,
            (high - low) / close * 100,
            0.0
        )

        return result

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature matrix suitable for ML model input.

        Returns:
            Tuple of (feature_matrix, feature_names) with NaN rows dropped.
        """
        featured_df = self.compute_all_features(df)
        feature_cols = self._feature_names
        X = featured_df[feature_cols].values

        # Find rows where all features are valid
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X_clean = X[valid_mask]

        return X_clean, feature_cols, valid_mask

    # ----------------------------------------------------------------
    # Core indicator implementations
    # ----------------------------------------------------------------

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        cumsum = np.cumsum(data)
        cumsum[period:] = cumsum[period:] - cumsum[:-period]
        result[period - 1:] = cumsum[period - 1:] / period
        return result

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average using precise recursive calculation."""
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result

        multiplier = 2.0 / (period + 1)
        # Seed with SMA
        result[period - 1] = np.mean(data[:period])

        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]

        return result

    @staticmethod
    def _rsi(close: np.ndarray, period: int) -> np.ndarray:
        """
        Relative Strength Index using Wilder's smoothing method.
        Returns values 0-100.
        """
        result = np.full_like(close, np.nan, dtype=np.float64)
        if len(close) < period + 1:
            return result

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # First average: simple mean
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))

        # Subsequent values: Wilder's smoothing
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                result[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return result

    def _macd(
        self, close: np.ndarray,
        fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD Line, Signal Line, and Histogram."""
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd_line = ema_fast - ema_slow

        # Signal line is EMA of MACD line (only compute where MACD is valid)
        valid_start = slow - 1
        macd_valid = macd_line[valid_start:]
        signal_line_partial = self._ema(macd_valid, signal)

        signal_line = np.full_like(close, np.nan, dtype=np.float64)
        signal_line[valid_start:] = signal_line_partial

        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def _bollinger_bands(
        close: np.ndarray, period: int = 20, num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands: upper, middle (SMA), lower."""
        middle = np.full_like(close, np.nan, dtype=np.float64)
        upper = np.full_like(close, np.nan, dtype=np.float64)
        lower = np.full_like(close, np.nan, dtype=np.float64)

        if len(close) < period:
            return upper, middle, lower

        for i in range(period - 1, len(close)):
            window = close[i - period + 1: i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=0)
            middle[i] = mean
            upper[i] = mean + num_std * std
            lower[i] = mean - num_std * std

        return upper, middle, lower

    @staticmethod
    def _atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Average True Range using Wilder's smoothing."""
        result = np.full_like(close, np.nan, dtype=np.float64)
        if len(close) < period + 1:
            return result

        # True range
        tr = np.zeros(len(close), dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Initial ATR: simple average
        result[period] = np.mean(tr[1: period + 1])

        # Wilder's smoothing
        for i in range(period + 1, len(close)):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

        return result

    @staticmethod
    def _adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """
        Average Directional Index.
        Measures trend strength regardless of direction. 0-100 scale.
        """
        n = len(close)
        result = np.full(n, np.nan, dtype=np.float64)
        if n < 2 * period + 1:
            return result

        # True Range
        tr = np.zeros(n, dtype=np.float64)
        plus_dm = np.zeros(n, dtype=np.float64)
        minus_dm = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smoothed TR, +DM, -DM using Wilder's method
        atr_s = np.mean(tr[1: period + 1])
        plus_dm_s = np.mean(plus_dm[1: period + 1])
        minus_dm_s = np.mean(minus_dm[1: period + 1])

        plus_di = np.zeros(n, dtype=np.float64)
        minus_di = np.zeros(n, dtype=np.float64)
        dx = np.zeros(n, dtype=np.float64)

        if atr_s > 0:
            plus_di[period] = (plus_dm_s / atr_s) * 100
            minus_di[period] = (minus_dm_s / atr_s) * 100

        di_sum = plus_di[period] + minus_di[period]
        if di_sum > 0:
            dx[period] = abs(plus_di[period] - minus_di[period]) / di_sum * 100

        for i in range(period + 1, n):
            atr_s = (atr_s * (period - 1) + tr[i]) / period
            plus_dm_s = (plus_dm_s * (period - 1) + plus_dm[i]) / period
            minus_dm_s = (minus_dm_s * (period - 1) + minus_dm[i]) / period

            if atr_s > 0:
                plus_di[i] = (plus_dm_s / atr_s) * 100
                minus_di[i] = (minus_dm_s / atr_s) * 100

            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100

        # ADX is smoothed DX
        adx_start = 2 * period
        if adx_start < n:
            result[adx_start] = np.mean(dx[period: adx_start + 1])
            for i in range(adx_start + 1, n):
                result[i] = (result[i - 1] * (period - 1) + dx[i]) / period

        return result

    @staticmethod
    def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On-Balance Volume - cumulative volume flow indicator."""
        obv = np.zeros(len(close), dtype=np.float64)
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        return obv

    @staticmethod
    def _vwap(
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        """
        Volume Weighted Average Price.
        Uses rolling cumulative calculation (resets conceptually per session,
        but for continuous futures we use a rolling 20-bar VWAP).
        """
        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume

        # Rolling 20-bar VWAP
        period = 20
        result = np.full_like(close, np.nan, dtype=np.float64)
        if len(close) < period:
            return result

        for i in range(period - 1, len(close)):
            window_tpv = tp_vol[i - period + 1: i + 1]
            window_vol = volume[i - period + 1: i + 1]
            vol_sum = np.sum(window_vol)
            if vol_sum > 0:
                result[i] = np.sum(window_tpv) / vol_sum
            else:
                result[i] = close[i]

        return result

    # ----------------------------------------------------------------
    # Smart Money Concept detections
    # ----------------------------------------------------------------

    def _order_block_score(
        self,
        open_: np.ndarray, high: np.ndarray,
        low: np.ndarray, close: np.ndarray,
        volume: np.ndarray, lookback: int = 20
    ) -> np.ndarray:
        """
        Detect order blocks (institutional supply/demand zones).

        An order block is the last opposing candle before a strong
        impulsive move. Bullish OB: last bearish candle before a
        strong bullish impulse. Bearish OB: last bullish candle
        before a strong bearish impulse.

        Returns a score from -1 to +1:
        - Positive: price is near a bullish order block (demand zone)
        - Negative: price is near a bearish order block (supply zone)
        - Zero: no nearby order block
        """
        n = len(close)
        scores = np.zeros(n, dtype=np.float64)
        if n < lookback + 3:
            return scores

        # Average candle body size for impulse detection
        body_sizes = np.abs(close - open_)
        avg_volume = self._sma(volume, lookback)

        for i in range(lookback + 2, n):
            best_score = 0.0

            # Scan recent candles for order block patterns
            for j in range(max(lookback, i - lookback), i - 1):
                is_bearish_candle = close[j] < open_[j]
                is_bullish_candle = close[j] > open_[j]

                # Check if followed by strong impulse move (2+ candles)
                move_after = close[i] - close[j]
                avg_body = np.mean(body_sizes[max(0, j - lookback): j]) if j > lookback else body_sizes[j]

                if avg_body == 0:
                    continue

                impulse_strength = abs(move_after) / (avg_body * 3)
                impulse_strength = min(impulse_strength, 1.0)

                # Bullish OB: bearish candle followed by strong up-move
                if is_bearish_candle and move_after > 0:
                    # Check if price is near the OB zone
                    ob_high = open_[j]
                    ob_low = close[j]
                    zone_size = ob_high - ob_low
                    if zone_size > 0:
                        dist = close[i] - ob_low
                        proximity = 1.0 - min(abs(dist) / (zone_size * 3), 1.0)

                        vol_factor = 1.0
                        if avg_volume[j] is not None and avg_volume[j] > 0 and not np.isnan(avg_volume[j]):
                            vol_factor = min(volume[j] / avg_volume[j], 2.0) / 2.0

                        score = proximity * impulse_strength * vol_factor
                        if score > best_score:
                            best_score = score

                # Bearish OB: bullish candle followed by strong down-move
                elif is_bullish_candle and move_after < 0:
                    ob_high = close[j]
                    ob_low = open_[j]
                    zone_size = ob_high - ob_low
                    if zone_size > 0:
                        dist = ob_high - close[i]
                        proximity = 1.0 - min(abs(dist) / (zone_size * 3), 1.0)

                        vol_factor = 1.0
                        if avg_volume[j] is not None and avg_volume[j] > 0 and not np.isnan(avg_volume[j]):
                            vol_factor = min(volume[j] / avg_volume[j], 2.0) / 2.0

                        score = -(proximity * impulse_strength * vol_factor)
                        if abs(score) > abs(best_score):
                            best_score = score

            scores[i] = np.clip(best_score, -1.0, 1.0)

        return scores

    @staticmethod
    def _fvg_score(
        open_: np.ndarray, high: np.ndarray,
        low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Detect Fair Value Gaps (imbalance zones).

        A bullish FVG occurs when candle 3's low is above candle 1's high,
        creating a gap that price may revisit. Bearish FVG is the opposite.

        Returns score -1 to +1:
        - Positive: price is near/in a bullish FVG (expect fill upward)
        - Negative: price is near/in a bearish FVG (expect fill downward)
        """
        n = len(close)
        scores = np.zeros(n, dtype=np.float64)
        if n < 5:
            return scores

        # Store detected FVGs with decay
        active_fvgs: List[Dict] = []

        for i in range(2, n):
            # Detect new FVGs at candle i (using candles i-2, i-1, i)
            # Bullish FVG: candle i low > candle i-2 high
            if low[i] > high[i - 2]:
                gap_size = low[i] - high[i - 2]
                mid_body = abs(close[i - 1] - open_[i - 1])
                if mid_body > 0:
                    strength = min(gap_size / mid_body, 1.0)
                    active_fvgs.append({
                        "type": "bullish",
                        "high": low[i],
                        "low": high[i - 2],
                        "strength": strength,
                        "birth": i,
                    })

            # Bearish FVG: candle i high < candle i-2 low
            if high[i] < low[i - 2]:
                gap_size = low[i - 2] - high[i]
                mid_body = abs(close[i - 1] - open_[i - 1])
                if mid_body > 0:
                    strength = min(gap_size / mid_body, 1.0)
                    active_fvgs.append({
                        "type": "bearish",
                        "high": low[i - 2],
                        "low": high[i],
                        "strength": strength,
                        "birth": i,
                    })

            # Score current price against active FVGs
            score = 0.0
            surviving_fvgs = []
            for fvg in active_fvgs:
                age = i - fvg["birth"]
                if age > 50:
                    continue  # FVG too old, discard

                decay = max(0.0, 1.0 - age / 50.0)

                # Check if price is in or near the FVG zone
                fvg_mid = (fvg["high"] + fvg["low"]) / 2
                fvg_range = fvg["high"] - fvg["low"]
                if fvg_range <= 0:
                    continue

                dist = abs(close[i] - fvg_mid)
                proximity = max(0.0, 1.0 - dist / (fvg_range * 2))

                if proximity > 0:
                    contribution = proximity * fvg["strength"] * decay
                    if fvg["type"] == "bullish":
                        score += contribution
                    else:
                        score -= contribution

                # Keep if not filled
                if fvg["type"] == "bullish" and close[i] > fvg["low"]:
                    surviving_fvgs.append(fvg)
                elif fvg["type"] == "bearish" and close[i] < fvg["high"]:
                    surviving_fvgs.append(fvg)

            active_fvgs = surviving_fvgs[-20:]  # Keep max 20 active FVGs
            scores[i] = np.clip(score, -1.0, 1.0)

        return scores

    def detect_order_blocks(
        self, open_: np.ndarray, high: np.ndarray,
        low: np.ndarray, close: np.ndarray,
        volume: np.ndarray, lookback: int = 50
    ) -> List[OrderBlock]:
        """
        Return list of detected order blocks for visualization/analysis.
        """
        n = len(close)
        blocks = []
        if n < lookback + 3:
            return blocks

        body_sizes = np.abs(close - open_)

        for i in range(lookback, n - 2):
            avg_body = np.mean(body_sizes[i - lookback: i])
            if avg_body == 0:
                continue

            # Check for impulse move after candle i
            impulse = close[i + 2] - close[i]
            if abs(impulse) < avg_body * 2:
                continue

            is_bearish = close[i] < open_[i]
            is_bullish = close[i] > open_[i]

            wick_ratio = 0.0
            body = abs(close[i] - open_[i])
            full_range = high[i] - low[i]
            if full_range > 0:
                wick_ratio = 1.0 - (body / full_range)

            vol_strength = volume[i] / np.mean(volume[max(0, i - lookback): i]) if np.mean(volume[max(0, i - lookback): i]) > 0 else 1.0

            strength = min((abs(impulse) / (avg_body * 3)) * min(vol_strength, 2.0), 1.0)

            if is_bearish and impulse > 0:
                blocks.append(OrderBlock(
                    price_high=open_[i],
                    price_low=close[i],
                    block_type="bullish",
                    strength=strength,
                    candle_index=i,
                ))
            elif is_bullish and impulse < 0:
                blocks.append(OrderBlock(
                    price_high=close[i],
                    price_low=open_[i],
                    block_type="bearish",
                    strength=strength,
                    candle_index=i,
                ))

        return blocks

    def detect_fvgs(
        self, open_: np.ndarray, high: np.ndarray,
        low: np.ndarray, close: np.ndarray
    ) -> List[FairValueGap]:
        """
        Return list of detected fair value gaps for visualization/analysis.
        """
        n = len(close)
        fvgs = []
        if n < 3:
            return fvgs

        for i in range(2, n):
            mid_body = abs(close[i - 1] - open_[i - 1])

            # Bullish FVG
            if low[i] > high[i - 2]:
                gap = low[i] - high[i - 2]
                size_pct = (gap / close[i - 1] * 100) if close[i - 1] != 0 else 0
                fvgs.append(FairValueGap(
                    high=low[i],
                    low=high[i - 2],
                    gap_type="bullish",
                    size_pct=size_pct,
                    candle_index=i,
                ))

            # Bearish FVG
            if high[i] < low[i - 2]:
                gap = low[i - 2] - high[i]
                size_pct = (gap / close[i - 1] * 100) if close[i - 1] != 0 else 0
                fvgs.append(FairValueGap(
                    high=low[i - 2],
                    low=high[i],
                    gap_type="bearish",
                    size_pct=size_pct,
                    candle_index=i,
                ))

        return fvgs
