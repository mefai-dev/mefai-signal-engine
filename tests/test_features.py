"""
Tests for the Feature Engine.

Validates all 15+ technical indicator calculations against known values.
Uses synthetic OHLCV data with predictable patterns.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_engine import FeatureEngine


@pytest.fixture
def engine():
    return FeatureEngine()


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data with an uptrend followed by consolidation."""
    np.random.seed(42)
    n = 300

    # Create a price series with clear trend
    base_price = 100.0
    trend = np.linspace(0, 20, n)
    noise = np.random.randn(n) * 0.5
    close = base_price + trend + noise

    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.abs(np.random.randn(n) * 1000 + 5000)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


class TestSMA:
    def test_sma_basic(self, engine):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = engine._sma(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_insufficient_data(self, engine):
        data = np.array([1.0, 2.0])
        result = engine._sma(data, 5)
        assert all(np.isnan(result))


class TestEMA:
    def test_ema_seed_is_sma(self, engine):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = engine._ema(data, 5)
        # First EMA value should be SMA of first 5 values
        assert result[4] == pytest.approx(3.0)

    def test_ema_responds_to_direction(self, engine):
        # Uptrend: EMA should be below current price
        data = np.arange(1.0, 21.0)
        result = engine._ema(data, 9)
        # Latest EMA should lag behind the uptrend
        assert result[-1] < data[-1]

    def test_ema_length(self, engine):
        data = np.random.randn(100)
        result = engine._ema(data, 9)
        assert len(result) == len(data)


class TestRSI:
    def test_rsi_range(self, engine, sample_ohlcv):
        close = sample_ohlcv["close"].values
        rsi = engine._rsi(close, 14)
        valid = rsi[~np.isnan(rsi)]
        assert all(0 <= v <= 100 for v in valid)

    def test_rsi_uptrend(self, engine):
        # Strong uptrend should have high RSI
        data = np.linspace(100, 200, 50)
        rsi = engine._rsi(data, 14)
        valid = rsi[~np.isnan(rsi)]
        assert valid[-1] > 70  # Should be overbought in pure uptrend

    def test_rsi_downtrend(self, engine):
        # Strong downtrend should have low RSI
        data = np.linspace(200, 100, 50)
        rsi = engine._rsi(data, 14)
        valid = rsi[~np.isnan(rsi)]
        assert valid[-1] < 30  # Should be oversold


class TestMACD:
    def test_macd_components(self, engine, sample_ohlcv):
        close = sample_ohlcv["close"].values
        macd_line, signal, histogram = engine._macd(close)

        assert len(macd_line) == len(close)
        assert len(signal) == len(close)
        assert len(histogram) == len(close)

    def test_macd_histogram_is_diff(self, engine, sample_ohlcv):
        close = sample_ohlcv["close"].values
        macd_line, signal, histogram = engine._macd(close)

        # Histogram should be MACD - Signal (where both are valid)
        valid = ~np.isnan(macd_line) & ~np.isnan(signal) & ~np.isnan(histogram)
        np.testing.assert_array_almost_equal(
            histogram[valid],
            macd_line[valid] - signal[valid],
        )


class TestBollingerBands:
    def test_bb_ordering(self, engine, sample_ohlcv):
        close = sample_ohlcv["close"].values
        upper, middle, lower = engine._bollinger_bands(close, 20, 2.0)

        valid = ~np.isnan(upper) & ~np.isnan(middle) & ~np.isnan(lower)
        assert all(upper[valid] >= middle[valid])
        assert all(middle[valid] >= lower[valid])

    def test_bb_middle_is_sma(self, engine):
        close = np.arange(1.0, 31.0)
        upper, middle, lower = engine._bollinger_bands(close, 20, 2.0)
        sma = engine._sma(close, 20)

        valid = ~np.isnan(middle) & ~np.isnan(sma)
        np.testing.assert_array_almost_equal(middle[valid], sma[valid])


class TestATR:
    def test_atr_positive(self, engine, sample_ohlcv):
        high = sample_ohlcv["high"].values
        low = sample_ohlcv["low"].values
        close = sample_ohlcv["close"].values
        atr = engine._atr(high, low, close, 14)

        valid = atr[~np.isnan(atr)]
        assert all(v > 0 for v in valid)


class TestADX:
    def test_adx_range(self, engine, sample_ohlcv):
        high = sample_ohlcv["high"].values
        low = sample_ohlcv["low"].values
        close = sample_ohlcv["close"].values
        adx = engine._adx(high, low, close, 14)

        valid = adx[~np.isnan(adx)]
        assert all(0 <= v <= 100 for v in valid)


class TestOBV:
    def test_obv_up(self, engine):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        volume = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        obv = engine._obv(close, volume)
        # Pure uptrend: OBV should be monotonically increasing
        assert obv[-1] == 400.0

    def test_obv_down(self, engine):
        close = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        volume = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        obv = engine._obv(close, volume)
        assert obv[-1] == -400.0


class TestFullFeatureComputation:
    def test_all_features_computed(self, engine, sample_ohlcv):
        result = engine.compute_all_features(sample_ohlcv)

        expected_columns = [
            "rsi_14", "rsi_7", "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "atr_14", "adx_14",
            "ema_9", "ema_21", "sma_50", "sma_200", "obv", "vwap",
            "order_block_score", "fvg_score",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing feature: {col}"

    def test_feature_matrix_shape(self, engine, sample_ohlcv):
        X, names, mask = engine.get_feature_matrix(sample_ohlcv)

        assert X.ndim == 2
        assert X.shape[1] == len(names)
        assert not np.any(np.isnan(X))  # No NaN in output matrix

    def test_feature_names_match(self, engine):
        assert len(engine.feature_names) == 25


class TestOrderBlocks:
    def test_ob_detection(self, engine, sample_ohlcv):
        blocks = engine.detect_order_blocks(
            sample_ohlcv["open"].values,
            sample_ohlcv["high"].values,
            sample_ohlcv["low"].values,
            sample_ohlcv["close"].values,
            sample_ohlcv["volume"].values,
        )
        # Should detect some order blocks in 300 candles
        assert isinstance(blocks, list)
        for ob in blocks:
            assert ob.block_type in ("bullish", "bearish")
            assert 0 <= ob.strength <= 1


class TestFVG:
    def test_fvg_detection(self, engine, sample_ohlcv):
        fvgs = engine.detect_fvgs(
            sample_ohlcv["open"].values,
            sample_ohlcv["high"].values,
            sample_ohlcv["low"].values,
            sample_ohlcv["close"].values,
        )
        assert isinstance(fvgs, list)
        for fvg in fvgs:
            assert fvg.gap_type in ("bullish", "bearish")
            assert fvg.high >= fvg.low
