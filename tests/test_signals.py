"""
Tests for the signal composition layers.

Tests technical analysis, correlation, and sentiment scoring
using synthetic data with known properties.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals.technical import TechnicalAnalyzer
from src.signals.correlation import CorrelationAnalyzer
from src.signals.sentiment import SentimentAnalyzer


@pytest.fixture
def uptrend_df():
    """Create a clear uptrend OHLCV dataset."""
    np.random.seed(42)
    n = 300
    base = 100.0
    trend = np.linspace(0, 30, n)
    noise = np.random.randn(n) * 0.3
    close = base + trend + noise

    return pd.DataFrame({
        "open": close - np.random.rand(n) * 0.2,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000 + 5000),
    })


@pytest.fixture
def downtrend_df():
    """Create a clear downtrend OHLCV dataset."""
    np.random.seed(42)
    n = 300
    base = 130.0
    trend = np.linspace(0, -30, n)
    noise = np.random.randn(n) * 0.3
    close = base + trend + noise

    return pd.DataFrame({
        "open": close + np.random.rand(n) * 0.2,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000 + 5000),
    })


@pytest.fixture
def sideways_df():
    """Create a sideways/ranging OHLCV dataset."""
    np.random.seed(42)
    n = 300
    close = 100.0 + np.random.randn(n) * 1.0

    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000 + 5000),
    })


class TestTechnicalAnalyzer:
    def test_uptrend_bullish_score(self, uptrend_df):
        analyzer = TechnicalAnalyzer()
        result = analyzer.score(uptrend_df)

        assert "score" in result
        assert "dimensions" in result
        assert result["score"] > 0, "Uptrend should produce positive score"

    def test_downtrend_bearish_score(self, downtrend_df):
        analyzer = TechnicalAnalyzer()
        result = analyzer.score(downtrend_df)

        assert result["score"] < 0, "Downtrend should produce negative score"

    def test_score_range(self, uptrend_df):
        analyzer = TechnicalAnalyzer()
        result = analyzer.score(uptrend_df)

        assert -100 <= result["score"] <= 100

    def test_dimensions_present(self, uptrend_df):
        analyzer = TechnicalAnalyzer()
        result = analyzer.score(uptrend_df)

        assert "trend" in result["dimensions"]
        assert "momentum" in result["dimensions"]
        assert "volatility" in result["dimensions"]
        assert "volume" in result["dimensions"]

        for dim in result["dimensions"].values():
            assert "score" in dim
            assert "details" in dim

    def test_key_levels(self, uptrend_df):
        analyzer = TechnicalAnalyzer()
        result = analyzer.score(uptrend_df)

        levels = result["key_levels"]
        assert "support" in levels
        assert "resistance" in levels
        assert "ema_9" in levels
        assert "bb_upper" in levels

    def test_support_below_price(self, uptrend_df):
        analyzer = TechnicalAnalyzer()
        result = analyzer.score(uptrend_df)

        support = result["key_levels"]["support"]
        if support is not None:
            assert support < result["current_price"]

    def test_resistance_above_price(self, uptrend_df):
        analyzer = TechnicalAnalyzer()
        result = analyzer.score(uptrend_df)

        resistance = result["key_levels"]["resistance"]
        if resistance is not None:
            assert resistance > result["current_price"]


class TestCorrelationAnalyzer:
    def test_self_correlation(self):
        analyzer = CorrelationAnalyzer()
        close = np.linspace(100, 150, 200)
        analyzer.update_prices("BTCUSDT", close)

        result = analyzer.score("BTCUSDT", pd.DataFrame({"close": close}))
        assert result["btc_correlation"]["note"] == "self_reference"

    def test_high_correlation_detection(self):
        analyzer = CorrelationAnalyzer()
        np.random.seed(42)

        btc = np.linspace(50000, 60000, 200) + np.random.randn(200) * 100
        eth = btc * 0.06 + np.random.randn(200) * 50  # Highly correlated

        analyzer.update_prices("BTCUSDT", btc)
        analyzer.update_prices("ETHUSDT", eth)

        df = pd.DataFrame({"close": eth})
        result = analyzer.score("ETHUSDT", df)

        corr = result["btc_correlation"]["long_term_corr"]
        assert corr > 0.8, f"Expected high correlation, got {corr}"

    def test_correlation_matrix(self):
        analyzer = CorrelationAnalyzer()
        np.random.seed(42)

        btc = np.linspace(50000, 55000, 100) + np.random.randn(100) * 100
        eth = btc * 0.06 + np.random.randn(100) * 20
        sol = btc * 0.002 + np.random.randn(100) * 5

        analyzer.update_prices("BTCUSDT", btc)
        analyzer.update_prices("ETHUSDT", eth)
        analyzer.update_prices("SOLUSDT", sol)

        matrix = analyzer.compute_correlation_matrix()
        assert len(matrix["symbols"]) == 3
        assert len(matrix["matrix"]) == 3
        # Diagonal should be 1.0
        for i in range(3):
            assert matrix["matrix"][i][i] == pytest.approx(1.0)

    def test_beta_calculation(self):
        analyzer = CorrelationAnalyzer()
        np.random.seed(42)

        btc = np.linspace(50000, 55000, 100) + np.random.randn(100) * 100
        # High beta asset: moves 2x relative to BTC
        high_beta = 100 + (btc - 50000) / 50000 * 200 * 2 + np.random.randn(100) * 2

        analyzer.update_prices("BTCUSDT", btc)

        _, beta_details = analyzer._calculate_beta("HIGHBETA", high_beta)
        assert beta_details["beta"] > 1.0, "High-beta asset should have beta > 1"


class TestSentimentAnalyzer:
    def test_keyword_scoring_bullish(self):
        bull_score, bear_score, matched = SentimentAnalyzer._score_text(
            "Bitcoin rally surges to all-time high as institutional adoption grows"
        )
        assert bull_score > bear_score
        assert len(matched) > 0

    def test_keyword_scoring_bearish(self):
        bull_score, bear_score, matched = SentimentAnalyzer._score_text(
            "Crypto crash liquidation fears as SEC crackdown continues"
        )
        assert bear_score > bull_score

    def test_keyword_scoring_neutral(self):
        bull_score, bear_score, matched = SentimentAnalyzer._score_text(
            "The weather today is sunny with clear skies"
        )
        assert bull_score == 0
        assert bear_score == 0

    def test_negation_detection(self):
        bull_score, bear_score, matched = SentimentAnalyzer._score_text(
            "Analysts say market is not bullish despite recent positive signals"
        )
        # Negation should reduce bullish impact
        assert "negation_detected" in matched

    def test_score_without_feeds(self):
        analyzer = SentimentAnalyzer(feeds=[])
        result = analyzer.score()
        assert result["score"] == 0.0
        assert "no_articles_cached" in result.get("note", "")

    def test_relevance_computation(self):
        from src.signals.sentiment import ArticleSentiment
        article = ArticleSentiment(
            title="Bitcoin price breaks through resistance",
            source="TestSource",
            published=None,
            bullish_score=5.0,
            bearish_score=1.0,
            net_score=4.0,
            matched_keywords=[],
            relevance=1.0,
        )

        relevance = SentimentAnalyzer._compute_relevance(article, ["bitcoin", "btc"])
        assert relevance > 0

        relevance_unrelated = SentimentAnalyzer._compute_relevance(article, ["solana", "sol"])
        assert relevance_unrelated < relevance
