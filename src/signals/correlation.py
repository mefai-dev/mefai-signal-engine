"""
Correlation Analysis Layer.

Evaluates cross-asset relationships to detect:
- BTC correlation: how closely an asset tracks Bitcoin
- Sector correlation: intra-sector divergence/convergence
- Cross-pair divergence: when correlated assets decouple (mean reversion signal)
- Beta calculation: asset volatility relative to market

Score range: -100 to +100
Positive = bullish divergence or favorable correlation shift
Negative = bearish divergence or unfavorable correlation shift
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Cross-asset correlation scoring engine.

    Maintains a rolling correlation matrix across tracked symbols
    and scores each asset based on its correlation dynamics.
    """

    def __init__(self, lookback: int = 100, short_lookback: int = 20):
        self.lookback = lookback
        self.short_lookback = short_lookback
        self._price_cache: Dict[str, np.ndarray] = {}

    def update_prices(self, symbol: str, closes: np.ndarray):
        """Update the price cache for a symbol."""
        self._price_cache[symbol] = closes

    def score(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Generate correlation score for a symbol.

        Args:
            symbol: The symbol being scored.
            df: OHLCV data for this symbol.

        Returns:
            Dict with composite correlation score and breakdown.
        """
        close = df["close"].values.astype(np.float64)
        self.update_prices(symbol, close)

        btc_score, btc_details = self._score_btc_correlation(symbol, close)
        sector_score, sector_details = self._score_sector_correlation(symbol)
        divergence_score, div_details = self._score_divergence(symbol, close)
        beta_val, beta_details = self._calculate_beta(symbol, close)

        # Weight the components
        composite = (
            btc_score * 0.35
            + sector_score * 0.25
            + divergence_score * 0.30
            + (beta_val - 1.0) * 10  # Beta deviation from 1.0 as signal
        )

        return {
            "score": float(np.clip(composite, -100, 100)),
            "btc_correlation": btc_details,
            "sector_correlation": sector_details,
            "divergence": div_details,
            "beta": beta_details,
        }

    def _score_btc_correlation(
        self, symbol: str, close: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Score based on BTC correlation dynamics.

        If an alt is normally highly correlated with BTC but suddenly
        diverges upward, that is a bullish signal (relative strength).
        If it diverges downward, bearish.
        """
        details = {}

        if symbol == "BTCUSDT":
            return 0.0, {"note": "self_reference"}

        btc_close = self._price_cache.get("BTCUSDT")
        if btc_close is None or len(btc_close) < self.lookback:
            return 0.0, {"note": "btc_data_unavailable"}

        # Align lengths
        min_len = min(len(close), len(btc_close))
        c = close[-min_len:]
        b = btc_close[-min_len:]

        if min_len < self.short_lookback + 5:
            return 0.0, {"note": "insufficient_data"}

        # Returns
        c_returns = np.diff(c) / c[:-1]
        b_returns = np.diff(b) / b[:-1]

        # Long-term correlation
        long_corr = self._pearson_correlation(
            c_returns[-self.lookback:] if len(c_returns) >= self.lookback else c_returns,
            b_returns[-self.lookback:] if len(b_returns) >= self.lookback else b_returns,
        )

        # Short-term correlation
        short_corr = self._pearson_correlation(
            c_returns[-self.short_lookback:],
            b_returns[-self.short_lookback:],
        )

        details["long_term_corr"] = float(long_corr)
        details["short_term_corr"] = float(short_corr)

        # Correlation shift (short vs long)
        corr_shift = short_corr - long_corr
        details["correlation_shift"] = float(corr_shift)

        # Relative performance (asset return vs BTC return over short period)
        asset_return = (c[-1] - c[-self.short_lookback]) / c[-self.short_lookback] * 100
        btc_return = (b[-1] - b[-self.short_lookback]) / b[-self.short_lookback] * 100
        relative_perf = asset_return - btc_return

        details["asset_return_pct"] = float(asset_return)
        details["btc_return_pct"] = float(btc_return)
        details["relative_performance"] = float(relative_perf)

        # Score: positive relative performance = bullish
        # De-correlation from BTC during BTC downtrend = bullish resilience
        score = 0.0

        # Relative strength/weakness
        score += np.clip(relative_perf * 5, -40, 40)

        # Correlation regime shift
        if corr_shift < -0.3:
            # De-correlating - potentially independent move
            if relative_perf > 0:
                score += 20  # Bullish independence
            else:
                score -= 20  # Bearish independence
        elif corr_shift > 0.3:
            # Re-correlating - following BTC
            if btc_return > 0:
                score += 10
            else:
                score -= 10

        details["signal"] = (
            "relative_strength" if score > 20
            else "relative_weakness" if score < -20
            else "neutral"
        )

        return float(np.clip(score, -100, 100)), details

    def _score_sector_correlation(self, symbol: str) -> Tuple[float, Dict]:
        """
        Score based on sector peer correlation.

        Groups symbols by sector and checks if this asset is
        leading or lagging its peers.
        """
        sectors = {
            "majors": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "l1_alts": ["SOLUSDT", "AVAXUSDT", "DOTUSDT", "ADAUSDT"],
            "defi": ["LINKUSDT", "MATICUSDT"],
            "memes": ["DOGEUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "1000FLOKIUSDT", "1000BONKUSDT"],
        }

        # Find which sector this symbol belongs to
        symbol_sector = None
        peers = []
        for sector_name, sector_symbols in sectors.items():
            if symbol in sector_symbols:
                symbol_sector = sector_name
                peers = [s for s in sector_symbols if s != symbol]
                break

        if not symbol_sector or not peers:
            return 0.0, {"note": "no_sector_peers"}

        details = {"sector": symbol_sector, "peers": peers}

        # Check how many peers have data
        peer_returns = []
        symbol_close = self._price_cache.get(symbol)
        if symbol_close is None or len(symbol_close) < self.short_lookback:
            return 0.0, {"note": "insufficient_symbol_data"}

        symbol_return = (
            (symbol_close[-1] - symbol_close[-self.short_lookback])
            / symbol_close[-self.short_lookback] * 100
        )

        for peer in peers:
            peer_close = self._price_cache.get(peer)
            if peer_close is not None and len(peer_close) >= self.short_lookback:
                ret = (peer_close[-1] - peer_close[-self.short_lookback]) / peer_close[-self.short_lookback] * 100
                peer_returns.append(ret)

        if not peer_returns:
            return 0.0, {"note": "no_peer_data"}

        avg_peer_return = np.mean(peer_returns)
        relative_to_sector = symbol_return - avg_peer_return

        details["symbol_return_pct"] = float(symbol_return)
        details["sector_avg_return_pct"] = float(avg_peer_return)
        details["relative_to_sector"] = float(relative_to_sector)

        # Score: outperforming sector = bullish, underperforming = bearish
        score = np.clip(relative_to_sector * 8, -100, 100)

        details["signal"] = (
            "sector_leader" if score > 20
            else "sector_laggard" if score < -20
            else "inline_with_sector"
        )

        return float(score), details

    def _score_divergence(
        self, symbol: str, close: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Detect divergence from correlated assets.

        When normally correlated assets diverge, mean reversion is likely.
        A bullish divergence: asset underperforms its expected move based
        on correlation (potential catch-up rally).
        """
        details = {}

        btc_close = self._price_cache.get("BTCUSDT")
        if btc_close is None or symbol == "BTCUSDT":
            return 0.0, {"note": "no_divergence_reference"}

        min_len = min(len(close), len(btc_close))
        if min_len < self.lookback:
            return 0.0, {"note": "insufficient_data"}

        c = close[-min_len:]
        b = btc_close[-min_len:]

        # Calculate rolling z-score of the spread
        c_norm = c / c[0]
        b_norm = b / b[0]

        spread = c_norm - b_norm
        if len(spread) < self.lookback:
            return 0.0, {"note": "spread_too_short"}

        spread_mean = np.mean(spread[-self.lookback:])
        spread_std = np.std(spread[-self.lookback:])

        if spread_std == 0:
            return 0.0, {"note": "zero_spread_variance"}

        z_score = (spread[-1] - spread_mean) / spread_std

        details["z_score"] = float(z_score)
        details["spread_mean"] = float(spread_mean)
        details["spread_std"] = float(spread_std)

        # Mean reversion signal:
        # z_score > 2: asset overperformed, expect reversion down
        # z_score < -2: asset underperformed, expect reversion up
        if z_score > 2:
            score = -min(z_score * 20, 80)  # Bearish (overextended)
            details["divergence_type"] = "bearish_overextension"
        elif z_score < -2:
            score = min(abs(z_score) * 20, 80)  # Bullish (undervalued)
            details["divergence_type"] = "bullish_undervaluation"
        elif z_score > 1:
            score = -z_score * 10
            details["divergence_type"] = "mild_bearish"
        elif z_score < -1:
            score = abs(z_score) * 10
            details["divergence_type"] = "mild_bullish"
        else:
            score = 0.0
            details["divergence_type"] = "no_divergence"

        return float(np.clip(score, -100, 100)), details

    def _calculate_beta(
        self, symbol: str, close: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Calculate beta (volatility relative to BTC).

        Beta > 1: more volatile than BTC (amplified moves)
        Beta < 1: less volatile than BTC (dampened moves)
        Beta < 0: inverse relationship (rare in crypto)
        """
        details = {}

        if symbol == "BTCUSDT":
            return 1.0, {"beta": 1.0, "note": "btc_self_beta"}

        btc_close = self._price_cache.get("BTCUSDT")
        if btc_close is None:
            return 1.0, {"beta": 1.0, "note": "btc_unavailable"}

        min_len = min(len(close), len(btc_close))
        if min_len < 30:
            return 1.0, {"beta": 1.0, "note": "insufficient_data"}

        c = close[-min_len:]
        b = btc_close[-min_len:]

        c_returns = np.diff(c) / c[:-1]
        b_returns = np.diff(b) / b[:-1]

        # Beta = Cov(asset, market) / Var(market)
        covariance = np.cov(c_returns, b_returns)[0, 1]
        variance = np.var(b_returns)

        if variance == 0:
            return 1.0, {"beta": 1.0, "note": "zero_market_variance"}

        beta = covariance / variance

        details["beta"] = float(beta)
        details["interpretation"] = (
            "high_beta" if beta > 1.5
            else "moderate_beta" if beta > 1.0
            else "low_beta" if beta > 0
            else "inverse_beta"
        )

        return float(beta), details

    @staticmethod
    def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        min_len = min(len(x), len(y))
        if min_len < 5:
            return 0.0

        x = x[-min_len:]
        y = y[-min_len:]

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x_diff = x - x_mean
        y_diff = y - y_mean

        numerator = np.sum(x_diff * y_diff)
        denominator = np.sqrt(np.sum(x_diff ** 2) * np.sum(y_diff ** 2))

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    def compute_correlation_matrix(self) -> Dict:
        """
        Compute full correlation matrix across all cached symbols.

        Returns:
            Dict with matrix data and symbol labels.
        """
        symbols = sorted(self._price_cache.keys())
        n = len(symbols)
        if n < 2:
            return {"symbols": symbols, "matrix": [], "note": "need_at_least_2_symbols"}

        # Compute returns for all symbols
        returns = {}
        min_len = float("inf")
        for sym in symbols:
            closes = self._price_cache[sym]
            if len(closes) >= 2:
                rets = np.diff(closes) / closes[:-1]
                returns[sym] = rets
                min_len = min(min_len, len(rets))

        if min_len < 10:
            return {"symbols": symbols, "matrix": [], "note": "insufficient_data"}

        # Build correlation matrix
        matrix = np.zeros((n, n))
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    ri = returns.get(sym_i, np.array([]))
                    rj = returns.get(sym_j, np.array([]))
                    matrix[i, j] = self._pearson_correlation(
                        ri[-int(min_len):], rj[-int(min_len):]
                    )

        return {
            "symbols": symbols,
            "matrix": matrix.tolist(),
        }
