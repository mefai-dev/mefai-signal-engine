"""
On-Chain Metrics Layer.

Evaluates Binance Futures market microstructure:
- Funding rate: positive = longs pay shorts (crowded long), negative = shorts pay longs
- Open interest change: rising OI + rising price = new money entering trend
- Long/short ratio: top trader positioning
- Liquidation levels: estimated price levels where cascading liquidations occur

Score range: -100 to +100
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import aiohttp

logger = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"


class OnChainAnalyzer:
    """
    Binance Futures on-chain/market microstructure analyzer.

    Fetches live data from Binance Futures API and scores
    market positioning and sentiment.
    """

    def __init__(self, base_url: str = BINANCE_FAPI):
        self.base_url = base_url
        self._cache: Dict[str, Dict] = {}

    async def score(self, symbol: str) -> Dict:
        """
        Generate on-chain score for a symbol.

        Fetches funding rate, open interest, and long/short ratio
        from Binance Futures API.
        """
        funding_score, funding_details = await self._score_funding(symbol)
        oi_score, oi_details = await self._score_open_interest(symbol)
        ls_score, ls_details = await self._score_long_short_ratio(symbol)
        liq_details = self._estimate_liquidation_levels(symbol)

        composite = (
            funding_score * 0.35
            + oi_score * 0.35
            + ls_score * 0.30
        )

        return {
            "score": float(np.clip(composite, -100, 100)),
            "funding_rate": funding_details,
            "open_interest": oi_details,
            "long_short_ratio": ls_details,
            "liquidation_levels": liq_details,
        }

    async def _fetch_json(self, path: str, params: Dict = None) -> Optional[Dict]:
        """Fetch JSON from Binance API."""
        url = f"{self.base_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        text = await resp.text()
                        logger.warning("API %s returned %d: %s", path, resp.status, text[:200])
                        return None
        except Exception as e:
            logger.error("Failed to fetch %s: %s", path, e)
            return None

    async def _score_funding(self, symbol: str) -> Tuple[float, Dict]:
        """
        Score based on funding rate.

        Funding rate interpretation:
        - Very positive (>0.01%): crowded long, contrarian bearish
        - Slightly positive: normal bullish lean
        - Near zero: neutral/balanced
        - Negative: crowded short, contrarian bullish
        - Very negative (<-0.01%): extreme short crowding, bullish
        """
        details = {}

        data = await self._fetch_json("/fapi/v1/fundingRate", {
            "symbol": symbol,
            "limit": 10,
        })

        if not data or not isinstance(data, list) or len(data) == 0:
            return 0.0, {"note": "funding_data_unavailable"}

        # Current funding rate (most recent)
        current_rate = float(data[-1]["fundingRate"])
        details["current_rate"] = current_rate
        details["current_rate_pct"] = current_rate * 100

        # Average funding rate (last 10 periods = ~80 hours)
        rates = [float(d["fundingRate"]) for d in data]
        avg_rate = np.mean(rates)
        details["avg_rate_10"] = float(avg_rate)

        # Funding trend (rising or falling)
        if len(rates) >= 3:
            recent_avg = np.mean(rates[-3:])
            older_avg = np.mean(rates[:3])
            details["funding_trend"] = "rising" if recent_avg > older_avg else "falling"

        # Score: contrarian to crowding
        # High positive funding = longs are paying, market too bullish -> bearish signal
        # High negative funding = shorts are paying, market too bearish -> bullish signal
        rate_pct = current_rate * 100

        if rate_pct > 0.05:
            score = -60  # Very crowded long
            details["interpretation"] = "extremely_crowded_long"
        elif rate_pct > 0.02:
            score = -35
            details["interpretation"] = "crowded_long"
        elif rate_pct > 0.01:
            score = -15
            details["interpretation"] = "slightly_long_biased"
        elif rate_pct > -0.01:
            score = 0
            details["interpretation"] = "neutral"
        elif rate_pct > -0.02:
            score = 15
            details["interpretation"] = "slightly_short_biased"
        elif rate_pct > -0.05:
            score = 35
            details["interpretation"] = "crowded_short"
        else:
            score = 60
            details["interpretation"] = "extremely_crowded_short"

        # Cache for liquidation estimation
        self._cache.setdefault(symbol, {})["funding_rate"] = current_rate

        return float(score), details

    async def _score_open_interest(self, symbol: str) -> Tuple[float, Dict]:
        """
        Score based on open interest changes.

        Interpretation matrix:
        - Rising OI + Rising Price = New longs entering (bullish)
        - Rising OI + Falling Price = New shorts entering (bearish)
        - Falling OI + Rising Price = Short squeeze (bullish momentum)
        - Falling OI + Falling Price = Long liquidation (bearish momentum)
        """
        details = {}

        # Fetch OI history
        data = await self._fetch_json("/futures/data/openInterestHist", {
            "symbol": symbol,
            "period": "5m",
            "limit": 30,
        })

        if not data or not isinstance(data, list) or len(data) < 5:
            return 0.0, {"note": "oi_data_unavailable"}

        oi_values = [float(d["sumOpenInterest"]) for d in data]
        oi_usd_values = [float(d["sumOpenInterestValue"]) for d in data]

        current_oi = oi_values[-1]
        prev_oi = oi_values[-6] if len(oi_values) >= 6 else oi_values[0]

        details["current_oi"] = current_oi
        details["current_oi_usd"] = oi_usd_values[-1]

        if prev_oi > 0:
            oi_change_pct = (current_oi - prev_oi) / prev_oi * 100
            details["oi_change_pct_30m"] = float(oi_change_pct)

            # Need price change to interpret OI change
            # Use cached price data if available
            price_cache = self._cache.get(symbol, {}).get("price_change")

            if oi_change_pct > 5:
                if price_cache and price_cache > 0:
                    score = 40  # Rising OI + Rising Price = bullish
                    details["interpretation"] = "new_longs_entering"
                elif price_cache and price_cache < 0:
                    score = -40  # Rising OI + Falling Price = bearish
                    details["interpretation"] = "new_shorts_entering"
                else:
                    score = 10  # Rising OI, unknown price direction
                    details["interpretation"] = "increasing_interest"
            elif oi_change_pct < -5:
                if price_cache and price_cache > 0:
                    score = 30  # Falling OI + Rising Price = short squeeze
                    details["interpretation"] = "short_squeeze"
                elif price_cache and price_cache < 0:
                    score = -30  # Falling OI + Falling Price = long liquidation
                    details["interpretation"] = "long_liquidation"
                else:
                    score = -10  # Falling OI
                    details["interpretation"] = "decreasing_interest"
            else:
                score = 0
                details["interpretation"] = "stable_oi"
        else:
            score = 0

        self._cache.setdefault(symbol, {})["open_interest"] = current_oi

        return float(np.clip(score, -100, 100)), details

    async def _score_long_short_ratio(self, symbol: str) -> Tuple[float, Dict]:
        """
        Score based on top trader long/short ratio.

        Uses Binance top trader positions as a proxy for smart money.
        """
        details = {}

        # Top trader long/short ratio (accounts)
        data = await self._fetch_json("/futures/data/topLongShortAccountRatio", {
            "symbol": symbol,
            "period": "5m",
            "limit": 12,
        })

        if not data or not isinstance(data, list) or len(data) < 3:
            return 0.0, {"note": "ls_ratio_unavailable"}

        ratios = [float(d["longShortRatio"]) for d in data]
        current_ratio = ratios[-1]
        avg_ratio = np.mean(ratios)

        details["current_ratio"] = float(current_ratio)
        details["avg_ratio_1h"] = float(avg_ratio)
        details["long_pct"] = float(data[-1].get("longAccount", 0.5)) * 100
        details["short_pct"] = float(data[-1].get("shortAccount", 0.5)) * 100

        # Ratio trend
        if len(ratios) >= 6:
            recent = np.mean(ratios[-3:])
            older = np.mean(ratios[:3])
            details["ratio_trend"] = "longs_increasing" if recent > older else "shorts_increasing"

        # Score: contrarian to extreme positioning
        # Ratio > 2.0: heavily long biased -> contrarian bearish
        # Ratio < 0.5: heavily short biased -> contrarian bullish
        # Ratio near 1.0: balanced
        if current_ratio > 3.0:
            score = -50
            details["interpretation"] = "extreme_long_bias"
        elif current_ratio > 2.0:
            score = -25
            details["interpretation"] = "strong_long_bias"
        elif current_ratio > 1.3:
            score = -10
            details["interpretation"] = "mild_long_bias"
        elif current_ratio > 0.7:
            score = 0
            details["interpretation"] = "balanced"
        elif current_ratio > 0.5:
            score = 10
            details["interpretation"] = "mild_short_bias"
        elif current_ratio > 0.3:
            score = 25
            details["interpretation"] = "strong_short_bias"
        else:
            score = 50
            details["interpretation"] = "extreme_short_bias"

        return float(score), details

    def _estimate_liquidation_levels(self, symbol: str) -> Dict:
        """
        Estimate major liquidation levels based on current price and leverage.

        This is a simplified estimation. In production, you would use
        order book depth data and known leverage tiers.
        """
        cache = self._cache.get(symbol, {})
        current_price = cache.get("current_price")

        if not current_price:
            return {"note": "price_unavailable_for_liquidation_estimate"}

        # Common leverage levels and their estimated liquidation distances
        leverage_levels = [5, 10, 20, 50, 100]
        liq_levels = {"long_liquidations": [], "short_liquidations": []}

        for lev in leverage_levels:
            # Approximate liquidation distance
            # For longs: liq_price = entry * (1 - 1/leverage * maintenance_margin_ratio)
            # Simplified: liq_distance roughly 1/leverage (minus some buffer)
            liq_dist_pct = (1.0 / lev) * 100 * 0.85  # 85% of theoretical max

            long_liq = current_price * (1 - liq_dist_pct / 100)
            short_liq = current_price * (1 + liq_dist_pct / 100)

            liq_levels["long_liquidations"].append({
                "leverage": lev,
                "price": float(round(long_liq, 2)),
                "distance_pct": float(round(liq_dist_pct, 2)),
            })
            liq_levels["short_liquidations"].append({
                "leverage": lev,
                "price": float(round(short_liq, 2)),
                "distance_pct": float(round(liq_dist_pct, 2)),
            })

        return liq_levels

    def update_price(self, symbol: str, price: float, price_change_pct: float = 0.0):
        """Update cached price for a symbol (used by pipeline)."""
        self._cache.setdefault(symbol, {})["current_price"] = price
        self._cache.setdefault(symbol, {})["price_change"] = price_change_pct
