"""
Free data providers that do not require API keys.
- DefiLlama: TVL data
- Alternative.me: Fear and Greed Index
- CoinGecko free tier: Market data
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class DefiLlamaProvider(BaseProvider):
    """DefiLlama - DeFi TVL tracking, no API key required."""

    @property
    def name(self) -> str:
        return "defillama"

    @property
    def base_url(self) -> str:
        return "https://api.llama.fi"

    @property
    def requires_api_key(self) -> bool:
        return False

    def _build_headers(self) -> Dict[str, str]:
        return {"Accept": "application/json"}

    async def get_protocols_tvl(self) -> List[MetricResult]:
        """Get TVL for all tracked protocols."""
        data = await self._get("protocols")
        results = []
        if isinstance(data, list):
            for protocol in data[:50]:  # top 50 by default
                results.append(MetricResult(
                    metric_name="protocol_tvl",
                    value=protocol.get("tvl", 0),
                    timestamp=int(time.time()),
                    source=self.name,
                    metadata={
                        "name": protocol.get("name", ""),
                        "slug": protocol.get("slug", ""),
                        "chain": protocol.get("chain", ""),
                        "category": protocol.get("category", ""),
                        "change_1d": protocol.get("change_1d", 0),
                        "change_7d": protocol.get("change_7d", 0),
                    },
                ))
        return results

    async def get_protocol_tvl(self, slug: str = "aave") -> List[MetricResult]:
        """Get historical TVL for a specific protocol."""
        data = await self._get(f"protocol/{slug}")
        results = []
        tvl_data = data.get("tvl", []) if isinstance(data, dict) else []
        for entry in tvl_data[-30:]:  # last 30 days
            results.append(MetricResult(
                metric_name="protocol_tvl_history",
                value=entry.get("totalLiquidityUSD", 0),
                timestamp=entry.get("date", int(time.time())),
                source=self.name,
                metadata={"protocol": slug},
            ))
        return results

    async def get_chains_tvl(self) -> List[MetricResult]:
        """Get TVL breakdown by chain."""
        data = await self._get("v2/chains")
        results = []
        if isinstance(data, list):
            for chain in data:
                results.append(MetricResult(
                    metric_name="chain_tvl",
                    value=chain.get("tvl", 0),
                    timestamp=int(time.time()),
                    source=self.name,
                    metadata={
                        "name": chain.get("name", ""),
                        "gecko_id": chain.get("gecko_id", ""),
                    },
                ))
        return results

    async def get_stablecoins(self) -> List[MetricResult]:
        """Get stablecoin market caps."""
        data = await self._get("stablecoins")
        results = []
        entries = data.get("peggedAssets", []) if isinstance(data, dict) else []
        for coin in entries[:20]:
            results.append(MetricResult(
                metric_name="stablecoin_mcap",
                value=coin.get("circulating", {}).get("peggedUSD", 0),
                timestamp=int(time.time()),
                source=self.name,
                metadata={
                    "name": coin.get("name", ""),
                    "symbol": coin.get("symbol", ""),
                },
            ))
        return results

    async def get_yields(self, limit: int = 20) -> List[MetricResult]:
        """Get top DeFi yields across protocols."""
        data = await self._get("pools")
        results = []
        entries = data.get("data", []) if isinstance(data, dict) else []
        # Sort by TVL descending
        entries.sort(key=lambda x: x.get("tvlUsd", 0), reverse=True)
        for pool in entries[:limit]:
            results.append(MetricResult(
                metric_name="defi_yield",
                value=pool.get("apy", 0),
                timestamp=int(time.time()),
                source=self.name,
                metadata={
                    "pool": pool.get("pool", ""),
                    "project": pool.get("project", ""),
                    "chain": pool.get("chain", ""),
                    "tvl": pool.get("tvlUsd", 0),
                    "symbol": pool.get("symbol", ""),
                },
            ))
        return results


class FearGreedProvider(BaseProvider):
    """Alternative.me Fear and Greed Index, no API key required."""

    @property
    def name(self) -> str:
        return "fear_greed"

    @property
    def base_url(self) -> str:
        return "https://api.alternative.me"

    @property
    def requires_api_key(self) -> bool:
        return False

    def _build_headers(self) -> Dict[str, str]:
        return {"Accept": "application/json"}

    async def get_current(self) -> List[MetricResult]:
        """Get current Fear and Greed Index value."""
        data = await self._get("fng/", params={"limit": 1})
        results = []
        entries = data.get("data", [])
        for entry in entries:
            results.append(MetricResult(
                metric_name="fear_greed_index",
                value=int(entry.get("value", 0)),
                timestamp=int(entry.get("timestamp", time.time())),
                source=self.name,
                metadata={
                    "classification": entry.get("value_classification", ""),
                },
            ))
        return results

    async def get_historical(self, days: int = 30) -> List[MetricResult]:
        """Get historical Fear and Greed Index."""
        data = await self._get("fng/", params={"limit": days})
        results = []
        for entry in data.get("data", []):
            results.append(MetricResult(
                metric_name="fear_greed_index",
                value=int(entry.get("value", 0)),
                timestamp=int(entry.get("timestamp", time.time())),
                source=self.name,
                metadata={
                    "classification": entry.get("value_classification", ""),
                },
            ))
        return results


class CoinGeckoFreeProvider(BaseProvider):
    """CoinGecko free tier - basic market data, no API key required."""

    @property
    def name(self) -> str:
        return "coingecko"

    @property
    def base_url(self) -> str:
        return "https://api.coingecko.com/api/v3"

    @property
    def requires_api_key(self) -> bool:
        return False

    def _build_headers(self) -> Dict[str, str]:
        return {"Accept": "application/json"}

    async def get_price(self, ids: str = "bitcoin,ethereum", vs: str = "usd") -> List[MetricResult]:
        """Get current price for multiple coins."""
        data = await self._get(
            "simple/price",
            params={
                "ids": ids,
                "vs_currencies": vs,
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            },
        )
        results = []
        for coin_id, values in data.items():
            results.append(MetricResult(
                metric_name="price",
                value=values.get(vs, 0),
                timestamp=int(time.time()),
                source=self.name,
                metadata={
                    "coin": coin_id,
                    "market_cap": values.get(f"{vs}_market_cap", 0),
                    "volume_24h": values.get(f"{vs}_24h_vol", 0),
                    "change_24h": values.get(f"{vs}_24h_change", 0),
                },
            ))
        return results

    async def get_market_data(self, vs: str = "usd", per_page: int = 50) -> List[MetricResult]:
        """Get market data for top coins by market cap."""
        data = await self._get(
            "coins/markets",
            params={
                "vs_currency": vs,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": 1,
                "sparkline": "false",
            },
        )
        results = []
        if isinstance(data, list):
            for coin in data:
                results.append(MetricResult(
                    metric_name="market_data",
                    value=coin.get("current_price", 0),
                    timestamp=int(time.time()),
                    source=self.name,
                    metadata={
                        "id": coin.get("id", ""),
                        "symbol": coin.get("symbol", ""),
                        "name": coin.get("name", ""),
                        "market_cap": coin.get("market_cap", 0),
                        "market_cap_rank": coin.get("market_cap_rank", 0),
                        "total_volume": coin.get("total_volume", 0),
                        "price_change_24h": coin.get("price_change_percentage_24h", 0),
                        "ath": coin.get("ath", 0),
                        "ath_change_pct": coin.get("ath_change_percentage", 0),
                    },
                ))
        return results

    async def get_global(self) -> List[MetricResult]:
        """Get global crypto market data."""
        data = await self._get("global")
        global_data = data.get("data", {})
        return [MetricResult(
            metric_name="global_market",
            value=global_data.get("total_market_cap", {}).get("usd", 0),
            timestamp=int(time.time()),
            source=self.name,
            metadata={
                "total_volume": global_data.get("total_volume", {}).get("usd", 0),
                "btc_dominance": global_data.get("market_cap_percentage", {}).get("btc", 0),
                "eth_dominance": global_data.get("market_cap_percentage", {}).get("eth", 0),
                "active_cryptos": global_data.get("active_cryptocurrencies", 0),
                "markets": global_data.get("markets", 0),
                "market_cap_change_24h": global_data.get("market_cap_change_percentage_24h_usd", 0),
            },
        )]

    async def get_trending(self) -> List[MetricResult]:
        """Get trending coins in the last 24 hours."""
        data = await self._get("search/trending")
        results = []
        coins = data.get("coins", [])
        for item in coins:
            coin = item.get("item", {})
            results.append(MetricResult(
                metric_name="trending",
                value=coin.get("score", 0),
                timestamp=int(time.time()),
                source=self.name,
                metadata={
                    "id": coin.get("id", ""),
                    "name": coin.get("name", ""),
                    "symbol": coin.get("symbol", ""),
                    "market_cap_rank": coin.get("market_cap_rank", 0),
                },
            ))
        return results
