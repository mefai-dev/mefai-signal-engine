"""
LunarCrush provider - social intelligence for crypto assets.
API docs: https://lunarcrush.com/developers/api
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class LunarCrushProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "lunarcrush"

    @property
    def base_url(self) -> str:
        return "https://lunarcrush.com/api4/public"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _parse_coin_data(self, data: Dict, metric_name: str) -> List[MetricResult]:
        results = []
        coin_data = data.get("data", [])
        if isinstance(coin_data, dict):
            coin_data = [coin_data]
        for entry in coin_data:
            results.append(MetricResult(
                metric_name=metric_name,
                value=entry.get(metric_name, entry),
                timestamp=entry.get("time", int(time.time())),
                source=self.name,
                metadata=entry,
            ))
        return results

    async def get_galaxy_score(self, symbol: str = "BTC") -> List[MetricResult]:
        """Galaxy Score - overall social and market health (0-100)."""
        data = await self._get(
            "coins/list",
            params={"sort": "galaxy_score", "limit": 1, "symbol": symbol},
        )
        return self._parse_coin_data(data, "galaxy_score")

    async def get_alt_rank(self, symbol: str = "BTC") -> List[MetricResult]:
        """AltRank - relative social/market performance vs BTC."""
        data = await self._get(
            "coins/list",
            params={"sort": "alt_rank", "limit": 1, "symbol": symbol},
        )
        return self._parse_coin_data(data, "alt_rank")

    async def get_social_metrics(self, symbol: str = "BTC") -> List[MetricResult]:
        """Social volume, engagement, and sentiment combined."""
        data = await self._get(
            "coins/list",
            params={"symbol": symbol, "limit": 1},
        )
        entries = data.get("data", [])
        results = []
        if entries:
            entry = entries[0] if isinstance(entries, list) else entries
            ts = entry.get("time", int(time.time()))
            for metric in ["social_volume", "social_score", "social_contributors", "sentiment"]:
                if metric in entry:
                    results.append(MetricResult(
                        metric_name=metric,
                        value=entry[metric],
                        timestamp=ts,
                        source=self.name,
                    ))
        return results

    async def get_influencer_tracking(self, symbol: str = "BTC", limit: int = 20) -> List[MetricResult]:
        """Top influencers mentioning the asset."""
        data = await self._get(
            "coins/influencers",
            params={"symbol": symbol, "limit": limit},
        )
        return self._parse_coin_data(data, "influencers")

    async def get_social_dominance(self, symbol: str = "BTC") -> List[MetricResult]:
        """Social dominance - share of total crypto social volume."""
        data = await self._get(
            "coins/list",
            params={"symbol": symbol, "limit": 1},
        )
        return self._parse_coin_data(data, "social_dominance")

    async def get_spam_score(self, symbol: str = "BTC") -> List[MetricResult]:
        """Spam detection score - percentage of posts flagged as spam."""
        data = await self._get(
            "coins/list",
            params={"symbol": symbol, "limit": 1},
        )
        return self._parse_coin_data(data, "spam_score")

    async def get_all_metrics(self, symbol: str = "BTC") -> List[MetricResult]:
        """Fetch all available metrics for a symbol."""
        import asyncio
        tasks = [
            self.get_galaxy_score(symbol),
            self.get_alt_rank(symbol),
            self.get_social_metrics(symbol),
            self.get_social_dominance(symbol),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list):
                combined.extend(r)
        return combined
