"""
Messari provider - crypto asset profiles, fundamentals, and market data.
API docs: https://messari.io/api/docs
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class MessariProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "messari"

    @property
    def base_url(self) -> str:
        return "https://data.messari.io/api/v2"

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-messari-api-key"] = self.api_key
        return headers

    def _extract_data(self, resp: Dict) -> Any:
        return resp.get("data", resp)

    async def get_token_unlock_schedule(self, slug: str = "bitcoin") -> List[MetricResult]:
        """Token unlock/vesting schedule."""
        data = await self._get(f"assets/{slug}/profile")
        profile = self._extract_data(data)
        economics = profile.get("economics", {}).get("token", {})
        return [MetricResult(
            metric_name="token_unlock_schedule",
            value=economics,
            timestamp=int(time.time()),
            source=self.name,
        )]

    async def get_governance_proposals(self, slug: str = "bitcoin") -> List[MetricResult]:
        """Active governance proposals for a project."""
        data = await self._get(f"assets/{slug}/profile")
        profile = self._extract_data(data)
        governance = profile.get("governance", {})
        return [MetricResult(
            metric_name="governance_proposals",
            value=governance,
            timestamp=int(time.time()),
            source=self.name,
        )]

    async def get_protocol_fundamentals(self, slug: str = "bitcoin") -> List[MetricResult]:
        """Protocol fundamentals - revenue, TVL, key metrics."""
        data = await self._get(f"assets/{slug}/metrics")
        metrics = self._extract_data(data)
        result = {
            "market_data": metrics.get("market_data", {}),
            "marketcap": metrics.get("marketcap", {}),
            "supply": metrics.get("supply", {}),
            "roi_data": metrics.get("roi_data", {}),
            "on_chain_data": metrics.get("on_chain_data", {}),
        }
        return [MetricResult(
            metric_name="protocol_fundamentals",
            value=result,
            timestamp=int(time.time()),
            source=self.name,
        )]

    async def get_asset_profile(self, slug: str = "bitcoin") -> List[MetricResult]:
        """Full asset profile and metadata."""
        data = await self._get(f"assets/{slug}/profile")
        profile = self._extract_data(data)
        return [MetricResult(
            metric_name="asset_profile",
            value={
                "name": profile.get("name", ""),
                "symbol": profile.get("symbol", ""),
                "tagline": profile.get("profile", {}).get("general", {}).get("overview", {}).get("tagline", ""),
                "sector": profile.get("profile", {}).get("general", {}).get("overview", {}).get("sector", ""),
                "category": profile.get("profile", {}).get("general", {}).get("overview", {}).get("category", ""),
            },
            timestamp=int(time.time()),
            source=self.name,
        )]

    async def get_market_data(self, slug: str = "bitcoin") -> List[MetricResult]:
        """Market data - OHLCV, ROI, market cap."""
        data = await self._get(f"assets/{slug}/metrics/market-data")
        market = self._extract_data(data).get("market_data", {})
        return [MetricResult(
            metric_name="market_data",
            value=market,
            timestamp=int(time.time()),
            source=self.name,
        )]

    async def get_all_metrics(self, slug: str = "bitcoin") -> List[MetricResult]:
        """Fetch all available metrics for an asset."""
        import asyncio
        tasks = [
            self.get_protocol_fundamentals(slug),
            self.get_market_data(slug),
            self.get_asset_profile(slug),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list) and r:
                combined.append(r[0])
        return combined
