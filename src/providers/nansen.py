"""
Nansen provider - blockchain analytics and smart money tracking.
API docs: https://docs.nansen.ai
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class NansenProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "nansen"

    @property
    def base_url(self) -> str:
        return "https://api.nansen.ai/v1"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "api-key": self.api_key,
        }

    def _parse_list(self, data: Any, metric_name: str) -> List[MetricResult]:
        results = []
        entries = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        for entry in entries:
            results.append(MetricResult(
                metric_name=metric_name,
                value=entry,
                timestamp=entry.get("timestamp", int(time.time())) if isinstance(entry, dict) else int(time.time()),
                source=self.name,
            ))
        return results

    async def get_smart_money_flow(self, token_address: str, chain: str = "ethereum") -> List[MetricResult]:
        """Smart money token flow - tracked wallets buying/selling."""
        data = await self._get(
            "smart-money/token-flow",
            params={"token_address": token_address, "chain": chain},
        )
        return self._parse_list(data, "smart_money_flow")

    async def get_hot_wallets(self, chain: str = "ethereum", limit: int = 20) -> List[MetricResult]:
        """Hot wallet tracking - most active wallets by transaction count."""
        data = await self._get(
            "wallet/hot",
            params={"chain": chain, "limit": limit},
        )
        return self._parse_list(data, "hot_wallets")

    async def get_token_god_mode(self, token_address: str, chain: str = "ethereum") -> List[MetricResult]:
        """Token God Mode - top holders, inflow/outflow breakdown."""
        data = await self._get(
            "token/god-mode",
            params={"token_address": token_address, "chain": chain},
        )
        return self._parse_list(data, "token_god_mode")

    async def get_nft_analytics(self, collection: str, chain: str = "ethereum") -> List[MetricResult]:
        """NFT collection analytics - floor price, volume, holder distribution."""
        data = await self._get(
            "nft/analytics",
            params={"collection": collection, "chain": chain},
        )
        return self._parse_list(data, "nft_analytics")

    async def get_defi_analytics(self, protocol: str, chain: str = "ethereum") -> List[MetricResult]:
        """DeFi protocol analytics - TVL, users, transaction volume."""
        data = await self._get(
            "defi/protocol",
            params={"protocol": protocol, "chain": chain},
        )
        return self._parse_list(data, "defi_analytics")

    async def get_bridge_flow(self, chain_from: str = "ethereum", chain_to: str = "bsc") -> List[MetricResult]:
        """Cross-chain bridge flow data."""
        data = await self._get(
            "bridge/flow",
            params={"chain_from": chain_from, "chain_to": chain_to},
        )
        return self._parse_list(data, "bridge_flow")

    async def get_label_data(self, address: str, chain: str = "ethereum") -> List[MetricResult]:
        """Label data - identify if address is exchange, fund, whale, etc."""
        data = await self._get(
            "labels",
            params={"address": address, "chain": chain},
        )
        return self._parse_list(data, "label_data")

    async def get_all_metrics(self, token_address: str = "", chain: str = "ethereum") -> List[MetricResult]:
        """Fetch available metrics. Requires token_address for most endpoints."""
        import asyncio
        tasks = []
        if token_address:
            tasks.append(self.get_smart_money_flow(token_address, chain))
            tasks.append(self.get_token_god_mode(token_address, chain))
        tasks.append(self.get_hot_wallets(chain))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list) and r:
                combined.append(r[0])
        return combined
