"""
IntoTheBlock provider - on-chain analytics and market intelligence.
API docs: https://api.intotheblock.com/docs
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class IntoTheBlockProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "intotheblock"

    @property
    def base_url(self) -> str:
        return "https://api.intotheblock.com/v1"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "x-api-key": self.api_key,
        }

    def _parse_response(self, data: Any, metric_name: str) -> List[MetricResult]:
        results = []
        if isinstance(data, dict):
            entries = data.get("data", [data])
        elif isinstance(data, list):
            entries = data
        else:
            entries = []
        for entry in entries:
            ts = entry.get("timestamp", entry.get("date", int(time.time())))
            if isinstance(ts, str):
                from datetime import datetime, timezone
                try:
                    ts = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    ts = int(time.time())
            results.append(MetricResult(
                metric_name=metric_name,
                value=entry.get("value", entry),
                timestamp=ts,
                source=self.name,
                metadata=entry if isinstance(entry, dict) else {},
            ))
        return results

    async def get_in_out_of_money(self, symbol: str = "bitcoin") -> List[MetricResult]:
        """In/Out of Money - percentage of addresses in profit vs loss."""
        data = await self._get(
            f"indicators/{symbol}/in-out-money",
        )
        return self._parse_response(data, "in_out_of_money")

    async def get_concentration(self, symbol: str = "bitcoin") -> List[MetricResult]:
        """Concentration - whale vs retail holder distribution."""
        data = await self._get(
            f"indicators/{symbol}/concentration",
        )
        return self._parse_response(data, "concentration")

    async def get_large_transactions(self, symbol: str = "bitcoin") -> List[MetricResult]:
        """Large transaction monitoring - transactions over 100K USD."""
        data = await self._get(
            f"indicators/{symbol}/large-transactions",
        )
        return self._parse_response(data, "large_transactions")

    async def get_network_health(self, symbol: str = "bitcoin") -> List[MetricResult]:
        """Network health indicators - combined on-chain health score."""
        data = await self._get(
            f"indicators/{symbol}/network-health",
        )
        return self._parse_response(data, "network_health")

    async def get_bid_ask_volume(self, symbol: str = "bitcoin") -> List[MetricResult]:
        """Bid/Ask volume imbalance across exchanges."""
        data = await self._get(
            f"indicators/{symbol}/bid-ask-volume",
        )
        return self._parse_response(data, "bid_ask_volume")

    async def get_all_metrics(self, symbol: str = "bitcoin") -> List[MetricResult]:
        """Fetch all available metrics for a symbol."""
        import asyncio
        tasks = [
            self.get_in_out_of_money(symbol),
            self.get_concentration(symbol),
            self.get_large_transactions(symbol),
            self.get_network_health(symbol),
            self.get_bid_ask_volume(symbol),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list) and r:
                combined.append(r[-1])
        return combined
