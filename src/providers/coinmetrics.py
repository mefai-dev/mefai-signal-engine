"""
Coin Metrics provider - network data and on-chain metrics.
API docs: https://docs.coinmetrics.io/api/v4
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class CoinMetricsProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "coinmetrics"

    @property
    def base_url(self) -> str:
        return "https://api.coinmetrics.io/v4"

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _parse_timeseries(self, data: Dict, metric_name: str) -> List[MetricResult]:
        results = []
        entries = data.get("data", [])
        for entry in entries:
            ts_str = entry.get("time", "")
            try:
                from datetime import datetime, timezone
                ts = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp())
            except (ValueError, AttributeError):
                ts = int(time.time())
            # Extract the first numeric value from the entry
            value = None
            for k, v in entry.items():
                if k != "time" and k != "asset":
                    try:
                        value = float(v)
                        break
                    except (TypeError, ValueError):
                        value = v
                        break
            results.append(MetricResult(
                metric_name=metric_name,
                value=value,
                timestamp=ts,
                source=self.name,
                metadata=entry,
            ))
        return results

    async def _get_metric(self, asset: str, metrics: str, frequency: str = "1d", limit: int = 30) -> Dict:
        """Fetch a timeseries metric from the asset-metrics endpoint."""
        return await self._get(
            "timeseries/asset-metrics",
            params={
                "assets": asset,
                "metrics": metrics,
                "frequency": frequency,
                "page_size": limit,
                "sort": "time",
                "sort_direction": "desc",
            },
        )

    async def get_hashrate(self, asset: str = "btc") -> List[MetricResult]:
        """Network hashrate."""
        data = await self._get_metric(asset, "HashRate")
        return self._parse_timeseries(data, "hashrate")

    async def get_difficulty(self, asset: str = "btc") -> List[MetricResult]:
        """Mining difficulty."""
        data = await self._get_metric(asset, "DiffMean")
        return self._parse_timeseries(data, "difficulty")

    async def get_fees(self, asset: str = "btc") -> List[MetricResult]:
        """Total transaction fees (USD)."""
        data = await self._get_metric(asset, "FeeTotUSD")
        return self._parse_timeseries(data, "fees_usd")

    async def get_active_addresses(self, asset: str = "btc") -> List[MetricResult]:
        """Active addresses count."""
        data = await self._get_metric(asset, "AdrActCnt")
        return self._parse_timeseries(data, "active_addresses")

    async def get_realized_cap(self, asset: str = "btc") -> List[MetricResult]:
        """Realized capitalization."""
        data = await self._get_metric(asset, "CapRealUSD")
        return self._parse_timeseries(data, "realized_cap")

    async def get_nvt_ratio(self, asset: str = "btc") -> List[MetricResult]:
        """NVT ratio - Network Value to Transactions."""
        data = await self._get_metric(asset, "NVTAdj")
        return self._parse_timeseries(data, "nvt_ratio")

    async def get_mvrv_ratio(self, asset: str = "btc") -> List[MetricResult]:
        """MVRV ratio - Market Value to Realized Value."""
        data = await self._get_metric(asset, "CapMVRVCur")
        return self._parse_timeseries(data, "mvrv_ratio")

    async def get_thermocap_ratio(self, asset: str = "btc") -> List[MetricResult]:
        """Thermocap ratio - cumulative miner revenue vs market cap."""
        data = await self._get_metric(asset, "RevAllTimeUSD")
        return self._parse_timeseries(data, "thermocap")

    async def get_supply_in_profit(self, asset: str = "btc") -> List[MetricResult]:
        """Percentage of supply currently in profit."""
        data = await self._get_metric(asset, "SplyActPctAbv1d")
        return self._parse_timeseries(data, "supply_in_profit")

    async def get_all_metrics(self, asset: str = "btc") -> List[MetricResult]:
        """Fetch all available metrics for an asset."""
        import asyncio
        tasks = [
            self.get_hashrate(asset),
            self.get_active_addresses(asset),
            self.get_realized_cap(asset),
            self.get_nvt_ratio(asset),
            self.get_mvrv_ratio(asset),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list) and r:
                combined.append(r[0])  # most recent (desc sort)
        return combined
