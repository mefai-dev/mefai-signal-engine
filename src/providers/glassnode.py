"""
Glassnode provider - on-chain analytics for Bitcoin and Ethereum.
API docs: https://docs.glassnode.com
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class GlassnodeProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "glassnode"

    @property
    def base_url(self) -> str:
        return "https://api.glassnode.com/v1/metrics"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
        }

    def _default_params(self, asset: str = "BTC", since: Optional[int] = None, until: Optional[int] = None, interval: str = "24h") -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "a": asset,
            "api_key": self.api_key,
            "i": interval,
        }
        if since:
            params["s"] = since
        if until:
            params["u"] = until
        return params

    def _parse_timeseries(self, data: List[Dict], metric_name: str) -> List[MetricResult]:
        results = []
        for entry in data:
            results.append(MetricResult(
                metric_name=metric_name,
                value=entry.get("v", entry.get("o", {})),
                timestamp=entry.get("t", 0),
                source=self.name,
            ))
        return results

    def _latest(self, results: List[MetricResult]) -> Optional[MetricResult]:
        if not results:
            return None
        return max(results, key=lambda r: r.timestamp)

    async def get_nupl(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """Net Unrealized Profit/Loss - measures overall market profit/loss."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("market/net_unrealized_profit_loss", params=params)
        return self._parse_timeseries(data, "nupl")

    async def get_sopr(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """Spent Output Profit Ratio - whether coins are sold at profit or loss."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("indicators/sopr", params=params)
        return self._parse_timeseries(data, "sopr")

    async def get_exchange_net_flow(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """Exchange net position change (inflow minus outflow)."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("transactions/transfers_volume_exchanges_net", params=params)
        return self._parse_timeseries(data, "exchange_net_flow")

    async def get_miner_revenue(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """Total miner revenue (block rewards + fees)."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("mining/revenue_sum", params=params)
        return self._parse_timeseries(data, "miner_revenue")

    async def get_hodl_waves(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """HODL Waves - UTXO age distribution (1d, 1w, 1m, 3m, 6m, 1y, 2y, 3y, 5y+ cohorts)."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("supply/hodl_waves", params=params)
        return self._parse_timeseries(data, "hodl_waves")

    async def get_active_addresses(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """Number of unique active addresses per day."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("addresses/active_count", params=params)
        return self._parse_timeseries(data, "active_addresses")

    async def get_stock_to_flow(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """Stock-to-Flow ratio and model deflection."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("indicators/stock_to_flow_ratio", params=params)
        return self._parse_timeseries(data, "stock_to_flow")

    async def get_mvrv_zscore(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """MVRV Z-Score - identifies over/undervaluation relative to realized value."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("market/mvrv_z_score", params=params)
        return self._parse_timeseries(data, "mvrv_zscore")

    async def get_puell_multiple(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """Puell Multiple - daily coin issuance vs 365-day moving average."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("mining/puell_multiple", params=params)
        return self._parse_timeseries(data, "puell_multiple")

    async def get_reserve_risk(self, asset: str = "BTC", **kwargs) -> List[MetricResult]:
        """Reserve Risk - confidence vs price, low = high confidence + low price."""
        params = self._default_params(asset, **kwargs)
        data = await self._get("indicators/reserve_risk", params=params)
        return self._parse_timeseries(data, "reserve_risk")

    async def get_all_metrics(self, asset: str = "BTC") -> List[MetricResult]:
        """Fetch all available metrics for an asset."""
        import asyncio
        tasks = [
            self.get_nupl(asset),
            self.get_sopr(asset),
            self.get_exchange_net_flow(asset),
            self.get_miner_revenue(asset),
            self.get_active_addresses(asset),
            self.get_mvrv_zscore(asset),
            self.get_puell_multiple(asset),
            self.get_reserve_risk(asset),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list):
                latest = self._latest(r)
                if latest:
                    combined.append(latest)
            elif isinstance(r, Exception):
                pass
        return combined
