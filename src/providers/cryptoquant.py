"""
CryptoQuant provider - on-chain and market data analytics.
API docs: https://docs.cryptoquant.com
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class CryptoQuantProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "cryptoquant"

    @property
    def base_url(self) -> str:
        return "https://api.cryptoquant.com/v1"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _parse_response(self, data: Dict, metric_name: str) -> List[MetricResult]:
        results = []
        entries = data.get("result", {}).get("data", [])
        for entry in entries:
            results.append(MetricResult(
                metric_name=metric_name,
                value=entry.get("value", entry),
                timestamp=entry.get("date", 0) if isinstance(entry.get("date"), int) else int(time.time()),
                source=self.name,
            ))
        return results

    async def get_exchange_reserve(self, asset: str = "btc", window: str = "day") -> List[MetricResult]:
        """Total BTC/ETH held on exchanges."""
        data = await self._get(
            f"{asset}/exchange-flows/reserve",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "exchange_reserve")

    async def get_exchange_inflow(self, asset: str = "btc", window: str = "day") -> List[MetricResult]:
        """Total inflow to exchanges with exchange breakdown."""
        data = await self._get(
            f"{asset}/exchange-flows/inflow",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "exchange_inflow")

    async def get_exchange_outflow(self, asset: str = "btc", window: str = "day") -> List[MetricResult]:
        """Total outflow from exchanges with exchange breakdown."""
        data = await self._get(
            f"{asset}/exchange-flows/outflow",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "exchange_outflow")

    async def get_miner_outflow(self, asset: str = "btc", window: str = "day") -> List[MetricResult]:
        """Miner outflow to exchanges."""
        data = await self._get(
            f"{asset}/miner-flows/outflow",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "miner_outflow")

    async def get_fund_flow_ratio(self, asset: str = "btc", window: str = "day") -> List[MetricResult]:
        """Fund flow ratio - exchange inflow relative to total on-chain transactions."""
        data = await self._get(
            f"{asset}/exchange-flows/fund-flow-ratio",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "fund_flow_ratio")

    async def get_stablecoin_supply_ratio(self, window: str = "day") -> List[MetricResult]:
        """Stablecoin Supply Ratio (SSR) - BTC market cap / stablecoin supply."""
        data = await self._get(
            "btc/market-data/stablecoin-supply-ratio",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "stablecoin_supply_ratio")

    async def get_utxo_age_bands(self, asset: str = "btc", window: str = "day") -> List[MetricResult]:
        """UTXO age band distribution."""
        data = await self._get(
            f"{asset}/utxo/age-bands",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "utxo_age_bands")

    async def get_estimated_leverage_ratio(self, asset: str = "btc", window: str = "day") -> List[MetricResult]:
        """Estimated leverage ratio - open interest / exchange reserve."""
        data = await self._get(
            f"{asset}/market-data/estimated-leverage-ratio",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "estimated_leverage_ratio")

    async def get_taker_buy_sell_ratio(self, asset: str = "btc", window: str = "day") -> List[MetricResult]:
        """Taker buy/sell ratio on futures markets."""
        data = await self._get(
            f"{asset}/market-data/taker-buy-sell-ratio",
            params={"window": window, "limit": 30},
        )
        return self._parse_response(data, "taker_buy_sell_ratio")

    async def get_all_metrics(self, asset: str = "btc") -> List[MetricResult]:
        """Fetch all available metrics for an asset."""
        import asyncio
        tasks = [
            self.get_exchange_reserve(asset),
            self.get_exchange_inflow(asset),
            self.get_exchange_outflow(asset),
            self.get_miner_outflow(asset),
            self.get_fund_flow_ratio(asset),
            self.get_estimated_leverage_ratio(asset),
            self.get_taker_buy_sell_ratio(asset),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list) and r:
                combined.append(r[-1])
        return combined
