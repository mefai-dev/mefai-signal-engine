"""
Kaiko provider - institutional-grade market data.
API docs: https://docs.kaiko.com
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class KaikoProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "kaiko"

    @property
    def base_url(self) -> str:
        return "https://us.market-api.kaiko.io/v2"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "X-Api-Key": self.api_key,
        }

    def _parse_data(self, data: Dict, metric_name: str) -> List[MetricResult]:
        results = []
        entries = data.get("data", [])
        for entry in entries:
            ts = entry.get("timestamp", entry.get("poll_timestamp", 0))
            if isinstance(ts, str):
                from datetime import datetime, timezone
                try:
                    ts = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    ts = int(time.time())
            results.append(MetricResult(
                metric_name=metric_name,
                value=entry,
                timestamp=ts,
                source=self.name,
            ))
        return results

    async def get_order_book_snapshot(
        self,
        exchange: str = "cbse",
        pair: str = "btc-usd",
        depth: str = "L2",
    ) -> List[MetricResult]:
        """Order book snapshot - L2 or L3 depth data."""
        data = await self._get(
            f"data/order_book_snapshots.v1/exchanges/{exchange}/spot/{pair}/snapshots",
            params={"depth": depth, "limit_events": 1},
        )
        return self._parse_data(data, f"orderbook_{depth.lower()}")

    async def get_vwap(
        self,
        exchange: str = "cbse",
        pair: str = "btc-usd",
        interval: str = "1h",
    ) -> List[MetricResult]:
        """Volume Weighted Average Price."""
        data = await self._get(
            f"data/trades.v1/exchanges/{exchange}/spot/{pair}/aggregations/vwap",
            params={"interval": interval, "page_size": 100},
        )
        return self._parse_data(data, "vwap")

    async def get_slippage(
        self,
        exchange: str = "cbse",
        pair: str = "btc-usd",
        slippage: str = "10000",
    ) -> List[MetricResult]:
        """Slippage estimation for a given order size (in USD)."""
        data = await self._get(
            f"data/order_book_snapshots.v1/exchanges/{exchange}/spot/{pair}/slippage",
            params={"slippage": slippage},
        )
        return self._parse_data(data, "slippage")

    async def get_trades(
        self,
        exchange: str = "cbse",
        pair: str = "btc-usd",
        limit: int = 100,
    ) -> List[MetricResult]:
        """Tick-level trade data."""
        data = await self._get(
            f"data/trades.v1/exchanges/{exchange}/spot/{pair}/trades",
            params={"page_size": limit},
        )
        return self._parse_data(data, "trades")

    async def get_ohlcv(
        self,
        exchange: str = "cbse",
        pair: str = "btc-usd",
        interval: str = "1h",
    ) -> List[MetricResult]:
        """OHLCV aggregated candle data."""
        data = await self._get(
            f"data/trades.v1/exchanges/{exchange}/spot/{pair}/aggregations/ohlcv",
            params={"interval": interval, "page_size": 100},
        )
        return self._parse_data(data, "ohlcv")

    async def get_cross_exchange_spread(
        self,
        pair: str = "btc-usd",
        exchanges: Optional[List[str]] = None,
    ) -> List[MetricResult]:
        """Cross-exchange spread comparison."""
        ex_list = exchanges or ["cbse", "bfnx", "krkn", "bnce"]
        import asyncio
        tasks = []
        for ex in ex_list:
            tasks.append(self.get_vwap(exchange=ex, pair=pair, interval="1h"))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for i, r in enumerate(results):
            if isinstance(r, list) and r:
                latest = r[-1]
                latest.metric_name = f"vwap_{ex_list[i]}"
                combined.append(latest)
        return combined

    async def get_all_metrics(self, pair: str = "btc-usd") -> List[MetricResult]:
        """Fetch all available metrics for a pair."""
        import asyncio
        tasks = [
            self.get_vwap(pair=pair),
            self.get_ohlcv(pair=pair),
            self.get_trades(pair=pair, limit=10),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list) and r:
                combined.append(r[-1])
        return combined
