"""
Token Terminal provider - protocol financial metrics.
API docs: https://docs.tokenterminal.com
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class TokenTerminalProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "token_terminal"

    @property
    def base_url(self) -> str:
        return "https://api.tokenterminal.com/v2"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _parse_timeseries(self, data: Any, metric_name: str) -> List[MetricResult]:
        results = []
        entries = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        for entry in entries:
            ts_str = entry.get("timestamp", entry.get("date", ""))
            try:
                from datetime import datetime, timezone
                ts = int(datetime.fromisoformat(str(ts_str).replace("Z", "+00:00")).timestamp())
            except (ValueError, AttributeError):
                ts = int(time.time())
            results.append(MetricResult(
                metric_name=metric_name,
                value=entry,
                timestamp=ts,
                source=self.name,
            ))
        return results

    async def get_protocol_revenue(self, project_id: str, granularity: str = "daily") -> List[MetricResult]:
        """Protocol revenue - daily and cumulative."""
        data = await self._get(
            f"projects/{project_id}/metrics",
            params={"metric_ids": "revenue", "granularity": granularity},
        )
        return self._parse_timeseries(data, "protocol_revenue")

    async def get_pe_ratio(self, project_id: str) -> List[MetricResult]:
        """P/E ratio - fully diluted market cap / annualized revenue."""
        data = await self._get(
            f"projects/{project_id}/metrics",
            params={"metric_ids": "pe"},
        )
        return self._parse_timeseries(data, "pe_ratio")

    async def get_ps_ratio(self, project_id: str) -> List[MetricResult]:
        """P/S ratio - fully diluted market cap / annualized fees."""
        data = await self._get(
            f"projects/{project_id}/metrics",
            params={"metric_ids": "ps"},
        )
        return self._parse_timeseries(data, "ps_ratio")

    async def get_tvl(self, project_id: str) -> List[MetricResult]:
        """Total Value Locked in the protocol."""
        data = await self._get(
            f"projects/{project_id}/metrics",
            params={"metric_ids": "tvl"},
        )
        return self._parse_timeseries(data, "tvl")

    async def get_active_users(self, project_id: str, granularity: str = "daily") -> List[MetricResult]:
        """Daily active users interacting with the protocol."""
        data = await self._get(
            f"projects/{project_id}/metrics",
            params={"metric_ids": "active_users", "granularity": granularity},
        )
        return self._parse_timeseries(data, "active_users")

    async def get_token_incentives(self, project_id: str) -> List[MetricResult]:
        """Token incentives - emissions and rewards distributed."""
        data = await self._get(
            f"projects/{project_id}/metrics",
            params={"metric_ids": "token_incentives"},
        )
        return self._parse_timeseries(data, "token_incentives")

    async def get_all_metrics(self, project_id: str = "aave") -> List[MetricResult]:
        """Fetch all available metrics for a protocol."""
        import asyncio
        tasks = [
            self.get_protocol_revenue(project_id),
            self.get_pe_ratio(project_id),
            self.get_ps_ratio(project_id),
            self.get_tvl(project_id),
            self.get_active_users(project_id),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list) and r:
                combined.append(r[-1])
        return combined
