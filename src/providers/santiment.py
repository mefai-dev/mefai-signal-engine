"""
Santiment provider - social and on-chain analytics via GraphQL.
API docs: https://academy.santiment.net/sanapi/
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .base import BaseProvider, MetricResult


class SantimentProvider(BaseProvider):

    @property
    def name(self) -> str:
        return "santiment"

    @property
    def base_url(self) -> str:
        return "https://api.santiment.net/graphql"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Apikey {self.api_key}",
        }

    def _time_range(self, days_back: int = 7) -> tuple:
        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(days=days_back)
        return from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ")

    async def _query(self, gql: str, variables: Optional[Dict] = None) -> Any:
        """Execute a GraphQL query against the Santiment API."""
        payload = {"query": gql}
        if variables:
            payload["variables"] = variables
        data = await self._post("", json_data=payload)
        if "errors" in data:
            raise ValueError(f"Santiment GraphQL error: {data['errors']}")
        return data.get("data", {})

    def _parse_timeseries(self, data: List[Dict], metric_name: str) -> List[MetricResult]:
        results = []
        for entry in data:
            ts = entry.get("datetime", "")
            try:
                timestamp = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
            except (ValueError, AttributeError):
                timestamp = 0
            results.append(MetricResult(
                metric_name=metric_name,
                value=entry.get("value", 0),
                timestamp=timestamp,
                source=self.name,
            ))
        return results

    async def get_social_volume(self, slug: str = "bitcoin", days_back: int = 7) -> List[MetricResult]:
        """Social volume - total mentions across crypto social channels."""
        from_dt, to_dt = self._time_range(days_back)
        gql = """
        {
          getMetric(metric: "social_volume_total") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (slug, from_dt, to_dt)
        data = await self._query(gql)
        series = data.get("getMetric", {}).get("timeseriesData", [])
        return self._parse_timeseries(series, "social_volume")

    async def get_dev_activity(self, slug: str = "bitcoin", days_back: int = 30) -> List[MetricResult]:
        """Development activity - GitHub commits per project."""
        from_dt, to_dt = self._time_range(days_back)
        gql = """
        {
          getMetric(metric: "dev_activity") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (slug, from_dt, to_dt)
        data = await self._query(gql)
        series = data.get("getMetric", {}).get("timeseriesData", [])
        return self._parse_timeseries(series, "dev_activity")

    async def get_whale_transactions(self, slug: str = "bitcoin", days_back: int = 7) -> List[MetricResult]:
        """Whale transaction count - transactions over 100K USD."""
        from_dt, to_dt = self._time_range(days_back)
        gql = """
        {
          getMetric(metric: "whale_transaction_count_100k_usd_to_inf") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (slug, from_dt, to_dt)
        data = await self._query(gql)
        series = data.get("getMetric", {}).get("timeseriesData", [])
        return self._parse_timeseries(series, "whale_transactions")

    async def get_token_age_consumed(self, slug: str = "bitcoin", days_back: int = 7) -> List[MetricResult]:
        """Token age consumed - dormant tokens that started moving."""
        from_dt, to_dt = self._time_range(days_back)
        gql = """
        {
          getMetric(metric: "age_consumed") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (slug, from_dt, to_dt)
        data = await self._query(gql)
        series = data.get("getMetric", {}).get("timeseriesData", [])
        return self._parse_timeseries(series, "token_age_consumed")

    async def get_daily_active_deposits(self, slug: str = "bitcoin", days_back: int = 7) -> List[MetricResult]:
        """Daily active deposit addresses on exchanges."""
        from_dt, to_dt = self._time_range(days_back)
        gql = """
        {
          getMetric(metric: "daily_active_deposits") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (slug, from_dt, to_dt)
        data = await self._query(gql)
        series = data.get("getMetric", {}).get("timeseriesData", [])
        return self._parse_timeseries(series, "daily_active_deposits")

    async def get_network_growth(self, slug: str = "bitcoin", days_back: int = 7) -> List[MetricResult]:
        """Network growth - new addresses created per day."""
        from_dt, to_dt = self._time_range(days_back)
        gql = """
        {
          getMetric(metric: "network_growth") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (slug, from_dt, to_dt)
        data = await self._query(gql)
        series = data.get("getMetric", {}).get("timeseriesData", [])
        return self._parse_timeseries(series, "network_growth")

    async def get_exchange_flow(self, slug: str = "bitcoin", days_back: int = 7) -> List[MetricResult]:
        """Exchange inflow and outflow combined."""
        from_dt, to_dt = self._time_range(days_back)
        gql = """
        {
          inflow: getMetric(metric: "exchange_inflow") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
          outflow: getMetric(metric: "exchange_outflow") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (slug, from_dt, to_dt, slug, from_dt, to_dt)
        data = await self._query(gql)
        inflow_data = data.get("inflow", {}).get("timeseriesData", [])
        outflow_data = data.get("outflow", {}).get("timeseriesData", [])
        results = self._parse_timeseries(inflow_data, "exchange_inflow")
        results.extend(self._parse_timeseries(outflow_data, "exchange_outflow"))
        return results

    async def get_mvrv_ratio(self, slug: str = "bitcoin", days_back: int = 30) -> List[MetricResult]:
        """MVRV ratio - Market Value to Realized Value."""
        from_dt, to_dt = self._time_range(days_back)
        gql = """
        {
          getMetric(metric: "mvrv_usd") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (slug, from_dt, to_dt)
        data = await self._query(gql)
        series = data.get("getMetric", {}).get("timeseriesData", [])
        return self._parse_timeseries(series, "mvrv_ratio")

    async def get_all_metrics(self, slug: str = "bitcoin") -> List[MetricResult]:
        """Fetch all available metrics for an asset."""
        import asyncio
        tasks = [
            self.get_social_volume(slug),
            self.get_dev_activity(slug),
            self.get_whale_transactions(slug),
            self.get_token_age_consumed(slug),
            self.get_network_growth(slug),
            self.get_mvrv_ratio(slug),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for r in results:
            if isinstance(r, list) and r:
                combined.append(r[-1])  # latest entry
            elif isinstance(r, Exception):
                pass
        return combined
