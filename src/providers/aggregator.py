"""
Provider aggregator - loads, manages, and queries all configured data providers.
Normalizes data to a common format and handles failures gracefully.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Type

from .base import BaseProvider, MetricResult, load_provider_config
from .glassnode import GlassnodeProvider
from .santiment import SantimentProvider
from .cryptoquant import CryptoQuantProvider
from .nansen import NansenProvider
from .kaiko import KaikoProvider
from .coinmetrics import CoinMetricsProvider
from .messari import MessariProvider
from .lunarcrush import LunarCrushProvider
from .token_terminal import TokenTerminalProvider
from .intotheblock import IntoTheBlockProvider
from .artemis import ArtemisProvider
from .defi_free import DefiLlamaProvider, FearGreedProvider, CoinGeckoFreeProvider

logger = logging.getLogger(__name__)

# Registry mapping config keys to provider classes
PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    "glassnode": GlassnodeProvider,
    "santiment": SantimentProvider,
    "cryptoquant": CryptoQuantProvider,
    "nansen": NansenProvider,
    "kaiko": KaikoProvider,
    "coinmetrics": CoinMetricsProvider,
    "messari": MessariProvider,
    "lunarcrush": LunarCrushProvider,
    "token_terminal": TokenTerminalProvider,
    "intotheblock": IntoTheBlockProvider,
    "artemis": ArtemisProvider,
    "defillama": DefiLlamaProvider,
    "fear_greed": FearGreedProvider,
    "coingecko": CoinGeckoFreeProvider,
}

# Classification of providers by data type
ONCHAIN_PROVIDERS = {"glassnode", "cryptoquant", "coinmetrics", "intotheblock"}
SENTIMENT_PROVIDERS = {"santiment", "lunarcrush", "fear_greed"}
FUNDAMENTAL_PROVIDERS = {"messari", "token_terminal", "artemis", "defillama"}
MARKET_PROVIDERS = {"kaiko", "coingecko", "nansen"}

# Symbol mapping: trading pair to provider-specific identifiers
SYMBOL_MAP = {
    "BTCUSDT": {"slug": "bitcoin", "asset": "BTC", "pair": "btc-usd", "cq_asset": "btc"},
    "ETHUSDT": {"slug": "ethereum", "asset": "ETH", "pair": "eth-usd", "cq_asset": "eth"},
    "BNBUSDT": {"slug": "binancecoin", "asset": "BNB", "pair": "bnb-usd", "cq_asset": "bnb"},
    "SOLUSDT": {"slug": "solana", "asset": "SOL", "pair": "sol-usd", "cq_asset": "sol"},
    "AVAXUSDT": {"slug": "avalanche-2", "asset": "AVAX", "pair": "avax-usd", "cq_asset": "avax"},
    "LINKUSDT": {"slug": "chainlink", "asset": "LINK", "pair": "link-usd", "cq_asset": "link"},
    "DOTUSDT": {"slug": "polkadot", "asset": "DOT", "pair": "dot-usd", "cq_asset": "dot"},
    "ADAUSDT": {"slug": "cardano", "asset": "ADA", "pair": "ada-usd", "cq_asset": "ada"},
    "MATICUSDT": {"slug": "matic-network", "asset": "MATIC", "pair": "matic-usd", "cq_asset": "matic"},
    "DOGEUSDT": {"slug": "dogecoin", "asset": "DOGE", "pair": "doge-usd", "cq_asset": "doge"},
}


def _resolve_symbol(symbol: str) -> Dict[str, str]:
    """Convert a trading pair to provider-specific identifiers."""
    mapped = SYMBOL_MAP.get(symbol)
    if mapped:
        return mapped
    # Fallback: strip USDT and lowercase
    base = symbol.replace("USDT", "").replace("1000", "").lower()
    return {"slug": base, "asset": base.upper(), "pair": f"{base}-usd", "cq_asset": base}


class ProviderAggregator:
    """
    Central aggregator that manages all data providers.

    Loads providers from config.yaml, checks which have valid API keys,
    fetches data concurrently, and normalizes results.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.providers: Dict[str, BaseProvider] = {}
        self._loaded = False

    def load(self) -> None:
        """Load and initialize all configured providers."""
        provider_configs = load_provider_config(self.config_path)
        for name, cls in PROVIDER_REGISTRY.items():
            cfg = provider_configs.get(name, {})
            if not cfg.get("enabled", True):
                logger.info("Provider %s is disabled in config", name)
                continue
            provider = cls(config=cfg)
            if provider.is_configured():
                self.providers[name] = provider
                logger.info("Loaded provider: %s (configured)", name)
            else:
                logger.info("Skipped provider: %s (no API key)", name)
        self._loaded = True
        logger.info(
            "Aggregator loaded %d/%d providers",
            len(self.providers), len(PROVIDER_REGISTRY),
        )

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Get a specific provider by name."""
        return self.providers.get(name)

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all loaded providers and their status."""
        result = []
        for name, provider in self.providers.items():
            result.append({
                "name": name,
                "configured": provider.is_configured(),
                "enabled": provider.enabled,
                "requires_key": provider.requires_api_key,
            })
        return result

    async def _gather_from_providers(
        self,
        provider_names: set,
        method_name: str,
        kwargs_map: Dict[str, Dict],
    ) -> List[MetricResult]:
        """
        Run a method on multiple providers concurrently.

        Args:
            provider_names: set of provider names to query
            method_name: method to call on each provider (e.g., "get_all_metrics")
            kwargs_map: provider-specific kwargs for each call
        """
        tasks = []
        task_names = []
        for name in provider_names:
            provider = self.providers.get(name)
            if provider is None:
                continue
            method = getattr(provider, method_name, None)
            if method is None:
                continue
            kwargs = kwargs_map.get(name, {})
            tasks.append(method(**kwargs))
            task_names.append(name)

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "Provider %s failed during %s: %s",
                    task_names[i], method_name, result,
                )
                continue
            if isinstance(result, list):
                combined.extend(result)
            elif isinstance(result, MetricResult):
                combined.append(result)
        return combined

    async def get_onchain_metrics(self, symbol: str = "BTCUSDT") -> List[MetricResult]:
        """
        Fetch on-chain metrics from all available on-chain providers.

        Returns normalized MetricResult list from:
        - Glassnode (NUPL, SOPR, exchange flow, miner revenue, etc.)
        - CryptoQuant (exchange reserve, leverage ratio, taker ratio, etc.)
        - Coin Metrics (hashrate, NVT, MVRV, realized cap, etc.)
        - IntoTheBlock (in/out of money, concentration, large txs, etc.)
        """
        sym = _resolve_symbol(symbol)
        kwargs_map = {
            "glassnode": {"asset": sym["asset"]},
            "cryptoquant": {"asset": sym["cq_asset"]},
            "coinmetrics": {"asset": sym["cq_asset"]},
            "intotheblock": {"symbol": sym["slug"]},
        }
        return await self._gather_from_providers(
            ONCHAIN_PROVIDERS, "get_all_metrics", kwargs_map,
        )

    async def get_sentiment(self, symbol: str = "BTCUSDT") -> List[MetricResult]:
        """
        Fetch sentiment metrics from all available sentiment providers.

        Returns normalized MetricResult list from:
        - Santiment (social volume, whale txs, dev activity, etc.)
        - LunarCrush (Galaxy Score, AltRank, social engagement, etc.)
        - Fear and Greed Index
        """
        sym = _resolve_symbol(symbol)
        kwargs_map = {
            "santiment": {"slug": sym["slug"]},
            "lunarcrush": {"symbol": sym["asset"]},
            "fear_greed": {},
        }
        # Fear and Greed only has get_current, not get_all_metrics
        results = []

        # Gather from santiment and lunarcrush
        standard_providers = {"santiment", "lunarcrush"}
        results.extend(
            await self._gather_from_providers(
                standard_providers, "get_all_metrics", kwargs_map,
            )
        )

        # Fear and Greed uses get_current
        fg = self.providers.get("fear_greed")
        if fg:
            try:
                fg_results = await fg.get_current()
                results.extend(fg_results)
            except Exception as e:
                logger.warning("Fear and Greed provider failed: %s", e)

        return results

    async def get_fundamentals(self, symbol: str = "BTCUSDT") -> List[MetricResult]:
        """
        Fetch fundamental metrics from all available fundamental providers.

        Returns normalized MetricResult list from:
        - Messari (asset profile, market data, protocol fundamentals)
        - Token Terminal (revenue, P/E, TVL, active users)
        - Artemis (chain DAA, transactions, fees)
        - DefiLlama (protocol TVL, chain TVL)
        """
        sym = _resolve_symbol(symbol)
        kwargs_map = {
            "messari": {"slug": sym["slug"]},
            "token_terminal": {"project_id": sym["slug"]},
            "artemis": {"chain": sym["slug"]},
            "defillama": {},
        }

        results = []

        # Standard get_all_metrics for messari, token_terminal, artemis
        standard = {"messari", "token_terminal", "artemis"}
        results.extend(
            await self._gather_from_providers(standard, "get_all_metrics", kwargs_map)
        )

        # DefiLlama: fetch chains TVL and protocol TVL
        dl = self.providers.get("defillama")
        if dl:
            try:
                chains = await dl.get_chains_tvl()
                results.extend(chains[:10])  # top 10 chains
            except Exception as e:
                logger.warning("DefiLlama failed: %s", e)

        return results

    async def get_market_data(self, symbol: str = "BTCUSDT") -> List[MetricResult]:
        """
        Fetch market microstructure data from available providers.

        Returns normalized MetricResult list from:
        - Kaiko (VWAP, order book, trades)
        - CoinGecko (price, market cap, volume)
        - Nansen (smart money flow, hot wallets)
        """
        sym = _resolve_symbol(symbol)
        kwargs_map = {
            "kaiko": {"pair": sym["pair"]},
            "coingecko": {},
            "nansen": {"chain": "ethereum"},
        }

        results = []

        # Kaiko
        kaiko = self.providers.get("kaiko")
        if kaiko:
            try:
                r = await kaiko.get_all_metrics(pair=sym["pair"])
                results.extend(r)
            except Exception as e:
                logger.warning("Kaiko failed: %s", e)

        # CoinGecko
        cg = self.providers.get("coingecko")
        if cg:
            try:
                prices = await cg.get_price(ids=sym["slug"])
                results.extend(prices)
                global_data = await cg.get_global()
                results.extend(global_data)
            except Exception as e:
                logger.warning("CoinGecko failed: %s", e)

        # Nansen
        nansen = self.providers.get("nansen")
        if nansen:
            try:
                hot = await nansen.get_hot_wallets()
                results.extend(hot[:5])
            except Exception as e:
                logger.warning("Nansen failed: %s", e)

        return results

    async def get_all(self, symbol: str = "BTCUSDT") -> Dict[str, List[MetricResult]]:
        """
        Fetch all available data for a symbol, organized by category.

        Returns dict with keys: onchain, sentiment, fundamentals, market
        """
        onchain, sentiment, fundamentals, market = await asyncio.gather(
            self.get_onchain_metrics(symbol),
            self.get_sentiment(symbol),
            self.get_fundamentals(symbol),
            self.get_market_data(symbol),
        )
        return {
            "onchain": onchain,
            "sentiment": sentiment,
            "fundamentals": fundamentals,
            "market": market,
        }

    async def health_check_all(self) -> List[Dict[str, Any]]:
        """Run health checks on all loaded providers."""
        tasks = [p.health_check() for p in self.providers.values()]
        return await asyncio.gather(*tasks)

    async def close_all(self) -> None:
        """Close all provider HTTP sessions."""
        for provider in self.providers.values():
            await provider.close()
