"""
Premium data providers for the Mefai Signal Engine.

Supports 11 premium providers (API key required) and 3 free providers.
All implementations make real HTTP calls to actual API endpoints.
"""

from .base import BaseProvider, MetricResult, ResponseCache, TokenBucket, load_provider_config
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
from .aggregator import ProviderAggregator, PROVIDER_REGISTRY

__all__ = [
    # Base
    "BaseProvider",
    "MetricResult",
    "ResponseCache",
    "TokenBucket",
    "load_provider_config",
    # Premium providers
    "GlassnodeProvider",
    "SantimentProvider",
    "CryptoQuantProvider",
    "NansenProvider",
    "KaikoProvider",
    "CoinMetricsProvider",
    "MessariProvider",
    "LunarCrushProvider",
    "TokenTerminalProvider",
    "IntoTheBlockProvider",
    "ArtemisProvider",
    # Free providers
    "DefiLlamaProvider",
    "FearGreedProvider",
    "CoinGeckoFreeProvider",
    # Aggregator
    "ProviderAggregator",
    "PROVIDER_REGISTRY",
]
