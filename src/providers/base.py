"""
Base provider class for premium data sources.
Handles API key management, rate limiting, retry logic, caching, and health checks.
"""

import asyncio
import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import yaml

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Standardized metric result returned by all providers."""
    metric_name: str
    value: Any
    timestamp: int
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class TokenBucket:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rate: float, capacity: Optional[float] = None):
        """
        Args:
            rate: tokens per second to refill
            capacity: max tokens (defaults to rate, allowing burst up to 1 second)
        """
        self.rate = rate
        self.capacity = capacity or rate
        self.tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until enough tokens are available, then consume them."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self._last_refill = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)


class ResponseCache:
    """TTL-based in-memory cache for API responses."""

    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def _make_key(self, url: str, params: Optional[Dict] = None) -> str:
        raw = url + (str(sorted(params.items())) if params else "")
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        key = self._make_key(url, params)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() > entry["expires"]:
            del self._cache[key]
            return None
        return entry["data"]

    def set(self, url: str, data: Any, params: Optional[Dict] = None, ttl: Optional[int] = None) -> None:
        key = self._make_key(url, params)
        self._cache[key] = {
            "data": data,
            "expires": time.time() + (ttl or self.default_ttl),
        }

    def clear(self) -> None:
        self._cache.clear()

    def remove_expired(self) -> int:
        now = time.time()
        expired = [k for k, v in self._cache.items() if now > v["expires"]]
        for k in expired:
            del self._cache[k]
        return len(expired)


class BaseProvider(ABC):
    """
    Abstract base class for all data providers.

    Subclasses must implement:
        - name (property)
        - base_url (property)
        - _build_headers() - returns auth headers
        - At least one data-fetching method
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_key = self._resolve_api_key()
        self.enabled = self.config.get("enabled", True)
        rate_per_minute = self.config.get("rate_limit", 10)
        self._rate_limiter = TokenBucket(rate=rate_per_minute / 60.0)
        self._cache = ResponseCache(default_ttl=self.config.get("cache_ttl", 300))
        self._session: Optional[aiohttp.ClientSession] = None
        self._healthy = True
        self._last_error: Optional[str] = None
        self._request_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string."""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL for API requests."""
        ...

    @abstractmethod
    def _build_headers(self) -> Dict[str, str]:
        """Return authentication and content-type headers."""
        ...

    @property
    def requires_api_key(self) -> bool:
        """Override to False for free providers."""
        return True

    def _resolve_api_key(self) -> str:
        """Resolve API key from config, then environment variable."""
        key = self.config.get("api_key", "")
        if key:
            return key
        env_name = f"MEFAI_{self.name.upper()}_API_KEY"
        return os.environ.get(env_name, "")

    def is_configured(self) -> bool:
        """Check if this provider has a valid API key (or doesn't need one)."""
        if not self.requires_api_key:
            return True
        return bool(self.api_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        cache_ttl: Optional[int] = None,
        max_retries: int = 3,
    ) -> Any:
        """
        Make an HTTP request with rate limiting, caching, and retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL or path (appended to base_url if not absolute)
            params: Query parameters
            json_data: JSON body for POST requests
            headers: Additional headers (merged with _build_headers)
            cache_ttl: Cache TTL override in seconds
            max_retries: Maximum retry attempts

        Returns:
            Parsed JSON response
        """
        if not url.startswith("http"):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"

        # Check cache for GET requests
        if method.upper() == "GET":
            cached = self._cache.get(url, params)
            if cached is not None:
                return cached

        merged_headers = self._build_headers()
        if headers:
            merged_headers.update(headers)

        last_exception = None
        for attempt in range(max_retries):
            await self._rate_limiter.acquire()
            self._request_count += 1

            try:
                session = await self._get_session()
                async with session.request(
                    method,
                    url,
                    params=params,
                    json=json_data,
                    headers=merged_headers,
                ) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                        logger.warning(
                            "%s: rate limited, retrying in %ds (attempt %d/%d)",
                            self.name, retry_after, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    resp.raise_for_status()
                    data = await resp.json()

                    # Cache successful GET responses
                    if method.upper() == "GET":
                        self._cache.set(url, data, params, cache_ttl)

                    self._healthy = True
                    self._last_error = None
                    return data

            except aiohttp.ClientResponseError as e:
                self._error_count += 1
                last_exception = e
                if e.status in (400, 401, 403, 404):
                    # Don't retry client errors
                    self._healthy = False
                    self._last_error = f"HTTP {e.status}: {e.message}"
                    logger.error("%s: client error %d - %s", self.name, e.status, e.message)
                    raise
                backoff = min(2 ** attempt, 30)
                logger.warning(
                    "%s: HTTP %d, retrying in %ds (attempt %d/%d)",
                    self.name, e.status, backoff, attempt + 1, max_retries,
                )
                await asyncio.sleep(backoff)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self._error_count += 1
                last_exception = e
                backoff = min(2 ** attempt, 30)
                logger.warning(
                    "%s: connection error (%s), retrying in %ds (attempt %d/%d)",
                    self.name, str(e), backoff, attempt + 1, max_retries,
                )
                await asyncio.sleep(backoff)

        self._healthy = False
        self._last_error = str(last_exception)
        raise last_exception

    async def _get(self, path: str, params: Optional[Dict] = None, **kwargs) -> Any:
        """Convenience method for GET requests."""
        return await self._request("GET", path, params=params, **kwargs)

    async def _post(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        """Convenience method for POST requests."""
        return await self._request("POST", path, json_data=json_data, **kwargs)

    async def health_check(self) -> Dict[str, Any]:
        """Run a health check against the provider."""
        return {
            "provider": self.name,
            "configured": self.is_configured(),
            "enabled": self.enabled,
            "healthy": self._healthy,
            "last_error": self._last_error,
            "requests": self._request_count,
            "errors": self._error_count,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} configured={self.is_configured()} healthy={self._healthy}>"


def load_provider_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load provider configuration from the config YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("providers", {})
    except FileNotFoundError:
        logger.warning("Config file not found at %s, using defaults", config_path)
        return {}
    except yaml.YAMLError as e:
        logger.error("Error parsing config file: %s", e)
        return {}
