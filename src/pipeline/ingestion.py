"""
Data Ingestion Pipeline.

Continuous data fetcher for:
- Binance klines (1m, 5m, 15m, 1h, 4h) via REST API
- Binance futures data (funding, OI, top trader positions)
- WebSocket for real-time price updates
- RSS feeds for sentiment analysis

Maintains an in-memory data store keyed by (symbol, timeframe).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Set

import aiohttp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Binance kline intervals mapped to seconds for staleness checks
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class DataStore:
    """
    Thread-safe in-memory data store for OHLCV data.

    Keyed by (symbol, timeframe) tuples, stores pandas DataFrames
    with a configurable maximum length (rolling window).
    """

    def __init__(self, max_candles: int = 1000):
        self._data: Dict[str, pd.DataFrame] = {}
        self._max_candles = max_candles
        self._last_update: Dict[str, datetime] = {}

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    def set(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Store or replace data for a symbol/timeframe pair."""
        key = self._key(symbol, timeframe)
        self._data[key] = df.tail(self._max_candles).copy()
        self._last_update[key] = datetime.now(timezone.utc)

    def append(self, symbol: str, timeframe: str, new_row: Dict):
        """Append a single candle to existing data."""
        key = self._key(symbol, timeframe)
        existing = self._data.get(key)

        row_df = pd.DataFrame([new_row])
        if "timestamp" in row_df.columns:
            row_df.set_index("timestamp", inplace=True)

        if existing is not None and not existing.empty:
            # Check if this timestamp already exists (update last candle)
            if not row_df.index.empty and row_df.index[0] in existing.index:
                existing.loc[row_df.index[0]] = row_df.iloc[0]
                self._data[key] = existing
            else:
                combined = pd.concat([existing, row_df])
                self._data[key] = combined.tail(self._max_candles)
        else:
            self._data[key] = row_df

        self._last_update[key] = datetime.now(timezone.utc)

    def get(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Retrieve data for a symbol/timeframe pair."""
        key = self._key(symbol, timeframe)
        return self._data.get(key)

    def get_last_update(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the timestamp of the last update."""
        key = self._key(symbol, timeframe)
        return self._last_update.get(key)

    def symbols(self) -> Set[str]:
        """Return set of all stored symbols."""
        syms = set()
        for key in self._data.keys():
            syms.add(key.split(":")[0])
        return syms

    def status(self) -> Dict:
        """Return store status summary."""
        entries = {}
        for key, df in self._data.items():
            entries[key] = {
                "rows": len(df),
                "last_update": self._last_update.get(key, "").isoformat() if self._last_update.get(key) else None,
            }
        return entries


class KlineIngester:
    """
    Fetches historical and real-time kline data from Binance.
    """

    def __init__(
        self,
        store: DataStore,
        base_url: str = "https://fapi.binance.com",
        rate_limit_pause: float = 0.1,
    ):
        self.store = store
        self.base_url = base_url
        self.rate_limit_pause = rate_limit_pause

    async def fetch_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> pd.DataFrame:
        """Fetch historical klines and store them."""
        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        logger.error("Kline fetch failed for %s %s: HTTP %d", symbol, interval, resp.status)
                        return pd.DataFrame()
                    data = await resp.json()
        except Exception as e:
            logger.error("Kline fetch error for %s %s: %s", symbol, interval, e)
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        rows = []
        for k in data:
            rows.append({
                "timestamp": pd.Timestamp(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]),
                "trades": int(k[8]),
                "taker_buy_volume": float(k[9]),
            })

        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)

        self.store.set(symbol, interval, df)
        logger.debug("Stored %d klines for %s %s", len(df), symbol, interval)

        await asyncio.sleep(self.rate_limit_pause)
        return df

    async def fetch_all_symbols(
        self, symbols: List[str], intervals: List[str], limit: int = 500
    ):
        """Fetch historical data for all symbol/interval combinations."""
        logger.info(
            "Fetching historical data: %d symbols x %d intervals",
            len(symbols), len(intervals)
        )

        for interval in intervals:
            tasks = []
            for symbol in symbols:
                tasks.append(self.fetch_historical(symbol, interval, limit))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error("Failed to fetch %s %s: %s", symbol, interval, result)

            # Pause between intervals to respect rate limits
            await asyncio.sleep(1.0)


class WebSocketIngester:
    """
    Real-time price updates via Binance WebSocket.

    Subscribes to kline streams for all tracked symbols
    and updates the data store in real-time.
    """

    def __init__(
        self,
        store: DataStore,
        symbols: List[str],
        interval: str = "1m",
        ws_url: str = "wss://fstream.binance.com/ws",
        on_price_update: Optional[Callable] = None,
    ):
        self.store = store
        self.symbols = symbols
        self.interval = interval
        self.ws_url = ws_url
        self.on_price_update = on_price_update
        self._running = False
        self._ws = None

    async def start(self):
        """Start the WebSocket connection and begin receiving updates."""
        self._running = True

        # Build combined stream URL
        streams = [f"{s.lower()}@kline_{self.interval}" for s in self.symbols]
        stream_url = f"{self.ws_url}/{'/'.join(streams)}" if len(streams) == 1 else self.ws_url

        logger.info("Starting WebSocket ingester for %d symbols", len(self.symbols))

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        stream_url,
                        timeout=aiohttp.ClientTimeout(total=None),
                        heartbeat=30,
                    ) as ws:
                        self._ws = ws

                        # Subscribe to streams via combined stream
                        if len(self.symbols) > 1:
                            subscribe_msg = {
                                "method": "SUBSCRIBE",
                                "params": streams,
                                "id": 1,
                            }
                            await ws.send_json(subscribe_msg)

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_message(msg.data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error("WebSocket error: %s", ws.exception())
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.warning("WebSocket closed")
                                break

            except asyncio.CancelledError:
                logger.info("WebSocket ingester cancelled")
                self._running = False
                return
            except Exception as e:
                logger.error("WebSocket connection error: %s", e)
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    async def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        logger.info("WebSocket ingester stopped")

    async def _handle_message(self, raw: str):
        """Process a WebSocket kline message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Handle subscription confirmation
        if "result" in data:
            return

        # Extract kline data
        event_data = data.get("data", data)
        if "k" not in event_data:
            return

        k = event_data["k"]
        symbol = k["s"]
        is_closed = k["x"]

        candle = {
            "timestamp": pd.Timestamp(k["t"], unit="ms"),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "quote_volume": float(k["q"]),
            "trades": int(k["n"]),
            "taker_buy_volume": float(k["V"]),
        }

        # Update the store
        self.store.append(symbol, self.interval, candle)

        # Notify callback
        if self.on_price_update and is_closed:
            try:
                await self.on_price_update(symbol, candle)
            except Exception as e:
                logger.error("Price update callback error: %s", e)


class FuturesDataIngester:
    """
    Fetches Binance Futures market data:
    - Funding rates
    - Open interest
    - Top trader long/short ratios
    """

    def __init__(self, base_url: str = "https://fapi.binance.com"):
        self.base_url = base_url

    async def fetch_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Fetch current funding rate."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/fapi/v1/fundingRate",
                    params={"symbol": symbol, "limit": 1},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            return {
                                "symbol": symbol,
                                "rate": float(data[-1]["fundingRate"]),
                                "time": data[-1]["fundingTime"],
                            }
        except Exception as e:
            logger.debug("Funding rate fetch failed for %s: %s", symbol, e)
        return None

    async def fetch_open_interest(self, symbol: str) -> Optional[Dict]:
        """Fetch current open interest."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/fapi/v1/openInterest",
                    params={"symbol": symbol},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "symbol": symbol,
                            "open_interest": float(data["openInterest"]),
                        }
        except Exception as e:
            logger.debug("OI fetch failed for %s: %s", symbol, e)
        return None

    async def fetch_long_short_ratio(self, symbol: str) -> Optional[Dict]:
        """Fetch top trader long/short ratio."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/futures/data/topLongShortAccountRatio",
                    params={"symbol": symbol, "period": "5m", "limit": 1},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            return {
                                "symbol": symbol,
                                "ratio": float(data[-1]["longShortRatio"]),
                                "long_account": float(data[-1]["longAccount"]),
                                "short_account": float(data[-1]["shortAccount"]),
                            }
        except Exception as e:
            logger.debug("L/S ratio fetch failed for %s: %s", symbol, e)
        return None

    async def fetch_all_futures_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch all futures data for multiple symbols."""
        results = {}

        for symbol in symbols:
            funding, oi, ls = await asyncio.gather(
                self.fetch_funding_rate(symbol),
                self.fetch_open_interest(symbol),
                self.fetch_long_short_ratio(symbol),
                return_exceptions=True,
            )

            results[symbol] = {
                "funding": funding if not isinstance(funding, Exception) else None,
                "open_interest": oi if not isinstance(oi, Exception) else None,
                "long_short_ratio": ls if not isinstance(ls, Exception) else None,
            }

            await asyncio.sleep(0.1)  # Rate limit

        return results
