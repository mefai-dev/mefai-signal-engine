"""
Data Processing Pipeline.

Coordinates data flow between ingestion, feature computation,
signal generation, and output distribution.

Processing stages:
1. Kline aggregation and validation
2. Feature computation via FeatureEngine
3. Signal generation via Composer
4. Result caching and WebSocket broadcast
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable

import pandas as pd

from src.pipeline.ingestion import DataStore, KlineIngester, FuturesDataIngester
from src.signals.composer import SignalComposer

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Central processing pipeline that orchestrates signal generation.

    Lifecycle:
    1. Initialization: load config, create data store, set up components
    2. Warm-up: fetch historical data for all symbols
    3. Processing loop: generate signals on schedule
    4. Output: cache results and notify subscribers
    """

    def __init__(
        self,
        composer: SignalComposer,
        store: DataStore,
        kline_ingester: KlineIngester,
        futures_ingester: FuturesDataIngester,
        symbols: List[str],
        primary_timeframe: str = "1h",
        signal_timeframes: List[str] = None,
    ):
        self.composer = composer
        self.store = store
        self.kline_ingester = kline_ingester
        self.futures_ingester = futures_ingester
        self.symbols = symbols
        self.primary_timeframe = primary_timeframe
        self.signal_timeframes = signal_timeframes or ["5m", "15m", "1h"]

        # Latest signals cache
        self._latest_signals: Dict[str, Dict] = {}
        self._signal_subscribers: List[Callable] = []
        self._processing = False

    def subscribe(self, callback: Callable):
        """Register a callback for new signal notifications."""
        self._signal_subscribers.append(callback)

    async def warm_up(self, limit: int = 500):
        """
        Fetch historical data for all symbols and timeframes.
        Must be called before processing starts.
        """
        logger.info("Warming up data store: %d symbols", len(self.symbols))

        all_timeframes = list(set(self.signal_timeframes + [self.primary_timeframe]))
        await self.kline_ingester.fetch_all_symbols(
            self.symbols, all_timeframes, limit
        )

        # Verify data availability
        ready_count = 0
        for symbol in self.symbols:
            df = self.store.get(symbol, self.primary_timeframe)
            if df is not None and len(df) >= 50:
                ready_count += 1
            else:
                logger.warning(
                    "Insufficient data for %s %s: %d rows",
                    symbol, self.primary_timeframe,
                    len(df) if df is not None else 0,
                )

        logger.info("Warm-up complete: %d/%d symbols ready", ready_count, len(self.symbols))
        return ready_count

    async def process_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Generate signal for a single symbol using all available data.

        Fetches fresh kline data, runs the composer, and caches the result.
        """
        # Refresh data for this symbol
        df = await self.kline_ingester.fetch_historical(
            symbol, self.primary_timeframe, limit=500
        )

        if df is None or df.empty:
            # Fall back to cached data
            df = self.store.get(symbol, self.primary_timeframe)

        if df is None or len(df) < 50:
            logger.warning("Skipping %s: insufficient data (%d rows)", symbol, len(df) if df is not None else 0)
            return None

        try:
            signal = await self.composer.generate_signal(symbol, df)

            # Cache the signal
            self._latest_signals[symbol] = signal

            # Notify subscribers
            for callback in self._signal_subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(signal)
                    else:
                        callback(signal)
                except Exception as e:
                    logger.error("Signal subscriber error: %s", e)

            return signal

        except Exception as e:
            logger.error("Signal generation failed for %s: %s", symbol, e, exc_info=True)
            return None

    async def process_all_symbols(self) -> Dict[str, Dict]:
        """
        Generate signals for all tracked symbols.

        Processes sequentially to avoid API rate limits, with a small
        pause between symbols.
        """
        if self._processing:
            logger.warning("Signal processing already in progress, skipping")
            return self._latest_signals

        self._processing = True
        start = datetime.now(timezone.utc)
        results = {}

        try:
            for symbol in self.symbols:
                signal = await self.process_symbol(symbol)
                if signal:
                    results[symbol] = signal
                await asyncio.sleep(0.2)  # Rate limit buffer

        finally:
            self._processing = False

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            "Processed %d/%d symbols in %.1fs",
            len(results), len(self.symbols), elapsed,
        )

        return results

    async def quick_price_check(self, symbol: str) -> Optional[Dict]:
        """
        Quick price update without full signal generation.

        Used for 1-minute updates between full signal cycles.
        Returns basic price info and checks for significant moves.
        """
        df = await self.kline_ingester.fetch_historical(symbol, "1m", limit=5)
        if df is None or df.empty:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        price_change_pct = 0.0
        if prev["close"] > 0:
            price_change_pct = (latest["close"] - prev["close"]) / prev["close"] * 100

        result = {
            "symbol": symbol,
            "price": float(latest["close"]),
            "change_pct": float(price_change_pct),
            "volume": float(latest["volume"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Update on-chain analyzer price cache
        self.composer.onchain.update_price(symbol, latest["close"], price_change_pct)

        # Check for significant moves that warrant immediate signal regeneration
        if abs(price_change_pct) > 2.0:
            logger.info(
                "Significant move detected for %s: %.2f%% - triggering signal refresh",
                symbol, price_change_pct,
            )
            await self.process_symbol(symbol)
            result["signal_refreshed"] = True

        return result

    def get_latest_signals(self) -> Dict[str, Dict]:
        """Return all cached signals."""
        return self._latest_signals.copy()

    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Return cached signal for a specific symbol."""
        return self._latest_signals.get(symbol)

    def get_status(self) -> Dict:
        """Return processor status."""
        return {
            "symbols_tracked": len(self.symbols),
            "signals_cached": len(self._latest_signals),
            "processing": self._processing,
            "data_store": self.store.status(),
            "subscribers": len(self._signal_subscribers),
        }
