"""
Training Pipeline - Fetches historical data and trains all three XGBoost models.

Orchestrates the full training cycle:
1. Fetch historical klines from Binance for each symbol category
2. Compute features via FeatureEngine
3. Train majors, alts, and memes models
4. Evaluate and log results
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import aiohttp
import yaml

from src.ml.majors_model import MajorsModel, MAJORS_SYMBOLS
from src.ml.alts_model import AltsModel, ALTS_SYMBOLS
from src.ml.memes_model import MemesModel, MEME_SYMBOLS

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config file not found at %s, using defaults", config_path)
        return {}


async def fetch_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
    base_url: str = "https://fapi.binance.com",
) -> pd.DataFrame:
    """
    Fetch historical klines from Binance Futures API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles (max 1500)
        base_url: Binance API base URL

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    url = f"{base_url}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1500),
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error("Binance API error for %s: %s - %s", symbol, resp.status, text)
                return pd.DataFrame()

            data = await resp.json()

    if not data:
        return pd.DataFrame()

    # Binance kline format:
    # [open_time, open, high, low, close, volume, close_time,
    #  quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
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
    return df


async def fetch_multi_symbol_klines(
    symbols: List[str],
    interval: str = "1h",
    limit: int = 500,
    base_url: str = "https://fapi.binance.com",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch klines for multiple symbols concurrently.

    Returns:
        Dict mapping symbol to DataFrame.
    """
    tasks = [
        fetch_klines(symbol, interval, limit, base_url)
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error("Failed to fetch %s: %s", symbol, result)
        elif not result.empty:
            data[symbol] = result
        else:
            logger.warning("Empty data for %s", symbol)

    return data


def concatenate_symbol_data(symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate data from multiple symbols for training.

    Each symbol's data is normalized (percentage returns) so the model
    learns patterns rather than absolute price levels.
    """
    frames = []
    for symbol, df in symbol_data.items():
        normalized = df.copy()

        # Normalize prices relative to first close
        first_close = df["close"].iloc[0]
        if first_close > 0:
            for col in ["open", "high", "low", "close"]:
                normalized[col] = df[col] / first_close * 100

        # Normalize volume relative to mean volume
        mean_vol = df["volume"].mean()
        if mean_vol > 0:
            normalized["volume"] = df["volume"] / mean_vol

        normalized["symbol"] = symbol
        frames.append(normalized)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=False)


class Trainer:
    """
    Orchestrates training for all three model types.

    Training pipeline:
    1. Fetch historical data for each symbol category
    2. Concatenate and normalize data
    3. Train each specialized model
    4. Log metrics and save models
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        model_config = self.config.get("models", {})
        model_dir = model_config.get("save_dir", "models")
        hyperparams = model_config.get("hyperparameters", {})
        horizon = model_config.get("prediction_horizon", 12)

        self.majors_model = MajorsModel(
            model_dir=model_dir,
            params=self._build_params(hyperparams.get("majors", {})),
            prediction_horizon=horizon,
            movement_threshold_pct=0.5,
        )
        self.alts_model = AltsModel(
            model_dir=model_dir,
            params=self._build_params(hyperparams.get("alts", {})),
            prediction_horizon=horizon,
            movement_threshold_pct=0.8,
        )
        self.memes_model = MemesModel(
            model_dir=model_dir,
            params=self._build_params(hyperparams.get("memes", {})),
            prediction_horizon=horizon,
            movement_threshold_pct=1.5,
        )

        binance_config = self.config.get("binance", {})
        self.base_url = binance_config.get("base_url", "https://fapi.binance.com")
        self.kline_limit = model_config.get("lookback_periods", 500)

        symbols_config = self.config.get("symbols", {})
        self.majors_symbols = symbols_config.get("majors", MAJORS_SYMBOLS)
        self.alts_symbols = symbols_config.get("alts", ALTS_SYMBOLS)
        self.memes_symbols = symbols_config.get("memes", MEME_SYMBOLS)

    @staticmethod
    def _build_params(custom: Dict) -> Optional[Dict]:
        """Build XGBoost params from config, or return None for defaults."""
        if not custom:
            return None
        params = custom.copy()
        params.update({
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        })
        return params

    async def train_all(self, interval: str = "1h") -> Dict:
        """
        Run the complete training pipeline for all three models.

        Args:
            interval: Candle interval to train on.

        Returns:
            Dictionary with training results for each model.
        """
        logger.info("Starting full training pipeline (interval=%s, limit=%d)", interval, self.kline_limit)
        start_time = datetime.utcnow()
        results = {}

        # Fetch data for all symbol categories concurrently
        majors_data, alts_data, memes_data = await asyncio.gather(
            fetch_multi_symbol_klines(self.majors_symbols, interval, self.kline_limit, self.base_url),
            fetch_multi_symbol_klines(self.alts_symbols, interval, self.kline_limit, self.base_url),
            fetch_multi_symbol_klines(self.memes_symbols, interval, self.kline_limit, self.base_url),
        )

        # Train majors model
        if majors_data:
            majors_df = concatenate_symbol_data(majors_data)
            if len(majors_df) >= 200:
                results["majors"] = self.majors_model.train(majors_df)
            else:
                results["majors"] = {"error": "insufficient_data", "rows": len(majors_df)}
        else:
            results["majors"] = {"error": "no_data_fetched"}

        # Train alts model
        if alts_data:
            alts_df = concatenate_symbol_data(alts_data)
            if len(alts_df) >= 200:
                results["alts"] = self.alts_model.train(alts_df)
            else:
                results["alts"] = {"error": "insufficient_data", "rows": len(alts_df)}
        else:
            results["alts"] = {"error": "no_data_fetched"}

        # Train memes model
        if memes_data:
            memes_df = concatenate_symbol_data(memes_data)
            if len(memes_df) >= 200:
                results["memes"] = self.memes_model.train(memes_df)
            else:
                results["memes"] = {"error": "insufficient_data", "rows": len(memes_df)}
        else:
            results["memes"] = {"error": "no_data_fetched"}

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        results["training_duration_seconds"] = elapsed
        results["timestamp"] = datetime.utcnow().isoformat()

        logger.info("Training pipeline complete in %.1fs", elapsed)
        return results

    def get_model_for_symbol(self, symbol: str):
        """Return the appropriate model for a given symbol."""
        if symbol in self.majors_symbols:
            return self.majors_model
        elif symbol in self.alts_symbols:
            return self.alts_model
        elif symbol in self.memes_symbols:
            return self.memes_model
        else:
            # Default to alts model for unknown symbols
            return self.alts_model

    def get_all_status(self) -> Dict:
        """Return status of all models."""
        return {
            "majors": self.majors_model.get_status(),
            "alts": self.alts_model.get_status(),
            "memes": self.memes_model.get_status(),
        }
