"""
FastAPI Server - REST API and WebSocket signal stream.

Endpoints:
- GET  /signals          - Latest signals for all tracked pairs
- GET  /signals/{symbol} - Signal for a specific pair
- GET  /models/status    - Model training status and metrics
- GET  /health           - System health check
- WS   /ws/signals       - Real-time signal stream via WebSocket

Port: 8300 (configurable via config.yaml)
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.ingestion import DataStore, KlineIngester, FuturesDataIngester
from src.pipeline.processor import SignalProcessor
from src.pipeline.scheduler import EngineScheduler
from src.signals.composer import SignalComposer
from src.ml.trainer import Trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------

class SignalResponse(BaseModel):
    symbol: str
    direction: str
    confidence: float
    composite_score: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    current_price: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    symbols_tracked: int
    signals_cached: int
    scheduler_running: bool
    models: Dict
    timestamp: str


class ModelStatusResponse(BaseModel):
    majors: Dict
    alts: Dict
    memes: Dict


# ---------------------------------------------------------------
# Application state
# ---------------------------------------------------------------

class AppState:
    """Global application state shared across all components."""

    def __init__(self):
        self.config: Dict = {}
        self.store: Optional[DataStore] = None
        self.trainer: Optional[Trainer] = None
        self.composer: Optional[SignalComposer] = None
        self.processor: Optional[SignalProcessor] = None
        self.scheduler: Optional[EngineScheduler] = None
        self.start_time: datetime = datetime.now(timezone.utc)
        self.ws_clients: Set[WebSocket] = set()


state = AppState()


def load_config(path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    config_path = os.environ.get("SIGNAL_ENGINE_CONFIG", path)
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config file %s not found, using defaults", config_path)
        return {}


# ---------------------------------------------------------------
# Lifespan management
# ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # --- STARTUP ---
    logger.info("Starting Mefai Signal Engine...")

    state.config = load_config()
    state.start_time = datetime.now(timezone.utc)

    # Initialize data store
    state.store = DataStore(max_candles=1000)

    # Get all symbols from config
    symbols_config = state.config.get("symbols", {})
    all_symbols = (
        symbols_config.get("majors", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        + symbols_config.get("alts", ["SOLUSDT", "AVAXUSDT", "LINKUSDT"])
        + symbols_config.get("memes", ["DOGEUSDT", "1000PEPEUSDT"])
    )

    binance_config = state.config.get("binance", {})
    base_url = binance_config.get("base_url", "https://fapi.binance.com")

    # Initialize components
    kline_ingester = KlineIngester(
        store=state.store,
        base_url=base_url,
        rate_limit_pause=binance_config.get("rate_limit_pause", 0.1),
    )
    futures_ingester = FuturesDataIngester(base_url=base_url)

    state.trainer = Trainer(config_path=os.environ.get("SIGNAL_ENGINE_CONFIG", "config.yaml"))

    state.composer = SignalComposer(
        config=state.config,
        trainer=state.trainer,
    )

    timeframes = state.config.get("timeframes", ["1m", "5m", "15m", "1h", "4h"])
    state.processor = SignalProcessor(
        composer=state.composer,
        store=state.store,
        kline_ingester=kline_ingester,
        futures_ingester=futures_ingester,
        symbols=all_symbols,
        primary_timeframe="1h",
        signal_timeframes=timeframes,
    )

    # Register WebSocket broadcast as signal subscriber
    state.processor.subscribe(broadcast_signal)

    state.scheduler = EngineScheduler(
        processor=state.processor,
        trainer=state.trainer,
        sentiment=state.composer.sentiment,
        config=state.config,
    )

    # Start the scheduler (runs warm-up, initial signals, then schedules tasks)
    asyncio.create_task(state.scheduler.start())

    logger.info("Signal engine started with %d symbols", len(all_symbols))
    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down signal engine...")
    if state.scheduler:
        await state.scheduler.stop()

    # Close all WebSocket connections
    for ws in list(state.ws_clients):
        try:
            await ws.close()
        except Exception:
            pass

    logger.info("Signal engine stopped")


# ---------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------

app = FastAPI(
    title="Mefai Signal Engine",
    description="Production-grade AI trading signal engine with XGBoost ensemble, multi-layer signal composition, and real-time data pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# WebSocket signal broadcast
# ---------------------------------------------------------------

async def broadcast_signal(signal: Dict):
    """Broadcast a new signal to all connected WebSocket clients."""
    if not state.ws_clients:
        return

    message = json.dumps(signal, default=str)
    disconnected = set()

    for ws in state.ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)

    state.ws_clients -= disconnected


# ---------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------

@app.get("/signals", response_model=Dict[str, SignalResponse])
async def get_all_signals():
    """
    Get latest signals for all tracked trading pairs.

    Returns a dictionary keyed by symbol, each containing the
    latest signal with direction, confidence, and price levels.
    """
    if not state.processor:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    signals = state.processor.get_latest_signals()

    if not signals:
        return {}

    # Format response
    result = {}
    for symbol, sig in signals.items():
        result[symbol] = SignalResponse(
            symbol=sig["symbol"],
            direction=sig["direction"],
            confidence=sig["confidence"],
            composite_score=sig["composite_score"],
            entry_price=sig["entry_price"],
            stop_loss=sig["stop_loss"],
            take_profit=sig["take_profit"],
            risk_reward_ratio=sig["risk_reward_ratio"],
            current_price=sig["current_price"],
            timestamp=sig["timestamp"],
        )

    return result


@app.get("/signals/{symbol}")
async def get_signal(symbol: str, full: bool = Query(False, description="Include full layer breakdown")):
    """
    Get signal for a specific trading pair.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        full: If true, include detailed layer-by-layer breakdown
    """
    if not state.processor:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    signal = state.processor.get_signal(symbol.upper())
    if not signal:
        raise HTTPException(status_code=404, detail=f"No signal available for {symbol}")

    if not full:
        # Return simplified response
        return SignalResponse(
            symbol=signal["symbol"],
            direction=signal["direction"],
            confidence=signal["confidence"],
            composite_score=signal["composite_score"],
            entry_price=signal["entry_price"],
            stop_loss=signal["stop_loss"],
            take_profit=signal["take_profit"],
            risk_reward_ratio=signal["risk_reward_ratio"],
            current_price=signal["current_price"],
            timestamp=signal["timestamp"],
        )

    # Return full signal with layer details
    return signal


@app.get("/signals/{symbol}/history")
async def get_signal_history(symbol: str, limit: int = Query(50, le=200)):
    """Get recent signal history for a symbol."""
    if not state.composer:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    history = state.composer.get_signal_history(symbol=symbol.upper(), limit=limit)
    return {"symbol": symbol.upper(), "history": history}


@app.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get training status, last retrain time, and accuracy metrics
    for all three XGBoost models (majors, alts, memes).
    """
    if not state.trainer:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    status = state.trainer.get_all_status()
    return ModelStatusResponse(**status)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check.

    Returns engine status, uptime, symbol count, and model states.
    """
    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()

    model_status = {}
    if state.trainer:
        all_status = state.trainer.get_all_status()
        model_status = {
            name: {"trained": info.get("is_trained", False)}
            for name, info in all_status.items()
        }

    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        symbols_tracked=len(state.processor.symbols) if state.processor else 0,
        signals_cached=len(state.processor.get_latest_signals()) if state.processor else 0,
        scheduler_running=state.scheduler._running if state.scheduler else False,
        models=model_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get detailed scheduler status including task execution history."""
    if not state.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    return state.scheduler.get_status()


@app.get("/data/status")
async def get_data_status():
    """Get data store status showing available data per symbol/timeframe."""
    if not state.store:
        raise HTTPException(status_code=503, detail="Data store not initialized")

    return state.store.status()


@app.get("/sentiment/status")
async def get_sentiment_status():
    """Get sentiment analysis status and feed health."""
    if not state.composer:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    return state.composer.sentiment.get_status()


@app.post("/train")
async def trigger_training():
    """Manually trigger model retraining."""
    if not state.trainer:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Run training in background
    asyncio.create_task(_run_training())
    return {"status": "training_started", "timestamp": datetime.now(timezone.utc).isoformat()}


async def _run_training():
    """Background training task."""
    try:
        results = await state.trainer.train_all(interval="1h")
        logger.info("Manual training complete: %s", json.dumps(
            {k: v.get("final_accuracy", "n/a") if isinstance(v, dict) else v for k, v in results.items()},
            default=str,
        ))
    except Exception as e:
        logger.error("Manual training failed: %s", e)


# ---------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------

@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    Real-time signal stream via WebSocket.

    Clients receive new signals as they are generated.
    On connection, sends the latest cached signals as initial state.
    """
    await websocket.accept()
    state.ws_clients.add(websocket)
    logger.info("WebSocket client connected (%d total)", len(state.ws_clients))

    try:
        # Send current signals on connect
        if state.processor:
            current = state.processor.get_latest_signals()
            if current:
                await websocket.send_text(json.dumps({
                    "type": "initial",
                    "signals": current,
                }, default=str))

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)

                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
                elif data.startswith("subscribe:"):
                    # Client can subscribe to specific symbols
                    symbol = data.split(":")[1].upper()
                    signal = state.processor.get_signal(symbol) if state.processor else None
                    if signal:
                        await websocket.send_text(json.dumps({
                            "type": "signal",
                            "data": signal,
                        }, default=str))

            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket error: %s", e)
    finally:
        state.ws_clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(state.ws_clients))


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main():
    """Run the signal engine server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config()
    server_config = config.get("server", {})

    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8300)

    logger.info("Starting signal engine on %s:%d", host, port)

    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
