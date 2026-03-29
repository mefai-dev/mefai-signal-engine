# Mefai Signal Engine

[![CI](https://github.com/mefai-dev/mefai-signal-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/mefai-dev/mefai-signal-engine/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)


A production grade trading signal engine that combines machine learning, on chain analytics, and multi source data fusion to generate actionable trading signals for cryptocurrency markets.

---

## Architecture Overview

The engine is built on three pillars of intelligence, each contributing to the final signal output.

```
                        +------------------+
                        |  Signal Output   |
                        |  Direction       |
                        |  Confidence      |
                        |  Position Size   |
                        |  SL / TP / RR    |
                        +--------+---------+
                                 |
                    +------------+------------+
                    |    Meta Ensemble        |
                    |    Dynamic Weights      |
                    |    Platt Calibration    |
                    |    Disagreement Check   |
                    +--+-------+----------+--+
                       |       |          |
          +------------+   +---+---+   +--+-----------+
          |                |       |                   |
  +-------v-------+ +-----v-----+ +---v-----------+   |
  | XGBoost       | | Transform | | RL Position   |   |
  | 3 Specialized | | LSTM+Attn | | Sizer (DQN)   |   |
  | Models        | | Predictor | | 5 Actions     |   |
  +-------+-------+ +-----+-----+ +---+-----------+   |
          |                |           |               |
  +-------v----------------v-----------v-------+       |
  |         Feature Engine (15+ indicators)    |       |
  |  RSI, MACD, BB, ATR, ADX, EMA, SMA, OBV   |       |
  |  Order Blocks, Fair Value Gaps, VWAP       |       |
  +---------------------+---------------------+       |
                        |                              |
          +-------------v--------------+    +----------v---------+
          |    5-Layer Signal          |    | Market Regime      |
          |    Composition             |    | Detection (HMM)    |
          |                            |    | 4 States:          |
          |  Technical (30-45%)        |    |  Bull / Bear /     |
          |  Correlation (10-15%)      |    |  Sideways / Volatile|
          |  On Chain (15-25%)         |    +--------------------+
          |  Sentiment (10-15%)        |
          |  ML Predictions (35%)      |    +--------------------+
          +-------------+--------------+    | Monte Carlo Risk   |
                        |                   | 10K Simulations    |
          +-------------v--------------+    | VaR / CVaR / MDD   |
          |    Data Pipeline           |    +--------------------+
          |                            |
          |  Binance (klines, futures) |    +--------------------+
          |  12 Premium Providers      |    | Walk Forward       |
          |  3 Free Providers          |    | Optimization       |
          |  12+ RSS News Feeds        |    | Overfitting Check  |
          +----------------------------+    +--------------------+
```

---

## Pillar 1: XGBoost Ensemble Engine

Three specialized gradient-boosted tree models, each tuned for a different market segment:

| Model | Symbols | Estimators | Movement Threshold | Key Features |
|-------|---------|------------|-------------------|--------------|
| **Majors** | BTC, ETH, BNB | 200 | 0.5% | Conservative, lower learning rate (0.03), deeper trees (max_depth=8) |
| **Alts** | SOL, AVAX, LINK, DOT, MATIC, NEAR, ARB, OP | 200 | 0.8% | Momentum heavy, balanced depth (7), moderate learning rate (0.05) |
| **Memes** | PEPE, DOGE, SHIB, FLOKI, BONK, WIF | 200 | 1.5% | Volume/sentiment weighted, shallow trees (6), higher learning rate (0.08), heavy subsampling |

**Feature Engine** extracts 15+ technical indicators from raw OHLCV data:

- Momentum: RSI(14), RSI(7), MACD, MACD Signal, MACD Histogram
- Trend: EMA(9), EMA(21), SMA(50), SMA(200), ADX(14)
- Volatility: Bollinger Bands (upper/mid/lower), ATR(14)
- Volume: OBV, VWAP, Volume SMA ratio
- Structure: Order Block detection (swing high/low), Fair Value Gap detection (3-candle imbalance)
- Derived: Price position within BB, EMA crossover signals, multi-timeframe RSI divergence

Models auto retrain every 6 hours on fresh market data.

---

## Pillar 2: Multi-Layer Signal Composition

Five independent scoring layers, each producing a score from -100 to +100:

| Layer | Weight | Data Source | What It Measures |
|-------|--------|-------------|-----------------|
| Technical Analysis | 30-45% | Price/Volume | Trend direction, momentum, support/resistance, pattern recognition |
| Correlation Analysis | 10-15% | Multi asset | BTC beta, sector correlation, cross pair divergence, decorrelation signals |
| On Chain Metrics | 15-25% | Binance + Providers | Funding rate, OI change, long/short ratio, liquidation levels, exchange flows |
| Sentiment | 10-15% | RSS + Social | News sentiment from 12+ sources, keyword scoring, source credibility weighting |
| ML Predictions | 35% | XGBoost + Transformer | Ensemble prediction with confidence, regime adjusted |

The composer combines all layers with dynamic weighting and outputs:

- **Direction**: LONG / SHORT / NEUTRAL
- **Confidence**: 0-100%
- **Entry, Stop Loss, Take Profit** levels
- **Risk/Reward Ratio**

---

## Pillar 3: Real-Time Data Pipeline

### Binance (Free, built-in)
- Klines: 1m, 5m, 15m, 1h, 4h via REST and WebSocket
- Futures: funding rate, open interest, top trader positions, liquidations
- Real-time price via WebSocket streams

### Premium Data Providers (12 sources, user provides own API keys)

| Provider | Data Type | Key Metrics | Pricing |
|----------|-----------|-------------|---------|
| **Glassnode** | On chain | NUPL, SOPR, exchange flows, HODL waves, MVRV Z Score, Puell Multiple, Reserve Risk | $29-799/mo |
| **Santiment** | Social + On chain | Social volume, dev activity, whale tx, token age, MVRV | $49-250/mo |
| **CryptoQuant** | Exchange analytics | Exchange reserve, inflow/outflow, miner flows, SSR, leverage ratio | $29-799/mo |
| **Nansen** | Smart money | Wallet labels, smart money flow, token god mode, bridge flow | $100-2500/mo |
| **Kaiko** | Institutional market data | Order book depth, VWAP, slippage, tick data, cross-exchange spread | Custom |
| **Coin Metrics** | Network fundamentals | Hashrate, NVT, realized cap, thermocap, supply in profit | $99-999/mo |
| **Messari** | Fundamentals | Token unlocks, governance, protocol revenue, asset profiles | $29-500/mo |
| **LunarCrush** | Social intelligence | Galaxy Score, AltRank, influencer tracking, spam detection | $49-299/mo |
| **Token Terminal** | Protocol finance | Revenue, P/E ratio, P/S ratio, TVL, active users | $325/mo |
| **IntoTheBlock** | Address analytics | In/out of money, concentration, large tx, bid/ask imbalance | $10-100/mo |
| **Artemis** | Ecosystem data | Chain activity, developer commits, fee revenue, ecosystem comparison | $100/mo |
| **The TIE** | NLP sentiment | Social media scoring, news sentiment, entity extraction | Enterprise |

### Free Data Sources (no API key needed)

| Provider | Data |
|----------|------|
| **DefiLlama** | TVL by protocol and chain, stablecoin flows, yield data |
| **Alternative.me** | Fear and Greed Index |
| **CoinGecko** | Market data, trending coins, global stats |

All providers are configured in `config.yaml`. The aggregator fetches from all enabled providers concurrently and normalizes data to a common format. The engine works without premium providers but accuracy improves significantly with more data sources.

---

## Advanced Models

### Transformer Price Predictor (LSTM + Multi-Head Attention)
- 4-layer transformer with 8 attention heads
- LSTM input embedding for sequential processing
- 96-candle lookback window, predicts next 12 candles
- AdamW optimizer with cosine annealing learning rate
- MC Dropout (50 forward passes) for uncertainty estimation
- Early stopping with patience=10

### Hidden Markov Model Regime Detector
- 4 market regimes: Bull Trend, Bear Trend, Sideways, High Volatility
- Features: log returns, rolling volatility, volume ratio, RSI
- Transition probability matrix for regime change prediction
- Position size multipliers per regime (e.g., reduce size in high volatility)

### Reinforcement Learning Position Sizer (Double DQN)
- Dueling architecture with 7-dimensional state vector
- 5 discrete actions: 0%, 25%, 50%, 75%, 100% position size
- Risk-adjusted reward function (Sharpe-like with drawdown penalty)
- 100K experience replay buffer, soft target updates
- Epsilon-greedy exploration (1.0 to 0.01 over 10K steps)

### Monte Carlo Risk Engine
- 10,000 Geometric Brownian Motion simulation paths
- Merton Jump Diffusion for tail risk modeling
- VaR (95%, 99%), CVaR (Expected Shortfall), max drawdown distribution
- Portfolio simulation with Cholesky decomposition for correlated assets
- Stress testing at 2x, 3x, 5x volatility multipliers

### Walk Forward Optimization
- Anchored and rolling window modes
- Per-window metrics: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
- Statistical significance testing (t-test on out-of-sample returns)
- Overfitting detection (in-sample vs out-of-sample ratio exceeding 2x)

### Meta Ensemble
- Stacking: XGBoost + Transformer + RL + HMM outputs combined
- Dynamic weight adjustment via EMA of recent accuracy
- Platt scaling for confidence calibration
- Disagreement detection reduces confidence when models conflict

---

## Training Requirements

| Model | Minimum Data | Recommended Data | Training Time (CPU) | Training Time (GPU) |
|-------|-------------|-----------------|--------------------|--------------------|
| XGBoost (x3) | 500 candles/symbol | 2000+ candles | 30-60 seconds | 10-20 seconds |
| Transformer | 5000 candles | 20000+ candles | 2-4 hours | 15-30 minutes |
| HMM Regime | 1000 candles | 5000+ candles | 5-15 seconds | N/A (CPU only) |
| RL Position Sizer | 10K transitions | 50K+ transitions | 1-2 hours | 20-40 minutes |

### Data Collection Timeline

| Interval | 500 candles | 2000 candles | 5000 candles | 20000 candles |
|----------|------------|-------------|-------------|--------------|
| 1m | 8 hours | 33 hours | 3.5 days | 14 days |
| 5m | 42 hours | 7 days | 17 days | 69 days |
| 15m | 5 days | 21 days | 52 days | 208 days |
| 1h | 21 days | 83 days | 208 days | 2.3 years |
| 4h | 83 days | 333 days | 2.3 years | 9.1 years |

**Recommended approach**: Train XGBoost on 1h data (available quickly). Train Transformer on 5m data for more granularity. Use 1h for HMM and RL.

### Retraining Schedule

| Model | Frequency | Trigger |
|-------|-----------|---------|
| XGBoost | Every 6 hours | Automatic (APScheduler) |
| Transformer | Every 24 hours | Automatic |
| HMM | Every 12 hours | Automatic |
| RL | Continuous | Online learning from live trades |

---

## API Endpoints

The engine exposes a FastAPI server on port 8300:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/signals` | Latest signals for all tracked pairs |
| GET | `/signals/{symbol}` | Signal for specific trading pair |
| GET | `/models/status` | Model training status, accuracy, last retrain |
| GET | `/health` | System health check |
| GET | `/providers/status` | Status of all data providers |
| WS | `/ws/signals` | Real-time signal stream via WebSocket |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Edit `config.yaml` and add your API keys for premium data providers. The engine works without premium providers (using Binance + free sources), but accuracy improves with more data.

### 3. Initial training

```bash
python -m src.ml.trainer
```

Fetches historical data from Binance and trains all three XGBoost models. First training takes 2-5 minutes.

### 4. Start the engine

```bash
python -m src.api.server
```

The engine starts on port 8300. Signals are generated every 5 minutes. Models retrain automatically.

### 5. Get signals

```bash
curl http://localhost:8300/signals/BTCUSDT
```

---

## Project Structure

```
mefai-signal-engine/
  config.yaml                    # All configuration
  requirements.txt               # Python dependencies
  pyproject.toml                 # Package metadata
  src/
    ml/                          # Machine Learning Models
      feature_engine.py          # 15+ technical indicator calculations
      majors_model.py            # XGBoost for BTC, ETH, BNB
      alts_model.py              # XGBoost for mid-cap alts
      memes_model.py             # XGBoost for meme coins
      trainer.py                 # Training pipeline orchestrator
    signals/                     # Signal Composition
      composer.py                # 5-layer signal combiner
      technical.py               # Technical analysis scoring
      correlation.py             # Cross-asset correlation analysis
      onchain.py                 # On chain metrics scoring
      sentiment.py               # RSS feed sentiment analysis
    pipeline/                    # Data Pipeline
      ingestion.py               # Multi-source data fetcher
      processor.py               # Data processing and feature computation
      scheduler.py               # APScheduler task runner
    providers/                   # Data Providers (12 premium + 3 free)
      base.py                    # Base class (rate limit, cache, retry)
      glassnode.py               # On chain analytics
      santiment.py               # Social + on chain (GraphQL)
      cryptoquant.py             # Exchange analytics
      nansen.py                  # Smart money tracking
      kaiko.py                   # Institutional market data
      coinmetrics.py             # Network fundamentals
      messari.py                 # Token fundamentals
      lunarcrush.py              # Social intelligence
      token_terminal.py          # Protocol financials
      intotheblock.py            # Address analytics
      artemis.py                 # Ecosystem data
      defi_free.py               # Free sources (DefiLlama, F&G, CoinGecko)
      aggregator.py              # Multi-provider data fusion
    advanced/                    # Advanced Models
      transformer_predictor.py   # LSTM + Multi-Head Attention
      regime_detector.py         # Hidden Markov Model (4 regimes)
      rl_position_sizer.py       # Double DQN position sizing
      monte_carlo.py             # 10K path risk simulation
      walk_forward.py            # Walk-forward backtesting
      ensemble.py                # Meta-ensemble combiner
      training_guide.py          # Training requirements calculator
    api/
      server.py                  # FastAPI server (port 8300)
  tests/
    test_features.py             # Feature engine tests
    test_models.py               # Model training/prediction tests
    test_signals.py              # Signal composition tests
```

---

## Performance

- Signal generation: under 500ms per symbol (with cached provider data)
- XGBoost inference: under 5ms per prediction
- Transformer inference: under 50ms (GPU), under 200ms (CPU)
- Memory: approximately 500MB base + 200MB per loaded model
- Recommended: 4+ CPU cores, 4GB+ RAM, GPU optional but recommended for Transformer

---

## License

MIT

---

Built by [Mefai](https://mefai.io)
