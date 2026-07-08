# Warlock

Warlock is a modular Reinforcement Learning framework for developing and evaluating cryptocurrency trading agents. The project provides an end-to-end pipeline covering historical market data collection, feature engineering, portfolio simulation, risk management, and a custom Gymnasium environment for training RL algorithms. The focus is on building a realistic trading environment that can be easily extended and experimented with.

## Key Features

- Modular data collection and preprocessing pipeline
- Automated feature engineering with visualization support
- Custom Gymnasium environment for reinforcement learning
- Realistic spot portfolio simulator with fees and slippage
- ATR-based Stop Loss, Take Profit, and dynamic position sizing
- Rolling Sharpe ratio based reward function with drawdown and overtrading penalties
- Config-driven architecture for rapid experimentation

## Project State & Functionality

The repository contains the foundational infrastructure required to train an RL trading agent and is actively being worked upon. All internal components, including the data pipelines, indicator creation, order execution simulation, and reward mechanics, are complete and validated by standalone test files. 

### Core Modules

1. **Data Management (`src/data_manager/`)**
   * **Downloader & Cleaner:** Automates downloading historical OHLCV data from exchanges (e.g., Binance) and does basic preproccessing such as duplicate removal and handling missing candles.
   * **Anomaly Detection:** Tracks market structural anomalies such as extreme wick deviations via rolling windows and wick multipliers.

2. **Feature Engineering Suite (`src/features/`)**
   * Implements a pipeline creating distinct features across 5 major categories: Price Action, Candlestick structures, Momentum, Volatility, and Volume indicators.
   * Includes automated feature profiling, producing visualization graphs (`graphs/features/`) to diagnose correlation profiles, rolling Sharpe ratios, trend strength, and distribution patterns.

3. **Custom Gymnasium Environment (`src/env/`)**
   * `gym_bitcoin.py` provides a custom Gymnasium interface designed to pass price tensors and historical lookback windows seamlessly to standard RL networks.

4. **Advanced Portfolio Simulator (`src/portfolio/`)**
   * Emulates realistic spot trading with configurable maker/taker fees, slippage models, minimum trade notional limits,and portfolio rebalancing.
   *  The simulator also includes ATR-based Stop Loss, Take Profit, dynamic position sizing,and portfolio-level drawdown protection while maintaining detailed trade and equity history throughout each episode.

6. **Asymmetric Reward Engineering (`src/env/rewards.py`)**
   * Implements a risk-aware reward function combining immediate portfolio returns with a rolling Sharpe ratio objective.
   * Additional penalties for portfolio drawdown and excessive trading encourage stable, risk-adjusted behaviour instead of maximizing raw profits alone.

---

## Configuration

All modules are completely decentralized and governed by `config.yaml`. Adjusting this file allows you to instantly alter exchange parameters, swap active technical indicators, fine-tune trading fees, configure lookback observation windows, or tweak reward structures without touching core code.

---

## Getting Started

### Prerequisites
* Python 3.8+
* `TA-Lib` C-library dependencies (required for technical market indicators)

### 1. Installation

Clone this repository and set up your local development environment:

```bash

# Clone the repository
git clone https://github.com/darkisthenight07/warlock
cd warlock

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```


## Running the Data & Feature Pipeline
To invoke the pipeline orchestration engine (which downloads data, builds clean feature matrices, prints dataset details, and builds diagnostic asset charts in your local graphs directory):
```bash

python main.py

```

## Component Verification Tests
The repository packages separate verification testing scripts to guarantee that your custom gym environment, simulated account portfolios, and mathematical reward components operate within perfect limits. Run them via the following scripts:

```bash

# Verify reward scaling properties, buffer mechanics, and penalties
python -m src.tests.test_rewards

# Verify trade execution flows, fee charges, slippage models, and liquidations
python -m src.tests.test_portfolio

# Verify Gymnasium state handling, lookback observations, step updates, and resets
python -m src.tests.test_env

```
