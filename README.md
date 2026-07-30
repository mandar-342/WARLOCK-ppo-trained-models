# Warlock

Warlock is a modular Reinforcement Learning framework for developing and evaluating cryptocurrency trading agents. The project provides an end-to-end pipeline covering historical market data collection, feature engineering, portfolio simulation, risk management, and a custom Gymnasium environment for training RL algorithms.

---

## Key Features

- **Modular data collection & preprocessing** — Automated OHLCV download from Binance (or any CCXT exchange), cleaning, anomaly detection, and temporal train/test splits
- **Automated feature engineering** — 17 technical indicators across 5 categories (Price Action, Candlestick, Momentum, Volatility, Volume) with per-asset z-score normalization fit on train only
- **Custom Gymnasium environment** — `GymBitcoinEnv` with Dict observation space (`market`: per-asset features, `portfolio`: portfolio state), recurrent policy support via `MultiInputLstmPolicy`, and deterministic evaluation mode
- **Unified portfolio simulator** — Single cash pool backing both long (spot, 1x) and short (futures, configurable leverage) positions; maker/taker fees, fixed-bps slippage, min-notional filters, ATR-based stop-loss/take-profit, dynamic position sizing, funding rates, and maintenance-margin liquidation
- **Risk-aware reward function** — Rolling Sharpe ratio (with configurable aggregation window), step-return component, drawdown penalty, and overtrading penalty; all components tanh-squashed for scale alignment
- **Config-driven architecture** — Entire pipeline controlled via `config.yaml` (data, features, env, portfolio, risk, reward, training, evaluation, benchmarks)
- **Experiment management** — Timestamped run directories with config snapshots, checkpoints (model + VecNormalize), TensorBoard logs, evaluation CSVs, plots, and PDF reports
- **Hyperparameter optimization** — Optuna integration with multi-seed evaluation for statistical robustness
- **Benchmark suite** — Buy & Hold, Always Cash, and Random Agent baselines with identical evaluation pipeline
- **Analytics & reporting** — 17 performance metrics (Sharpe, Sortino, Calmar, CAGR, max DD, win rate, profit factor, expectancy…), equity/drawdown/rolling-Sharpe plots, and auto-generated PDF reports

---

## Project Structure

```
WARLOCK/
├── config.yaml                 # Single source of truth for all configuration
├── main.py                     # Entry point: runs data + feature pipeline
├── requirements.txt            # Python dependencies
├── data/                       # Data directories (git-kept, contents ignored)
│   ├── raw/                    # Raw OHLCV from exchange
│   ├── processed/              # Cleaned & anomaly-filtered
│   └── features/               # Train/test parquet with engineered features
├── graphs/features/            # Feature diagnostic plots (auto-generated)
├── src/
│   ├── data_manager/           # Download, clean, anomaly detection
│   ├── features/               # 5 feature modules + pipeline + plotting
│   ├── env/                    # Gymnasium env + reward calculator
│   ├── portfolio/              # Unified portfolio + positions + orders
│   ├── agent/                  # PPO trainer, callbacks, eval, HPO, features extractor
│   ├── analytics/              # Metrics, plots, reports, leaderboard
│   ├── benchmark/              # Buy&Hold, Cash, Random baselines
│   ├── tests/                  # Unit tests for env, portfolio, rewards
│   └── utils/                  # Config loader, project root, seeding
├── experiments/                # Auto-created per run (checkpoints, logs, eval)
├── benchmarks/                 # Baseline evaluation outputs
├── notebooks/                  # Exploratory notebooks
└── scripts/                    # Helper scripts (Optuna launchers)
```

---

## Prerequisites

- **Python 3.10+**
- **TA-Lib C library** (required for technical indicators)
  - Windows: `conda install -c conda-forge ta-lib` or download wheel from [LFD](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
  - Linux: `sudo apt-get install ta-lib`
  - macOS: `brew install ta-lib`
- **CUDA-enabled PyTorch** (recommended for training) — install per [PyTorch site](https://pytorch.org/get-started/locally/)

---

## Installation

```bash
# Clone
git clone https://github.com/darkisthenight07/warlock
cd warlock

# Virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Dependencies
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes `stable-baselines3[extra]`, `sb3-contrib`, `optuna`, `tensorboard`, `ccxt`, `pyarrow`, `loguru`, `matplotlib`, `seaborn`, `reportlab`, `TA-Lib`, `gymnasium`, `pandas`, `numpy`, `pyyaml`.

---

## Quick Start (Full Pipeline)

### 1. Download & Prepare Data + Features

```bash
python main.py
```

This runs the complete data pipeline:
1. Downloads raw OHLCV from Binance (configured symbols, timeframe, date range)
2. Cleans data (removes duplicates, fills small gaps, detects wick anomalies)
3. Generates 17 features per asset
4. Performs **strict temporal split** (train: ~4.5 years, test: 6 months) — **no lookahead leakage**
5. Fits volume z-score on train only, applies frozen stats to test
6. Writes `data/features/train.parquet` and `data/features/test.parquet`
7. Emits feature diagnostic plots to `graphs/features/`

### 2. Train a PPO Agent

```bash
python -m src.agent.train
```

- Creates timestamped experiment dir under `experiments/ppo_baseline/`
- Trains `RecurrentPPO` with `MultiInputLstmPolicy` + custom `MultiAssetFeaturesExtractor`
- Saves checkpoints every `checkpoint_frequency` steps (model.zip + vecnormalize.pkl)
- Logs to TensorBoard (`tensorboard --logdir experiments/ppo_baseline/<run>/tensorboard`)
- On completion: final `model.zip`, `best_model.zip`, `config.yaml` snapshot, `metadata.json`

**Resume from checkpoint:**
```yaml
# In config.yaml (training section):
resume_checkpoint: "experiments/ppo_baseline/<run>/checkpoints/checkpoint_10000_steps.zip"
resume_experiment: "experiments/ppo_baseline/<run>"
```

### 3. Evaluate on Held-Out Test Set

```bash
python -m src.agent.evaluate --experiment experiments/ppo_baseline/<run_id>
```

Produces in `experiments/ppo_baseline/<run_id>/evaluation/`:
- `equity_curve.csv`, `portfolio.csv`, `trades.csv`, `reward_components.csv`, `action_diagnostics.csv`
- `metrics.json` (17 metrics)
- Plots: `equity_curve.png`, `drawdown.png`, `returns_distribution.png`, `rolling_sharpe.png`
- `report.pdf` (full tear sheet)

### 4. Run Benchmarks (for comparison)

```bash
# Buy & Hold
python -m src.benchmark.buy_hold_runner --experiment experiments/ppo_baseline/<run_id>

# Always Cash
python -m src.benchmark.always_cash_runner --experiment experiments/ppo_baseline/<run_id>

# Random Agent
python -m src.benchmark.random_agent_runner --experiment experiments/ppo_baseline/<run_id>

# Or run all benchmarks + comparison table
python -m src.benchmark.benchmark_runner --experiment experiments/ppo_baseline/<run_id>
```

Outputs go to `benchmarks/<benchmark_name>/` and a summary `benchmarks/comparison.md`.

---

## Hyperparameter Optimization (Optuna)

```bash
# Windows PowerShell
.\scripts\launch_optuna.ps1

# Linux/macOS
bash scripts/launch_optuna.sh
```

- Defined in `src/agent/hpo_optuna.py` (search space: LR, ent_coef, clip_range, n_steps, batch_size, net_arch, lstm_hidden, sharpe_weight, etc.)
- Each trial runs a **short training** + evaluation
- Multi-seed evaluation (`src/agent/multi_seed.py`) for statistical significance
- Best trial config saved; launch full training with those params

---

## Configuration Guide (`config.yaml`)

All knobs are in one file. Key sections:

| Section | Purpose |
|---------|---------|
| `data` | Exchange, symbols, timeframe, date range, anomaly detection |
| `features` | 17 selected features + per-category parameters (windows, periods) |
| `paths` | All I/O directories |
| `env` | `window_len`, `max_trade_step`, `transaction_cost_rate`, `max_drawdown` |
| `portfolio` | Initial capital, assets, fees, slippage, leverage, margin, rebalance, risk |
| `risk` | ATR stop-loss/TP multiples, target ATR% for position sizing |
| `reward` | Sharpe window, step-return weight, sharpe weight, drawdown/overtrade penalties, aggregation steps |
| `training` | Seed, device, timesteps, checkpoint/eval frequency, tensorboard, resume |
| `ppo` | Policy, policy_kwargs (net_arch, LSTM, features_extractor), all PPO hyperparams |
| `evaluation` | Risk-free rate, periods/year, rolling window, benchmarks, plots, report |

**Example: switch to single-asset (BTC only)**
```yaml
data:
  symbols: ["BTC/USDT"]
portfolio:
  assets:
    - symbol: "BTC/USDT"
      min_trade_notional: 10.0
```

---

## Running Unit Tests

```bash
# Environment (Gymnasium API compliance, obs spaces, step/reset)
python -m src.tests.test_env

# Portfolio (orders, fees, slippage, liquidation, PnL)
python -m src.tests.test_portfolio

# Rewards (scaling, buffers, penalties, Sharpe calc)
python -m src.tests.test_rewards
```

---

## Outputs & Artifacts

| Location | Contents |
|----------|----------|
| `data/features/train.parquet` `test.parquet` | Multi-asset feature matrices (asset-prefixed columns + shared timestamp) |
| `graphs/features/*.png` | Per-feature distributions, rolling Sharpe, correlation heatmaps |
| `experiments/ppo_baseline/<run_id>/` | Complete training run artifact |
| `experiments/.../checkpoints/` | `checkpoint_<step>_steps.zip` + `checkpoint_vecnormalize_<step>_steps.pkl` |
| `experiments/.../evaluation/` | CSVs, `metrics.json`, plots, `report.pdf` |
| `benchmarks/<name>/` | Baseline equity, trades, metrics, plots, report |
| `benchmarks/comparison.{csv,json,md}` | Side-by-side comparison table |

---

## Key Design Decisions

1. **Dict observation space** — `{"market": (n_assets, n_features), "portfolio": (k,)}` enables shared per-asset encoder (`MultiAssetFeaturesExtractor`) instead of flattening
2. **Recurrent policy** — LSTM carries temporal context; `window_len` only used for feature warm-up, not observation stacking
3. **Single cash pool** — Longs (1x) and shorts (leverage) draw from same equity; realistic margin coupling
4. **Train-fit normalization** — Volume z-score (and any rolling stats) fit on train only, frozen for test
5. **Deterministic evaluation** — `deterministic_start=True` scores full held-out window identically every run
6. **Per-asset risk management** — Independent ATR stop-loss/TP per asset; short equity impact scaled by leverage

---

## Citation

If you use Warlock in research, please cite:

```bibtex
@software{warlock,
  title = {Warlock: A Modular RL Framework for Cryptocurrency Trading},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/darkisthenight07/warlock}
}
```

---

## License

MIT License — see `LICENSE` for details.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `TA-Lib` import error | Install C library first (see Prerequisites), then `pip install TA-Lib` |
| `No module named 'src'` | Run from repo root; `src/` is a package (has `__init__.py`) |
| CUDA OOM | Reduce `batch_size`, `n_steps`, or `n_envs` in `config.yaml` |
| `VecNormalize` mismatch on resume | Ensure `resume_checkpoint` AND `resume_experiment` both point to same run |
| Empty `train.parquet` | Check date range, symbol availability, and `max_fill_candles` in config |
| Optuna study not saving | Check `storage` URL in `hpo_optuna.py` (defaults to SQLite file) |

---
