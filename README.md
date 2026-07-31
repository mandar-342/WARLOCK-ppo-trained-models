<div align="center">

![Warlock Banner](https://capsule-render.vercel.app/api?type=waving&color=0:0B0714,35:2E1065,70:5B21B6,100:7C3AED&height=220&section=header&text=WARLOCK%20%2F%20LSTM-OPTUNA&fontSize=44&fontColor=EDE9FE&animation=fadeIn&fontAlignY=38&desc=Experiment%20Log%20%E2%80%94%20Recurrent%20PPO%20%2B%20Optuna%20HPO%20Branch&descAlignY=58&descSize=16)

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=18&duration=3000&pause=1000&color=C4B5FD&center=true&vCenter=true&width=760&lines=Branch%3A+ppo-trained-models-lstm-optuna;Investigating+Reward+Hacking+in+a+Trading+Policy;Optuna-Driven+Search+for+a+Stable+Reward+Surface;Status%3A+Experimental+%E2%80%94+Not+Production+Ready" alt="Typing SVG" />

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-6D28D9?style=flat-square&logo=python&logoColor=EDE9FE&labelColor=17092e)](https://www.python.org/)
[![SB3-Contrib](https://img.shields.io/badge/SB3--Contrib-RecurrentPPO-7C3AED?style=flat-square&logo=pytorch&logoColor=EDE9FE&labelColor=17092e)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-8B5CF6?style=flat-square&logoColor=EDE9FE&labelColor=17092e)](https://optuna.org/)
[![Branch Status](https://img.shields.io/badge/Branch-Experimental-A78BFA?style=flat-square&labelColor=17092e)]()

</div>

<br>

> This README documents a specific **experiment branch**, not a finished product. It exists to record what was tried, what the numbers actually said, and what's being changed next. Nothing here is a performance claim — the honest headline is that the policy is not yet beating the baselines, and this branch is the attempt to find out why and fix it.

---

## 🧪 Why This Branch Exists

The `main` branch of WARLOCK trains a baseline PPO agent on the custom Bitcoin trading environment. Early runs on that baseline surfaced a classic RL failure mode — **reward hacking** — where the agent found a way to inflate its reward signal without producing genuinely good trading behaviour. Observed symptoms going into this branch:

- Circuit-breaker (max drawdown) trigger rate jumped from **0% → 100%** across otherwise similar seeds
- Sharpe ratio during training regressed from **≈ -1.1 → ≈ -3.62**
- Policy behaviour looked unstable and seed-dependent rather than convergent

This branch swaps the plain MLP policy for a **Recurrent PPO (LSTM) policy** — to give the agent temporal memory over the `window_len=48` lookback — and adds a full **Optuna hyperparameter search harness** (`src/agent/hpo_optuna.py`) aimed specifically at the reward/environment parameters most implicated in the hacking behaviour.

---

## 📉 Latest Evaluation Results (this branch, as recorded)

Pulled directly from `experiments/ppo_baseline/*/logs/evaluations.npz` — SB3's `EvalCallback` output across 5 evaluation episodes per checkpoint. Values are episodic reward (not portfolio return), reflecting the reward function under test, not raw PnL.

| Run | Timesteps | Eval rewards (5 episodes) | Mean reward | Notes |
|---|---|---|---|---|
| `20260730_191504_pid9152` | 200,000 | -349.2, -88.7, -125.0, -132.4, -168.5 | ≈ **-172.8** | Long single-checkpoint run |
| `20260728_192439_pid32320` | 5k → 20k | see below | trending down then partially recovering | Early-training snapshots, high variance |
| `20260730_184545_pid42888` | 200,000 | -349.2, -88.7, -125.0, -132.4, -168.5 | ≈ **-172.8** | Duplicate config, same result profile |

Early-checkpoint run in detail (`20260728_192439_pid32320`):

| Timestep | Episode rewards | Mean |
|---|---|---|
| 5,000 | -99.8, -62.1, -197.7, -128.9, -100.3 | -117.7 |
| 10,000 | -183.3, -281.8, -351.0, -75.0, -502.6 | -278.7 |
| 15,000 | -414.2, -87.7, -132.5, -297.7, -40.5 | -194.5 |
| 20,000 | -195.8, -35.4, -59.8, -50.9, -57.6 | **-79.9** |

**Reading of the data:** reward is still consistently negative and episode length is highly variable (roughly 1.4k–31.5k steps per episode across runs), which is itself informative — the agent is being cut short by the drawdown circuit breaker or stop-loss/take-profit logic on a large share of episodes rather than completing full backtests. The 20k-step checkpoint shows the best mean reward observed so far, which is the current lead for the Optuna search to build on.

### Static baselines for reference (`benchmarks/comparison.md`, same environment)

| Strategy | Total Return | Sharpe | Max Drawdown | Trades | Win Rate |
|---|---|---|---|---|---|
| Buy & Hold | -26.16% | -3.20 | 30.22% | 637 | 28.4% |
| Always Cash | 0.00% | 0.00 | 0.00% | 0 | — |
| Random Agent | -29.98% | -5.98 | 30.00% | 2,356 | 0.0% |

These are portfolio-return metrics, not directly comparable unit-for-unit to the reward-signal numbers above, but they set the bar the trained policy still needs to clear on a like-for-like backtest.

---

## 🔧 What Changed vs. the PPO Baseline

| Area | Baseline (`main`) | This branch |
|---|---|---|
| Policy architecture | Standard MLP PPO | **Recurrent PPO with LSTM** backbone (`sb3-contrib`) |
| Hyperparameter search | Manual / fixed | **Optuna**, multi-tier search spaces (`core`, `core+lstm`, `core+env`, `full`) |
| Search objective | N/A | `median(Sharpe) − λ·breaker_rate − μ·std(Sharpe)` across seeds |
| Pruning | N/A | `MedianPruner` / `HyperbandPruner`, inter-seed |
| Persistence | N/A | `JournalStorage` / SQLite, concurrent-safe, resumable sweeps |
| Priority search tier | — | `core+env` first — targets the reward-hacking fix directly |

### Parameters currently under search (`core+env` tier)

| Parameter | Current default | Search range | Rationale |
|---|---|---|---|
| `reward.drawdown_penalty_scale` | 0.1 | [0.05, 1.0] (log) | Primary suspect — strengthen per-step drawdown penalty |
| `env.max_drawdown` | 0.30 | [0.15, 0.40] | Trip the circuit breaker earlier, before hacking compounds |
| `reward.sharpe_weight` | 0.10 | [0.0, 0.15] | Reduce noise from the rolling-Sharpe term |
| `reward.sharpe_aggregation_steps` | 12 | {1, 6, 12, 24, 48} | Stabilize the Sharpe estimate the reward relies on |
| `ppo.ent_coef` | 0.02 | [1e-4, 0.05] (log) | Guard against entropy collapse into a degenerate policy |
| `reward.overtrade_penalty_scale` | 0.01 | [1e-4, 0.05] (log) | Discourage churn-for-reward behaviour |

---

## 🗂️ Branch-Relevant Structure

```
WARLOCK-ppo-trained-models-lstm-optuna/
├── src/
│   ├── agent/           # PPO/RecurrentPPO trainer + hpo_optuna.py
│   ├── env/              # gym_bitcoin.py, rewards.py (the file under active tuning)
│   ├── portfolio/         # fee/slippage/SL-TP simulator
│   ├── features/          # 5-category feature engineering pipeline
│   └── tests/            # test_rewards.py / test_portfolio.py / test_env.py
├── experiments/
│   └── ppo_baseline/      # per-run dirs: logs/evaluations.npz, model.zip
├── benchmarks/            # buy_hold, always_cash, random_agent + comparison.{csv,json,md}
├── docs/
│   └── OPTUNA_HPO.md      # full HPO usage + search space documentation
├── scripts/
│   ├── launch_optuna.sh
│   └── launch_optuna.ps1
└── config.yaml            # single source of truth for data/env/reward/portfolio params
```

---

## ▶️ Reproducing This Branch's Runs

```bash
# Quick sweep on the reward/env parameters implicated in the hacking behaviour
./scripts/launch_optuna.sh quick

# Full multi-seed sweep
./scripts/launch_optuna.sh full

# Resume an interrupted sweep
./scripts/launch_optuna.sh resume warlock_full

# Or drive it directly for full control
python -m src.agent.hpo_optuna \
    --n-trials 20 \
    --n-jobs 2 \
    --seeds 0,1,2 \
    --timesteps 50000 \
    --search-space core+env
```

Component sanity checks (unchanged from `main`, still pass on this branch):

```bash
python -m src.tests.test_rewards
python -m src.tests.test_portfolio
python -m src.tests.test_env
```

---

## 📝 Status & Next Steps

- [x] Reward-hacking failure mode identified and characterized (breaker rate, Sharpe regression)
- [x] RecurrentPPO (LSTM) policy wired in
- [x] Optuna HPO harness built with a search space targeted at the failure mode
- [ ] `core+env` sweep completed and best config exported to YAML
- [ ] Trained policy beats `buy_hold` / `random_agent` on a full backtest, not just eval-episode reward
- [ ] Promote a stable config back into `main`

This README will be updated once a sweep completes and a checkpoint clears the static baselines above — until then, treat every number in this document as a mid-experiment reading, not a result.

<br>

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:7C3AED,50:5B21B6,100:0B0714&height=100&section=footer)

</div>
