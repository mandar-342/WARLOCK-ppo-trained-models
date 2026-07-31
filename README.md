<div align="center">

![Warlock Banner](https://capsule-render.vercel.app/api?type=waving&color=0:0B0714,35:1E1B4B,70:4338CA,100:6366F1&height=220&section=header&text=WARLOCK%20%2F%20LSTM-LONG-ONLY&fontSize=40&fontColor=E0E7FF&animation=fadeIn&fontAlignY=38&desc=Experiment%20Log%20%E2%80%94%20Constraining%20the%20Action%20Space%20to%20Fix%20a%20Broken%20Agent&descAlignY=58&descSize=15)

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=17&duration=3000&pause=1000&color=A5B4FC&center=true&vCenter=true&width=780&lines=Branch%3A+ppo-trained-models-lstm-long-only;Action+Space+Restricted+to+%5B0%2C+1%5D+Net+Weight+(No+Shorting);20-Trial+Optuna+Sweep+Logged+%2B+Best+Config+Exported;Status%3A+Experimental+%E2%80%94+Breaker+Rate+Still+Elevated" alt="Typing SVG" />

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-4338CA?style=flat-square&logo=python&logoColor=E0E7FF&labelColor=10102b)](https://www.python.org/)
[![SB3-Contrib](https://img.shields.io/badge/SB3--Contrib-RecurrentPPO-4F46E5?style=flat-square&logo=pytorch&logoColor=E0E7FF&labelColor=10102b)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
[![Optuna](https://img.shields.io/badge/Optuna-20%20Trials%20Logged-6366F1?style=flat-square&logoColor=E0E7FF&labelColor=10102b)](https://optuna.org/)
[![Action Space](https://img.shields.io/badge/Action%20Space-Long--Only-818CF8?style=flat-square&labelColor=10102b)]()
[![Branch Status](https://img.shields.io/badge/Branch-Experimental-818CF8?style=flat-square&labelColor=10102b)]()

</div>

<br>

> Like the sibling `lstm-optuna` branch, this is an **experiment log**, not a finished agent. It records a specific hypothesis test — does removing shorting from the action space reduce the reward-hacking/circuit-breaker behaviour seen in earlier runs — along with the actual sweep results, good and bad.

---

## 🧪 The Hypothesis Behind This Branch

Earlier PPO runs (see the `lstm-optuna` branch) kept tripping the drawdown circuit breaker on effectively every seed, and Sharpe ratios were deeply negative. One suspected contributor: an unrestricted **long/short action space** gives the policy more surface area to exploit the reward function through leveraged or oscillating short positions rather than learning a coherent trading strategy.

This branch tests that directly by **constraining the action space to long-only**:

```python
# src/env/gym_bitcoin.py
# Long-only: net weight per asset stays in [0, 1] (no short leg).
self.action_space = spaces.Box(...)
```

The agent can now only ever hold between 0% and 100% net long exposure — no shorting, no leverage past 1x. Everything else (reward shaping, Optuna search harness, portfolio simulator) carries over from the other branch.

---

## 📊 Multi-Seed / Ablation Sweep Results (`experiments/multi_seed/*.json`)

These are hand-labelled reward-config experiments run before and alongside the formal Optuna sweep, each evaluated on a full backtest (not just eval-episode reward). **`breaker_hit`** means the drawdown circuit breaker fired during that run.

| Config | Sharpe | Total Return | Max DD | Breaker Hit | Trades | Win Rate | Notes |
|---|---|---|---|---|---|---|---|
| **`quick_check`** | **-1.10** | -20.8% | 31.1% | **No** | 1,550 | 39.3% | **Best result on this branch — only config where the breaker never fired** |
| `longer_train` | -3.62 | -38.1% | 41.2% | Yes | 941 | 38.7% | Same overrides as `quick_check`, longer training — regressed |
| `simple_reward` | -3.46 | -38.1% | 41.1% | Yes | 903 | 34.5% | Dropping the Sharpe term entirely didn't help |
| `pure_return_strong_dd` | -5.74 | -49.8% | 50.1% | Yes | 2,965 | 40.8% | Strong DD penalty alone isn't sufficient |
| `strong_dd_penalty` | -6.09 | -49.9% | 50.1% | Yes | 2,219 | 36.8% | Even stronger DD penalty made it worse, not better |
| `optuna_trial_0000–0004` | -3.86 | -30.2% | 30.3% | Yes | 2,134 | 39.8% | Early Optuna trials, all converged to the same point (search space too narrow at this stage) |

**Reading of the data:** long-only constraint alone did not fix the breaker problem — most configurations still hit it on effectively every seed. The one exception, `quick_check`, is the current best-known configuration on this branch: it's the only run where the drawdown circuit breaker never triggered, and it has the least-negative Sharpe ratio (-1.10) of anything recorded here. Notably, simply training that same config for longer (`longer_train`) made results worse, suggesting overfitting to reward-hacking behaviour with extended training time is still a live risk even long-only.

---

## 🔎 Optuna Sweep Summary

A formal Optuna study (`experiments/optuna_warlock_fix_breaker_unknown.db`) was run against this branch's environment:

| | |
|---|---|
| Trials logged | 20 (18 complete, 2 running at time of export) |
| Best objective observed | **-10.57** (up from -10.86 at trial 0) |
| Objective | `median(Sharpe) − λ·breaker_rate − μ·std(Sharpe)` across seeds |

The best-performing configuration was exported to `experiments/best_config_warlock_fix_breaker_unknown.yaml`:

| Parameter | Best Value |
|---|---|
| `ppo.learning_rate` | 0.000213 |
| `ppo.gamma` | 0.9740 |
| `ppo.gae_lambda` | 0.9716 |
| `ppo.clip_range` | 0.1038 |
| `ppo.ent_coef` | 0.000866 |
| `ppo.vf_coef` | 0.4875 |
| `ppo.n_steps` / `batch_size` | 512 / 32 |
| `env.max_drawdown` | 0.4075 |
| `env.max_trade_step` | 0.0558 |
| `reward.drawdown_penalty_scale` | 0.3181 |
| `reward.sharpe_weight` | 0.1690 |
| `reward.sharpe_aggregation_steps` | 6 |
| `reward.step_return_weight` | 1.9399 |
| `reward.overtrade_penalty_scale` | 0.0549 |

The objective value only moved modestly across the whole 20-trial sweep (-10.86 → -10.57), which is itself a finding: this search space, on the long-only environment, hadn't yet located a configuration that clears the reward-hacking problem outright. `quick_check`'s hand-picked overrides currently outperform anything the sweep found, which is the open question this branch leaves for the next iteration.

### Static baselines for reference (`benchmarks/comparison.md`, same environment)

| Strategy | Total Return | Sharpe | Max Drawdown | Trades | Win Rate |
|---|---|---|---|---|---|
| Buy & Hold | -26.16% | -3.20 | 30.22% | 637 | 28.4% |
| Always Cash | 0.00% | 0.00 | 0.00% | 0 | — |
| Random Agent | -29.98% | -5.98 | 30.00% | 2,356 | 0.0% |

`quick_check`'s -1.10 Sharpe is the best number on this branch across any comparison point, trained or baseline — but total return (-20.8%) is still well short of `always_cash`, so "best so far" and "good" are not yet the same thing here.

---

## 🗂️ Branch-Relevant Structure

```
WARLOCK-ppo-trained-models-lstm-long-only/
├── src/
│   ├── agent/                   # PPO/RecurrentPPO trainer + hpo_optuna.py
│   ├── env/
│   │   └── gym_bitcoin.py        # long-only action space change lives here
│   ├── portfolio/                 # fee/slippage/SL-TP simulator
│   └── tests/
├── experiments/
│   ├── multi_seed/                 # hand-labelled ablation configs (this README's data)
│   ├── ppo_baseline/                # raw per-run training dirs
│   ├── best_config_warlock_fix_breaker_unknown.yaml
│   ├── optuna_warlock_fix_breaker_unknown.db
│   └── optuna_futures.journal
├── benchmarks/                     # buy_hold, always_cash, random_agent + comparison.{csv,json,md}
├── df.py                           # ad-hoc analysis / dataframe scratch script
└── config.yaml
```

---

## ▶️ Reproducing the Best-Known Run

```bash
# Train using the currently best hand-picked config (breaker never fired)
python main.py --overrides \
    env.max_drawdown=0.4 \
    ppo.ent_coef=0.005 \
    ppo.learning_rate=0.0001 \
    ppo.n_steps=512 \
    reward.drawdown_penalty_scale=0.3 \
    reward.sharpe_aggregation_steps=24 \
    reward.sharpe_weight=0.05

# Or resume/extend the Optuna study that produced best_config_warlock_fix_breaker_unknown.yaml
python -m src.agent.hpo_optuna \
    --study-name warlock_fix_breaker_unknown \
    --storage sqlite:///experiments/optuna_warlock_fix_breaker_unknown.db \
    --n-trials 20 \
    --search-space core+env
```

Component sanity checks:

```bash
python -m src.tests.test_rewards
python -m src.tests.test_portfolio
python -m src.tests.test_env
```

---

## 📝 Status & Next Steps

- [x] Long-only action space implemented and validated
- [x] Ablation sweep run across 10 hand-picked reward/env configurations
- [x] Found one config (`quick_check`) where the drawdown breaker never fires
- [x] Formal 20-trial Optuna sweep completed, best config exported to YAML
- [ ] Understand why `longer_train` (same config, more steps) regressed — overfitting vs. instability
- [ ] Re-run Optuna with a search space anchored closer to `quick_check`'s overrides
- [ ] Get total return positive, or at minimum ahead of `always_cash`, on a held-out backtest

As with the other branch, every number above is a mid-experiment reading. `quick_check` is the current best lead, not a result to ship.

<br>

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:6366F1,50:4338CA,100:0B0714&height=100&section=footer)

</div>
