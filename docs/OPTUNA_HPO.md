# WARLOCK Optuna Hyperparameter Optimization

## Overview

This project includes a comprehensive Optuna-based hyperparameter optimization system built on top of the existing multi-seed evaluation harness. The enhanced script (`src/agent/hpo_optuna.py`) provides:

- **Multiple search space tiers** (core, core+lstm, core+env, full)
- **Multi-objective scoring**: median Sharpe - λ·breaker_rate - μ·std(Sharpe)
- **Inter-seed pruning** via MedianPruner/HyperbandPruner
- **Process-based parallelism** (safe for CUDA + shared config dict)
- **JournalStorage/SQLite** for concurrent-safe persistence
- **Top-trial reporting** with full parameter breakdown
- **Best-config export** as YAML for reproduction

---

## Search Spaces

| Tier | Parameters | Use Case |
|------|------------|----------|
| `core` | PPO: lr, batch_size, gamma, gae_lambda, clip_range, ent_coef, vf_coef | Baseline PPO tuning |
| `core+lstm` | core + LSTM: hidden_size, n_layers, shared_lstm, enable_critic_lstm, net_arch | Architecture search |
| `core+env` | core + Env/Reward: max_drawdown, drawdown_penalty_scale, overtrade_penalty_scale, sharpe_weight, sharpe_aggregation_steps | Fix reward hacking (recommended first) |
| `full` | All of the above + futures leverage/margin (futures branch) | Final sweep |

### Key Parameters for Current Issue (Reward Hacking)

Based on the catastrophic regression (breaker_rate 0% → 100%, Sharpe -1.1 → -3.62):

| Parameter | Current | Search Range | Rationale |
|-----------|---------|--------------|-----------|
| `reward.drawdown_penalty_scale` | 0.1 | [0.05, 1.0] log | Primary fix: stronger per-step DD penalty |
| `env.max_drawdown` | 0.3 | [0.15, 0.40] | Earlier circuit breaker |
| `reward.sharpe_weight` | 0.10 | [0.0, 0.15] | Reduce noisy Sharpe term |
| `reward.sharpe_aggregation_steps` | 12 | [1, 6, 12, 24, 48] | Stabilize Sharpe estimate |
| `ppo.ent_coef` | 0.02 | [1e-4, 0.05] log | Prevent entropy collapse |
| `reward.overtrade_penalty_scale` | 0.01 | [1e-4, 0.05] log | Discourage churn |

---

## Quick Start

### PowerShell (Windows)
```powershell
# Quick test sweep (recommended first run)
.\scripts\launch_optuna.ps1 quick

# Full sweep with 4 workers
.\scripts\launch_optuna.ps1 full

# Analyze results
.\scripts\launch_optuna.ps1 analyze warlock_full

# Resume interrupted sweep
.\scripts\launch_optuna.ps1 resume warlock_full
```

### Bash (WSL/Linux/Mac)
```bash
# Quick test sweep
./scripts/launch_optuna.sh quick

# Full sweep
./scripts/launch_optuna.sh full

# Analyze
./scripts/launch_optuna.sh analyze warlock_full

# Resume
./scripts/launch_optuna.sh resume warlock_full
```

### Direct Python (full control)
```powershell
python -m src.agent.hpo_optuna `
    --n-trials 20 `
    --n-jobs 2 `
    --seeds 0,1,2 `
    --timesteps 50000 `
    --search-space core+env `
    --lambda-penalty 7.0 `
    --volatility-penalty 0.5 `
    --pruner median `
    --study-name warlock_reward_debug
```

---

## Output & Artifacts

### Study Storage
- Default location: `experiments/optuna_<branch>.journal` (Optuna ≥3.1) or `.db`
- All workers share one storage → single study, concurrent trials

### Per-Trial Artifacts
- `experiments/multi_seed/optuna_trial_XXXX.json` — multi-seed summary per trial
- Checkpoints/TensorBoard: auto-cleaned by default (`--keep-checkpoints` to preserve)

### Final Outputs
- `experiments/best_config_<study_name>.yaml` — Best trial's dotted-path overrides
- Console: Top-N trials with full params, breaker_rate, Sharpe stats
- Parameter importance (requires ≥10 completed trials)

---

## Interpreting Results

### Objective Score
```
score = median_sharpe - lambda * breaker_rate - volatility_penalty * std_sharpe
```
- `median_sharpe`: Robust central tendency (odd # seeds)
- `breaker_rate`: Fraction of seeds hitting max_drawdown (0.0–1.0)
- `std_sharpe`: Instability penalty (optional)

### Good Trial Indicators
| Metric | Target |
|--------|--------|
| `median_sharpe` | > 0.0 (positive risk-adjusted return) |
| `breaker_rate` | < 0.2 (few catastrophic failures) |
| `std_sharpe` | < 1.0 (stable across seeds) |
| `mean_return` | > 0% (profitable on average) |

### Red Flags
- High `score` but `breaker_rate` > 0.5 → trial gamed the penalty weight
- `median_sharpe` ≈ `mean_sharpe` but both negative → consistently losing
- High `std_sharpe` with good median → unstable, not production-ready

---

## Architecture Notes

### Why Process-Based Parallelism?
1. **Config isolation**: `src/utils/config.py` loads YAML into a module-level dict. `config_overrides.py` mutates it in-place. Concurrent threads would race.
2. **CUDA safety**: PyTorch CUDA contexts don't fork safely. Processes get fresh contexts.

### Inter-Seed Pruning
- Pruning operates on **seeds completed within a trial**, not mid-training steps
- After each seed, reports running median Sharpe to Optuna
- If trial falls below population median at that seed-count, remaining seeds skipped
- This is a **simplification** (can't stop a single seed mid-training) but:
  - Works without modifying `PPOTrainer`/`callbacks.py`
  - Most doomed trials fail early anyway (low Sharpe on seed 0)

### Storage Backends
- **JournalStorage** (Optuna ≥3.1): File-lock based, no SQLite locking issues, recommended
- **SQLite**: Fallback for older Optuna, works but can lock under heavy concurrent writes

---

## Recommended Workflow

1. **Quick reward/env sweep** (fixes the immediate breaker issue):
   ```powershell
   .\scripts\launch_optuna.ps1 quick
   ```
   → Check `breaker_rate` drops below 0.3, Sharpe positive

2. **Analyze & export best config**:
   ```powershell
   .\scripts\launch_optuna.ps1 analyze warlock_quick
   ```

3. **Apply best config, run 5-seed validation**:
   ```powershell
   python -m src.agent.multi_seed --seeds 0,1,2,3,4 --group-name validation_best `
       --set $(Get-Content experiments\best_config_warlock_quick.yaml | ConvertFrom-StringData)
   ```

4. **Full architecture + PPO sweep** (if reward fixed):
   ```powershell
   .\scripts\launch_optuna.ps1 full
   ```

5. **Test set evaluation** on best 3 configs from full sweep.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM with `--n-jobs > 1` | Reduce `ppo.batch_size` in search space, or `--n-jobs 1` |
| Workers fail immediately | Check `experiments/optuna_*.journal` permissions; delete and retry |
| Pruner never triggers | Increase `--n-startup-trials` or `--n-warmup-seeds` |
| Study seems stuck | `Ctrl+C` and resume with same `--study-name` (storage persists) |
| Import errors (optuna) | `pip install optuna>=3.1` for JournalStorage |

---

## Extending Search Space

Add new parameters in `sample_params()` in `hpo_optuna.py`:

```python
params["env.some_new_param"] = trial.suggest_float(
    "some_new_param", low, high, log=True/False
)
```

For categorical choices:
```python
params["ppo.activation_fn"] = trial.suggest_categorical(
    "activation_fn", ["ReLU", "Tanh", "ELU"]
)
```

Then ensure the config path exists in `config.yaml` or is handled by `config_overrides.py`.

---

## References

- `src/agent/hpo_optuna.py` — Main optimization script
- `src/agent/multi_seed.py` — Multi-seed evaluation harness (used by each trial)
- `src/agent/trainer.py` — PPO training loop
- `src/env/rewards.py` — Reward calculation (key failure mode)
- `config.yaml` — Base configuration