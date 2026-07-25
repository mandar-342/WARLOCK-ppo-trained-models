#!/usr/bin/env python
"""
Enhanced Optuna hyperparameter optimization for WARLOCK PPO.

This script extends the base hpo.py with:
- Expanded search spaces covering LSTM architecture, portfolio/risk params
- Multi-objective optimization (Sharpe + return - breaker penalty)
- Advanced pruning with trial-level (mid-training) and seed-level pruning
- Automatic study resumption with SQLite/JournalStorage
- Rich logging and post-sweep analysis utilities
- Support for both 'main' (long-only) and 'futures' (long+short) branches

Usage
-----
# Quick sweep (10 trials, 3 seeds each, 50k timesteps)
python -m src.agent.hpo_optuna --n-trials 10 --seeds 0,1,2 --timesteps 50000

# Full sweep with 4 parallel workers (process-based for CUDA isolation)
python -m src.agent.hpo_optuna --n-trials 50 --n-jobs 4 --seeds 0,1,2 --timesteps 100000

# Resume an interrupted sweep (same --storage, --study-name)
python -m src.agent.hpo_optuna --n-trials 50 --study-name warlock_ppo_sweep

# Analyze completed study
python -m src.agent.hpo_optuna --analyze --study-name warlock_ppo_sweep

Key Search Spaces
-----------------
PPO Core:
    learning_rate: 1e-5 .. 3e-4 (log)
    n_steps: [128, 256, 512, 1024]
    batch_size: [32, 64, 128, 256]
    n_epochs: [5, 10, 15]
    gamma: 0.95 .. 0.999
    gae_lambda: 0.90 .. 0.99
    clip_range: 0.1 .. 0.3
    ent_coef: 1e-4 .. 0.1 (log)  -- critical for entropy collapse
    vf_coef: 0.3 .. 0.8
    max_grad_norm: 0.3 .. 1.0

LSTM Architecture:
    lstm_hidden_size: [128, 256, 384, 512]
    n_lstm_layers: [1, 2]
    shared_lstm: [True, False]
    enable_critic_lstm: [True, False]
    net_arch_pi: [[256,128,64], [256,256,128], [512,256,128], [128,64]]
    net_arch_vf: [[256,128,64], [256,256,128], [512,256,128], [128,64]]
    activation_fn: [ReLU, Tanh, ELU]

Env/Reward (targeting breaker_rate failure mode):
    env.max_drawdown: 0.15 .. 0.50
    env.max_trade_step: 0.05 .. 0.40
    reward.drawdown_penalty_scale: 0.05 .. 2.0 (log)
    reward.overtrade_penalty_scale: 1e-4 .. 0.1 (log)
    reward.sharpe_weight: 0.0 .. 0.3
    reward.sharpe_aggregation_steps: [1, 6, 12, 24, 48]
    reward.step_return_weight: 0.5 .. 2.0
    reward.min_buffer_size: [3, 5, 10, 20]

Risk/Portfolio (futures branch):
    portfolio.short.leverage: 2.0 .. 5.0
    portfolio.short.maintenance_margin_ratio: 0.01 .. 0.15
    risk.stop_loss_atr_multiple: 0.5 .. 3.0
    risk.take_profit_atr_multiple: 1.0 .. 5.0
    risk.target_atr_pct: 0.5 .. 2.0

Objective
---------
score = median_sharpe - lambda_penalty * breaker_rate - volatility_penalty * sharpe_std

Where:
- median_sharpe: median Sharpe across seeds (more robust than mean)
- breaker_rate: fraction of seeds hitting drawdown circuit breaker
- sharpe_std: std of Sharpe across seeds (penalizes unstable configs)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna
from loguru import logger
from optuna.pruners import MedianPruner, NopPruner

from src.agent.multi_seed import run_multi_seed
from src.utils import root

try:
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    _HAS_JOURNAL_STORAGE = True
except ImportError:  # Optuna < 3.1
    _HAS_JOURNAL_STORAGE = False

# JournalStorage uses symlinks for locking; on Windows this requires
# admin/Developer Mode. Default to SQLite for portability.
import platform
_USE_JOURNAL = _HAS_JOURNAL_STORAGE and platform.system() != "Windows"


# =============================================================================
# Search Space Definitions
# =============================================================================

PPO_CORE_PARAMS = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 3e-4, "log": True},
    "n_steps": {"type": "categorical", "choices": [128, 256, 512, 1024]},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "n_epochs": {"type": "categorical", "choices": [5, 10, 15, 20]},
    "gamma": {"type": "float", "low": 0.95, "high": 0.999},
    "gae_lambda": {"type": "float", "low": 0.90, "high": 0.99},
    "clip_range": {"type": "float", "low": 0.1, "high": 0.3},
    "ent_coef": {"type": "float", "low": 1e-4, "high": 0.1, "log": True},
    "vf_coef": {"type": "float", "low": 0.3, "high": 0.8},
    "max_grad_norm": {"type": "float", "low": 0.3, "high": 1.0},
}

LSTM_ARCH_PARAMS = {
    "lstm_hidden_size": {"type": "categorical", "choices": [128, 256, 384, 512]},
    "n_lstm_layers": {"type": "categorical", "choices": [1, 2]},
    "shared_lstm": {"type": "categorical", "choices": [True, False]},
    "enable_critic_lstm": {"type": "categorical", "choices": [True, False]},
    # net_arch choices as tuples for immutability/hashing
    "net_arch_pi": {"type": "categorical", "choices": [
        [256, 128, 64], [256, 256, 128], [512, 256, 128], [128, 64], [512, 256], [384, 192, 96]
    ]},
    "net_arch_vf": {"type": "categorical", "choices": [
        [256, 128, 64], [256, 256, 128], [512, 256, 128], [128, 64], [512, 256], [384, 192, 96]
    ]},
    "activation_fn": {"type": "categorical", "choices": ["ReLU", "Tanh", "ELU"]},
}

ENV_REWARD_PARAMS = {
    "max_drawdown": {"type": "float", "low": 0.15, "high": 0.50},
    "max_trade_step": {"type": "float", "low": 0.05, "high": 0.40},
    "drawdown_penalty_scale": {"type": "float", "low": 0.05, "high": 2.0, "log": True},
    "overtrade_penalty_scale": {"type": "float", "low": 1e-4, "high": 0.1, "log": True},
    "sharpe_weight": {"type": "float", "low": 0.0, "high": 0.3},
    "sharpe_aggregation_steps": {"type": "categorical", "choices": [1, 6, 12, 24, 48]},
    "step_return_weight": {"type": "float", "low": 0.5, "high": 2.0},
    "min_buffer_size": {"type": "categorical", "choices": [3, 5, 10, 20]},
}

FUTURES_PARAMS = {
    "short_leverage": {"type": "float", "low": 2.0, "high": 5.0},
    "maintenance_margin_ratio": {"type": "float", "low": 0.01, "high": 0.15},
    "stop_loss_atr_multiple": {"type": "float", "low": 0.5, "high": 3.0},
    "take_profit_atr_multiple": {"type": "float", "low": 1.0, "high": 5.0},
    "target_atr_pct": {"type": "float", "low": 0.5, "high": 2.0},
}


# =============================================================================
# Parameter Sampling
# =============================================================================

def current_branch() -> str:
    """Detect current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sample_params(trial: optuna.Trial, branch: str, search_space: str = "full") -> dict[str, Any]:
    """
    Sample hyperparameters for a single trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    branch : str
        Git branch name ('main' or 'futures').
    search_space : str
        'core' - PPO core only
        'core+lstm' - PPO + LSTM architecture
        'core+env' - PPO + Env/Reward
        'full' - everything (default)
    """
    params: dict[str, Any] = {}

    # Always include PPO core
    for name, spec in PPO_CORE_PARAMS.items():
        _suggest_param(trial, f"ppo.{name}", spec)

    if search_space in ("core+lstm", "full"):
        for name, spec in LSTM_ARCH_PARAMS.items():
            _suggest_param(trial, f"ppo.policy_kwargs.{name}", spec)

    if search_space in ("core+env", "full"):
        # Env parameters
        for name, spec in [("max_drawdown", ENV_REWARD_PARAMS["max_drawdown"]),
                            ("max_trade_step", ENV_REWARD_PARAMS["max_trade_step"])]:
            _suggest_param(trial, f"env.{name}", spec)

        # Reward parameters
        for name, spec in {k: v for k, v in ENV_REWARD_PARAMS.items()
                           if k not in ("max_drawdown", "max_trade_step")}.items():
            _suggest_param(trial, f"reward.{name}", spec)

    if branch == "futures" and search_space == "full":
        for name, spec in FUTURES_PARAMS.items():
            if name in ("short_leverage", "maintenance_margin_ratio"):
                _suggest_param(trial, f"portfolio.short.{name}", spec)
            else:
                _suggest_param(trial, f"risk.{name}", spec)

    return params


def _suggest_param(trial: optuna.Trial, key: str, spec: dict[str, Any]) -> None:
    """Suggest a single parameter based on its spec."""
    param_type = spec["type"]
    if param_type == "float":
        trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
    elif param_type == "categorical":
        trial.suggest_categorical(key, spec["choices"])
    else:
        raise ValueError(f"Unknown param type: {param_type}")


# =============================================================================
# Objective Function
# =============================================================================

def make_objective(
    seeds: list[int],
    timesteps: int | None,
    lambda_penalty: float,
    volatility_penalty: float,
    branch: str,
    search_space: str,
    no_checkpoints: bool = True,
):
    """
    Create the Optuna objective function.

    The objective combines:
    - median_sharpe: primary signal (robust to outliers)
    - breaker_rate: hard penalty (circuit breaker = failed episode)
    - sharpe_std: stability penalty (unstable configs unreliable)
    """
    def objective(trial: optuna.Trial) -> float:
        start_time = time.time()

        params = sample_params(trial, branch, search_space)

        summary = run_multi_seed(
            seeds=seeds,
            overrides=params,
            timesteps=timesteps,
            group_name=f"optuna_trial_{trial.number:04d}",
            no_checkpoints=no_checkpoints,
            trial=trial,
        )

        median_sharpe = summary["sharpe_ratio"]["median"]
        mean_sharpe = summary["sharpe_ratio"]["mean"]
        std_sharpe = summary["sharpe_ratio"]["std"]
        mean_return = summary["total_return"]["mean"]
        std_return = summary["total_return"]["std"]
        breaker_rate = summary["breaker_rate"]

        # Composite objective: maximize median Sharpe, penalize breakers and instability
        score = (
            median_sharpe
            - lambda_penalty * breaker_rate
            - volatility_penalty * std_sharpe
        )

        # Store all metrics for post-hoc analysis
        trial.set_user_attr("median_sharpe", median_sharpe)
        trial.set_user_attr("mean_sharpe", mean_sharpe)
        trial.set_user_attr("std_sharpe", std_sharpe)
        trial.set_user_attr("mean_return", mean_return)
        trial.set_user_attr("std_return", std_return)
        trial.set_user_attr("breaker_rate", breaker_rate)
        trial.set_user_attr("elapsed_seconds", time.time() - start_time)

        # Multi-objective: also report mean_return as secondary objective
        # (handled via user_attrs; Optuna single-objective but we track both)
        trial.set_user_attr("composite_score", score)

        logger.info(
            "Trial #{:04d} done: score={:.4f} median_sharpe={:.4f} "
            "breaker_rate={:.2f} std_sharpe={:.4f} return={:.4f}",
            trial.number, score, median_sharpe, breaker_rate, std_sharpe, mean_return
        )

        return score

    return objective


# =============================================================================
# Storage Helpers
# =============================================================================

def _resolve_storage_arg(branch: str, study_name: str) -> str:
    """Pick a default storage location shared by all worker processes."""
    if _USE_JOURNAL:
        journal_path = root("experiments") / f"optuna_{study_name}_{branch}.journal"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        return f"journal://{journal_path}"

    db_path = root("experiments") / f"optuna_{study_name}_{branch}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def _build_storage(storage_arg: str):
    """Turn a --storage value into whatever optuna.create_study expects."""
    if storage_arg.startswith("journal://"):
        journal_path = storage_arg[len("journal://"):]
        return JournalStorage(JournalFileBackend(journal_path))
    return storage_arg


# =============================================================================
# Reporting & Analysis
# =============================================================================

def _report_top_trials(study: optuna.Study, show_top: int, sort_by: str = "score") -> None:
    """Print top trials with key metrics."""
    completed = [t for t in study.trials if t.value is not None]

    if sort_by == "score":
        key_fn = lambda t: t.value
    elif sort_by == "median_sharpe":
        key_fn = lambda t: t.user_attrs.get("median_sharpe", float("-inf"))
    elif sort_by == "return":
        key_fn = lambda t: t.user_attrs.get("mean_return", float("-inf"))
    elif sort_by == "breaker_rate":
        key_fn = lambda t: -t.user_attrs.get("breaker_rate", float("inf"))  # lower is better
    else:
        key_fn = lambda t: t.value

    top_trials = sorted(completed, key=key_fn, reverse=True)[:show_top]

    logger.info("Top {} trials (sorted by {}):", len(top_trials), sort_by)
    for t in top_trials:
        logger.info(
            "  #{:04d} score={:.4f} | median_sharpe={:.4f} mean_sharpe={:.4f} "
            "std_sharpe={:.4f} | return={:.4f}±{:.4f} | breaker={:.2f} | "
            "lr={:.2e} ent={:.4f} n_steps={} bs={} gamma={:.3f} "
            "dd_pen={:.3f} sharpe_w={:.3f} agg={} dd_max={:.2f}",
            t.number,
            t.value,
            t.user_attrs.get("median_sharpe", float("nan")),
            t.user_attrs.get("mean_sharpe", float("nan")),
            t.user_attrs.get("std_sharpe", float("nan")),
            t.user_attrs.get("mean_return", float("nan")),
            t.user_attrs.get("std_return", float("nan")),
            t.user_attrs.get("breaker_rate", float("nan")),
            t.params.get("ppo.learning_rate", float("nan")),
            t.params.get("ppo.ent_coef", float("nan")),
            t.params.get("ppo.n_steps", "?"),
            t.params.get("ppo.batch_size", "?"),
            t.params.get("ppo.gamma", float("nan")),
            t.params.get("reward.drawdown_penalty_scale", float("nan")),
            t.params.get("reward.sharpe_weight", float("nan")),
            t.params.get("reward.sharpe_aggregation_steps", "?"),
            t.params.get("env.max_drawdown", float("nan")),
        )


def _save_best_config(study: optuna.Study, output_path: Path) -> None:
    """Save best trial's config as a YAML overrides file."""
    best = study.best_trial
    overrides = {k: v for k, v in best.params.items()}

    import yaml
    with output_path.open("w") as f:
        yaml.dump(overrides, f, default_flow_style=False, sort_keys=True)

    logger.success("Best config saved to {}", output_path)


def analyze_study(study_name: str, branch: str, storage: str | None, show_top: int = 20) -> None:
    """Load and analyze a completed study."""
    storage_arg = storage or _resolve_storage_arg(branch, study_name)
    study = optuna.load_study(study_name=f"{study_name}_{branch}", storage=_build_storage(storage_arg))

    logger.info(f"Loaded study '{study_name}_{branch}' with {len(study.trials)} trials")
    _report_top_trials(study, show_top, sort_by="score")
    _report_top_trials(study, min(10, show_top), sort_by="median_sharpe")
    _report_top_trials(study, min(10, show_top), sort_by="breaker_rate")

    # Parameter importance (requires completed trials with values)
    completed = [t for t in study.trials if t.value is not None]
    if len(completed) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            logger.info("Parameter importance (top 15):")
            for param, imp in sorted(importance.items(), key=lambda x: -x[1])[:15]:
                logger.info("  {}: {:.4f}", param, imp)
        except Exception as e:
            logger.warning("Could not compute parameter importance: {}", e)

    # Save best config
    output_path = root("experiments") / f"best_config_{study_name}_{branch}.yaml"
    _save_best_config(study, output_path)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhanced Optuna hyperparameter optimization for WARLOCK PPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sweep configuration
    parser.add_argument("--n-trials", type=int, default=30, help="Total trials to run.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of concurrent worker PROCESSES. "
                             "Each worker runs trials sequentially; all share one study storage.")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seeds evaluated per trial.")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override training.timesteps for speed during sweep.")
    parser.add_argument("--search-space", type=str, default="full",
                        choices=["core", "core+lstm", "core+env", "full"],
                        help="Which hyperparameter groups to search.")

    # Objective weighting
    parser.add_argument("--lambda-penalty", type=float, default=7.0,
                        help="Weight on breaker_rate in objective.")
    parser.add_argument("--volatility-penalty", type=float, default=0.5,
                        help="Weight on Sharpe std (instability penalty).")

    # Study management
    parser.add_argument("--study-name", type=str, default="warlock_ppo",
                        help="Base study name (branch appended automatically).")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL. Defaults to JournalStorage/SQLite in experiments/.")
    parser.add_argument("--pruner", type=str, choices=["median", "none", "hyperband"],
                        default="median", help="Optuna pruner type.")
    parser.add_argument("--n-startup-trials", type=int, default=5,
                        help="Trials before pruner activates.")
    parser.add_argument("--n-warmup-seeds", type=int, default=1,
                        help="Seeds per trial before that trial becomes eligible for pruning.")

    # Artifact management
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep all trial x seed checkpoints (default: cleanup).")

    # Analysis mode
    parser.add_argument("--analyze", action="store_true",
                        help="Load and analyze existing study instead of running new trials.")
    parser.add_argument("--show-top", type=int, default=15,
                        help="Number of top trials to display in analysis.")

    # Internal (worker mode)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    branch = current_branch()
    logger.info("Detected git branch: {}", branch)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    storage_arg = args.storage or _resolve_storage_arg(branch, args.study_name)
    study_name = f"{args.study_name}_{branch}"

    # Pruner setup
    if args.pruner == "median":
        pruner = MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_seeds,
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=len(seeds),
            reduction_factor=3,
        )
    else:
        pruner = NopPruner()

    # ---------------------------------------------------------------------
    # Analysis mode: just load and report
    # ---------------------------------------------------------------------
    if args.analyze:
        analyze_study(args.study_name, branch, args.storage, args.show_top)
        return 0

    # ---------------------------------------------------------------------
    # Orchestrator path: spawn worker processes
    # ---------------------------------------------------------------------
    if args.n_jobs > 1 and not args.worker:
        # Ensure study exists before workers race to create it
        optuna.create_study(
            study_name=study_name,
            storage=_build_storage(storage_arg),
            direction="maximize",
            pruner=pruner,
            load_if_exists=True,
        )

        base, remainder = divmod(args.n_trials, args.n_jobs)
        trials_per_worker = [base + (1 if i < remainder else 0) for i in range(args.n_jobs)]
        trials_per_worker = [n for n in trials_per_worker if n > 0]

        logger.info(
            "Launching {} worker processes on branch '{}' (trials per worker: {})",
            len(trials_per_worker), branch, trials_per_worker
        )

        processes: list[subprocess.Popen] = []
        for worker_trials in trials_per_worker:
            cmd = [
                sys.executable, "-m", "src.agent.hpo_optuna",
                "--n-trials", str(worker_trials),
                "--seeds", args.seeds,
                "--lambda-penalty", str(args.lambda_penalty),
                "--volatility-penalty", str(args.volatility_penalty),
                "--study-name", args.study_name,
                "--storage", storage_arg,
                "--pruner", args.pruner,
                "--n-startup-trials", str(args.n_startup_trials),
                "--n-warmup-seeds", str(args.n_warmup_seeds),
                "--search-space", args.search_space,
                "--worker",
            ]
            if args.timesteps is not None:
                cmd += ["--timesteps", str(args.timesteps)]
            if args.keep_checkpoints:
                cmd += ["--keep-checkpoints"]

            processes.append(subprocess.Popen(cmd, cwd=str(root())))

        failures = 0
        for process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures += 1
                logger.error("Worker pid={} exited with code {}", process.pid, return_code)

        if failures:
            logger.warning("{} of {} workers failed -- see logs above.", failures, len(processes))

        # Final report from orchestrator
        study = optuna.load_study(study_name=study_name, storage=_build_storage(storage_arg))
        logger.success(
            "Sweep complete. Best score={:.4f} (trial #{})",
            study.best_value, study.best_trial.number
        )
        _report_top_trials(study, args.show_top)
        _save_best_config(study, root("experiments") / f"best_config_{study_name}.yaml")

        return 1 if failures else 0

    # ---------------------------------------------------------------------
    # Single-process path (also used by each worker with --worker)
    # ---------------------------------------------------------------------
    study = optuna.create_study(
        study_name=study_name,
        storage=_build_storage(storage_arg),
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        make_objective(
            seeds=seeds,
            timesteps=args.timesteps,
            lambda_penalty=args.lambda_penalty,
            volatility_penalty=args.volatility_penalty,
            branch=branch,
            search_space=args.search_space,
            no_checkpoints=not args.keep_checkpoints,
        ),
        n_trials=args.n_trials,
        n_jobs=1,  # this process does one trial at a time
    )

    # Worker processes leave reporting to orchestrator
    if args.worker:
        return 0

    logger.success(
        "Sweep complete. Best score={:.4f} (trial #{})",
        study.best_value, study.best_trial.number
    )
    _report_top_trials(study, args.show_top)
    _save_best_config(study, root("experiments") / f"best_config_{study_name}.yaml")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())