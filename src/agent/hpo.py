from __future__ import annotations

"""
Optuna sweep on top of the multi-seed harness.

Objective (per the discussion this implements):
    score = median(sharpe_across_seeds) - lambda * breaker_rate

Median rather than mean so one blown-up/lucky seed doesn't dominate with
only 3-4 seeds per trial. The breaker-rate penalty is a *soft* term
(rather than a hard filter) so the sweep still gets gradient signal even
when most/all configs currently hit the drawdown breaker (true for this
project's baseline runs) -- a hard filter would disqualify almost every
trial early on and starve the search. Treat the penalized ranking as a
shortlist: manually sanity-check the top trials' breaker_rate afterwards
(see --show-top) rather than trusting the top-1 blindly, since a config
can still score well by being great on its non-breaking seeds alone.

Branch awareness
-----------------
The search space is assembled from three parts:
  1. PPO hyperparameters -- identical on both branches.
  2. Env/reward hyperparameters -- identical on both branches (both read
     `env.*` / `reward.*`; the futures branch just interprets weights as
     signed).
  3. Futures-only hyperparameters (`portfolio.short.leverage`,
     `portfolio.short.maintenance_margin_ratio`) -- only added to the
     search space when running on the `futures` branch, since `main`'s
     SpotPortfolio has no such keys and PPOTrainer/GymBitcoinEnv on main
     never reads them.

Branch is detected via git, not hardcoded, so the same script works
whichever branch you have checked out.

Concurrency: kept sequential (n_jobs=1). Config overrides are applied by
mutating the shared, module-level config dict in place (see
config_overrides.py); running trials in parallel processes/threads would
race on that dict. If you need parallel trials, give each worker its own
process with its own config.yaml/import, or refactor config threading
before increasing n_jobs.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Any

import optuna
from loguru import logger

from src.agent.multi_seed import run_multi_seed
from src.utils import root


def current_branch() -> str:
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


def sample_params(trial: optuna.Trial, branch: str) -> dict[str, Any]:
    """Builds the dotted-path config overrides for one trial."""

    params: dict[str, Any] = {
        # -- PPO hyperparameters (both branches) --
        "ppo.learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        ),
        "ppo.n_steps": trial.suggest_categorical(
            "n_steps", [1024, 2048, 4096]
        ),
        "ppo.batch_size": trial.suggest_categorical(
            "batch_size", [32, 64, 128]
        ),
        "ppo.gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "ppo.gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "ppo.clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ppo.ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
        "ppo.vf_coef": trial.suggest_float("vf_coef", 0.3, 0.8),

        # -- Env / reward hyperparameters (both branches) --
        # These directly target the observed failure mode (3/4 baseline
        # runs terminating via the drawdown breaker), so they're in the
        # search space rather than left fixed.
        "env.max_drawdown": trial.suggest_float("max_drawdown", 0.15, 0.40),
        "reward.drawdown_penalty_scale": trial.suggest_float(
            "drawdown_penalty_scale", 0.05, 0.5, log=True
        ),
        "reward.overtrade_penalty_scale": trial.suggest_float(
            "overtrade_penalty_scale", 1e-4, 0.05, log=True
        ),
        "reward.sharpe_weight": trial.suggest_float(
            "sharpe_weight", 0.0, 0.5
        ),
    }

    if branch == "futures":
        # Futures-only knobs -- absent on main's SpotPortfolio, and
        # meaningless there, so only sampled on this branch.
        params["portfolio.short.leverage"] = trial.suggest_float(
            "short_leverage", 1.0, 5.0
        )
        params["portfolio.short.maintenance_margin_ratio"] = trial.suggest_float(
            "maintenance_margin_ratio", 0.02, 0.10
        )

    return params


def make_objective(
    seeds: list[int],
    timesteps: int | None,
    lambda_penalty: float,
    branch: str,
    no_checkpoints: bool = True,
):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, branch)

        summary = run_multi_seed(
            seeds=seeds,
            overrides=params,
            timesteps=timesteps,
            group_name=f"optuna_trial_{trial.number:04d}",
            no_checkpoints=no_checkpoints,
        )

        median_sharpe = summary["sharpe_ratio"]["median"]
        breaker_rate = summary["breaker_rate"]

        score = median_sharpe - lambda_penalty * breaker_rate

        # Keep the full picture attached to the trial so a good penalized
        # score can still be sanity-checked (or thrown out) afterwards --
        # e.g. discard top trials whose breaker_rate is still > 0.5 even
        # if their score looks good.
        trial.set_user_attr("median_sharpe", median_sharpe)
        trial.set_user_attr("mean_sharpe", summary["sharpe_ratio"]["mean"])
        trial.set_user_attr("std_sharpe", summary["sharpe_ratio"]["std"])
        trial.set_user_attr("mean_return", summary["total_return"]["mean"])
        trial.set_user_attr("std_return", summary["total_return"]["std"])
        trial.set_user_attr("breaker_rate", breaker_rate)

        return score

    return objective


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna sweep for WARLOCK PPO.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds evaluated per trial.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override training.timesteps for speed during the sweep.",
    )
    parser.add_argument(
        "--lambda-penalty",
        type=float,
        default=7.0,
        help="Weight on breaker_rate in the objective "
        "(score = median_sharpe - lambda * breaker_rate).",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="warlock_ppo_sweep",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///experiments/optuna.db. "
        "Defaults to experiments/optuna_<branch>.db so trials resume "
        "across interrupted runs.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=5,
        help="Print the top N trials (with breaker_rate) at the end.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Disable the default no-checkpoints cleanup and keep every "
        "trial x seed's full checkpoint/tensorboard artifacts. Off by "
        "default because a sweep of n_trials x n_seeds full checkpoint "
        "sets adds up fast; model.zip/best_model.zip are always kept "
        "either way.",
    )

    args = parser.parse_args()

    branch = current_branch()
    logger.info("Detected git branch: {}", branch)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    storage = args.storage
    if storage is None:
        db_path = root("experiments") / f"optuna_{branch}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"{args.study_name}_{branch}",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(
        make_objective(
            seeds=seeds,
            timesteps=args.timesteps,
            lambda_penalty=args.lambda_penalty,
            branch=branch,
            no_checkpoints=not args.keep_checkpoints,
        ),
        n_trials=args.n_trials,
        n_jobs=1,  # sequential -- see module docstring on config mutation
    )

    logger.success(
        "Sweep complete. Best score={:.4f} (trial #{})",
        study.best_value,
        study.best_trial.number,
    )

    top_trials = sorted(
        (t for t in study.trials if t.value is not None),
        key=lambda t: t.value,
        reverse=True,
    )[: args.show_top]

    logger.info("Top {} trials (score | median_sharpe | breaker_rate):", len(top_trials))
    for t in top_trials:
        logger.info(
            "  #{:04d} score={:.4f} median_sharpe={:.4f} breaker_rate={:.2f} params={}",
            t.number,
            t.value,
            t.user_attrs.get("median_sharpe", float("nan")),
            t.user_attrs.get("breaker_rate", float("nan")),
            t.params,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())