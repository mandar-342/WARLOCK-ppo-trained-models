from __future__ import annotations

"""
Optuna sweep on top of the multi-seed harness, with support for running
multiple trials concurrently (--n-jobs > 1).

Concurrency model
------------------
Trials run as **separate OS processes**, not threads, for two reasons
specific to this project:

1. Config isolation. `src/utils/config.py` loads config.yaml once into a
   module-level dict that every module (`PPOTrainer`, `GymBitcoinEnv`,
   ...) holds a reference to, and `config_overrides.py` mutates that dict
   in place per run. Within a single process, concurrent trials
   mutating the same dict would race. Across processes there's no such
   risk: each process gets its own independent copy of everything at
   startup, so config.py itself needs no restructuring for this to be
   safe -- the isolation is free.
2. CUDA. PyTorch/CUDA contexts don't fork safely, so with a shared GPU
   (this project's setup), process-based parallelism is the technically
   correct choice here, not just the simpler one.

With --n-jobs > 1, this script spawns N subprocesses (each re-invoking
this same module with `--worker`), all pointed at the same Optuna
storage, and waits for them. Each worker independently pulls trials via
`study.optimize(n_trials=trials_per_worker)`. This is Optuna's standard
local-multiprocess pattern.

Storage: defaults to `optuna.storages.JournalStorage` with a
`JournalFileBackend` (safe for concurrent local processes via file
locking) when available (Optuna >= 3.1), falling back to a SQLite URL
otherwise. Either way, all worker processes must share one storage
location -- that's what makes them one sweep rather than N independent
sweeps.

GPU note: all workers share whatever `training.device` resolves to
(default "auto" -> the single GPU here). CUDA contexts from multiple
processes on one GPU is normal and supported, but memory is shared and
finite -- if you see OOMs, reduce --n-jobs or `ppo.batch_size` in the
search space.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
from loguru import logger

from src.agent.multi_seed import run_multi_seed
from src.utils import root

try:
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    _HAS_JOURNAL_STORAGE = True
except ImportError:  # Optuna < 3.1
    _HAS_JOURNAL_STORAGE = False


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
            "learning_rate", 1e-5, 3e-4, log=True
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
            "short_leverage", 2.5, 3.5
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
            trial=trial,
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


def _resolve_storage_arg(branch: str) -> str:
    """
    Picks a default storage location shared by all worker processes.
    Prefers Optuna's JournalStorage (file-lock based, safe for
    concurrent local processes) when available; falls back to SQLite
    for older Optuna installs.
    """
    if _HAS_JOURNAL_STORAGE:
        journal_path = root("experiments") / f"optuna_{branch}.journal"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        return f"journal://{journal_path}"

    db_path = root("experiments") / f"optuna_{branch}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def _build_storage(storage_arg: str):
    """Turns a --storage value into whatever optuna.create_study expects."""
    if storage_arg.startswith("journal://"):
        journal_path = storage_arg[len("journal://"):]
        return JournalStorage(JournalFileBackend(journal_path))
    return storage_arg  # sqlite:// or any other optuna-native URL


def _report_top_trials(study: optuna.Study, show_top: int) -> None:
    top_trials = sorted(
        (t for t in study.trials if t.value is not None),
        key=lambda t: t.value,
        reverse=True,
    )[:show_top]

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna sweep for WARLOCK PPO.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of concurrent worker PROCESSES sharing one sweep "
        "(see module docstring for why processes, not threads). "
        "--n-trials is split across them.",
    )
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
        help="Optuna storage location shared by all workers. Defaults to "
        "a JournalStorage file (or SQLite if Optuna < 3.1) under "
        "experiments/, keyed by branch, so trials resume across "
        "interrupted/re-launched sweeps.",
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
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: set on the subprocesses this script spawns itself
    )
    parser.add_argument(
        "--pruner",
        type=str,
        choices=["median", "none"],
        default="median",
        help="Optuna pruner. 'median' stops a trial's remaining seeds "
        "once its running-median Sharpe (after each completed seed) "
        "falls below the median of other trials at the same "
        "seeds-completed checkpoint. 'none' disables pruning.",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=5,
        help="Trials always run to completion before the pruner starts "
        "judging (needs a baseline population to compare against).",
    )
    parser.add_argument(
        "--n-warmup-seeds",
        type=int,
        default=1,
        help="Seeds completed within a trial before that trial becomes "
        "eligible for pruning (per-trial warmup, on the "
        "seeds-completed step axis -- not related to --n-startup-trials).",
    )

    args = parser.parse_args()

    branch = current_branch()
    logger.info("Detected git branch: {}", branch)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    storage_arg = args.storage or _resolve_storage_arg(branch)

    study_name = f"{args.study_name}_{branch}"

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_seeds,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # Orchestrator path: spawn --n-jobs worker subprocesses, each running
    # this same module with --worker and a share of --n-trials, all
    # pointed at the same storage. This process itself does no training.
    if args.n_jobs > 1 and not args.worker:
        # Make sure the study exists before any worker races to create it.
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
            len(trials_per_worker),
            branch,
            trials_per_worker,
        )

        processes: list[subprocess.Popen] = []
        for worker_trials in trials_per_worker:
            cmd = [
                sys.executable,
                "-m",
                "src.agent.hpo",
                "--n-trials",
                str(worker_trials),
                "--seeds",
                args.seeds,
                "--lambda-penalty",
                str(args.lambda_penalty),
                "--study-name",
                args.study_name,
                "--storage",
                storage_arg,
                "--pruner",
                args.pruner,
                "--n-startup-trials",
                str(args.n_startup_trials),
                "--n-warmup-seeds",
                str(args.n_warmup_seeds),
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

        study = optuna.load_study(study_name=study_name, storage=_build_storage(storage_arg))
        logger.success(
            "Sweep complete. Best score={:.4f} (trial #{})",
            study.best_value,
            study.best_trial.number,
        )
        _report_top_trials(study, args.show_top)

        return 1 if failures else 0

    # Single-process path: also used for each spawned worker (--worker),
    # and for the default --n-jobs=1 case.
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
            branch=branch,
            no_checkpoints=not args.keep_checkpoints,
        ),
        n_trials=args.n_trials,
        n_jobs=1,  # this process contributes one trial at a time; see --n-jobs for multi-process concurrency
    )

    # A spawned worker leaves the summary to the orchestrator, which
    # reports over the full shared study once every worker has finished.
    if args.worker:
        return 0

    logger.success(
        "Sweep complete. Best score={:.4f} (trial #{})",
        study.best_value,
        study.best_trial.number,
    )
    _report_top_trials(study, args.show_top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())