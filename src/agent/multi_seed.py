from __future__ import annotations

"""
Runs the same config across multiple seeds and reports the *distribution*
of results (mean/std/median Sharpe & return, plus how often the drawdown
circuit breaker fired) instead of a single run's numbers.

Works unchanged on both `main` (long-only / SpotPortfolio) and `futures`
(long+short / unified Portfolio): everything here goes through
PPOTrainer / GymBitcoinEnv / the global config dict, none of which differ
in their public interface between the two branches (only the portfolio
internals do). Branch-specific hyperparameters (e.g. futures' short
leverage) are the sweep's concern (see hpo.py), not this harness's.

CLI
---
python -m src.agent.multi_seed --seeds 0,1,2 --group-name baseline
python -m src.agent.multi_seed --seeds 0,1,2,3,4 --timesteps 20000 \
    --set ppo.learning_rate=1e-4 --set env.max_drawdown=0.25
"""

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.config_overrides import override_config
from src.agent.quick_eval import run_quick_episode
from src.agent.trainer import PPOTrainer
from src.utils import config, root

# Large enough that CheckpointCallback's periodic save_freq never
# triggers within any realistic training run, without touching
# callbacks.py/trainer.py. Used by the `no_checkpoints` option below.
_NO_CHECKPOINT_FREQUENCY = 10**9


def _cleanup_run_directory(experiment) -> None:
    """
    Removes the bulky, disposable parts of an experiment directory
    (per-step checkpoints, tensorboard event files) while keeping what
    the harness/analytics actually need afterwards: model.zip,
    best_model.zip, metadata.json, and the config.yaml snapshot.

    Safe to call even if checkpoint saving was already suppressed via
    `training.checkpoint_frequency` (checkpoints_directory will just be
    empty in that case).
    """
    for directory in (experiment.checkpoints_directory, experiment.tensorboard_directory):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


def _parse_value(raw: str) -> Any:
    """Best-effort type coercion for `--set key=value` CLI overrides."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def run_single_seed(
    seed: int,
    overrides: dict[str, Any] | None = None,
    timesteps: int | None = None,
    no_checkpoints: bool = False,
) -> dict[str, Any]:
    """
    Trains and evaluates one PPO run for a given seed + config overrides.

    Parameters
    ----------
    no_checkpoints
        If True, suppresses CheckpointCallback's periodic checkpoint_*.zip
        saves (via a config override, so callbacks.py is untouched) and
        deletes the run's tensorboard event files afterwards. model.zip,
        best_model.zip, metadata.json, and the config.yaml snapshot are
        always kept -- they're small and needed for eval/traceability.
        Meant for sweeps/multi-seed batches where dozens of full
        checkpoint sets would otherwise pile up in experiments/.

    Returns a dict with the evaluation metrics, the breaker flag, and the
    experiment directory used, for traceability.
    """

    seed_overrides = dict(overrides or {})
    seed_overrides["training.seed"] = seed

    if no_checkpoints:
        seed_overrides["training.checkpoint_frequency"] = _NO_CHECKPOINT_FREQUENCY

    with override_config(seed_overrides):
        trainer = PPOTrainer(total_timesteps=timesteps)
        try:
            trainer.train()

            # Prefer the EvalCallback's best_model.zip when it exists --
            # it's selected on the eval env during training, so it's a
            # slightly less noisy pick than whatever the final checkpoint
            # happens to be. Fall back to the final model otherwise.
            best_model_path = trainer.experiment.best_model_path
            model_path = (
                best_model_path
                if best_model_path.exists()
                else trainer.experiment.model_path
            )
            model = trainer.load(model_path=str(model_path))

            metrics, breaker_hit = run_quick_episode(model)
        finally:
            trainer.close()
            if no_checkpoints:
                _cleanup_run_directory(trainer.experiment)

    return {
        "seed": seed,
        "experiment_dir": str(trainer.experiment.run_directory),
        "breaker_hit": breaker_hit,
        **metrics,
    }


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def run_multi_seed(
    seeds: list[int],
    overrides: dict[str, Any] | None = None,
    timesteps: int | None = None,
    group_name: str = "multi_seed",
    save: bool = True,
    no_checkpoints: bool = False,
    trial: Any = None,
) -> dict[str, Any]:
    """
    Runs `run_single_seed` for every seed in `seeds`, then aggregates.

    Parameters
    ----------
    trial
        Optional `optuna.Trial`. When given, reports the running median
        Sharpe of completed seeds after each seed finishes (step = number
        of seeds completed so far), and raises `optuna.TrialPruned` if
        the trial's pruner says to stop -- skipping this trial's
        remaining seeds entirely.

        Pruning here is *inter-seed*, not mid-training: it can only save
        the cost of seeds not yet run, not partial training within a
        seed. That's a deliberate simplification -- it needs no changes
        to trainer.py/callbacks.py (no hook into SB3's training loop),
        and most of a doomed trial's wasted cost is exactly its later
        seeds, since one seed's Sharpe is usually a decent early signal
        of whether a config is competitive at all.

        Because pruning decisions happen between seeds, results are only
        comparable trial-to-trial at the same "seeds completed" count --
        this is what the pruner's step axis represents, not a wall-clock
        or training-timestep axis.

    Returns a summary dict with per-seed results plus mean/std/median for
    sharpe_ratio and total_return, and the fraction of seeds that hit the
    drawdown circuit breaker. If pruned, a summary with
    `"pruned": True` and only the completed seeds is saved (when `save`)
    before `optuna.TrialPruned` is raised.
    """

    per_seed_results: list[dict[str, Any]] = []

    def _save_partial(pruned: bool) -> None:
        if not save or not per_seed_results:
            return
        sharpe_values = [r["sharpe_ratio"] for r in per_seed_results]
        return_values = [r["total_return"] for r in per_seed_results]
        breaker_rate = sum(r["breaker_hit"] for r in per_seed_results) / len(per_seed_results)
        partial_summary = {
            "group_name": group_name,
            "overrides": overrides or {},
            "timesteps": timesteps,
            "seeds": seeds,
            "pruned": pruned,
            "breaker_rate": breaker_rate,
            "sharpe_ratio": _summarize(sharpe_values),
            "total_return": _summarize(return_values),
            "per_seed": per_seed_results,
        }
        output_dir = root("experiments", "multi_seed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{group_name}.json"
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(partial_summary, fp, indent=4, sort_keys=True, default=str)

    for seed in seeds:
        logger.info(
            "[{}] Running seed {} ({}/{})",
            group_name,
            seed,
            len(per_seed_results) + 1,
            len(seeds),
        )
        result = run_single_seed(
            seed=seed,
            overrides=overrides,
            timesteps=timesteps,
            no_checkpoints=no_checkpoints,
        )
        per_seed_results.append(result)

        logger.info(
            "[{}] seed={} sharpe={:.4f} total_return={:.4f} breaker_hit={}",
            group_name,
            seed,
            result["sharpe_ratio"],
            result["total_return"],
            result["breaker_hit"],
        )

        if trial is not None:
            import optuna  # local import: multi_seed.py stays usable without optuna installed when trial is None

            running_median = statistics.median(
                r["sharpe_ratio"] for r in per_seed_results
            )
            trial.report(running_median, step=len(per_seed_results))

            if trial.should_prune():
                logger.info(
                    "[{}] pruned after {}/{} seeds (running median sharpe={:.4f})",
                    group_name,
                    len(per_seed_results),
                    len(seeds),
                    running_median,
                )
                _save_partial(pruned=True)
                raise optuna.TrialPruned(
                    f"Pruned after {len(per_seed_results)}/{len(seeds)} seeds "
                    f"(running median sharpe={running_median:.4f})"
                )

    sharpe_values = [r["sharpe_ratio"] for r in per_seed_results]
    return_values = [r["total_return"] for r in per_seed_results]
    breaker_rate = sum(r["breaker_hit"] for r in per_seed_results) / len(per_seed_results)

    summary = {
        "group_name": group_name,
        "overrides": overrides or {},
        "timesteps": timesteps,
        "seeds": seeds,
        "pruned": False,
        "breaker_rate": breaker_rate,
        "sharpe_ratio": _summarize(sharpe_values),
        "total_return": _summarize(return_values),
        "per_seed": per_seed_results,
    }

    if save:
        output_dir = root("experiments", "multi_seed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{group_name}.json"
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=4, sort_keys=True, default=str)
        logger.success("Saved multi-seed summary to {}", output_path)

    logger.info(
        "[{}] sharpe mean={:.4f} std={:.4f} median={:.4f} | "
        "return mean={:.4f} std={:.4f} | breaker_rate={:.2f}",
        group_name,
        summary["sharpe_ratio"]["mean"],
        summary["sharpe_ratio"]["std"],
        summary["sharpe_ratio"]["median"],
        summary["total_return"]["mean"],
        summary["total_return"]["std"],
        breaker_rate,
    )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a config across multiple seeds and report distributions."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated list of seeds, e.g. '0,1,2'.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override training.timesteps. Defaults to config.yaml value.",
    )
    parser.add_argument(
        "--group-name",
        type=str,
        default="multi_seed",
        help="Name used for the saved summary file and logging.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Dotted-path config override, e.g. --set ppo.learning_rate=1e-4. "
        "Repeatable.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Suppress periodic checkpoint_*.zip saves and delete "
        "tensorboard event files for each run once it's evaluated. "
        "model.zip, best_model.zip, and metadata.json are always kept.",
    )

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    overrides: dict[str, Any] = {}
    for item in args.set:
        key, _, raw_value = item.partition("=")
        overrides[key] = _parse_value(raw_value)

    run_multi_seed(
        seeds=seeds,
        overrides=overrides,
        timesteps=args.timesteps,
        group_name=args.group_name,
        no_checkpoints=args.no_checkpoints,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())