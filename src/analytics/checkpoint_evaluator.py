from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.agent.evaluate import Evaluator


class CheckpointEvaluator:
    """
    Evaluates every checkpoint in an experiment directory and
    produces a summary dataframe.
    """

    def __init__(
        self,
        experiment_directory: str | Path,
    ) ->None:

        self._experiment = Path(experiment_directory)

        self._checkpoint_dir = (
            self._experiment / "checkpoints"
        )

        if not self._checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self._checkpoint_dir}"
            )

        self._output_dir = (
            self._experiment / "checkpoint_evaluation"
        )

        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _discover_checkpoints(self) -> list[tuple[int, Path, Path]]:
        checkpoints = []

        models = sorted(
        self._checkpoint_dir.glob("checkpoint_*_steps.zip"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

        for model in models:
            step = int(model.stem.split("_")[1])

            vec = self._checkpoint_dir / (
            f"checkpoint_vecnormalize_{step}_steps.pkl"
        )

            if not vec.exists():
                logger.warning(
                "Missing VecNormalize for checkpoint {}",
                step,
            )
                continue

            checkpoints.append(
            (
                step,
                model,
                vec,
            )
        )

        return checkpoints

    def run(self) -> pd.DataFrame:

        rows = []

        checkpoints = self._discover_checkpoints()

        logger.info(
            "Found {} checkpoints.",
            len(checkpoints),
        )

        for step, model, vec in checkpoints:

            logger.info(
                "Evaluating checkpoint {}",
                step,
            )

            evaluator = Evaluator(
                experiment_directory=self._experiment,
                model_path=model,
                vecnormalize_path=vec,
            )

            metrics = evaluator.evaluate()

            metrics["checkpoint"] = step

            rows.append(metrics)

        summary = (
            pd.DataFrame(rows)
            .sort_values("checkpoint")
            .reset_index(drop=True)
        )

        summary.to_csv(
            self._output_dir / "checkpoint_summary.csv",
            index=False,
        )

        best = summary.sort_values(
            "sharpe_ratio",
            ascending=False,
        ).iloc[0]

        logger.success(
            "Best checkpoint: {}",
            int(best["checkpoint"]),
        )

        logger.success(
            "Sharpe: {:.4f}",
            best["sharpe_ratio"],
        )

        logger.success(
            "Return: {:.2%}",
            best["total_return"],
        )

        logger.success(
            "Profit Factor: {:.4f}",
            best["profit_factor"],
        )

        return summary