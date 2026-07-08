from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from stable_baselines3 import PPO

from src.analytics.metrics import MetricsCalculator
from src.analytics.plots import PlotGenerator
from src.analytics.report import ReportGenerator
from src.env.gym_bitcoin import GymBitcoinEnv
from src.utils import config,root


class Evaluator:
    """
    Evaluates a trained PPO model on the held-out test dataset.
    """

    def __init__(
        self,
        experiment_directory: str | Path,
    ) -> None:

        self._experiment_dir = Path(experiment_directory)

        if not self._experiment_dir.exists():
            raise FileNotFoundError(
                f"Experiment directory not found: {self._experiment_dir}"
            )

        self._model_path = self._experiment_dir / "model.zip"

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self._model_path}"
            )

        self._evaluation_cfg = config["evaluation"]
        self._training_cfg = config["training"]

        logger.info(
            "Loading PPO model from {}",
            self._model_path,
        )

        self._model = PPO.load(
            self._model_path,
            device=self._training_cfg["device"],
        )

        logger.success("Model loaded successfully.")

        test_data = (
            root("data", "features", "test.parquet")
)

        self._environment = GymBitcoinEnv(
    data_path=str(test_data),
)
        self._evaluation_dir = (
            self._experiment_dir / "evaluation"
        )

        self._plots_dir = (
            self._evaluation_dir / "plots"
        )

        self._evaluation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._equity_curve: list[float] = []
        self._trade_returns: list[float] = []
        self._portfolio_history: list[dict] = []
        self._trade_history: list[dict] = []

        logger.info(
            "Evaluation initialized."
        )
        
    def _run_episode(self) -> None:
        """
        Runs one deterministic evaluation episode and records
        portfolio and trade history.
        """

        logger.info("Starting evaluation episode.")

        observation, _ = self._environment.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action, _ = self._model.predict(
                observation,
                deterministic=self._evaluation_cfg["deterministic"],
            )

            observation, reward, terminated, truncated, info = (
                self._environment.step(action)
            )

            self._equity_curve.append(
                float(info["capital"])
            )

            self._portfolio_history.append(
                {
                    "step": int(info["step"]),
                    "price": float(info["price"]),
                    "capital": float(info["capital"]),
                    "cash": float(info["cash"]),
                    "drawdown": float(info["drawdown"]),
                    "realized_pnl": float(info["realized_pnl"]),
                    "unrealized_pnl": float(info["unrealized_pnl"]),
                    "reward": float(reward),
                    "weights": list(info["weights"]),
                    "forced_exit": bool(info["forced_exit"]),
                    "exit_reason": info["exit_reason"],
                }
            )

            if int(info["n_trades_this_step"]) > 0:

                realized = float(info["realized_pnl"])

                self._trade_returns.append(realized)

                self._trade_history.append(
                    {
                        "step": int(info["step"]),
                        "price": float(info["price"]),
                        "realized_pnl": realized,
                        "forced_exit": bool(info["forced_exit"]),
                        "exit_reason": info["exit_reason"],
                    }
                )

        logger.success(
            "Evaluation episode completed."
        )

    def _save_csv_outputs(self) -> None:
        """
        Persist evaluation history to CSV files.
        """

        if self._evaluation_cfg["save_equity_curve"]:

            pd.DataFrame(
                {
                    "step": range(len(self._equity_curve)),
                    "capital": self._equity_curve,
                }
            ).to_csv(
                self._evaluation_dir / "equity_curve.csv",
                index=False,
            )

        if self._evaluation_cfg["save_portfolio_history"]:

            pd.DataFrame(
                self._portfolio_history,
            ).to_csv(
                self._evaluation_dir / "portfolio.csv",
                index=False,
            )

        if self._evaluation_cfg["save_trade_history"]:

            pd.DataFrame(
                self._trade_history,
            ).to_csv(
                self._evaluation_dir / "trades.csv",
                index=False,
            )

        logger.success(
            "Evaluation CSV files saved."
        )
        
    def _generate_analytics(self) -> None:
        """
        Generate metrics, plots and PDF report.
        """

        logger.info("Generating evaluation analytics.")

        metrics = MetricsCalculator(
            equity_curve=pd.Series(self._equity_curve),
            trade_returns=pd.Series(self._trade_returns),
            risk_free_rate=self._evaluation_cfg["risk_free_rate"],
        )

        metrics.save_json(
            self._evaluation_dir / "metrics.json",
        )

        if self._evaluation_cfg["generate_plots"]:

            PlotGenerator(
                equity_curve=pd.Series(self._equity_curve),
                trade_returns=pd.Series(self._trade_returns),
                output_directory=self._plots_dir,
            ).generate_all(
                rolling_sharpe_window=self._evaluation_cfg[
                    "rolling_sharpe_window"
                ]
            )

        if self._evaluation_cfg["generate_report"]:

            ReportGenerator(
                evaluation_directory=self._evaluation_dir,
                plots_directory=self._plots_dir,
            ).generate()

        logger.success(
            "Analytics generated successfully."
        )

    def evaluate(self) -> None:
        """
        Execute the complete evaluation pipeline.
        """

        logger.info("=" * 80)
        logger.info("WARLOCK Evaluation")
        logger.info("=" * 80)

        self._run_episode()

        self._save_csv_outputs()

        self._generate_analytics()

        logger.success(
            "Evaluation completed successfully."
        )


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Evaluate a trained WARLOCK experiment."
    )

    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to experiment directory.",
    )

    args = parser.parse_args()

    evaluator = Evaluator(
        experiment_directory=args.experiment,
    )

    evaluator.evaluate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())