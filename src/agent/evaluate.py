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
    deterministic_start=True,
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
        self._equity_steps: list[int] = []
        self._trade_returns: list[float] = []
        self._portfolio_history: list[dict] = []
        self._trade_history: list[dict] = []
        self._reward_history: list[dict] = []
        self._action_history: list[dict] = []
        # `info["realized_pnl"]` is the portfolio's *cumulative* realized
        # P&L since episode start (see Portfolio.realized_pnl()), not the
        # individual trade's own P&L. We track the previous cumulative
        # value so we can compute the marginal delta attributable to each
        # trade event below, instead of mistakenly treating the running
        # total as a per-trade return.
        self._prev_realized_pnl: float = 0.0

        logger.info(
            "Evaluation initialized."
        )
        
    def _run_episode(self) -> None:
        """
        Runs one deterministic evaluation episode and records
        portfolio and trade history.
        """

        logger.info("Starting evaluation episode.")

        # Defensive reset: guards against stale state if `evaluate()` is
        # ever called more than once on the same Evaluator instance (the
        # lists otherwise accumulate across calls instead of representing
        # a single clean episode).
        self._equity_curve = []
        self._equity_steps = []
        self._trade_returns = []
        self._portfolio_history = []
        self._trade_history = []
        self._reward_history = []
        self._action_history = []
        self._prev_realized_pnl = 0.0

        # `deterministic_start=True` already removes the only source of
        # randomness in this env (see gym_bitcoin.py), but we pass an
        # explicit seed too as defense-in-depth in case a stochastic
        # element (e.g. slippage noise) is ever added later.
        observation, _ = self._environment.reset(
            seed=self._evaluation_cfg.get("seed", 42)
        )

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
            self._action_history.append(
       {
        "step": int(info["step"]),
        "raw_action": float(info["raw_action"][0]),
        "target_weight": float(info["target_weights"][0]),
        "position_weight": float(info["position_sized_weights"][0]),
        "risk_multiplier": float(info["risk_multiplier"]),
        "forced_exit": bool(info["forced_exit"]),
        "exit_reason": info["exit_reason"],
        }
            )
            reward_components = info["reward_components"]
            
            self._reward_history.append(
                {
                    "step_return": float(reward_components["step_return"]),
                    "reward_return": float(reward_components["reward_return"]),
                    "drawdown_penalty": float(reward_components["drawdown_penalty"]),
                    "overtrade_penalty": float(reward_components["overtrade_penalty"]),
                    "total_reward": float(reward_components["total_reward"]),
                }
            )

            self._equity_curve.append(
                float(info["capital"])
            )
            self._equity_steps.append(
                int(info["step"])
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

                cumulative_realized = float(info["realized_pnl"])
                # Marginal P&L realized by trade(s) executed this step. If
                # multiple trades landed in the same step, this is their
                # combined effect; there isn't enough granularity in `info`
                # to split them further without also changing the env.
                realized = cumulative_realized - self._prev_realized_pnl

                self._trade_returns.append(realized)

                self._trade_history.append(
                    {
                        "step": int(info["step"]),
                        "price": float(info["price"]),
                        "realized_pnl": realized,
                        "cumulative_realized_pnl": cumulative_realized,
                        "forced_exit": bool(info["forced_exit"]),
                        "exit_reason": info["exit_reason"],
                    }
                )

            # Keep the tracker in sync every step (not just on trade steps)
            # so the next delta is always measured against the latest total.
            self._prev_realized_pnl = float(info["realized_pnl"])

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
                    "step": self._equity_steps,
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
        pd.DataFrame(
    self._reward_history,
            ).to_csv(
    self._evaluation_dir / "reward_components.csv",
    index=False,
           )
        pd.DataFrame(
    self._action_history,
).to_csv(
    self._evaluation_dir / "action_diagnostics.csv",
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
            
        reward_df = pd.DataFrame(self._reward_history)
        logger.info("-" * 80)
        logger.info("Reward Diagnostics")
        for column in reward_df.columns:
            logger.info(
        "{} | mean={:.6f} std={:.6f} min={:.6f} max={:.6f}",
        column,
        reward_df[column].mean(),
        reward_df[column].std(),
        reward_df[column].min(),
        reward_df[column].max(),
    )
        logger.info("-" * 80)



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