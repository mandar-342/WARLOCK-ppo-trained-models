from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import config

PERIODS_PER_YEAR = int(
    config["evaluation"]["periods_per_year"]
)


class PlotGenerator:
    """
    Generates research-quality performance plots
    for WARLOCK evaluation experiments.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trade_returns: pd.Series,
        output_directory: str | Path,
    ) -> None:

        self._equity = equity_curve.astype(float).reset_index(drop=True)
        self._trade_returns = (
            trade_returns.astype(float).reset_index(drop=True)
        )

        self._returns = self._equity.pct_change().fillna(0.0)

        self._output_dir = Path(output_directory)
        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def output_directory(self) -> Path:
        return self._output_dir

    def _save_figure(
        self,
        filename: str,
    ) -> None:

        plt.tight_layout()

        plt.savefig(
            self._output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        
    def plot_equity_curve(self) -> None:
        """
        Plot portfolio equity over time.
        """

        plt.figure(figsize=(12, 6))

        plt.plot(
            self._equity.values,
            linewidth=2,
            label="Portfolio Value",
        )

        plt.title("Equity Curve")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.grid(True, alpha=0.3)
        plt.legend()

        self._save_figure("equity_curve.png")

    def plot_drawdown(self) -> None:
        """
        Plot portfolio drawdown over time.
        """

        running_peak = self._equity.cummax()

        drawdown = (
            (self._equity - running_peak)
            / running_peak
        ) * 100.0

        plt.figure(figsize=(12, 5))

        plt.plot(
            drawdown.values,
            linewidth=2,
            label="Drawdown",
        )

        plt.fill_between(
            np.arange(len(drawdown)),
            drawdown.values,
            0,
            alpha=0.25,
        )

        plt.title("Portfolio Drawdown")
        plt.xlabel("Time Step")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        self._save_figure("drawdown.png")
        
    def plot_returns_distribution(self) -> None:
        """
        Plot the distribution of portfolio returns.
        """

        plt.figure(figsize=(12, 6))

        plt.hist(
            self._returns.values,
            bins=50,
            density=True,
        )

        plt.title("Returns Distribution")
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)

        self._save_figure("returns_distribution.png")

    def plot_rolling_sharpe(
        self,
        window: int = 252,
    ) -> None:
        """
        Plot the rolling Sharpe ratio.
        """

        rolling_mean = self._returns.rolling(
            window=window,
            min_periods=window,
        ).mean()

        rolling_std = self._returns.rolling(
            window=window,
            min_periods=window,
        ).std()

        rolling_sharpe = (
            rolling_mean
            / rolling_std
        ) * np.sqrt(PERIODS_PER_YEAR)

        rolling_sharpe = rolling_sharpe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        plt.figure(figsize=(12, 6))

        plt.plot(
            rolling_sharpe.values,
            linewidth=2,
            label=f"{window}-Period Rolling Sharpe",
        )

        plt.axhline(
            y=0.0,
            linestyle="--",
            linewidth=1,
        )

        plt.title("Rolling Sharpe Ratio")
        plt.xlabel("Time Step")
        plt.ylabel("Sharpe Ratio")
        plt.grid(True, alpha=0.3)
        plt.legend()

        self._save_figure("rolling_sharpe.png")
        
        
    def generate_all(
        self,
        rolling_sharpe_window: int = 252,
    ) -> None:
        """
        Generate every evaluation plot.
        """

        self.plot_equity_curve()
        self.plot_drawdown()
        self.plot_returns_distribution()
        self.plot_rolling_sharpe(
            window=rolling_sharpe_window,
        )