from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.analytics.metrics import MetricsCalculator
from src.analytics.plots import PlotGenerator
from src.analytics.report import ReportGenerator
from src.utils import config


class BenchmarkBase(ABC):
    """
    Base class for all benchmark strategies.
    Every benchmark must implement the run() method.
    """

    def __init__(
        self,
        output_directory: str | Path,
    ) -> None:

        self._output_dir = Path(output_directory)

        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._plots_dir = self._output_dir / "plots"

        self._plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._equity_curve: list[float] = []
        self._trade_returns: list[float] = []
        self._portfolio_history: list[dict] = []
        self._trade_history: list[dict] = []

    @abstractmethod
    def run(self) -> None:
        """
        Execute the benchmark strategy.
        """
        raise NotImplementedError
    
    def _save_csv_outputs(self) -> None:
        """
        Save benchmark outputs to CSV files.
        """

        pd.DataFrame(
            {
                "step": range(len(self._equity_curve)),
                "capital": self._equity_curve,
            }
        ).to_csv(
            self._output_dir / "equity_curve.csv",
            index=False,
        )

        pd.DataFrame(
            self._portfolio_history,
        ).to_csv(
            self._output_dir / "portfolio.csv",
            index=False,
        )

        pd.DataFrame(
            self._trade_history,
        ).to_csv(
            self._output_dir / "trades.csv",
            index=False,
        )

    def _generate_analytics(self) -> None:
        """
        Generate metrics, plots and PDF report.
        """

        metrics = MetricsCalculator(
            equity_curve=pd.Series(self._equity_curve),
            trade_returns=pd.Series(self._trade_returns),
            risk_free_rate=config["evaluation"]["risk_free_rate"],
        )

        metrics.save_json(
            self._output_dir / "metrics.json",
        )

        if config["evaluation"]["generate_plots"]:

            PlotGenerator(
                equity_curve=pd.Series(self._equity_curve),
                trade_returns=pd.Series(self._trade_returns),
                output_directory=self._plots_dir,
            ).generate_all(
                rolling_sharpe_window=config["evaluation"][
                    "rolling_sharpe_window"
                ]
            )

        if config["evaluation"]["generate_report"]:

            ReportGenerator(
                evaluation_directory=self._output_dir,
                plots_directory=self._plots_dir,
            ).generate()

    def evaluate(self) -> None:
        self.run()
        self._save_csv_outputs()
        self._generate_analytics()
         
        
        
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable benchmark name.
        """
        raise NotImplementedError

    @property
    def output_directory(self) -> Path:
        """
        Output directory of the benchmark.
        """

        return self._output_dir
    
    @property
    def equity_curve(self) -> pd.Series:
        """
        Portfolio equity curve.
        """

        return pd.Series(
            self._equity_curve,
            name="capital",
        )

    @property
    def trade_returns(self) -> pd.Series:
        """
        Realized trade returns.
        """

        return pd.Series(
            self._trade_returns,
            name="trade_return",
        )

    @property
    def portfolio_history(self) -> pd.DataFrame:
        """
        Portfolio history.
        """

        return pd.DataFrame(
            self._portfolio_history,
        )

    @property
    def trade_history(self) -> pd.DataFrame:
        """
        Trade history.
        """

        return pd.DataFrame(
            self._trade_history,
        )

    def __str__(self) -> str:
        return self.name