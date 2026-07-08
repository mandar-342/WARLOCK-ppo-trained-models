from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from src.utils import config
PERIODS_PER_YEAR = int(
    config["evaluation"]["periods_per_year"]
)


@dataclass(slots=True)
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    average_drawdown: float
    longest_drawdown: int
    final_capital: float
    peak_capital: float
    minimum_capital: float
    number_of_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    expectancy: float


class MetricsCalculator:
    """
    Computes research-quality portfolio and trading metrics
    from evaluation outputs.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trade_returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> None:

        self._equity = equity_curve.astype(float).reset_index(drop=True)
        self._trade_returns = trade_returns.astype(float).reset_index(drop=True)
        self._risk_free_rate = float(risk_free_rate)

        if len(self._equity) < 2:
            raise ValueError(
                "Equity curve must contain at least two observations."
            )

        self._returns = self._equity.pct_change().fillna(0.0)
        
    def _total_return(self) -> float:
        return float(
            (self._equity.iloc[-1] / self._equity.iloc[0]) - 1.0
        )

    def _annualized_return(self) -> float:
        periods = len(self._equity)

        if periods <= 1:
            return 0.0

        growth = self._equity.iloc[-1] / self._equity.iloc[0]

        return float(
            growth ** (PERIODS_PER_YEAR / periods) - 1.0
        )

    def _cagr(self) -> float:
        years = len(self._equity) / PERIODS_PER_YEAR

        if years <= 0.0:
            return 0.0

        growth = self._equity.iloc[-1] / self._equity.iloc[0]

        return float(
            growth ** (1.0 / years) - 1.0
        )

    def _annualized_volatility(self) -> float:
        volatility = self._returns.std(ddof=0)

        return float(
            volatility * np.sqrt(PERIODS_PER_YEAR)
        )

    def _downside_volatility(self) -> float:
        downside = self._returns[self._returns < 0.0]

        if downside.empty:
            return 0.0

        return float(
            downside.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)
        )

    def _excess_returns(self) -> pd.Series:
        rf_per_period = (
            self._risk_free_rate / PERIODS_PER_YEAR
        )

        return self._returns - rf_per_period

    def _safe_divide(
        self,
        numerator: float,
        denominator: float,
    ) -> float:
        if abs(denominator) < 1e-12:
            return 0.0

        return float(numerator / denominator)

    def _max_drawdown(self) -> float:
        running_peak = self._equity.cummax()
        drawdowns = (self._equity - running_peak) / running_peak

        return float(abs(drawdowns.min()))

    def _average_drawdown(self) -> float:
        running_peak = self._equity.cummax()
        drawdowns = (self._equity - running_peak) / running_peak

        active_drawdowns = drawdowns[drawdowns < 0.0]

        if active_drawdowns.empty:
            return 0.0

        return float(abs(active_drawdowns.mean()))

    def _longest_drawdown(self) -> int:
        running_peak = self._equity.cummax()
        drawdowns = (self._equity - running_peak) / running_peak

        longest = 0
        current = 0

        for value in drawdowns:
            if value < 0.0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        return longest

    def _sharpe_ratio(self) -> float:
        excess_returns = self._excess_returns()

        return self._safe_divide(
            excess_returns.mean() * np.sqrt(PERIODS_PER_YEAR),
            excess_returns.std(ddof=0),
        )

    def _sortino_ratio(self) -> float:
        excess_returns = self._excess_returns()

        return self._safe_divide(
            excess_returns.mean() * np.sqrt(PERIODS_PER_YEAR),
            self._downside_volatility(),
        )

    def _calmar_ratio(self) -> float:
        return self._safe_divide(
            self._annualized_return(),
            self._max_drawdown(),
        )
        
    def _number_of_trades(self) -> int:
        return int(len(self._trade_returns))

    def _winning_trades(self) -> pd.Series:
        return self._trade_returns[self._trade_returns > 0.0]

    def _losing_trades(self) -> pd.Series:
        return self._trade_returns[self._trade_returns < 0.0]

    def _win_rate(self) -> float:
        if self._trade_returns.empty:
            return 0.0

        return float(
            len(self._winning_trades()) / len(self._trade_returns)
        )

    def _average_win(self) -> float:
        winners = self._winning_trades()

        if winners.empty:
            return 0.0

        return float(winners.mean())

    def _average_loss(self) -> float:
        losers = self._losing_trades()

        if losers.empty:
            return 0.0

        return float(abs(losers.mean()))

    def _gross_profit(self) -> float:
        winners = self._winning_trades()

        if winners.empty:
            return 0.0

        return float(winners.sum())

    def _gross_loss(self) -> float:
        losers = self._losing_trades()

        if losers.empty:
            return 0.0

        return float(abs(losers.sum()))

    def _profit_factor(self) -> float:
        gross_loss = self._gross_loss()

        if gross_loss <= 1e-12:
            return float("inf")

        return float(
            self._gross_profit() / gross_loss
        )

    def _expectancy(self) -> float:
        if self._trade_returns.empty:
            return 0.0

        win_rate = self._win_rate()
        loss_rate = 1.0 - win_rate

        return float(
            (win_rate * self._average_win())
            -
            (loss_rate * self._average_loss())
        )
        
    def compute_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return=self._total_return(),
            annualized_return=self._annualized_return(),
            cagr=self._cagr(),
            annualized_volatility=self._annualized_volatility(),
            sharpe_ratio=self._sharpe_ratio(),
            sortino_ratio=self._sortino_ratio(),
            calmar_ratio=self._calmar_ratio(),
            max_drawdown=self._max_drawdown(),
            average_drawdown=self._average_drawdown(),
            longest_drawdown=self._longest_drawdown(),
            final_capital=float(self._equity.iloc[-1]),
            peak_capital=float(self._equity.max()),
            minimum_capital=float(self._equity.min()),
            number_of_trades=self._number_of_trades(),
            win_rate=self._win_rate(),
            average_win=self._average_win(),
            average_loss=self._average_loss(),
            profit_factor=self._profit_factor(),
            expectancy=self._expectancy(),
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self.compute_metrics())

    def save_json(
        self,
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=4,
            )