from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import vectorbt as vbt

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
    number_of_closing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    expectancy: float


class VectorBTMetricsCalculator:
    """
    Computes research-quality portfolio and trading metrics using
    VectorBT's `Portfolio` simulation engine, replacing the hand-rolled
    statistics that used to live in `MetricsCalculator`.

    This is built directly from the (price, target-weight) series
    recorded during policy rollout, rather than from a pre-computed
    equity curve: VectorBT owns order simulation, fee/slippage
    application, position/trade bookkeeping, and every statistic below.
    This class only maps VectorBT's output onto the same
    `PerformanceMetrics` schema the rest of the pipeline
    (`ReportGenerator`, `PlotGenerator`, checkpoint comparisons) already
    expects, so nothing downstream has to change.

    Weights use VectorBT's `targetpercent` sizing: at each step the
    simulated portfolio is rebalanced to hold `weight * portfolio_value`
    of the asset (0.0 = flat/cash, 1.0 = fully long; negative values
    open a short if the branch's action space allows it).
    """

    def __init__(
        self,
        prices: pd.Series,
        weights: pd.Series,
        init_cash: float = 10_000.0,
        fees: float = 0.0,
        slippage: float = 0.0,
        risk_free_rate: float = 0.0,
        freq: str = "1min",
    ) -> None:

        self._prices = prices.astype(float).reset_index(drop=True)
        self._weights = weights.astype(float).reset_index(drop=True)
        self._risk_free_rate = float(risk_free_rate)
        self._freq = freq

        if len(self._prices) < 2:
            raise ValueError(
                "Price series must contain at least two observations."
            )

        self.portfolio = vbt.Portfolio.from_orders(
            close=self._prices,
            size=self._weights,
            size_type="targetpercent",
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq=freq,
        )

        self._equity = self.portfolio.value()
        self._trades = self.portfolio.trades
        self._drawdowns = self.portfolio.drawdowns

    def _annual_factor(self) -> float:
        return float(PERIODS_PER_YEAR)

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(value):
            return default
        return value

    def _total_return(self) -> float:
        return self._safe_float(self.portfolio.total_return())

    def _annualized_return(self) -> float:
        return self._safe_float(
            self.portfolio.annualized_return(freq=self._freq)
        )

    def _cagr(self) -> float:
        # VectorBT's `annualized_return` already compounds from total
        # return over the elapsed period at PERIODS_PER_YEAR -- same
        # definition as CAGR, so the two coincide here.
        return self._annualized_return()

    def _annualized_volatility(self) -> float:
        returns = self._equity.pct_change().fillna(0.0)
        return self._safe_float(
            returns.std(ddof=0) * np.sqrt(self._annual_factor())
        )

    def _sharpe_ratio(self) -> float:
        return self._safe_float(
            self.portfolio.sharpe_ratio(
                risk_free=self._risk_free_rate / self._annual_factor(),
                freq=self._freq,
            )
        )

    def _sortino_ratio(self) -> float:
        return self._safe_float(
            self.portfolio.sortino_ratio(freq=self._freq)
        )

    def _calmar_ratio(self) -> float:
        value = self._safe_float(self.portfolio.calmar_ratio(freq=self._freq))
        # calmar_ratio() blows up towards +/-inf when max_drawdown is
        # ~0 (e.g. a flat/no-trade episode); clip rather than report a
        # meaningless astronomical number.
        return float(np.clip(value, -1e6, 1e6))

    def _max_drawdown(self) -> float:
        return abs(self._safe_float(self.portfolio.max_drawdown()))

    def _average_drawdown(self) -> float:
        if self._drawdowns.count() == 0:
            return 0.0
        return abs(self._safe_float(self._drawdowns.avg_drawdown()))

    def _longest_drawdown(self) -> int:
        if self._drawdowns.count() == 0:
            return 0
        max_duration = self._drawdowns.max_duration()
        if pd.isna(max_duration):
            return 0
        # Duration comes back as a Timedelta (or bar count if freq is
        # unset); normalize to an integer number of bars.
        if isinstance(max_duration, pd.Timedelta):
            bar = pd.Timedelta(self._freq)
            return int(max_duration / bar) if bar else 0
        return int(max_duration)

    def _number_of_trades(self) -> int:
        return int(self._trades.count())

    def _number_of_closing_trades(self) -> int:
        return int(self._trades.closed.count())

    def _win_rate(self) -> float:
        closed = self._trades.closed
        if closed.count() == 0:
            return 0.0
        return self._safe_float(closed.win_rate())

    def _average_win(self) -> float:
        winning = self._trades.winning
        if winning.count() == 0:
            return 0.0
        return self._safe_float(winning.pnl.mean())

    def _average_loss(self) -> float:
        losing = self._trades.losing
        if losing.count() == 0:
            return 0.0
        return abs(self._safe_float(losing.pnl.mean()))

    def _profit_factor(self) -> float:
        closed = self._trades.closed
        if closed.count() == 0:
            return 0.0
        value = closed.profit_factor()
        if not np.isfinite(value):
            return float("inf") if value > 0 else 0.0
        return self._safe_float(value)

    def _expectancy(self) -> float:
        closed = self._trades.closed
        if closed.count() == 0:
            return 0.0
        return self._safe_float(closed.expectancy())

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
            number_of_closing_trades=self._number_of_closing_trades(),
            win_rate=self._win_rate(),
            average_win=self._average_win(),
            average_loss=self._average_loss(),
            profit_factor=self._profit_factor(),
            expectancy=self._expectancy(),
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self.compute_metrics())

    def equity_curve(self) -> pd.Series:
        """VectorBT-simulated portfolio value series (replaces the
        manually-tracked `capital` column as the source of truth for
        plots/reports)."""
        return self._equity.reset_index(drop=True)

    def trade_returns(self) -> pd.Series:
        """Realized P&L per closed trade, as simulated by VectorBT
        (replaces the manual `realized_pnl` delta-tracking)."""
        closed = self._trades.closed.records_readable
        if len(closed) == 0:
            return pd.Series(dtype=float)
        return closed["PnL"].reset_index(drop=True)

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
