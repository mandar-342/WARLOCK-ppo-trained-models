"""
Portfolio performance metrics.

`PortfolioMetrics` computes standard performance statistics (returns,
drawdown, exposure, trade stats) purely from a `TradeHistory` instance.

All ratio-style metrics (Sharpe, etc.) are computed on per-step returns
derived from the equity curve, consistent with the `periods_per_year`
convention already used in `src/features` (e.g. rolling_sharpe).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from src.portfolio.history import TradeHistory

_EPSILON = 1e-12

@dataclass
class MetricsSummary:
    """A snapshot of portfolio performance statistics over an episode."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    n_trades: int
    total_fees_paid: float
    total_slippage_cost: float
    total_realized_pnl: float
    win_rate: float
    avg_exposure: float

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "n_trades": self.n_trades,
            "total_fees_paid": self.total_fees_paid,
            "total_slippage_cost": self.total_slippage_cost,
            "total_realized_pnl": self.total_realized_pnl,
            "win_rate": self.win_rate,
            "avg_exposure": self.avg_exposure,
        }

class PortfolioMetrics:
    """Computes performance statistics from a `TradeHistory`.

    Args:
        periods_per_year: Number of timesteps per year, used to annualize
            return and volatility. Defaults to 8760 (hourly candles),
            matching `features.volatility.periods_per_year` in config.yaml.
    """

    def __init__(self, periods_per_year: int = 8760):
        self.periods_per_year = periods_per_year

    def step_returns(self, history: TradeHistory) -> np.ndarray:
        """Per-step simple returns derived from the equity curve."""
        values = np.asarray(history.equity_values(), dtype=np.float64)
        if len(values) < 2:
            return np.array([], dtype=np.float64)
        prev = values[:-1]
        curr = values[1:]
        # Guard against a zeroed-out equity curve (e.g. total wipeout).
        safe_prev = np.where(prev == 0, _EPSILON, prev)
        return (curr - prev) / safe_prev

    def total_return(self, history: TradeHistory) -> float:
        values = history.equity_values()
        if len(values) < 2 or values[0] == 0:
            return 0.0
        return (values[-1] - values[0]) / values[0]

    def annualized_return(self, history: TradeHistory) -> float:
        values = history.equity_values()
        n_periods = len(values) - 1
        if n_periods <= 0:
            return 0.0
        total = self.total_return(history)
        years = n_periods / self.periods_per_year
        if years <= 0:
            return 0.0
        return (1.0 + total) ** (1.0 / years) - 1.0

    def annualized_volatility(self, history: TradeHistory) -> float:
        returns = self.step_returns(history)
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns) * np.sqrt(self.periods_per_year))

    def sharpe_ratio(self, history: TradeHistory, risk_free_rate: float = 0.0) -> float:
        returns = self.step_returns(history)
        if len(returns) < 2:
            return 0.0
        excess = returns - (risk_free_rate / self.periods_per_year)
        mean_r = np.mean(excess)
        std_r = np.std(excess) + _EPSILON
        return float((mean_r / std_r) * np.sqrt(self.periods_per_year))

    def drawdown_series(self, history: TradeHistory) -> np.ndarray:
        """Drawdown at every step: (peak - value) / peak."""
        values = np.asarray(history.equity_values(), dtype=np.float64)
        if len(values) == 0:
            return np.array([], dtype=np.float64)
        running_peak = np.maximum.accumulate(values)
        safe_peak = np.where(running_peak == 0, _EPSILON, running_peak)
        return (running_peak - values) / safe_peak

    def max_drawdown(self, history: TradeHistory) -> float:
        dd = self.drawdown_series(history)
        return float(np.max(dd)) if len(dd) else 0.0

    def current_drawdown(self, history: TradeHistory) -> float:
        dd = self.drawdown_series(history)
        return float(dd[-1]) if len(dd) else 0.0

    def win_rate(self, history: TradeHistory) -> float:
        """Fraction of closing (sell) trades with positive realized PnL."""
        closing_trades = [t for t in history.trades if t.side.value == "sell"]
        if not closing_trades:
            return 0.0
        wins = sum(1 for t in closing_trades if t.realized_pnl > 0)
        return wins / len(closing_trades)

    def avg_exposure(self, history: TradeHistory) -> float:
        """Average fraction of portfolio value held in assets (vs. cash)."""
        points = history.equity_curve
        if not points:
            return 0.0
        exposures = [
            1.0 - (p.cash / p.total_value) if p.total_value > 0 else 0.0
            for p in points
        ]
        return float(np.mean(exposures))

    def summary(self, history: TradeHistory) -> MetricsSummary:
        return MetricsSummary(
            total_return=self.total_return(history),
            annualized_return=self.annualized_return(history),
            annualized_volatility=self.annualized_volatility(history),
            sharpe_ratio=self.sharpe_ratio(history),
            max_drawdown=self.max_drawdown(history),
            current_drawdown=self.current_drawdown(history),
            n_trades=history.n_trades,
            total_fees_paid=history.total_fees_paid(),
            total_slippage_cost=history.total_slippage_cost(),
            total_realized_pnl=history.total_realized_pnl(),
            win_rate=self.win_rate(history),
            avg_exposure=self.avg_exposure(history),
        )