"""
`TradeHistory` is the append-only log of everything that happened in a
portfolio over an episode: every executed `Trade`, plus a snapshot of
total portfolio value at each step (the equity curve).
"""

from __future__ import annotations

from dataclasses import dataclass
from src.portfolio.trade import Trade

@dataclass
class EquityPoint:
    """A single snapshot of total portfolio value at a given step."""

    step: int
    timestamp: object | None
    total_value: float
    cash: float

class TradeHistory:
    """Append-only record of trades and equity snapshots for one episode.

    Attributes:
        trades: All executed trades, in execution order.
        equity_curve: All recorded `EquityPoint`s, in step order.
    """

    def __init__(self):
        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []

    def reset(self) -> None:
        """Clear all recorded trades and equity points (new episode)."""
        self.trades = []
        self.equity_curve = []

    def record_trades(self, trades: list[Trade]) -> None:
        self.trades.extend(trades)

    def record_equity(
        self, step: int, total_value: float, cash: float, timestamp: object | None = None
    ) -> None:
        self.equity_curve.append(
            EquityPoint(step=step, timestamp=timestamp, total_value=total_value, cash=cash)
        )

    def trades_for_symbol(self, symbol: str) -> list[Trade]:
        return [t for t in self.trades if t.symbol == symbol]

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def total_fees_paid(self) -> float:
        return sum(t.fee for t in self.trades)

    def total_slippage_cost(self) -> float:
        return sum(t.slippage_cost for t in self.trades)

    def total_realized_pnl(self) -> float:
        return sum(t.realized_pnl for t in self.trades)

    def equity_values(self) -> list[float]:
        """Convenience accessor: just the total_value series from equity_curve."""
        return [p.total_value for p in self.equity_curve]

    def to_records(self) -> list[dict]:
        """All trades as plain dicts, e.g. for pd.DataFrame(history.to_records())."""
        return [t.to_dict() for t in self.trades]