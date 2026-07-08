"""
A `Position` tracks how much of an asset is held and at what average cost,
which is everything needed to compute unrealized PnL given a current price
and realized PnL when the position is reduced or closed.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Position:
    """
    Attributes:
        symbol: Trading pair symbol, e.g. "BTC/USDT".
        quantity: Units of the base asset currently held (>= 0 for spot;
            spot trading does not support short positions).
        avg_entry_price: Volume-weighted average price paid for the
            currently held quantity. Meaningless when quantity == 0.
        realized_pnl: Cumulative realized profit/loss from all sells of
            this asset so far, net of fees and slippage already applied
            at execution time.
    """

    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        if self.quantity == 0.0:
            return 0.0
        return (price - self.avg_entry_price) * self.quantity

    def unrealized_pnl_pct(self, price: float) -> float:
        if self.quantity == 0.0 or self.avg_entry_price == 0.0:
            return 0.0
        return (price - self.avg_entry_price) / self.avg_entry_price

    def is_flat(self) -> bool:
        return self.quantity == 0.0

    def increase(self, quantity: float, fill_price: float) -> None:
        if quantity <= 0.0:
            return
        new_total_qty = self.quantity + quantity
        new_cost = self.avg_entry_price * self.quantity + fill_price * quantity
        self.avg_entry_price = new_cost / new_total_qty
        self.quantity = new_total_qty

    def decrease(self, quantity: float, fill_price: float) -> float:
        if quantity <= 0.0:
            return 0.0
        realized = (fill_price - self.avg_entry_price) * quantity
        self.realized_pnl += realized
        self.quantity -= quantity
        if self.quantity <= 1e-12:
            # Fully closed: reset cost basis so a future re-entry starts clean.
            self.quantity = 0.0
            self.avg_entry_price = 0.0
        return realized

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
        }