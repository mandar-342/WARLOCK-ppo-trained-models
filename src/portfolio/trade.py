from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass(frozen=True)
class Trade:
    """
    Attributes:
        step: The environment timestep at which the trade executed.
        timestamp: The candle timestamp associated with `step`, if known.
        symbol: Trading pair symbol, e.g. "BTC/USDT".
        side: BUY or SELL.
        quantity: Units of the base asset traded (always positive).
        requested_price: The reference price before slippage (e.g. candle close).
        fill_price: The effective execution price after slippage.
        fee: Fee paid on this trade, denominated in the base currency (e.g. USDT).
        slippage_cost: Cost attributable to slippage, denominated in base currency.
            Equal to `quantity * abs(fill_price - requested_price)`.
        realized_pnl: PnL realized by this specific trade. Always 0.0 for buys;
            for sells, the gain/loss versus average entry price, before fees.
        cash_after: Cash balance immediately after this trade settled.
    """

    step: int
    timestamp: object | None
    symbol: str
    side: TradeSide
    quantity: float
    requested_price: float
    fill_price: float
    fee: float
    slippage_cost: float
    realized_pnl: float
    cash_after: float

    @property
    def notional(self) -> float:
        """Gross notional value of the trade at the fill price."""
        return self.quantity * self.fill_price

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "requested_price": self.requested_price,
            "fill_price": self.fill_price,
            "fee": self.fee,
            "slippage_cost": self.slippage_cost,
            "realized_pnl": self.realized_pnl,
            "cash_after": self.cash_after,
            "notional": self.notional,
        }