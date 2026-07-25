"""
Order execution engine.

Takes resolved per-asset trade deltas (from `sizing.py`) and turns them
into actual fills: applies slippage to determine the effective fill price,
applies fees, updates `Position` objects and cash in place, and returns
the resulting `Trade` records.

This module is intentionally the only place that mutates `Position`
objects or cash during a rebalance, which keeps `Portfolio.step()` a thin
orchestrator and makes the cost model easy to test in isolation.
"""
from __future__ import annotations

from typing import Sequence
from src.portfolio.exceptions import InsufficientCashError, InsufficientPositionError
from src.portfolio.position import Position
from src.portfolio.sizing import AssetDelta
from src.portfolio.trade import Trade, TradeSide

_EPSILON = 1e-12

class SlippageModel:
    """Computes the effective fill price given a reference price and side.

    Only a fixed-bps model is implemented now; the `model` field is read
    from config so additional models (e.g. volume/ATR-scaled slippage)
    can be added later without touching the executor.
    """

    def __init__(self, model: str = "fixed_bps", fixed_bps: float = 0.0):
        self.model = model
        self.fixed_bps = float(fixed_bps)

    def fill_price(self, reference_price: float, side: TradeSide) -> float:
        if self.model == "none" or self.fixed_bps == 0.0:
            return reference_price
        if self.model != "fixed_bps":
            raise ValueError(f"Unknown slippage model: {self.model}")

        adverse_fraction = self.fixed_bps / 10_000.0
        if side is TradeSide.BUY:
            return reference_price * (1.0 + adverse_fraction)
        return reference_price * (1.0 - adverse_fraction)

class FeeModel:
    """Computes the fee owed on a trade's notional value.

    Spot trading here always uses the taker fee, since rebalancing trades
    are market orders against the current candle close. `maker_fee_rate`
    is retained in config for future limit-order execution modes.
    """

    def __init__(self, taker_fee_rate: float = 0.0, maker_fee_rate: float = 0.0):
        self.taker_fee_rate = float(taker_fee_rate)
        self.maker_fee_rate = float(maker_fee_rate)

    def fee(self, notional: float) -> float:
        return abs(notional) * self.taker_fee_rate

class OrderExecutor:
    """Executes resolved asset deltas against positions and cash.

    Attributes:
        fee_model: Computes fees per trade.
        slippage_model: Computes effective fill prices per trade.
    """

    def __init__(self, fee_model: FeeModel, slippage_model: SlippageModel):
        self.fee_model = fee_model
        self.slippage_model = slippage_model

    def execute(
        self,
        deltas: Sequence[AssetDelta],
        positions: dict[str, Position],
        cash: float,
        step: int,
        timestamp: object | None = None,
    ) -> tuple[list[Trade], float]:
        """Execute all non-zero deltas, mutating `positions` in place.

        Args:
            deltas: Resolved per-asset trade requirements.
            positions: Mapping of symbol -> Position, mutated in place.
            cash: Current cash balance before this step's trades.
            step: Current environment timestep, recorded on each Trade.
            timestamp: Optional candle timestamp, recorded on each Trade.

        Returns:
            (trades, updated_cash): the list of executed Trade records and
            the cash balance after all trades settled.

        Raises:
            InsufficientPositionError: if a sell delta implies selling more
                units than the position holds (should not happen if deltas
                were derived from current state, but guarded defensively).
            InsufficientCashError: if a buy delta (plus fees) would drive
                cash negative.
        """
        trades: list[Trade] = []

        # Process sells before buys so freed-up cash is available for buys
        # within the same step (matters when multiple assets rebalance at once).
        ordered = sorted(deltas, key=lambda d: d.delta_value)

        for delta in ordered:
            if abs(delta.delta_value) < _EPSILON:
                continue

            position = positions[delta.symbol]
            reference_price = delta.price

            if delta.is_buy:
                trade, cash = self._execute_buy(
                    delta, position, reference_price, cash, step, timestamp
                )
            else:
                trade, cash = self._execute_sell(
                    delta, position, reference_price, cash, step, timestamp
                )
            trades.append(trade)

        return trades, cash

    def _execute_buy(
        self,
        delta: AssetDelta,
        position: Position,
        reference_price: float,
        cash: float,
        step: int,
        timestamp: object | None,
    ) -> tuple[Trade, float]:
        fill_price = self.slippage_model.fill_price(reference_price, TradeSide.BUY)

        """
        Cap the buy notional so notional + fee never exceeds available cash.
        """
        desired_notional = delta.delta_value
        max_notional_from_cash = cash / (1.0 + self.fee_model.taker_fee_rate)
        notional = min(desired_notional, max_notional_from_cash)
        notional = max(notional, 0.0)

        quantity = notional / fill_price if fill_price > 0 else 0.0
        notional = quantity * fill_price
        fee = self.fee_model.fee(notional)
        total_cost = notional + fee

        if total_cost > cash + _EPSILON:
            raise InsufficientCashError(
                f"Buy of {delta.symbol} requires {total_cost:.8f} cash, "
                f"only {cash:.8f} available."
            )

        position.increase(quantity, fill_price)
        cash -= total_cost
        slippage_cost = quantity * abs(fill_price - reference_price)

        trade = Trade(
            step=step,
            timestamp=timestamp,
            symbol=delta.symbol,
            side=TradeSide.BUY,
            quantity=quantity,
            requested_price=reference_price,
            fill_price=fill_price,
            fee=fee,
            slippage_cost=slippage_cost,
            realized_pnl=0.0,
            cash_after=cash,
        )
        return trade, cash

    def _execute_sell(
        self,
        delta: AssetDelta,
        position: Position,
        reference_price: float,
        cash: float,
        step: int,
        timestamp: object | None,
    ) -> tuple[Trade, float]:
        fill_price = self.slippage_model.fill_price(reference_price, TradeSide.SELL)
        quantity = abs(delta.delta_value) / reference_price if reference_price > 0 else 0.0

        if quantity > position.quantity + _EPSILON:
            raise InsufficientPositionError(
                f"Sell of {delta.symbol} requires {quantity:.8f} units, "
                f"only {position.quantity:.8f} held."
            )
        quantity = min(quantity, position.quantity)

        realized = position.decrease(quantity, fill_price)
        notional = quantity * fill_price
        fee = self.fee_model.fee(notional)
        cash += notional - fee
        slippage_cost = quantity * abs(fill_price - reference_price)

        trade = Trade(
            step=step,
            timestamp=timestamp,
            symbol=delta.symbol,
            side=TradeSide.SELL,
            quantity=quantity,
            requested_price=reference_price,
            fill_price=fill_price,
            fee=fee,
            slippage_cost=slippage_cost,
            realized_pnl=realized,
            cash_after=cash,
        )
        return trade, cash