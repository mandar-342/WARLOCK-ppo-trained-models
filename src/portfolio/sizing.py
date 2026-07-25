"""
Translates target allocations into the trade deltas needed to reach them.
 
This module sits between "what does the agent want" (a weight vector) and
"what orders does the executor need to place" (per-asset buy/sell deltas
in base-asset units).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from src.portfolio.utils import validate_weights

@dataclass
class AssetDelta:
    """
    Attributes:
        symbol: Trading pair symbol.
        price: Reference market price used to compute target/current value
            (the same price the executor will apply slippage to).
        target_value: Desired notional value to hold in this asset.
        current_value: Current notional value held in this asset.
        delta_value: target_value - current_value. Positive means buy,
            negative means sell, zero (or below `min_rebalance_delta`
            of portfolio value) means no trade.
    """

    symbol: str
    price: float
    target_value: float
    current_value: float
    delta_value: float

    @property
    def is_buy(self) -> bool:
        return self.delta_value > 0

    @property
    def is_sell(self) -> bool:
        return self.delta_value < 0


def resolve_target_deltas(
    symbols: Sequence[str],
    target_weights: Sequence[float],
    current_quantities: Sequence[float],
    prices: Sequence[float],
    portfolio_value: float,
    min_rebalance_delta: float = 0.0,
) -> list[AssetDelta]:
    """Compute the per-asset notional deltas needed to reach target_weights.

    Args:
        symbols: Configured symbols, defines ordering for all sequence args.
        target_weights: Desired weight per asset (cash is implied as the
            remainder); validated to be non-negative and sum to <= 1.0.
        current_quantities: Current units held per asset.
        prices: Current price per asset, same order as `symbols`.
        portfolio_value: Total portfolio value (cash + all positions) used
            as the base for converting weights into target notional values.
        min_rebalance_delta: Minimum |delta_value| / portfolio_value below
            which a trade is skipped, to avoid dust trades from tiny policy
            output fluctuations.

    Returns:
        One `AssetDelta` per symbol, in the same order as `symbols`.
    """
    validate_weights(target_weights, n_assets=len(symbols))

    deltas: list[AssetDelta] = []
    for symbol, weight, qty, price in zip(
        symbols, target_weights, current_quantities, prices
    ):
        target_value = weight * portfolio_value
        current_value = qty * price
        delta_value = target_value - current_value

        if portfolio_value > 0 and abs(delta_value) / portfolio_value < min_rebalance_delta:
            delta_value = 0.0

        deltas.append(
            AssetDelta(
                symbol=symbol,
                price=price,
                target_value=target_value,
                current_value=current_value,
                delta_value=delta_value,
            )
        )
    return deltas