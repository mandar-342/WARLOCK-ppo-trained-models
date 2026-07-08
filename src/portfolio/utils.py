"""
Shared helpers for the portfolio subsystem.
"""

from __future__ import annotations

from typing import Sequence
from src.portfolio.exceptions import InvalidActionError
WEIGHT_SUM_TOLERANCE = 1e-6

def normalize_symbol(symbol: str) -> str:
    """Canonicalize a trading pair symbol, e.g. 'btc/usdt' -> 'BTC/USDT'."""
    return symbol.strip().upper()

def base_asset(symbol: str) -> str:
    """Extract the base asset from a pair symbol, e.g. 'BTC/USDT' -> 'BTC'."""
    return normalize_symbol(symbol).split("/")[0]

def validate_weights(weights: Sequence[float], n_assets: int) -> None:
    """Validate a target-allocation vector before it reaches the executor.

    `weights` represents the target allocation to each risky asset, in the
    same order as the configured `portfolio.assets` list. The implied cash
    weight is `1 - sum(weights)` and is therefore not passed explicitly.

    Raises:
        InvalidActionError: if the vector has the wrong length, contains
            negative weights, or the assets' weights sum to more than 1.0
            beyond tolerance (which would imply negative cash).
    """
    if len(weights) != n_assets:
        raise InvalidActionError(
            f"Expected {n_assets} weight(s), got {len(weights)}."
        )
    for w in weights:
        if w < -WEIGHT_SUM_TOLERANCE:
            raise InvalidActionError(f"Weights must be non-negative, got {w}.")

    total = sum(weights)
    if total > 1.0 + WEIGHT_SUM_TOLERANCE:
        raise InvalidActionError(
            f"Asset weights sum to {total:.6f}, which exceeds 1.0 "
            "(implied cash weight would be negative)."
        )

def clip_weights(weights: Sequence[float]) -> list[float]:
    """Clip each weight to [0, 1] and rescale down (never up) if the sum exceeds 1.

    Used as a defensive last line before execution, e.g. when an RL policy's
    raw output drifts slightly outside the valid simplex due to numerical noise.
    """
    clipped = [min(max(w, 0.0), 1.0) for w in weights]
    total = sum(clipped)
    if total > 1.0:
        clipped = [w / total for w in clipped]
    return clipped

def round_decimal(value: float, precision: int = 8) -> float:
    """Round a float to a fixed precision"""
    return round(value, precision)