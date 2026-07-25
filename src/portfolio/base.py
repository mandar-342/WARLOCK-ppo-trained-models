"""
Abstract portfolio interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from src.portfolio.trade import Trade

class PortfolioBase(ABC):
    """Common interface for all portfolio implementations.
    """

    @abstractmethod
    def reset(self, initial_capital: float | None = None) -> None:
        """Reset all state (cash, positions, history) to a fresh episode start.

        Args:
            initial_capital: Override the configured starting cash for this
                episode. If None, the configured default is used.
        """

    @abstractmethod
    def step(
        self,
        target_weights: Sequence[float],
        prices: Sequence[float],
        step: int,
        timestamp: object | None = None,
    ) -> list[Trade]:
        """Advance the portfolio by one timestep.

        Rebalances current holdings toward `target_weights` at `prices`,
        applying fees and slippage, then returns the list of trades
        executed (empty if no rebalancing was needed).

        Args:
            target_weights: Desired allocation to each configured asset,
                in the same order as `self.symbols`. The implied cash
                weight is `1 - sum(target_weights)`.
            prices: Current fill price for each configured asset, in the
                same order as `self.symbols`.
            step: The current environment timestep, recorded on each Trade.
            timestamp: Optional candle timestamp, recorded on each Trade.

        Returns:
            The list of `Trade` records executed this step.
        """

    @abstractmethod
    def total_value(self, prices: Sequence[float]) -> float:
        """Total portfolio value (cash + all positions) at the given prices."""

    @abstractmethod
    def current_weights(self, prices: Sequence[float]) -> list[float]:
        """Current allocation to each configured asset, in `self.symbols` order."""

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Configured tradable symbols, in canonical action-vector order."""

    @property
    @abstractmethod
    def cash(self) -> float:
        """Current uninvested cash balance."""