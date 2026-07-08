"""
Spot trading portfolio implementation.
    1. Resolves target weights -> per-asset notional deltas (`sizing.py`).
    2. Executes those deltas against cash/positions, applying fees and
       slippage (`order_executor.py`).
    3. Returns the resulting trades for the caller to log/inspect.
    - All weights must be in [0, 1]; no shorting.
    - A position's quantity can never go negative (no margin).
"""

from __future__ import annotations

from typing import Sequence
from src.portfolio.base import PortfolioBase
from src.portfolio.exceptions import InvalidConfigError, UnknownAssetError
from src.portfolio.history import TradeHistory
from src.portfolio.metrics import MetricsSummary, PortfolioMetrics
from src.portfolio.order_executor import FeeModel, OrderExecutor, SlippageModel
from src.portfolio.position import Position
from src.portfolio.sizing import resolve_target_deltas
from src.portfolio.trade import Trade
from src.portfolio.utils import normalize_symbol
from src.utils import config as default_config

class SpotPortfolio(PortfolioBase):
    """
    Args:
        cfg: Optional config dict to use instead of the module-level
            `src.utils.config`. Primarily for tests.

    Attributes:
        cash: Current uninvested cash balance (base currency, e.g. USDT).
        positions: Mapping of symbol -> Position for every configured asset.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or default_config
        try:
            pf_cfg = cfg["portfolio"]
        except KeyError as e:
            raise InvalidConfigError(
                "config.yaml is missing a top-level 'portfolio' section."
            ) from e

        self._symbols: list[str] = [
            normalize_symbol(a["symbol"]) for a in pf_cfg["assets"]
        ]
        if not self._symbols:
            raise InvalidConfigError("portfolio.assets must contain at least one asset.")

        self._initial_capital = float(pf_cfg.get("initial_capital", 10_000.0))
        self._min_rebalance_delta = float(
            pf_cfg.get("rebalance", {}).get("min_rebalance_delta", 0.0)
        )

        fees_cfg = pf_cfg.get("fees", {})
        self._fee_model = FeeModel(
            taker_fee_rate=fees_cfg.get("taker_fee_rate", 0.0),
            maker_fee_rate=fees_cfg.get("maker_fee_rate", 0.0),
        )

        slippage_cfg = pf_cfg.get("slippage", {})
        self._slippage_model = SlippageModel(
            model=slippage_cfg.get("model", "none"),
            fixed_bps=slippage_cfg.get("fixed_bps", 0.0),
        )

        self._executor = OrderExecutor(self._fee_model, self._slippage_model)

        self._cash: float = self._initial_capital
        self._positions: dict[str, Position] = {
            symbol: Position(symbol=symbol) for symbol in self._symbols
        }
        self.history = TradeHistory()

        periods_per_year = (
            cfg.get("features", {}).get("volatility", {}).get("periods_per_year", 8760)
        )
        self._metrics_engine = PortfolioMetrics(periods_per_year=periods_per_year)

    def reset(self, initial_capital: float | None = None) -> None:
        self._cash = float(initial_capital) if initial_capital is not None else self._initial_capital
        self._positions = {symbol: Position(symbol=symbol) for symbol in self._symbols}
        self.history.reset()

    def step(
        self,
        target_weights: Sequence[float],
        prices: Sequence[float],
        step: int,
        timestamp: object | None = None,
    ) -> list[Trade]:
        if len(prices) != len(self._symbols):
            raise UnknownAssetError(
                f"Expected {len(self._symbols)} price(s), got {len(prices)}."
            )

        portfolio_value = self.total_value(prices)
        quantities = [self._positions[s].quantity for s in self._symbols]

        deltas = resolve_target_deltas(
            symbols=self._symbols,
            target_weights=target_weights,
            current_quantities=quantities,
            prices=prices,
            portfolio_value=portfolio_value,
            min_rebalance_delta=self._min_rebalance_delta,
        )

        trades, self._cash = self._executor.execute(
            deltas=deltas,
            positions=self._positions,
            cash=self._cash,
            step=step,
            timestamp=timestamp,
        )

        self.history.record_trades(trades)
        self.history.record_equity(
            step=step,
            total_value=self.total_value(prices),
            cash=self._cash,
            timestamp=timestamp,
        )
        return trades

    def total_value(self, prices: Sequence[float]) -> float:
        value = self._cash
        for symbol, price in zip(self._symbols, prices):
            value += self._positions[symbol].market_value(price)
        return value

    def current_weights(self, prices: Sequence[float]) -> list[float]:
        value = self.total_value(prices)
        if value <= 0:
            return [0.0 for _ in self._symbols]
        return [
            self._positions[s].market_value(p) / value
            for s, p in zip(self._symbols, prices)
        ]

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    def position(self, symbol: str) -> Position:
        symbol = normalize_symbol(symbol)
        if symbol not in self._positions:
            raise UnknownAssetError(f"Unknown asset: {symbol}")
        return self._positions[symbol]

    def unrealized_pnl(self, prices: Sequence[float]) -> float:
        return sum(
            self._positions[s].unrealized_pnl(p)
            for s, p in zip(self._symbols, prices)
        )

    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

    def metrics(self) -> MetricsSummary:
        """Compute the full performance metrics summary for this episode so far."""
        return self._metrics_engine.summary(self.history)