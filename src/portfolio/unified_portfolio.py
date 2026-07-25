"""
Unified trading portfolio: one cash/equity pool backing both long (spot,
unleveraged) and short (futures-style, leveraged margin) exposure per
asset.

This replaces the previous two-portfolio design (`SpotPortfolio` +
`FuturesPortfolio`, each with its own cash pool) with a single `Portfolio`
that the env drives with one signed weight per asset in [-1, 1]:
    - positive weight -> long, cash-backed 1:1, no leverage
    - negative weight -> short, backed by margin drawn from the same cash
      pool, at `portfolio.futures.leverage`

Why one pool: a real trading account has one balance. Longs and shorts on
the same asset draw against (and free up) the same capital, so a losing
short can eat into the cash available to open a long and vice versa. That
coupling is the whole point of unifying — it's more realistic than two
walled-off sub-accounts, and it's also simpler for the env to drive (no
more splitting a signed weight into two separate `.step()` calls).

An asset's long and short legs are tracked independently (`Position` for
long, `FuturesPosition` for short) so both can, in principle, be open at
once — though the env only ever asks for one side per asset per step,
since a single net weight can't request both directions simultaneously.

Fee/slippage/leverage config is kept separate for the two legs
(`portfolio.long.fees`/`portfolio.long.slippage` for longs,
`portfolio.short.*` for shorts) since real venues price spot and margin
trading differently.
"""

from __future__ import annotations

from typing import Sequence

from src.portfolio.base import PortfolioBase
from src.portfolio.exceptions import InvalidConfigError, UnknownAssetError
from src.portfolio.history import TradeHistory
from src.portfolio.metrics import MetricsSummary, PortfolioMetrics
from src.portfolio.order_executor import FeeModel, SlippageModel
from src.portfolio.position import FuturesPosition, Position
from src.portfolio.trade import Trade, TradeSide
from src.portfolio.utils import normalize_symbol
from src.utils import config as default_config

_EPSILON = 1e-12


class Portfolio(PortfolioBase):
    """Single-pool portfolio supporting both long and short exposure.

    Args:
        cfg: Optional config dict to use instead of the module-level
            `src.utils.config`. Primarily for tests.

    Attributes:
        cash: Shared cash/margin balance backing both legs.
        long_positions: Mapping of symbol -> Position (long, unleveraged).
        short_positions: Mapping of symbol -> FuturesPosition (short,
            leveraged).
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

        # Long leg: spot-style, unleveraged.
        long_cfg = pf_cfg.get("long", {})
        long_fees_cfg = long_cfg.get("fees", {})
        self._long_fee_model = FeeModel(
            taker_fee_rate=long_fees_cfg.get("taker_fee_rate", 0.0),
            maker_fee_rate=long_fees_cfg.get("maker_fee_rate", 0.0),
        )
        long_slippage_cfg = long_cfg.get("slippage", {})
        self._long_slippage_model = SlippageModel(
            model=long_slippage_cfg.get("model", "none"),
            fixed_bps=long_slippage_cfg.get("fixed_bps", 0.0),
        )

        # Short leg: futures-style margin, its own (typically wider) fee
        # schedule and its own leverage/liquidation parameters.
        try:
            short_cfg = pf_cfg["short"]
        except KeyError as e:
            raise InvalidConfigError(
                "config.yaml is missing a 'portfolio.short' section."
            ) from e

        self._leverage = float(short_cfg.get("leverage", 3.0))
        if self._leverage <= 0:
            raise InvalidConfigError("portfolio.short.leverage must be positive.")
        self._maintenance_margin_ratio = float(short_cfg.get("maintenance_margin_ratio", 0.05))
        self._funding_rate_per_step = float(short_cfg.get("funding_rate_per_step", 0.0))

        short_fees_cfg = short_cfg.get("fees", {})
        self._short_fee_model = FeeModel(
            taker_fee_rate=short_fees_cfg.get("taker_fee_rate", 0.0),
            maker_fee_rate=short_fees_cfg.get("maker_fee_rate", 0.0),
        )
        short_slippage_cfg = short_cfg.get("slippage", {})
        self._short_slippage_model = SlippageModel(
            model=short_slippage_cfg.get("model", "none"),
            fixed_bps=short_slippage_cfg.get("fixed_bps", 0.0),
        )

        self._cash: float = self._initial_capital
        self._long_positions: dict[str, Position] = {
            symbol: Position(symbol=symbol) for symbol in self._symbols
        }
        self._short_positions: dict[str, FuturesPosition] = {
            symbol: FuturesPosition(symbol=symbol) for symbol in self._symbols
        }
        self.history = TradeHistory()

        periods_per_year = (
            cfg.get("features", {}).get("volatility", {}).get("periods_per_year", 8760)
        )
        self._metrics_engine = PortfolioMetrics(periods_per_year=periods_per_year)

    def reset(self, initial_capital: float | None = None) -> None:
        self._cash = float(initial_capital) if initial_capital is not None else self._initial_capital
        self._long_positions = {symbol: Position(symbol=symbol) for symbol in self._symbols}
        self._short_positions = {symbol: FuturesPosition(symbol=symbol) for symbol in self._symbols}
        self.history.reset()

    def step(
        self,
        target_weights: Sequence[float],
        prices: Sequence[float],
        step: int,
        timestamp: object | None = None,
    ) -> list[Trade]:
        """Rebalance toward signed `target_weights` in [-1, 1] per asset.

        Positive weight -> target long notional = weight * equity.
        Negative weight -> target short notional = |weight| * equity * leverage.
        Both legs draw from / return to the single shared `cash` balance.
        """
        if len(prices) != len(self._symbols):
            raise UnknownAssetError(
                f"Expected {len(self._symbols)} price(s), got {len(prices)}."
            )

        trades: list[Trade] = []

        # 1. Funding cost (opt-in, defaults to zero) on open shorts, charged
        # on notional and settled against the shared cash pool.
        if self._funding_rate_per_step != 0.0:
            for symbol, price in zip(self._symbols, prices):
                short_position = self._short_positions[symbol]
                if short_position.short_quantity > 0.0:
                    self._cash -= short_position.short_quantity * price * self._funding_rate_per_step

        # 2. Liquidation check: if equity has fallen below the maintenance
        # margin requirement on total short notional, force-cover every
        # short position before any further rebalancing this step.
        equity = self.total_value(prices)
        total_short_notional = sum(
            self._short_positions[s].notional(p) for s, p in zip(self._symbols, prices)
        )
        if total_short_notional > 0.0 and equity < self._maintenance_margin_ratio * total_short_notional:
            trades.extend(self._liquidate_all_shorts(prices, step, timestamp))
            equity = self.total_value(prices)

        # 3. Split each signed target weight into a long leg (>=0) and a
        # short leg (>=0, magnitude of a negative weight).
        long_targets = [max(0.0, w) * equity for w in target_weights]
        short_targets = [max(0.0, -w) * equity * self._leverage for w in target_weights]

        # Process every decrease (sells + covers) first so freed-up cash is
        # available to fund increases (buys + new shorts) within the same
        # step, then process increases.
        decreases: list[tuple[str, float, float]] = []  # (symbol, kind_sign, price)
        increases: list[tuple[str, float, float]] = []
        for symbol, price, long_target, short_target in zip(
            self._symbols, prices, long_targets, short_targets
        ):
            if price <= 0.0:
                continue

            long_delta = long_target - self._long_positions[symbol].market_value(price)
            if equity > 0 and abs(long_delta) / equity < self._min_rebalance_delta:
                long_delta = 0.0
            if long_delta < -_EPSILON:
                decreases.append(("long_sell", symbol, price, -long_delta))
            elif long_delta > _EPSILON:
                increases.append(("long_buy", symbol, price, long_delta))

            short_delta = short_target - self._short_positions[symbol].notional(price)
            denom = equity * self._leverage
            if denom > 0 and abs(short_delta) / denom < self._min_rebalance_delta:
                short_delta = 0.0
            if short_delta < -_EPSILON:
                decreases.append(("short_cover", symbol, price, -short_delta))
            elif short_delta > _EPSILON:
                increases.append(("short_open", symbol, price, short_delta))

        for kind, symbol, price, notional in decreases + increases:
            trade = self._execute(kind, symbol, price, notional, step, timestamp)
            if trade is not None:
                trades.append(trade)

        self.history.record_trades(trades)
        self.history.record_equity(
            step=step,
            total_value=self.total_value(prices),
            cash=self._cash,
            timestamp=timestamp,
        )
        return trades

    def _execute(
        self,
        kind: str,
        symbol: str,
        reference_price: float,
        notional: float,
        step: int,
        timestamp: object | None,
    ) -> Trade | None:
        if kind == "long_buy":
            return self._buy_long(symbol, notional, reference_price, step, timestamp)
        if kind == "long_sell":
            return self._sell_long(symbol, notional, reference_price, step, timestamp)
        if kind == "short_open":
            return self._increase_short(symbol, notional, reference_price, step, timestamp)
        if kind == "short_cover":
            return self._decrease_short(symbol, notional, reference_price, step, timestamp)
        raise ValueError(f"Unknown execution kind: {kind}")

    def _buy_long(
        self, symbol: str, desired_notional: float, reference_price: float,
        step: int, timestamp: object | None,
    ) -> Trade | None:
        position = self._long_positions[symbol]
        fill_price = self._long_slippage_model.fill_price(reference_price, TradeSide.BUY)
        if fill_price <= 0.0:
            return None

        max_notional_from_cash = self._cash / (1.0 + self._long_fee_model.taker_fee_rate)
        notional = max(0.0, min(desired_notional, max_notional_from_cash))
        if notional <= _EPSILON:
            return None

        quantity = notional / fill_price
        notional = quantity * fill_price
        fee = self._long_fee_model.fee(notional)
        total_cost = notional + fee
        if total_cost > self._cash + _EPSILON:
            return None

        position.increase(quantity, fill_price)
        self._cash -= total_cost
        slippage_cost = quantity * abs(fill_price - reference_price)

        return Trade(
            step=step, timestamp=timestamp, symbol=symbol, side=TradeSide.BUY,
            quantity=quantity, requested_price=reference_price, fill_price=fill_price,
            fee=fee, slippage_cost=slippage_cost, realized_pnl=0.0, cash_after=self._cash,
        )

    def _sell_long(
        self, symbol: str, desired_notional: float, reference_price: float,
        step: int, timestamp: object | None,
    ) -> Trade | None:
        position = self._long_positions[symbol]
        fill_price = self._long_slippage_model.fill_price(reference_price, TradeSide.SELL)
        if fill_price <= 0.0 or reference_price <= 0.0:
            return None

        quantity = desired_notional / reference_price
        quantity = min(quantity, position.quantity)
        if quantity <= _EPSILON:
            return None

        realized = position.decrease(quantity, fill_price)
        notional = quantity * fill_price
        fee = self._long_fee_model.fee(notional)
        self._cash += notional - fee
        slippage_cost = quantity * abs(fill_price - reference_price)

        return Trade(
            step=step, timestamp=timestamp, symbol=symbol, side=TradeSide.SELL,
            quantity=quantity, requested_price=reference_price, fill_price=fill_price,
            fee=fee, slippage_cost=slippage_cost, realized_pnl=realized, cash_after=self._cash,
        )

    def _increase_short(
        self, symbol: str, desired_notional: float, reference_price: float,
        step: int, timestamp: object | None,
    ) -> Trade | None:
        position = self._short_positions[symbol]
        fill_price = self._short_slippage_model.fill_price(reference_price, TradeSide.SELL)
        if fill_price <= 0.0:
            return None

        # Cap so required_margin + fee never exceeds available shared cash.
        denom = (1.0 / self._leverage) + self._short_fee_model.taker_fee_rate
        max_notional_from_cash = self._cash / denom if denom > 0 else 0.0
        notional = max(0.0, min(desired_notional, max_notional_from_cash))
        if notional <= _EPSILON:
            return None

        quantity = notional / fill_price
        notional = quantity * fill_price
        fee = self._short_fee_model.fee(notional)
        required_margin = notional / self._leverage
        if required_margin + fee > self._cash + _EPSILON:
            return None

        position.increase_short(quantity, fill_price)
        self._cash -= fee
        slippage_cost = quantity * abs(fill_price - reference_price)

        return Trade(
            step=step, timestamp=timestamp, symbol=symbol, side=TradeSide.SELL,
            quantity=quantity, requested_price=reference_price, fill_price=fill_price,
            fee=fee, slippage_cost=slippage_cost, realized_pnl=0.0, cash_after=self._cash,
        )

    def _decrease_short(
        self, symbol: str, desired_notional: float, reference_price: float,
        step: int, timestamp: object | None,
    ) -> Trade | None:
        position = self._short_positions[symbol]
        fill_price = self._short_slippage_model.fill_price(reference_price, TradeSide.BUY)
        if fill_price <= 0.0 or reference_price <= 0.0:
            return None

        quantity = desired_notional / reference_price
        quantity = min(quantity, position.short_quantity)
        if quantity <= _EPSILON:
            return None

        realized = position.decrease_short(quantity, fill_price)
        notional = quantity * fill_price
        fee = self._short_fee_model.fee(notional)
        self._cash += realized - fee
        slippage_cost = quantity * abs(fill_price - reference_price)

        return Trade(
            step=step, timestamp=timestamp, symbol=symbol, side=TradeSide.BUY,
            quantity=quantity, requested_price=reference_price, fill_price=fill_price,
            fee=fee, slippage_cost=slippage_cost, realized_pnl=realized, cash_after=self._cash,
        )

    def _liquidate_all_shorts(
        self, prices: Sequence[float], step: int, timestamp: object | None
    ) -> list[Trade]:
        trades: list[Trade] = []
        for symbol, price in zip(self._symbols, prices):
            position = self._short_positions[symbol]
            if position.short_quantity <= 0.0 or price <= 0.0:
                continue
            trade = self._decrease_short(symbol, position.notional(price), price, step, timestamp)
            if trade is not None:
                trades.append(trade)
        return trades

    def total_value(self, prices: Sequence[float]) -> float:
        value = self._cash
        for symbol, price in zip(self._symbols, prices):
            value += self._long_positions[symbol].market_value(price)
            value += self._short_positions[symbol].unrealized_pnl(price)
        return value

    def current_weights(self, prices: Sequence[float]) -> list[float]:
        """Net signed weight per asset in [-1, 1]: positive is long
        (fraction of equity, 1x), negative is short (fraction of
        equity * leverage)."""
        equity = self.total_value(prices)
        if equity <= 0:
            return [0.0 for _ in self._symbols]
        weights = []
        for symbol, price in zip(self._symbols, prices):
            long_w = self._long_positions[symbol].market_value(price) / equity
            short_denom = equity * self._leverage
            short_w = self._short_positions[symbol].notional(price) / short_denom if short_denom > 0 else 0.0
            weights.append(long_w - short_w)
        return weights

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def leverage(self) -> float:
        return self._leverage

    @property
    def long_positions(self) -> dict[str, Position]:
        return self._long_positions

    @property
    def short_positions(self) -> dict[str, FuturesPosition]:
        return self._short_positions

    def long_position(self, symbol: str) -> Position:
        symbol = normalize_symbol(symbol)
        if symbol not in self._long_positions:
            raise UnknownAssetError(f"Unknown asset: {symbol}")
        return self._long_positions[symbol]

    def short_position(self, symbol: str) -> FuturesPosition:
        symbol = normalize_symbol(symbol)
        if symbol not in self._short_positions:
            raise UnknownAssetError(f"Unknown asset: {symbol}")
        return self._short_positions[symbol]

    def unrealized_pnl(self, prices: Sequence[float]) -> float:
        total = 0.0
        for symbol, price in zip(self._symbols, prices):
            total += self._long_positions[symbol].unrealized_pnl(price)
            total += self._short_positions[symbol].unrealized_pnl(price)
        return total

    def realized_pnl(self) -> float:
        total = sum(p.realized_pnl for p in self._long_positions.values())
        total += sum(p.realized_pnl for p in self._short_positions.values())
        return total

    def metrics(self) -> MetricsSummary:
        """Compute the full performance metrics summary for this episode so far."""
        return self._metrics_engine.summary(self.history)