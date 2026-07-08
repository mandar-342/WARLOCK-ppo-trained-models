from __future__ import annotations

import random
import sys
import traceback

from loguru import logger

from src.portfolio import SpotPortfolio, InsufficientPositionError
from src.portfolio.order_executor import FeeModel, OrderExecutor, SlippageModel
from src.portfolio.position import Position
from src.portfolio.sizing import AssetDelta
from src.utils import config, root

log_dir = root(config["paths"]["logs_dir"])
log_dir.mkdir(exist_ok=True, parents=True)
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level: <7}|{message}")
logger.add(
    str(log_dir / "portfolio_test_{time}.log"),
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
)

def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol

def test_single_asset_full_buy_and_sell():
    pf = SpotPortfolio()
    assert pf.symbols == ["BTC/USDT"]
    assert pf.cash == 10000.0

    price = 50_000.0

    trades = pf.step(target_weights=[1.0], prices=[price], step=1)
    assert len(trades) == 1
    t = trades[0]
    assert t.side.value == "buy"
    logger.info(
        f"BUY  qty={t.quantity:.6f} fill={t.fill_price:.2f} "
        f"fee={t.fee:.4f} cash_after={t.cash_after:.4f}"
    )

    assert pf.cash < 50.0
    btc_pos = pf.position("BTC/USDT")
    assert btc_pos.quantity > 0

    total_value_before = pf.total_value([price])
    logger.info(
        f"Total value after buy: {total_value_before:.4f} "
        f"(started at 10000, fee+slippage drag expected)"
    )
    assert total_value_before < 10000.0
    assert total_value_before > 9900.0

    new_price = price * 1.10
    unrealized = pf.unrealized_pnl([new_price])
    logger.info(f"Unrealized PnL after +10% move: {unrealized:.4f}")
    assert unrealized > 0

    trades2 = pf.step(target_weights=[0.0], prices=[new_price], step=2)
    assert len(trades2) == 1
    t2 = trades2[0]
    assert t2.side.value == "sell"
    logger.info(
        f"SELL qty={t2.quantity:.6f} fill={t2.fill_price:.2f} fee={t2.fee:.4f} "
        f"realized_pnl={t2.realized_pnl:.4f} cash_after={t2.cash_after:.4f}"
    )

    assert btc_pos.is_flat()
    assert pf.cash > 10000.0
    final_value = pf.total_value([new_price])
    logger.info(f"Final value: {final_value:.4f}")

def test_partial_allocation():
    pf = SpotPortfolio()
    price = 30_000.0
    trades = pf.step(target_weights=[0.5], prices=[price], step=1)
    assert len(trades) == 1
    weights = pf.current_weights([price])
    logger.info(f"Current weights after 50% target: {weights}")
    assert approx(weights[0], 0.5, tol=0.01)

def test_min_rebalance_delta_skips_dust_trades():
    pf = SpotPortfolio()
    price = 30_000.0
    pf.step(target_weights=[0.5], prices=[price], step=1)
    trades = pf.step(target_weights=[0.5005], prices=[price], step=2)
    logger.info(f"Trades from tiny nudge: {len(trades)}")
    assert len(trades) == 0

def test_zero_cash_buy_resolves_to_noop():
    pf = SpotPortfolio()
    pf._cash = 0.0
    trades = pf.step(target_weights=[1.0], prices=[100.0], step=1)
    assert len(trades) == 0
    logger.info("Zero-cash buy correctly resolved to no trade (no exception)")

def test_insufficient_position_guard():
    executor = OrderExecutor(FeeModel(0.0005, 0.0005), SlippageModel("fixed_bps", 5.0))
    position = Position(symbol="BTC/USDT", quantity=0.01, avg_entry_price=50_000.0)
    positions = {"BTC/USDT": position}

    bad_delta = AssetDelta(
        symbol="BTC/USDT", price=50_000.0,
        target_value=0.0, current_value=5000.0, delta_value=-100_000.0,
    )
    try:
        executor.execute([bad_delta], positions, cash=0.0, step=1)
        raise AssertionError("Expected InsufficientPositionError")
    except InsufficientPositionError as e:
        logger.info(f"Correctly raised InsufficientPositionError: {e}")

def test_history_and_metrics_random_walk():
    random.seed(42)

    pf = SpotPortfolio()
    price = 50_000.0
    weight = 0.0
    for step in range(1, 101):
        price *= (1.0 + random.uniform(-0.01, 0.01))
        weight = min(max(weight + random.uniform(-0.1, 0.1), 0.0), 1.0)
        pf.step(target_weights=[weight], prices=[price], step=step)

    assert pf.history.n_trades > 0
    assert len(pf.history.equity_curve) == 100
    logger.info(
        f"n_trades={pf.history.n_trades} "
        f"total_fees={pf.history.total_fees_paid():.4f} "
        f"total_slippage={pf.history.total_slippage_cost():.4f}"
    )

    summary = pf.metrics()
    logger.info(f"Metrics summary: {summary.to_dict()}")
    assert isinstance(summary.total_return, float)
    assert 0.0 <= summary.max_drawdown <= 1.0
    assert summary.n_trades == pf.history.n_trades
    assert 0.0 <= summary.avg_exposure <= 1.0
    assert 0.0 <= summary.win_rate <= 1.0

def test_reset_clears_history():
    pf = SpotPortfolio()
    pf.step(target_weights=[1.0], prices=[50_000.0], step=1)
    assert pf.history.n_trades == 1
    pf.reset()
    assert pf.history.n_trades == 0
    assert len(pf.history.equity_curve) == 0
    assert pf.cash == 10000.0
    logger.info("Reset correctly clears history and restores initial capital")

TESTS = [
    test_single_asset_full_buy_and_sell,
    test_partial_allocation,
    test_min_rebalance_delta_skips_dust_trades,
    test_zero_cash_buy_resolves_to_noop,
    test_insufficient_position_guard,
    test_history_and_metrics_random_walk,
    test_reset_clears_history,
]

def run_all() -> bool:
    logger.info("Portfolio subsystem test run starting")

    results: list[tuple[str, bool, str]] = []

    for test_fn in TESTS:
        name = test_fn.__name__
        logger.info(f" RUNNING {name} ")
        try:
            test_fn()
            logger.success(f" PASSED {name} \n")
            results.append((name, True, ""))
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f" FAILED {name}: {e} ")
            logger.debug(tb)
            results.append((name, False, str(e)))

    n_passed = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)

    logger.info("SUMMARY")
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {name}"
        if not ok:
            line += f"  ({err})"
        logger.info(line)

    if n_passed == n_total:
        logger.success(f"ALL TESTS PASSED ({n_passed}/{n_total})")
    else:
        logger.error(f"{n_total - n_passed} TEST(S) FAILED ({n_passed}/{n_total} passed)")

    return n_passed == n_total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)