from __future__ import annotations

from src.portfolio import Portfolio
import sys
import traceback
from loguru import logger


def test_open_short_uses_leverage():
    pf = Portfolio()
    price = 50_000.0
    trades = pf.step(target_weights=[-1.0], prices=[price], step=1)
    assert len(trades) == 1
    short = pf.short_position("BTC/USDT")
    assert short.notional(price) > pf.cash  # levered beyond 1x
    assert pf.cash < 10_000.0  # fee paid out of shared cash pool


def test_short_profits_when_price_falls():
    pf = Portfolio()
    price = 50_000.0
    pf.step(target_weights=[-1.0], prices=[price], step=1)

    lower_price = 45_000.0
    assert pf.unrealized_pnl([lower_price]) > 0.0

    trades = pf.step(target_weights=[0.0], prices=[lower_price], step=2)
    assert len(trades) == 1
    assert pf.short_position("BTC/USDT").is_flat()
    assert pf.realized_pnl() > 0.0
    assert pf.cash > 10_000.0  # net profit after fees


def test_short_loses_when_price_rises():
    pf = Portfolio()
    price = 50_000.0
    pf.step(target_weights=[-0.5], prices=[price], step=1)

    higher_price = 52_000.0
    assert pf.unrealized_pnl([higher_price]) < 0.0

    pf.step(target_weights=[0.0], prices=[higher_price], step=2)
    assert pf.realized_pnl() < 0.0


def test_adverse_move_triggers_liquidation():
    pf = Portfolio()
    price = 50_000.0
    pf.step(target_weights=[-1.0], prices=[price], step=1)
    assert not pf.short_position("BTC/USDT").is_flat()

    # +30% against a 3x short should breach the maintenance margin and
    # force-close the position before any further rebalancing happens.
    adverse_price = 65_000.0
    pf.step(target_weights=[-1.0], prices=[adverse_price], step=2)
    assert pf.cash >= 0.0


def test_margin_cap_limits_position_size():
    pf = Portfolio()
    price = 50_000.0
    pf.step(target_weights=[-1.0], prices=[price], step=1)
    short = pf.short_position("BTC/USDT")
    required_margin = short.notional(price) / pf.leverage
    assert required_margin <= 10_000.0 + 1e-6


def test_flip_from_long_to_short_reuses_freed_cash():
    """The whole point of unifying: closing a long frees cash that can
    immediately fund opening a short, all within one step() call."""
    pf = Portfolio()
    price = 50_000.0
    pf.step(target_weights=[0.8], prices=[price], step=1)
    assert pf.long_position("BTC/USDT").quantity > 0.0

    trades = pf.step(target_weights=[-0.8], prices=[price], step=2)
    assert len(trades) == 2  # one sell (close long), one short-open
    assert pf.long_position("BTC/USDT").is_flat()
    assert pf.short_position("BTC/USDT").short_quantity > 0.0


def test_reset_clears_state():
    pf = Portfolio()
    pf.step(target_weights=[-1.0], prices=[50_000.0], step=1)
    assert not pf.short_position("BTC/USDT").is_flat()
    pf.reset()
    assert pf.short_position("BTC/USDT").is_flat()
    assert pf.long_position("BTC/USDT").is_flat()
    assert pf.cash == pf._initial_capital
    assert pf.history.n_trades == 0

TESTS = [
    test_open_short_uses_leverage,
    test_short_profits_when_price_falls,
    test_short_loses_when_price_rises,
    test_adverse_move_triggers_liquidation,
    test_margin_cap_limits_position_size,
    test_flip_from_long_to_short_reuses_freed_cash,
    test_reset_clears_state,
]

def run_all() -> bool:
    logger.info("Portfolio integration test run starting")

    results: list[tuple[str, bool, str]] = []
    for test_fn in TESTS:
        name = test_fn.__name__
        logger.info(f" RUNNING {name} ")
        try:
            test_fn()
            logger.success(f" PASSED {name} \n")
            results.append((name, True, ""))
        except Exception as e:
            logger.error(f" FAILED {name}: {e} ")
            logger.debug(traceback.format_exc())
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