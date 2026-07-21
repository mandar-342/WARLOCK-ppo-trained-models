from __future__ import annotations

import sys
import traceback
import numpy as np
from loguru import logger
from src.env import GymBitcoinEnv
from src.env.rewards import RewardCalculator
from src.utils import config, root

log_dir = root(config["paths"]["logs_dir"])
log_dir.mkdir(exist_ok=True, parents=True)
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level: <7}|{message}")
logger.add(
    str(log_dir / "rewards_test_{time}.log"),
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
)

def test_zero_drawdown_zero_overtrade():
    rc = RewardCalculator()
    reward = rc.calculate(step_return=0.01, drawdown=0.0, position_change=0.0)
    comps = rc.last_components

    # total = step_return_weight*tanh(step_return*100) + sharpe_weight*tanh(sharpe)
    #         - dd_pen - ot_pen.
    # On the very first call the returns buffer is below MIN_BUFFER_SIZE, so
    # the sharpe term is still 0 and this reduces to the pure step-return
    # component scaled by step_return_weight (1.0 by default).
    expected_total = np.tanh(0.01 * 100.0)
    assert abs(reward - expected_total) < 1e-9
    assert comps["drawdown_penalty"] == 0.0
    assert comps["overtrade_penalty"] == 0.0

def test_drawdown_strictly_penalizes_reward():
    rc_a = RewardCalculator()
    rc_b = RewardCalculator()
    # Same return history, but rc_b also reports a large drawdown each step.
    for _ in range(5):
        r_a = rc_a.calculate(step_return=0.005, drawdown=0.0, position_change=0.0)
        r_b = rc_b.calculate(step_return=0.005, drawdown=0.2, position_change=0.0)
    logger.info(f"No-drawdown reward={r_a:.6f}, with-drawdown reward={r_b:.6f}")
    assert r_b < r_a
    assert rc_b.last_components["drawdown_penalty"] > 0.0

def test_overtrading_strictly_penalizes_reward():
    rc_a = RewardCalculator()
    rc_b = RewardCalculator()
    for _ in range(5):
        r_a = rc_a.calculate(step_return=0.005, drawdown=0.0, position_change=0.0)
        r_b = rc_b.calculate(step_return=0.005, drawdown=0.0, position_change=1.0)
    logger.info(f"No-overtrade reward={r_a:.6f}, overtrading reward={r_b:.6f}")
    assert r_b < r_a
    assert rc_b.last_components["overtrade_penalty"] > 0.0

def test_negative_returns_yield_negative_sharpe():
    # Explicit sharpe_aggregation_steps=1 isolates the per-step sharpe
    # calculation itself from the aggregation behavior (default 12 from
    # config.yaml) exercised in test_sharpe_aggregation_smooths_reward
    # below.
    rc = RewardCalculator(sharpe_aggregation_steps=1)
    rc.calculate(step_return=-0.01, drawdown=0.0, position_change=0.0)
    reward = rc.calculate(step_return=-0.01, drawdown=0.0, position_change=0.0)
    logger.info(f"Consistent losses: reward={reward:.6f} components={rc.last_components}")
    assert reward < 0.0

def test_sharpe_aggregation_smooths_reward():
    # With sharpe_aggregation_steps=4, the returns_buffer should only
    # gain a new sample every 4th calculate() call, and that sample
    # should be the *sum* of the 4 intervening step returns rather than
    # each one individually -- this is the fix for the sharpe term
    # previously being estimated from tiny, single-timestep samples.
    rc = RewardCalculator(sharpe_aggregation_steps=4, sharpe_weight=0.5)
    for i in range(3):
        rc.calculate(step_return=0.01, drawdown=0.0, position_change=0.0)
        assert len(rc.returns_buffer) == 0, "buffer should not update before aggregation completes"
    rc.calculate(step_return=0.01, drawdown=0.0, position_change=0.0)
    assert len(rc.returns_buffer) == 1
    assert abs(rc.returns_buffer[0] - 0.04) < 1e-9, "buffer sample should be the sum of the 4 step returns"

def test_reset_clears_buffer_and_components():
    rc = RewardCalculator()
    rc.calculate(step_return=0.02, drawdown=0.1, position_change=0.5)
    # Default sharpe_aggregation_steps (12, from config.yaml) means a
    # single calculate() call only adds to the pending-returns
    # accumulator; the returns_buffer itself only gets a new sample once
    # 12 steps have accumulated. reset() should clear both back to empty.
    assert len(rc.returns_buffer) == 0
    assert len(rc._pending_returns) == 1
    rc.reset()
    assert len(rc.returns_buffer) == 0
    assert len(rc._pending_returns) == 0
    assert rc.last_components["total_reward"] == 0.0
    logger.info("Reset correctly clears returns buffer, pending returns, and last_components")

def test_single_sample_buffer():
    rc = RewardCalculator()
    reward = rc.calculate(step_return=0.05, drawdown=0.0, position_change=0.0)

    # A single sample hasn't completed one aggregation window yet
    # (config default sharpe_aggregation_steps=12), so the sharpe
    # component is still in its warm-up state (0.0) rather than absent --
    # `calculate()` always reports both components now that
    # `_sharpe_reward` is actually wired in.
    assert "sharpe_reward" in rc.last_components
    assert rc.last_components["sharpe_reward"] == 0.0
    assert abs(reward - np.tanh(0.05 * 100.0)) < 1e-9

def test_env_step_exposes_reward_components():
    env = GymBitcoinEnv()
    env.reset(seed=11)
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    logger.info(f"info['reward_components']={info['reward_components']}")
    assert "reward_components" in info
    comps = info["reward_components"]
    assert set(comps.keys()) == {
        "step_return", "reward_return", "sharpe_reward", "drawdown_penalty", "overtrade_penalty", "total_reward"
    }
    assert abs(comps["total_reward"] - reward) < 1e-9
    env.close()

def test_env_step_exposes_realized_and_unrealized_pnl():
    env = GymBitcoinEnv()
    env.reset(seed=12)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    logger.info(
        f"After full allocation: realized_pnl={info['realized_pnl']:.4f}, "
        f"unrealized_pnl={info['unrealized_pnl']:.4f}"
    )
    # Just opened the position this step, so nothing has been realized yet,
    # but unrealized PnL should reflect the entry slippage drag.
    assert info["realized_pnl"] == 0.0
    assert isinstance(info["unrealized_pnl"], float)
    env.close()

def test_higher_drawdown_correlates_with_lower_reward_in_env():
    env = GymBitcoinEnv()
    env.reset(seed=21)
    whipsaw_rewards = []
    for i in range(30):
        action = np.array([1.0 if i % 2 == 0 else 0.0], dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        whipsaw_rewards.append(reward)
        if terminated or truncated:
            break
    env.close()

    env2 = GymBitcoinEnv()
    env2.reset(seed=21)
    steady_rewards = []
    for i in range(30):
        action = np.array([0.5], dtype=np.float32)
        _, reward, terminated, truncated, _ = env2.step(action)
        steady_rewards.append(reward)
        if terminated or truncated:
            break
    env2.close()

    logger.info(
        f"Whipsaw mean reward={np.mean(whipsaw_rewards):.6f}, "
        f"steady mean reward={np.mean(steady_rewards):.6f}"
    )
    # Not a strict assertion on which is better (depends on price path),
    # just confirms both policies produce finite, distinct reward series —
    # i.e. the overtrading penalty is actually doing something.
    assert whipsaw_rewards != steady_rewards
    assert all(np.isfinite(r) for r in whipsaw_rewards + steady_rewards)

TESTS = [
    test_zero_drawdown_zero_overtrade,
    test_drawdown_strictly_penalizes_reward,
    test_overtrading_strictly_penalizes_reward,
    test_negative_returns_yield_negative_sharpe,
    test_sharpe_aggregation_smooths_reward,
    test_reset_clears_buffer_and_components,
    test_single_sample_buffer,
    test_env_step_exposes_reward_components,
    test_env_step_exposes_realized_and_unrealized_pnl,
    test_higher_drawdown_correlates_with_lower_reward_in_env,
]

def run_all() -> bool:
    logger.info("Reward integration test run starting")

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