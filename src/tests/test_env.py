from __future__ import annotations
import sys
import traceback
import numpy as np
from loguru import logger
from src.env import GymBitcoinEnv
from src.utils import config, root

log_dir = root(config["paths"]["logs_dir"])
log_dir.mkdir(exist_ok=True, parents=True)
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level: <7}|{message}")
logger.add(
    str(log_dir / "env_test_{time}.log"),
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
)

def test_env_creation_and_spaces():
    env = GymBitcoinEnv()
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Max steps: {env.max_steps}, features: {env.n_features}")
    assert env.action_space.shape == (1,)  # BTC-only for now
    assert env.action_space.low[0] == 0.0
    assert env.action_space.high[0] == 1.0
    env.close()

def test_reset_contract():
    env = GymBitcoinEnv()
    obs, info = env.reset(seed=42)
    logger.info(f"Reset obs shape: {obs.shape} (expected {env.observation_space.shape})")
    logger.info(f"Reset info: {info}")
    assert obs.shape == env.observation_space.shape
    assert "step" in info and "price" in info and "capital" in info
    assert info["capital"] == env.portfolio.cash  # flat at reset, all cash
    env.close()

def test_full_allocation_buys_btc():
    env = GymBitcoinEnv()
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    logger.info(
        f"After action=1.0: weights={info['weights']} cash={info['cash']:.4f} "
        f"capital={info['capital']:.4f} cost={info['cost']:.4f} reward={reward:.6f}"
    )
    # max_trade_step in config caps the move, so weight should be > 0 but
    # likely not yet at 1.0 after a single step.
    assert info["weights"][0] > 0.0
    assert info["cost"] > 0.0  # fee + slippage charged
    env.close()

def test_full_episode_random_policy():
    env = GymBitcoinEnv()
    obs, info = env.reset(seed=7)
    total_reward = 0.0
    n_steps = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            logger.info(f"Episode terminated early at step {info['step']}")
            break
    logger.info(
        f"Ran {n_steps} steps, total_reward={total_reward:.4f}, "
        f"final capital={info['capital']:.4f}, final drawdown={info['drawdown']:.4f}"
    )
    summary = env.portfolio.metrics()
    logger.info(f"Portfolio metrics after episode: {summary.to_dict()}")
    env.close()

def test_action_clipping():
    env = GymBitcoinEnv()
    env.reset(seed=1)
    # Out-of-range action should clip to [0, 1], not error.
    obs, r, t, tr, info = env.step(np.array([5.0], dtype=np.float32))
    assert 0.0 <= info["weights"][0] <= 1.0
    logger.info(f"Out-of-range action clipped correctly: weights={info['weights']}")
    env.close()

def test_max_trade_step_rate_limit():
    env = GymBitcoinEnv()
    env.reset(seed=3)
    env.step(np.array([1.0], dtype=np.float32))
    w1 = env.current_weights[0]
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    w2 = env.current_weights[0]
    logger.info(f"Weight after target=1.0: {w1:.4f}, after target=0.0: {w2:.4f}")
    # max_trade_step (config: 0.2) should prevent an instant full reversal.
    assert w2 >= w1 - env.position_sizer.max_step_change - 1e-6
    env.close()

def test_reset_clears_portfolio_state():
    env = GymBitcoinEnv()
    env.reset(seed=1)
    env.step(np.array([1.0], dtype=np.float32))
    assert env.portfolio.history.n_trades > 0
    env.reset(seed=1)
    assert env.portfolio.history.n_trades == 0
    assert env.portfolio.cash == env.initial_capital
    logger.info("Reset correctly clears portfolio trade history and cash")
    env.close()

TESTS = [
    test_env_creation_and_spaces,
    test_reset_contract,
    test_full_allocation_buys_btc,
    test_full_episode_random_policy,
    test_action_clipping,
    test_max_trade_step_rate_limit,
    test_reset_clears_portfolio_state,
]

def run_all() -> bool:
    logger.info("GymBitcoinEnv integration test run starting")

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