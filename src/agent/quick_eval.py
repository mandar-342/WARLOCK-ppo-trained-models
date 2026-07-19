from __future__ import annotations

"""
Cheap, no-file-I/O evaluation used by the multi-seed harness and the
Optuna sweep. `src.agent.evaluate.Evaluator` is the full pipeline (CSVs,
plots, PDF report) meant for one-off runs; running that for every
seed x trial combination would be needlessly slow and would flood the
experiments directory. This module reruns the same deterministic
evaluation episode (fixed held-out `test.parquet`, `deterministic_start`)
and only keeps what the seed harness needs: the performance metrics and
whether the episode ended via the drawdown circuit breaker rather than
simply running out of data.
"""

from typing import Any

import pandas as pd
from stable_baselines3 import PPO

from src.analytics.metrics import MetricsCalculator
from src.env.gym_bitcoin import GymBitcoinEnv
from src.utils import config, root


def run_quick_episode(model: PPO) -> tuple[dict[str, Any], bool]:
    """
    Runs one deterministic evaluation episode on the held-out test set.

    Returns
    -------
    metrics
        Dict of performance metrics (same fields as evaluate.py's
        metrics.json).
    breaker_hit
        True if the episode was terminated early by the portfolio
        drawdown circuit breaker (`drawdown >= env.max_drawdown`)
        rather than by reaching the end of the test data.
    """

    evaluation_cfg = config["evaluation"]

    test_data = root("data", "features", "test.parquet")

    env = GymBitcoinEnv(
        data_path=str(test_data),
        deterministic_start=True,
    )

    observation, _ = env.reset(seed=evaluation_cfg.get("seed", 42))

    equity_curve: list[float] = []
    trade_returns: list[float] = []
    prev_realized_pnl = 0.0

    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}

    while not (terminated or truncated):

        action, _ = model.predict(
            observation,
            deterministic=evaluation_cfg["deterministic"],
        )

        observation, reward, terminated, truncated, info = env.step(action)
        final_info = info

        equity_curve.append(float(info["capital"]))

        if int(info.get("n_trades_this_step", 0)) > 0:
            cumulative_realized = float(info["realized_pnl"])
            trade_returns.append(cumulative_realized - prev_realized_pnl)
            prev_realized_pnl = cumulative_realized

    metrics = MetricsCalculator(
        equity_curve=pd.Series(equity_curve),
        trade_returns=pd.Series(trade_returns),
        risk_free_rate=evaluation_cfg["risk_free_rate"],
    ).to_dict()

    # A run that ends because it ran out of data has current_step at/near
    # max_steps; a run stopped by the circuit breaker ends with
    # drawdown >= max_drawdown at a step short of that. Checking the
    # drawdown directly (rather than inferring from step count) is exact
    # and matches the condition in GymBitcoinEnv.step()/reset() verbatim.
    breaker_hit = bool(
        final_info
        and float(final_info.get("drawdown", 0.0)) >= env.max_drawdown
    )

    env.close()

    return metrics, breaker_hit