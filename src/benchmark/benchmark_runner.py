from __future__ import annotations

from collections.abc import Callable

import numpy as np
from loguru import logger

from src.env.gym_bitcoin import GymBitcoinEnv
from src.utils import config


class BenchmarkRunner:
    """
    Executes a single evaluation episode using an arbitrary
    action-selection function.

    The runner is environment-agnostic and strategy-agnostic.
    """

    def __init__(
        self,
        environment: GymBitcoinEnv,
    ) -> None:

        self._environment = environment

        self.equity_curve: list[float] = []
        self.trade_returns: list[float] = []
        self.portfolio_history: list[dict] = []
        self.trade_history: list[dict] = []

    def reset(self) -> None:

        self.equity_curve.clear()

        self.trade_returns.clear()

        self.portfolio_history.clear()

        self.trade_history.clear()

    def run(
        self,
        action_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Execute one complete episode.

        Parameters
        ----------
        action_fn
            Function mapping observation -> action.
        """

        logger.info(
            "Starting benchmark episode."
        )

        self.reset()

        # Previously an unseeded `reset()`: for envs without
        # `deterministic_start=True` this left the episode start step
        # (and, for callers like the random-agent benchmark,
        # `action_space.sample()`) tied to OS entropy, so results
        # differed on every run even with an otherwise identical setup.
        seed = int(config.get("evaluation", {}).get("seed", 42))
        self._environment.action_space.seed(seed)
        observation, _ = self._environment.reset(seed=seed)

        terminated = False

        truncated = False
        while not (terminated or truncated):

            action = action_fn(observation)

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = self._environment.step(action)

            self._record_step(
        info=info,
        reward=reward,
    )

            

            

        logger.success(
            "Benchmark episode completed."
        )
        
    def _record_step(
    self,
    info: dict,
    reward: float,
      ) -> None:
        
        
       """
    Record one environment step into the benchmark history.
       """

       self.equity_curve.append(
        float(info["capital"])
    )

       self.portfolio_history.append(
        {
            "step": int(info["step"]),
            "price": float(info["price"]),
            "capital": float(info["capital"]),
            "cash": float(info["cash"]),
            "drawdown": float(info["drawdown"]),
            "realized_pnl": float(
                info["realized_pnl"]
            ),
            "unrealized_pnl": float(
                info["unrealized_pnl"]
            ),
            "reward": float(reward),
            "weights": list(info["weights"]),
            "forced_exit": bool(
                info["forced_exit"]
            ),
            "exit_reason": info["exit_reason"],
           }
       )

       if int(info["n_trades_this_step"]) > 0:
            

            realized = float(
            info["realized_pnl"]
        )

            self.trade_returns.append(
            realized
        )

            self.trade_history.append(
            {
                "step": int(info["step"]),
                "price": float(info["price"]),
                "realized_pnl": realized,
                "forced_exit": bool(
                    info["forced_exit"]
                ),
                "exit_reason": info[
                    "exit_reason"
                ],
            }
        )
    
    @property
    def results(self) -> dict:
        """
        Return all benchmark outputs collected during the episode.
        """

        return {
            "equity_curve": self.equity_curve,
            "trade_returns": self.trade_returns,
            "portfolio_history": self.portfolio_history,
            "trade_history": self.trade_history,
        }

    @property
    def final_capital(self) -> float:
        """
        Final portfolio value.
        """

        if not self.equity_curve:
            return 0.0

        return float(self.equity_curve[-1])

    @property
    def total_trades(self) -> int:
        """
        Number of completed trades.
        """

        return len(self.trade_history)

    def __str__(self) -> str:
        return (
            f"BenchmarkRunner("
            f"capital={self.final_capital:.2f}, "
            f"trades={self.total_trades})"
        )