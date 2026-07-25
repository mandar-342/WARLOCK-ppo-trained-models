from __future__ import annotations

import numpy as np

from loguru import logger

from src.benchmark.runner import BenchmarkRunner


class BuyHoldRunner(BenchmarkRunner):

    """
    Executes a true Buy & Hold benchmark.

    Buy once on the first step and then hold the position
    until the episode finishes.
    """

    def run(self) -> None:

        logger.info(
            "Starting Buy & Hold benchmark."
        )

        self.reset()

        observation, _ = self._environment.reset()

        terminated = False
        truncated = False

        first_step = True
        while not (terminated or truncated):
            if first_step:
                
                (
               observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self._environment.step(
            np.array(
                [1.0],
                dtype=np.float32,
            )
        )
              

                first_step = False

            else:
                (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self._environment.hold_step()
                
                

            
            self._record_step(
        info=info,
        reward=reward,
    )
            
        logger.success(
    "Buy & Hold benchmark completed."
)