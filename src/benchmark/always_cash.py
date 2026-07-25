from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from src.benchmark.base import BenchmarkBase
from src.benchmark.runner import BenchmarkRunner
from src.env.gym_bitcoin import GymBitcoinEnv
from src.utils import config, root


class AlwaysCashBenchmark(BenchmarkBase):
    """
    Always Cash benchmark.

    The strategy never enters the market and keeps
    the portfolio fully in cash.
    """

    def __init__(
        self,
        output_directory: str | Path,
    ) -> None:

        super().__init__(output_directory)

        test_data = (
            root(
                config["paths"]["feature_engineered_dir"]
            )
            / "test.parquet"
        )

        self._environment = GymBitcoinEnv(
            data_path=str(test_data),
        )

        logger.info(
            "Always Cash benchmark initialized."
        )

    @property
    def name(self) -> str:
        return "always_cash"
    
    def run(self) -> None:
        """
        Execute the Always Cash benchmark.
        """

        logger.info(
            "Running Always Cash benchmark."
        )

        runner = BenchmarkRunner(
            self._environment,
        )

        runner.run(
            lambda _: np.array(
                [0.0],
                dtype=np.float32,
            )
        )

        results = runner.results

        self._equity_curve = results["equity_curve"]
        self._trade_returns = results["trade_returns"]
        self._portfolio_history = results["portfolio_history"]
        self._trade_history = results["trade_history"]

        logger.success(
            "Always Cash benchmark completed."
        )
        
def main() -> int:

    output_directory = root(
        "benchmarks",
        "always_cash",
    )

    benchmark = AlwaysCashBenchmark(
        output_directory=output_directory,
    )

    logger.info("=" * 80)
    logger.info("Always Cash Benchmark")
    logger.info("=" * 80)

    benchmark.evaluate()

    logger.success(
        "Always Cash benchmark completed successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())