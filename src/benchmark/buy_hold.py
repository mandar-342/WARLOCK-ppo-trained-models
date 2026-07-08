from __future__ import annotations

import numpy as np

from src.benchmark.runner import BenchmarkRunner

from pathlib import Path

from loguru import logger

from src.benchmark.base import BenchmarkBase
from src.env.gym_bitcoin import GymBitcoinEnv
from src.utils import config, root



class BuyHoldBenchmark(BenchmarkBase):
    """
    Buy-and-Hold benchmark.

    The strategy invests 100% of available capital at the
    beginning of the evaluation and never changes the position.
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
            "Buy & Hold benchmark initialized."
        )

    @property
    def name(self) -> str:
        return "buy_hold"
    
    
    def run(self) -> None:
        """
        Execute the Buy & Hold benchmark.
        """

        logger.info(
            "Running Buy & Hold benchmark."
        )

        runner = BenchmarkRunner(
            self._environment,
        )

        runner.run(
            lambda _: np.array(
                [1.0],
                dtype=np.float32,
            )
        )

        results = runner.results

        self._equity_curve = results["equity_curve"]
        self._trade_returns = results["trade_returns"]
        self._portfolio_history = results["portfolio_history"]
        self._trade_history = results["trade_history"]

        logger.success(
            "Buy & Hold benchmark completed."
        )
        



def main() -> int:

    output_directory = (
        root(
            "benchmarks",
            "buy_hold",
        )
    )

    benchmark = BuyHoldBenchmark(
        output_directory=output_directory,
    )

    logger.info("=" * 80)
    logger.info("Buy & Hold Benchmark")
    logger.info("=" * 80)
    benchmark.evaluate()
    logger.success(
    "Buy & Hold benchmark completed successfully."
)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())