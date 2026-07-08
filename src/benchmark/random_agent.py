from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.benchmark.base import BenchmarkBase
from src.benchmark.runner import BenchmarkRunner
from src.env.gym_bitcoin import GymBitcoinEnv
from src.utils import config, root


class RandomAgentBenchmark(BenchmarkBase):
    """
    Random benchmark.

    A random action is sampled from the environment's
    continuous action space at every timestep.
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
            "Random Agent benchmark initialized."
        )

    @property
    def name(self) -> str:
        return "random_agent"
    
    def run(self) -> None:
        """
        Execute the Random Agent benchmark.
        """

        logger.info(
            "Running Random Agent benchmark."
        )

        runner = BenchmarkRunner(
            self._environment,
        )

        runner.run(
            lambda _: self._environment.action_space.sample()
        )

        results = runner.results

        self._equity_curve = results["equity_curve"]
        self._trade_returns = results["trade_returns"]
        self._portfolio_history = results["portfolio_history"]
        self._trade_history = results["trade_history"]

        logger.success(
            "Random Agent benchmark completed."
        )
        
def main() -> int:

    output_directory = root(
        "benchmarks",
        "random_agent",
    )

    benchmark = RandomAgentBenchmark(
        output_directory=output_directory,
    )

    logger.info("=" * 80)
    logger.info("Random Agent Benchmark")
    logger.info("=" * 80)

    benchmark.evaluate()

    logger.success(
        "Random Agent benchmark completed successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())