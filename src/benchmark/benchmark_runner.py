from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.benchmark.always_cash import AlwaysCashBenchmark
from src.benchmark.buy_hold import BuyHoldBenchmark
from src.benchmark.random_agent import RandomAgentBenchmark
from src.utils import root


class BenchmarkSuite:
    """
    Executes every benchmark strategy and
    aggregates the results into a comparison.
    """

    def __init__(
        self,
        output_directory: str | Path,
    ) -> None:

        self._output_dir = Path(output_directory)

        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._comparison: list[dict] = []

    @staticmethod
    def _load_metrics(
        metrics_path: Path,
    ) -> dict:

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)
        
    def _run_benchmarks(self) -> None:
        """
        Execute every benchmark strategy and collect metrics.
        """

        benchmarks = [
            BuyHoldBenchmark(
                output_directory=root(
                    "benchmarks",
                    "buy_hold",
                ),
            ),
            AlwaysCashBenchmark(
                output_directory=root(
                    "benchmarks",
                    "always_cash",
                ),
            ),
            RandomAgentBenchmark(
                output_directory=root(
                    "benchmarks",
                    "random_agent",
                ),
            ),
        ]

        for benchmark in benchmarks:

            logger.info(
                "Running {}...",
                benchmark.name,
            )

            benchmark.evaluate()

            metrics = self._load_metrics(
                benchmark.output_directory
                / "metrics.json"
            )

            metrics = {
                "strategy": benchmark.name,
                **metrics,
            }

            self._comparison.append(metrics)

        logger.success(
            "Benchmark suite completed successfully."
        )

    def _comparison_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Build the benchmark comparison table.
        """

        dataframe = pd.DataFrame(
            self._comparison,
        )

        if "strategy" in dataframe.columns:

            columns = (
                ["strategy"]
                + [
                    column
                    for column in dataframe.columns
                    if column != "strategy"
                ]
            )

            dataframe = dataframe[
                columns
            ]

        return dataframe

    def _save_results(self) -> None:
        """
        Save benchmark comparison results.
        """

        comparison_df = (
            self._comparison_dataframe()
        )

        comparison_df.to_csv(
            self._output_dir
            / "comparison.csv",
            index=False,
        )

        comparison_df.to_json(
            self._output_dir
            / "comparison.json",
            orient="records",
            indent=4,
        )

        logger.success(
            "Comparison files saved."
        )
        
    def _save_markdown(self) -> None:
        """
        Save a Markdown leaderboard.
        """

        dataframe = self._comparison_dataframe()

        markdown = "# WARLOCK Benchmark Results\n\n"

        markdown += dataframe.to_markdown(
            index=False,
        )

        with (
            self._output_dir / "comparison.md"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(markdown)

    def run(self) -> None:
        """
        Execute the benchmark suite.
        """

        logger.info("=" * 80)
        logger.info("WARLOCK Benchmark Suite")
        logger.info("=" * 80)

        self._run_benchmarks()

        self._save_results()

        self._save_markdown()

        logger.success(
            "Benchmark suite completed successfully."
        )
        
        
def main() -> int:

    suite = BenchmarkSuite(
        output_directory=root(
            "benchmarks",
        ),
    )

    suite.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())