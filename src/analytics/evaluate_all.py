from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.analytics.checkpoint_evaluator import (
    CheckpointEvaluator,
)


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint of an experiment."
    )

    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to experiment directory.",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("WARLOCK Checkpoint Evaluation")
    logger.info("=" * 80)

    summary = CheckpointEvaluator(
        experiment_directory=args.experiment,
    ).run()

    print()
    print(summary)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())