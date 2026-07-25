from __future__ import annotations

import sys


from loguru import logger

from src.agent.trainer import PPOTrainer
from src.utils import root


def configure_logger() -> None:
    """
    Configures Loguru for console and file logging.
    """

    log_directory = root ("logs") 
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        log_directory / "training.log",
        level="INFO",
        rotation="10 MB",
        retention=10,
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    
    
def main() -> int:
    """
    Application entry point.

    Returns
    -------
    int
        Exit status code.
    """

    configure_logger()

    trainer: PPOTrainer | None = None

    try:
        logger.info("=" * 80)
        logger.info("WARLOCK PPO Training")
        logger.info("=" * 80)

        trainer = PPOTrainer()

        trainer.train()

        logger.success("Training completed successfully.")

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130

    except Exception:
        logger.exception("Training failed due to an unexpected error.")
        return 1

    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())