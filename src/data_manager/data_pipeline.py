from pathlib import Path
from loguru import logger
from src.utils import config, root
from src.data_manager.downloader import download
from src.data_manager.data_cleaning import clean_ohlcv

def data_pipeline():
    log_path = Path(root(config["paths"]["logs_dir"]))
    log_path.mkdir(exist_ok=True)
    logger.add(
        "logs/cleaning_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    symbols       = config["data"]["symbols"]
    timeframe     = config["data"]["timeframe"]
    raw_dir       = root(config["paths"]["raw_dir"])
    processed_dir = root(config["paths"]["processed_dir"])

    logger.info(f"Symbols: {symbols} | Timeframe: {timeframe}")
    logger.info(f"Raw → {raw_dir} | Processed → {processed_dir}")
    download()
    for symbol in symbols:
        try:
            clean_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
            )
        except FileNotFoundError as e:
            logger.error(f"Skipping {symbol} — {e}")
        except Exception as e:
            logger.exception(f"Failed on {symbol} — {e}")
            raise

if __name__ == "__main__":
    data_pipeline()
