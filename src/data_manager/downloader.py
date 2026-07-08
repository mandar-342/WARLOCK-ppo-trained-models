import ccxt
import time
import pandas as pd
import sys
from loguru import logger
from pathlib import Path
from src.utils import config, root

def download():
    symbols = config['data']['symbols']
    data_dir = root(config['paths']['raw_dir'])
    exchange = getattr(ccxt, config['data']['exchange'])()
    
    for symbol in symbols:
        logger.info(f'Downloading {symbol}...')
        all_candles = []
        cursor = exchange.parse8601(f"{config['data']['start_date']}T00:00:00Z")
        end = exchange.parse8601(f"{config['data']['end_date']}T23:59:59Z")

        while cursor < end:
            try:
                batch = exchange.fetch_ohlcv(symbol, config['data']['timeframe'], cursor)
                if not batch:
                    break
                all_candles.extend(batch)
                cursor = batch[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)
            except ccxt.NetworkError as e:
                logger.error(f"Network error: {e}. Retrying...")
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
        
        all_candles = [candle for candle in all_candles if candle[0] <= end]
        data = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        symbol = symbol.split('/')[0]
        file_path = data_dir / (symbol + "_raw.parquet")
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        data.to_parquet(file_path)
        logger.info(f'Saved {symbol} data to {file_path}')

if __name__ == "__main__":
    download()
