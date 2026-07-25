from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.utils import root, config
from loguru import logger

#Order of the imports matters here
from .price      import price_features
from .candle     import candle_features
from .momentum   import momentum_features
from .volatility import volatility_features
from .volume     import volume_features
from .plot_features import plot_features

def load_cleaned(symbol: str, timeframe: str,
                processed_dir: str = config['paths']['processed_dir']) -> pd.DataFrame:
    filename = f"{symbol.replace('/', '_')}_{timeframe}_cleaned.parquet"
    path = Path(root(processed_dir)) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned data not found: {path}")
    return pd.read_parquet(path)

def split_temporal(df: pd.DataFrame,
                    train_years: int = 4,
                    train_months: int = 5,
                    test_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)

    start = df["timestamp"].min()
    end   = df["timestamp"].max()

    train_cutoff = start + pd.DateOffset(years=train_years, months=train_months)
    test_start   = end - pd.DateOffset(months=test_months)

    if test_start <= train_cutoff:
        raise ValueError(
            f"Temporal split conflict: test_start ({test_start}) <= train_cutoff ({train_cutoff})"
        )

    train_df = df[df["timestamp"] <= train_cutoff].copy()
    test_df  = df[(df["timestamp"] > train_cutoff) & (df["timestamp"] >= test_start)].copy()

    return train_df, test_df

def apply_train_stats(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    def recompute_ratio(raw_col: str, ratio_col: str, window: int):
        train_raw = train[raw_col]
        test_raw  = test[raw_col]

        train_mean = train_raw.rolling(window, min_periods=window).mean()
        train_std  = train_raw.rolling(window, min_periods=window).std()

        train[ratio_col] = (train_raw - train_mean) / train_std

        final_mean = train_mean.iloc[-1]
        final_std  = train_std.iloc[-1]
        test[ratio_col] = (test_raw - final_mean) / final_std

    recompute_ratio(
        "volume", "volume_zscore",
        window=config["features"]["volume"]["zscore_window"],
    )

    return train, test



def generate_and_plot_features(symbol: str = "BTC/USDT",
                    timeframe: str = "1h",
                    processed_dir: str = config['paths']['processed_dir'],
                    out_dir: str = config['paths']['feature_engineered_dir']) -> None:
    """
    Full pipeline:
    1️⃣ Load cleaned parquet.
    2️⃣ Run each feature‑group module in deterministic order.
    3️⃣ Perform the strict 4.5‑year / 6‑month temporal split.
    4️⃣ Re‑fit any rolling‑ratio columns on the train set only.
    5️⃣ Write ``train.parquet`` and ``test.parquet`` (snappy compressed).
    """
    log_path = Path(root(config["paths"]["logs_dir"]))
    log_path.mkdir(exist_ok=True)
    logger.add(
        "logs/feature_engineering_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    df = load_cleaned(symbol, timeframe, processed_dir)

    df = price_features(df)       # log_return, log_return_6h, EMA_ratio, distance_from_high/low_24h
    df = candle_features(df)      # close_location, body_pct, wick_anomaly_flag
    df = momentum_features(df)    # RSI, ADX, up_streak
    df = volatility_features(df)  # ATR_pct, volatility_ratio, rolling_sharpe (needs log_return)
    df = volume_features(df)      # volume_zscore, OBV
    plot_features(df)

    train_df, test_df = split_temporal(df, train_years=4, train_months=5, test_months=6)
    
    

    train_df, test_df = apply_train_stats(train_df, test_df)
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True) 

    out_path = Path(root(out_dir))
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train.parquet"
    test_path  = out_path / "test.parquet"

    train_df.to_parquet(train_path, compression="snappy")
    test_df.to_parquet(test_path,  compression="snappy")

    logger.success(
        f"\nFeature pipeline completed:\n"
        f"   • Train file: {train_path}   ({len(train_df):,} rows)\n"
        f"   • Test  file: {test_path}    ({len(test_df):,} rows)\n"
        f"   • No data from the test period was used for any computation or tuning.\n"
    )