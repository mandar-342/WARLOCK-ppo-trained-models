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

def symbol_tag(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC'. Used to prefix per-asset feature columns when
    multiple assets are merged into a single combined parquet."""
    return symbol.split("/")[0]

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

def apply_train_stats(train: pd.DataFrame, test: pd.DataFrame,
                       raw_col: str = "volume", ratio_col: str = "volume_zscore") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-fit the volume z-score using train-set rolling stats only, then
    apply the frozen (final train-window) mean/std to the test set. Column
    names are parameterized so this can be called once per asset with
    asset-prefixed column names (e.g. 'BTC_volume' / 'BTC_volume_zscore')
    without pooling statistics across assets."""

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
        raw_col, ratio_col,
        window=config["features"]["volume"]["zscore_window"],
    )

    return train, test


def build_asset_features(symbol: str, timeframe: str = "1h",
                          processed_dir: str = config['paths']['processed_dir']) -> pd.DataFrame:
    """Run the deterministic per-asset feature-engineering chain (steps
    1-5 of the old single-asset pipeline) and return the raw, un-split,
    un-prefixed feature frame for one symbol, indexed by 'timestamp'.
    Used as the building block for both the single-asset and multi-asset
    pipelines so the feature math itself never differs between them."""
    df = load_cleaned(symbol, timeframe, processed_dir)

    df = price_features(df)
    df = candle_features(df)
    df = momentum_features(df)
    df = volatility_features(df)
    df = volume_features(df)
    return df



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
    df = build_asset_features(symbol, timeframe, processed_dir)
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

def generate_multiasset_features(symbols: list[str] | None = None,
                                  timeframe: str = "1h",
                                  processed_dir: str = config['paths']['processed_dir'],
                                  out_dir: str = config['paths']['feature_engineered_dir']) -> None:
    """
    Multi-asset feature pipeline:
    1. Run the per-asset feature chain for every symbol independently.
    2. Prefix each asset's feature columns with its ticker (BTC_/ETH_/...)
       and inner-join on 'timestamp' so all assets share one master step
       index. Rows where any asset is missing (differing listing dates,
       exchange downtime, etc.) are dropped at this stage rather than
       silently forward-filled, so the env never sees a fabricated candle.
    3. Perform the temporal split ONCE on the shared timestamp axis, so
       every asset's train/test boundary lines up with every other's.
    4. Re-fit the volume z-score per asset, independently, on that asset's
       own train-window stats (never pooled across assets).
    5. Write a single combined train.parquet / test.parquet with
       asset-prefixed columns plus one shared 'timestamp' column.
    """
    symbols = symbols or config["data"]["symbols"]

    log_path = Path(root(config["paths"]["logs_dir"]))
    log_path.mkdir(exist_ok=True)
    logger.add(
        "logs/feature_engineering_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    per_asset_raw: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = build_asset_features(symbol, timeframe, processed_dir)
        plot_features(df)
        per_asset_raw[symbol] = df

    # --- align on shared timestamp axis ---
    merged = None
    for symbol, df in per_asset_raw.items():
        tag = symbol_tag(symbol)
        feature_cols = [c for c in df.columns if c != "timestamp"]
        renamed = df[["timestamp"] + feature_cols].rename(
            columns={c: f"{tag}_{c}" for c in feature_cols}
        )
        merged = renamed if merged is None else merged.merge(renamed, on="timestamp", how="inner")

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    train_df, test_df = split_temporal(merged, train_years=4, train_months=5, test_months=6)

    # --- per-asset train-fit normalization, applied independently ---
    for symbol in symbols:
        tag = symbol_tag(symbol)
        train_df, test_df = apply_train_stats(
            train_df, test_df,
            raw_col=f"{tag}_volume", ratio_col=f"{tag}_volume_zscore",
        )

    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    out_path = Path(root(out_dir))
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train.parquet"
    test_path = out_path / "test.parquet"

    train_df.to_parquet(train_path, compression="snappy")
    test_df.to_parquet(test_path, compression="snappy")

    logger.success(
        f"\nMulti-asset feature pipeline completed ({', '.join(symbols)}):\n"
        f"   • Train file: {train_path}   ({len(train_df):,} rows)\n"
        f"   • Test  file: {test_path}    ({len(test_df):,} rows)\n"
        f"   • Assets aligned on shared timestamp index (inner join, no fill).\n"
    )
