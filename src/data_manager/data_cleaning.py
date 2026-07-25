import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from loguru import logger
from src.utils import config

timeframe_to_freq = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}

def load_raw_ohlcv(raw_dir: str, symbol: str, timeframe: str) -> pd.DataFrame:
    
    symbol_safe = symbol.split("/")[0]
    filepath = Path(raw_dir) / f"{symbol_safe}_raw.parquet"
    
    if not filepath.exists():
        
        raise FileNotFoundError(f"No data found : {filepath}")

    df = pd.read_parquet(filepath)
    df = df.reset_index()
    logger.info(f"Loaded {len(df):,} candles from {filepath}")
    return df

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    required={"timestamp", "open", "high", "low", "close", "volume"}
    df.columns=[c.lower().strip() for c in df.columns]
    
    missing=required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df[list(required)].copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]): 
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df=df.sort_values("timestamp").reset_index(drop=True)
    return df

def remove_duplicate_timestamps(df:pd.DataFrame)->pd.DataFrame:
    n_before=len(df)
    duplicated_mask=df.duplicated(subset=["timestamp"],keep=False)
    n_duplicated = int(duplicated_mask.sum())
    
    if n_duplicated==0:
        logger.info("No duplicates found")
        return df
    
    logger.warning(f"Duplicates: {n_duplicated} rows share a timestamp — merging")
    
    df=(
        df.groupby("timestamp", as_index=False).agg(
            open=("open", "first"),
            high=("high","max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume","sum"),
        ).sort_values(by="timestamp").reset_index(drop=True)    )
    
    
    logger.info(f"Duplicates: merged down to {len(df):,} rows (removed {n_before - len(df)})")
    return df

def _compute_gap_lengths(missing_mask: pd.Series) -> pd.Series:
    gaps: dict = {}
    gap_start = None
    gap_len = 0
    
    for ts, is_missing in missing_mask.items():
        if is_missing and gap_start is None:
            gap_start = ts
            gap_len = 1
        elif is_missing:
            gap_len += 1
        elif gap_start is not None:
            gaps[gap_start] = gap_len
            gap_start = None
            gap_len = 0
 
    if gap_start is not None:
        gaps[gap_start] = gap_len
 
    return pd.Series(gaps)

def handle_missing_candles(
    df: pd.DataFrame,
    timeframe: str,
    max_fill: int = config['data']['max_fill_candles'],
) -> pd.DataFrame:
    freq = timeframe_to_freq.get(timeframe)
    if freq is None:
        raise ValueError(f"Timeframe not supported '{timeframe}'. Supported: {list(timeframe_to_freq.keys())}")
 
    complete_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=freq,
    )
 
    df_indexed = df.set_index("timestamp").reindex(complete_index)
    df_indexed.index.name = "timestamp"
 
    missing_mask = df_indexed["close"].isna()
    n_missing = int(missing_mask.sum())
 
    if n_missing == 0:
        logger.info("Missing candles: none found")
        return df_indexed.reset_index()
 
    logger.warning(f"Missing candles: {n_missing} out of {len(complete_index):,} total")
 
    gap_lengths = _compute_gap_lengths(missing_mask)
    short_gaps = gap_lengths[gap_lengths <= max_fill]
    long_gaps  = gap_lengths[gap_lengths > max_fill]
 
    logger.info(f"Missing candles: {len(short_gaps)} short gaps (≤{max_fill}) → forward-fill")
    logger.info(f"Missing candles: {len(long_gaps)} long gaps (>{max_fill}) → drop")
 
    df_indexed = df_indexed.ffill(limit=max_fill)
 
    still_missing = df_indexed["close"].isna()
    if still_missing.any():
        n_drop = int(still_missing.sum())
        logger.warning(f"Missing candles: dropping {n_drop} unfillable rows")
        df_indexed = df_indexed[~still_missing]
 
    df_indexed = df_indexed.reset_index()
    logger.info(f"Missing candles: {len(df_indexed):,} candles after handling")
    return df_indexed

def detect_wick_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    ad_cfg = config['data']['anomaly_detection']
    window     = int(ad_cfg.get("rolling_window", 30))
    multiplier = float(ad_cfg.get("wick_multiplier", 5.0))
    percent    = float(ad_cfg.get("wick_percent", 0.75))

    price_range = df["high"] - df["low"]
    rolling_median = price_range.rolling(window=window, min_periods=10).median()

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    cond_upper = (upper_wick > multiplier * rolling_median) & (upper_wick > percent * price_range)
    cond_lower = (lower_wick > multiplier * rolling_median) & (lower_wick > percent * price_range)

    df["wick_anomaly_flag"] = cond_upper | cond_lower

    flagged = df[df["wick_anomaly_flag"]]
    total_flagged = int(flagged.shape[0])
    ts_list = flagged["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    ts_str = ", ".join(ts_list) if ts_list else "(none)"
    logger.info(f"[wick] {total_flagged:,} anomalous wicks (no symbol column) – timestamps: {ts_str}")

    logger.success(f"Wick‑anomaly detection completed – {total_flagged:,} wicks flagged")
    return df
 
def save_cleaned_data(
    df: pd.DataFrame,
    processed_dir: str,
    symbol: str,
    timeframe: str,
) -> Path:
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    symbol_safe = symbol.replace("/", "_")
    filepath = Path(processed_dir) / f"{symbol_safe}_{timeframe}_cleaned.parquet"
 
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, filepath, compression="snappy")
    logger.info(f"Saved → {filepath} ({len(df):,} rows)")
    return filepath

def generate_cleaning_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    symbol: str,
) -> dict:
    report = {
        "symbol":           symbol,
        "raw_candles":      len(df_raw),
        "clean_candles":    len(df_clean),
        "dropped_candles":  len(df_raw) - len(df_clean),
        "wick_anomalies":   int(df_clean["wick_anomaly_flag"].sum()) if "wick_anomaly_flag" in df_clean.columns else 0,
        "date_start":       str(df_clean["timestamp"].min()),
        "date_end":         str(df_clean["timestamp"].max()),
        "missing_pct":      round(100 * (1 - len(df_clean) / len(df_raw)), 4) if len(df_raw) else 0,
    }
 
    logger.info("Cleaning Report ")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")
 
    return report
  
def clean_ohlcv(
    symbol: str,
    timeframe: str,
    raw_dir: str,
    processed_dir: str,
    max_fill: int = config['data']['max_fill_candles'],
) -> pd.DataFrame:
    logger.info(f" Cleaning pipeline start: {symbol} [{timeframe}] ")
 
    df = load_raw_ohlcv(raw_dir, symbol, timeframe)
    df_raw_snapshot = df.copy()
 
    df = standardise_columns(df)
    df = remove_duplicate_timestamps(df)
    df = handle_missing_candles(df, timeframe, max_fill=max_fill)
    df = detect_wick_anomalies(df)
 
    save_cleaned_data(df, processed_dir, symbol, timeframe)
    generate_cleaning_report(df_raw_snapshot, df, symbol)
 
    logger.success(f"Cleaning pipeline done: {symbol}")
    return df
