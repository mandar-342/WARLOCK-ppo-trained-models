from __future__ import annotations
import pandas as pd
import numpy as np
from .base import safe_divide
from src.utils import config

_CFG = config["features"]["volatility"]

def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires `log_return` (from price_features) to already be present on
    `df` — run price_features() before volatility_features() in the
    pipeline.
    """
    out = df.copy()
    if "log_return" not in out.columns:
        raise KeyError(
            "volatility_features requires 'log_return'; run price_features first."
        )

    atr_w = _CFG["atr_window"]
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_w, adjust=False).mean()
    out["ATR_pct"] = safe_divide(atr, out["close"]) * 100

    short_w = _CFG["vol_ratio_short_window"]
    long_w = _CFG["vol_ratio_long_window"]
    realvol_short = out["log_return"].rolling(short_w, min_periods=short_w).std()
    realvol_long = out["log_return"].rolling(long_w, min_periods=long_w).std()
    out["volatility_ratio"] = safe_divide(realvol_short, realvol_long)

    sharpe_w = _CFG["sharpe_window"]
    periods_per_year = _CFG["periods_per_year"]
    roll_mean = out["log_return"].rolling(sharpe_w, min_periods=sharpe_w).mean()
    roll_std = out["log_return"].rolling(sharpe_w, min_periods=sharpe_w).std()
    out["rolling_sharpe"] = safe_divide(roll_mean, roll_std) * np.sqrt(periods_per_year)

    return out
