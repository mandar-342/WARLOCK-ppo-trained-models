from __future__ import annotations
import pandas as pd
import numpy as np
from .base import safe_divide
from src.utils import config

_CFG = config["features"]["price"]

def price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    short_p = _CFG["log_return_short_period"]
    long_p = _CFG["log_return_long_period"]
    ema_w = _CFG["ema_window"]
    hl_w = _CFG["dist_high_low_window"]

    out["log_return"] = np.log(out["close"] / out["close"].shift(short_p))
    out[f"log_return_{long_p}h"] = np.log(out["close"] / out["close"].shift(long_p))

    ema = out["close"].ewm(span=ema_w, adjust=False).mean()
    out["EMA_ratio"] = safe_divide(out["close"], ema)

    roll_high = out["high"].rolling(hl_w, min_periods=hl_w).max()
    roll_low = out["low"].rolling(hl_w, min_periods=hl_w).min()
    out["distance_from_high_24h"] = safe_divide(out["close"], roll_high) - 1
    out["distance_from_low_24h"] = safe_divide(out["close"], roll_low) - 1

    return out
