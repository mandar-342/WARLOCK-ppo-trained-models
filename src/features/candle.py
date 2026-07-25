from __future__ import annotations

import numpy as np
import pandas as pd


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-bar candle geometry features.
    """

    out = df.copy()

    bar_range = out["high"] - out["low"]
    valid_range = bar_range.replace(0, np.nan)

    out["close_location"] = (
        (out["close"] - out["low"]) / valid_range
    ).fillna(0.5)

    out["body_pct"] = (
        ((out["close"] - out["open"]).abs() / valid_range) * 100
    ).fillna(0.0)

    return out