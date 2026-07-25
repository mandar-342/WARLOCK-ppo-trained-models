from __future__ import annotations
import pandas as pd
import numpy as np

def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den == 0, np.nan, num / den)

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Causal rolling z-score: (x - rolling_mean) / rolling_std, computed over
    the trailing `window` periods only (no look-ahead).
    """
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return safe_divide(series - mean, std)

def consecutive_streak(condition: pd.Series) -> pd.Series:
    """
    Length of the current run of consecutive True values in `condition`,
    reset to 0 wherever `condition` is False. Used for things like
    "N consecutive up bars".
    """
    condition = condition.fillna(False)
    groups = (condition != condition.shift()).cumsum()
    streak = condition.groupby(groups).cumcount() + 1
    return streak.where(condition, 0)