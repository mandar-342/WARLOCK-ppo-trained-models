import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from loguru import logger
from src.utils import root, config

def last_n_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise KeyError("Dataframe must contain a 'timestamp' column.")
    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    return df[df["timestamp"] >= cutoff].copy()

def _plot_line(df, col, graph_path, days=None, hlines=None, title=None, ylabel=None):
    plot_df = last_n_days(df, days) if days else df
    plt.figure(figsize=(12, 4))
    plt.plot(plot_df["timestamp"], plot_df[col], linewidth=0.9, color="steelblue", label=col)
    for y in (hlines or []):
        plt.axhline(y, color="black", lw=0.7, ls="--")
    plt.title(title or f"{col} over time")
    plt.xlabel("Timestamp")
    plt.ylabel(ylabel or col)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(graph_path / f"{col}.png", dpi=150)
    plt.close()

def _plot_hist(df, col, graph_path, bins=100, title=None):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=bins, kde=True, color="steelblue", edgecolor="white")
    plt.title(title or f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(graph_path / f"{col}_hist.png", dpi=150)
    plt.close()

def _plot_ema_ratio(df, graph_path):
    pct_dist = (df["EMA_ratio"] - 1) * 100
    tmp = df.copy()
    tmp["EMA_ratio_pct"] = pct_dist
    plot_df = last_n_days(tmp, days=90)
    plt.figure(figsize=(12, 4))
    plt.plot(plot_df["timestamp"], plot_df["EMA_ratio_pct"],
              label="EMA_ratio (% diff)", color="steelblue", linewidth=0.9)
    plt.axhline(0, color="black", lw=0.7, ls="--")
    plt.title("EMA_ratio expressed as % distance from EMA (90-day window)")
    plt.xlabel("Timestamp")
    plt.ylabel("% diff")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(graph_path / "EMA_ratio_pct.png", dpi=150)
    plt.close()

def _plot_rsi(df, graph_path):
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["RSI"], label="RSI (14)", color="steelblue", linewidth=0.9)
    plt.axhline(70, color="red",   lw=0.7, ls="--", label="Over-bought (70)")
    plt.axhline(30, color="green", lw=0.7, ls="--", label="Over-sold (30)")
    plt.title("RSI – over-bought / over-sold zones")
    plt.xlabel("Timestamp")
    plt.ylabel("RSI")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(graph_path / "RSI.png", dpi=150)
    plt.close()

def _plot_adx(df, graph_path):
    roll_mean = df["ADX"].rolling(14, min_periods=1).mean()
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["ADX"], linewidth=0.8, label="ADX", color="steelblue")
    plt.plot(df["timestamp"], roll_mean, linewidth=1.2, label="ADX (14-period MA)",
              color="gray", alpha=0.8)
    plt.title("ADX over time")
    plt.xlabel("Timestamp")
    plt.ylabel("ADX")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(graph_path / "ADX.png", dpi=150)
    plt.close()

def _plot_obv(df, graph_path):
    obv_sub = last_n_days(df, days=14)
    fig, (ax_price, ax_obv) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_price.plot(obv_sub["timestamp"], obv_sub["close"], color="steelblue", linewidth=0.9)
    ax_price.set_ylabel("Close")
    ax_price.set_title("Price vs. OBV (14-day window)")

    ax_obv.plot(obv_sub["timestamp"], obv_sub["OBV"], color="darkorange", linewidth=0.9)
    ax_obv.set_ylabel("OBV")
    ax_obv.set_xlabel("Timestamp")
    plt.tight_layout()
    plt.savefig(graph_path / "OBV_dual.png", dpi=150)
    plt.close()

def _plot_up_streak(df, graph_path):
    sub = last_n_days(df, days=14)
    plt.figure(figsize=(12, 4))
    plt.bar(sub["timestamp"], sub["up_streak"], width=0.03, color="steelblue")
    plt.title("up_streak – consecutive up bars (14-day window)")
    plt.xlabel("Timestamp")
    plt.ylabel("Streak length")
    plt.tight_layout()
    plt.savefig(graph_path / "up_streak.png", dpi=150)
    plt.close()

def _plot_wick_anomaly(df, graph_path):
    sub = last_n_days(df, days=30)
    flagged = sub[sub["wick_anomaly_flag"] == 1]

    plt.figure(figsize=(12, 4))
    plt.plot(sub["timestamp"], sub["close"], color="steelblue", linewidth=0.9, label="Close")
    plt.scatter(flagged["timestamp"], flagged["close"], color="red", s=18,
                zorder=5, label="Wick anomaly")
    plt.title("wick_anomaly_flag – flagged bars over price (30-day window)")
    plt.xlabel("Timestamp")
    plt.ylabel("Close")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(graph_path / "wick_anomaly_flag.png", dpi=150)
    plt.close()

# columns plotted as distribution histograms rather than time series
_HIST_FEATURES = {"log_return", "log_return_6h", "close_location", "body_pct"}
# columns plotted as a simple 90-day line, optionally with a reference hline
_LINE_90D_FEATURES = {
    "distance_from_high_24h": [0.0],
    "distance_from_low_24h": [0.0],
    "ATR_pct": [],
    "volatility_ratio": [1.0],
    "rolling_sharpe": [0.0],
    "volume_zscore": [0.0]
}

# columns with a fully custom handler
_CUSTOM_HANDLERS = {
    "EMA_ratio": _plot_ema_ratio,
    "RSI": _plot_rsi,
    "ADX": _plot_adx,
    "OBV": _plot_obv,
    "up_streak": _plot_up_streak,
    "wick_anomaly_flag": _plot_wick_anomaly,
}

def plot_features(df: pd.DataFrame, out_dir: str = config['paths']['feature_graphs_dir']) -> None:
    graph_path = root(out_dir)
    graph_path.mkdir(parents=True, exist_ok=True)

    raw_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    feature_cols = [
        c for c in df.select_dtypes(include=["float", "int"]).columns
        if c not in raw_cols
    ]

    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.9)

    for col in feature_cols:
        if col in _CUSTOM_HANDLERS:
            _CUSTOM_HANDLERS[col](df, graph_path)
        elif col in _HIST_FEATURES:
            _plot_hist(df, col, graph_path)
        elif col in _LINE_90D_FEATURES:
            _plot_line(df, col, graph_path, days=90, hlines=_LINE_90D_FEATURES[col])
        else:
            # fallback for any future feature not explicitly categorized above
            _plot_line(df, col, graph_path)

    logger.success(f"\nAll feature graphs written to: {graph_path}\n")
