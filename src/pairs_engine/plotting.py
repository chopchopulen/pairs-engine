from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity(equity: pd.Series, outpath: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, color="#2563eb", linewidth=1.2, label="Equity")
    ax1.set_ylabel("Equity (normalised to 1)")
    ax1.set_title("Out-of-Sample Equity Curve — GLD/IAU Pairs Strategy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#dc2626", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_spread_z(df: pd.DataFrame, outpath: str) -> None:
    """Two-panel: spread on top, z-score with threshold lines below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(df.index, df["spread"], color="#059669", linewidth=0.8)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Spread (Kalman innovation)")
    ax1.set_title("Spread and Z-Score — GLD/IAU")
    ax1.grid(True, alpha=0.3)

    z = df["z"]
    ax2.plot(df.index, z, color="#7c3aed", linewidth=0.8, label="Z-score")
    ax2.axhline(2.0, color="#dc2626", linewidth=1.0, linestyle="--", label="Entry ±2.0")
    ax2.axhline(-2.0, color="#dc2626", linewidth=1.0, linestyle="--")
    ax2.axhline(0.5, color="#d97706", linewidth=0.8, linestyle=":", label="Exit ±0.5")
    ax2.axhline(-0.5, color="#d97706", linewidth=0.8, linestyle=":")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")

    pos = df["position"]
    for i in range(1, len(df)):
        if pos.iloc[i] == -1:
            ax2.axvspan(df.index[i-1], df.index[i], alpha=0.15, color="#dc2626")
        elif pos.iloc[i] == 1:
            ax2.axvspan(df.index[i-1], df.index[i], alpha=0.15, color="#059669")

    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_beta(df: pd.DataFrame, outpath: str) -> None:
    """Kalman beta over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["beta"], color="#2563eb", linewidth=0.9, label="Kalman beta")
    ax.set_ylabel("Hedge Ratio (beta)")
    ax.set_title("Dynamic Hedge Ratio — Kalman Filter")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_johansen_stability(log_prices: pd.DataFrame, formation_min: int, trading_len: int, outpath: str) -> None:
    """Expanding-window Johansen trace statistic vs 95% critical value."""
    from pairs_engine.cointegration import johansen_test

    dates = []
    trace_stats = []
    crit_95 = None

    for end in range(formation_min, len(log_prices), trading_len):
        slice_ = log_prices.iloc[:end]
        try:
            result = johansen_test(slice_)
        except ValueError:
            continue
        dates.append(log_prices.index[end - 1])
        trace_stats.append(result["trace_stat"])
        if crit_95 is None:
            crit_95 = result["crit_95"]

    if not dates:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, trace_stats, color="#2563eb", linewidth=1.2, label="Johansen trace stat")
    ax.axhline(crit_95, color="#dc2626", linewidth=1.0, linestyle="--",
               label=f"95% critical value ({crit_95:.2f})")
    ax.fill_between(dates, trace_stats, crit_95,
                    where=[t > crit_95 for t in trace_stats],
                    alpha=0.15, color="#059669", label="Cointegrated region")
    ax.set_ylabel("Trace Statistic")
    ax.set_xlabel("Formation Window End Date")
    ax.set_title("Johansen Cointegration Stability (Expanding Window)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
