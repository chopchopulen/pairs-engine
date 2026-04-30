from __future__ import annotations
import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, ann: int = 252) -> float:
    """Annualised Sharpe ratio. Returns 0.0 if std is zero."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ann))


def max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough maximum drawdown as a positive fraction (e.g. 0.25 = 25%)."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(-drawdown.min())


def trade_stats(ledger: pd.DataFrame) -> dict:
    """Compute per-trade statistics from a ledger with columns: net_pnl, holding_days."""
    n = len(ledger)
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "avg_holding_days": 0.0, "avg_pnl_per_trade": 0.0}
    wins = (ledger["net_pnl"] > 0).sum()
    return {
        "n_trades": n,
        "win_rate": float(wins / n),
        "avg_holding_days": float(ledger["holding_days"].mean()),
        "avg_pnl_per_trade": float(ledger["net_pnl"].mean()),
    }


def summary(
    returns: pd.Series,
    equity: pd.Series,
    ledger: pd.DataFrame,
    windows_total: int,
    windows_skipped: int,
) -> dict:
    """Aggregate all performance metrics for reporting."""
    stats = trade_stats(ledger)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    return {
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(equity),
        "total_return": total_return,
        "pct_windows_skipped": windows_skipped / windows_total if windows_total > 0 else 0.0,
        **stats,
    }
