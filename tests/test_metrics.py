import numpy as np
import pandas as pd
import pytest
from pairs_engine.metrics import sharpe, max_drawdown, trade_stats, summary


def test_sharpe_all_positive():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 252))
    s = sharpe(rets)
    assert 1.0 < s < 2.5


def test_sharpe_zero_return():
    rets = pd.Series(np.zeros(100))
    s = sharpe(rets)
    assert s == 0.0


def test_max_drawdown_no_drawdown():
    equity = pd.Series([1.0, 1.1, 1.2, 1.3])
    mdd = max_drawdown(equity)
    assert mdd == 0.0


def test_max_drawdown_simple():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    mdd = max_drawdown(equity)
    # peak=1.2, trough=0.9: drawdown = (0.9-1.2)/1.2 = 0.25
    assert abs(mdd - 0.25) < 1e-10


def test_trade_stats_basic():
    ledger = pd.DataFrame({
        "net_pnl": [0.01, -0.005, 0.02],
        "holding_days": [5, 3, 10],
    })
    stats = trade_stats(ledger)
    assert stats["n_trades"] == 3
    assert abs(stats["win_rate"] - 2/3) < 1e-10
    assert abs(stats["avg_holding_days"] - 6.0) < 1e-10
    assert abs(stats["avg_pnl_per_trade"] - (0.01 - 0.005 + 0.02) / 3) < 1e-10


def test_summary_keys():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0005, 0.01, 200))
    equity = (1 + rets).cumprod()
    ledger = pd.DataFrame({"net_pnl": [0.01], "holding_days": [5]})
    result = summary(rets, equity, ledger, windows_total=10, windows_skipped=1)
    for key in ["sharpe", "max_drawdown", "n_trades", "win_rate", "avg_holding_days",
                "total_return", "pct_windows_skipped"]:
        assert key in result
