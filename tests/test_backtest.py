import numpy as np
import pandas as pd
import pytest
from pairs_engine.backtest import walk_forward, _build_trade_ledger


def make_cointegrated_log_prices(n=1000, beta=1.0, seed=42):
    """Synthetic cointegrated pair in log-price space."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, n).cumsum()
    spread = rng.normal(0, 0.005, n)
    y = beta * x + spread
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"Y": y, "X": x}, index=dates)


def test_walk_forward_returns_expected_keys():
    prices = make_cointegrated_log_prices(n=800)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    for key in ["equity_curve", "trade_ledger", "windows_total", "windows_skipped"]:
        assert key in result


def test_walk_forward_formation_only_uses_past_data():
    """
    The Johansen test at each window uses only formation_data = log_prices.iloc[:formation_end].
    Verify by checking that the formation slice index max is < trading window start.
    We patch johansen_test to assert this invariant on every call.
    """
    import pairs_engine.backtest as bt_mod
    import pairs_engine.cointegration as coint_mod

    original_johansen = coint_mod.johansen_test
    formation_end_seen = []

    def checking_johansen(log_prices_slice):
        formation_end_seen.append(log_prices_slice.index[-1])
        return original_johansen(log_prices_slice)

    original_ref = bt_mod.johansen_test
    bt_mod.johansen_test = checking_johansen

    prices = make_cointegrated_log_prices(n=800)
    try:
        result = walk_forward(prices, formation_min=252, trading_len=63,
                              delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    finally:
        bt_mod.johansen_test = original_ref

    # Each johansen call's last index must be < corresponding trading_start
    # Window 1: formation=[0:252], trading=[252:315] → last formation date < trading_start date
    assert len(formation_end_seen) > 0
    for i, last_formation_date in enumerate(formation_end_seen):
        trading_start_idx = 252 + i * 63
        if trading_start_idx < len(prices):
            trading_start_date = prices.index[trading_start_idx]
            assert last_formation_date < trading_start_date, (
                f"Formation data leaked into trading window at window {i}: "
                f"last formation date {last_formation_date} >= trading start {trading_start_date}"
            )


def test_walk_forward_skips_window_on_no_coint():
    """Independent random walks — Johansen should fail → at least one window skipped."""
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(0, 0.01, n).cumsum()
    y = rng.normal(0, 0.01, n).cumsum()
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"Y": y, "X": x}, index=dates)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    assert result["windows_skipped"] > 0


def test_equity_curve_starts_at_one():
    prices = make_cointegrated_log_prices(n=800)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    equity = result["equity_curve"]["equity"]
    if len(equity) > 0:
        assert abs(equity.iloc[0] - 1.0) < 1e-10


def test_position_lagged_correctly():
    """Returns at bar t use position from bar t-1. Flat bars should have ~0 gross_ret."""
    prices = make_cointegrated_log_prices(n=800)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    ec = result["equity_curve"]
    flat_bars = ec[ec["position"].shift(1).fillna(0) == 0]
    assert (flat_bars["gross_ret"].abs() < 1e-10).all()


def test_trade_ledger_columns():
    prices = make_cointegrated_log_prices(n=800)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    ledger = result["trade_ledger"]
    if len(ledger) > 0:
        for col in ["entry_date", "exit_date", "side", "gross_pnl", "costs", "net_pnl", "holding_days"]:
            assert col in ledger.columns
