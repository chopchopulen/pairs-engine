from __future__ import annotations
import numpy as np
import pandas as pd
from pairs_engine.cointegration import johansen_test
from pairs_engine.kalman import KalmanHedge
from pairs_engine.signals import compute_zscore, generate_positions
from pairs_engine.costs import apply_costs


def walk_forward(
    log_prices: pd.DataFrame,
    formation_min: int = 504,
    trading_len: int = 126,
    delta: float = 1e-4,
    R: float = 1e-3,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 5,
) -> dict:
    """
    Expanding-window walk-forward backtest.

    log_prices: DataFrame with exactly 2 columns [Y, X] in log-price space.
    formation_min: minimum bars in the first formation window.
    trading_len: fixed trading-window length in bars.

    Returns dict: equity_curve, trade_ledger, windows_total, windows_skipped.
    """
    y_col, x_col = log_prices.columns[0], log_prices.columns[1]

    all_bars: list[pd.DataFrame] = []
    trade_ledger_rows: list[dict] = []
    windows_total = 0
    windows_skipped = 0
    formation_end = formation_min

    while formation_end + trading_len <= len(log_prices):
        trading_start = formation_end
        trading_end = formation_end + trading_len
        windows_total += 1

        # Formation data only — strict slice, no trading data
        formation_data = log_prices.iloc[:formation_end]

        # Cointegration gate
        try:
            coint = johansen_test(formation_data)
        except ValueError:
            windows_skipped += 1
            formation_end += trading_len
            continue

        if not coint["rejects_no_coint"]:
            windows_skipped += 1
            formation_end += trading_len
            continue

        # OLS prior for Kalman warm-up
        X_form = formation_data[x_col].values
        Y_form = formation_data[y_col].values
        A = np.column_stack([X_form, np.ones_like(X_form)])
        ols_coef, _, _, _ = np.linalg.lstsq(A, Y_form, rcond=None)
        prior_mean = ols_coef
        prior_cov = np.eye(2) * 100.0

        kf = KalmanHedge(delta=delta, R=R)
        # Warm up through formation — kf internal state reaches formation_end
        kf.run(formation_data[y_col], formation_data[x_col],
               prior_mean=prior_mean, prior_cov=prior_cov)

        # Trading window
        trading_data = log_prices.iloc[trading_start:trading_end]
        y_trade = trading_data[y_col]
        x_trade = trading_data[x_col]

        kf_results = kf.run(y_trade, x_trade)  # continues from formation-end state

        z = compute_zscore(kf_results["spread"], kf_results["spread_var"])
        raw_pos = generate_positions(z, entry_z=entry_z, exit_z=exit_z)

        # Lag position and beta by 1 bar — CRITICAL for no look-ahead
        pos_lagged = raw_pos.shift(1).fillna(0)
        beta_lagged = kf_results["beta"].shift(1).fillna(kf_results["beta"].iloc[0])

        r_y = y_trade.diff().fillna(0)
        r_x = x_trade.diff().fillna(0)

        gross_ret = pos_lagged * (r_y - beta_lagged * r_x)
        costs = apply_costs(raw_pos, kf_results["beta"], bps=cost_bps)
        net_ret = gross_ret - costs

        equity_start = all_bars[-1]["equity"].iloc[-1] if all_bars else 1.0
        equity = equity_start * (1 + net_ret).cumprod()

        window_df = pd.DataFrame({
            "beta": kf_results["beta"],
            "alpha": kf_results["alpha"],
            "spread": kf_results["spread"],
            "z": z,
            "position": raw_pos,
            "gross_ret": gross_ret,
            "cost": costs,
            "net_ret": net_ret,
            "equity": equity,
        })
        all_bars.append(window_df)
        trade_ledger_rows.extend(_build_trade_ledger(window_df))
        formation_end += trading_len

    equity_curve = pd.concat(all_bars) if all_bars else pd.DataFrame()
    trade_ledger = (
        pd.DataFrame(trade_ledger_rows) if trade_ledger_rows
        else pd.DataFrame(columns=["entry_date", "exit_date", "side",
                                    "gross_pnl", "costs", "net_pnl", "holding_days"])
    )

    return {
        "equity_curve": equity_curve,
        "trade_ledger": trade_ledger,
        "windows_total": windows_total,
        "windows_skipped": windows_skipped,
    }


def _build_trade_ledger(window_df: pd.DataFrame) -> list[dict]:
    """Extract individual closed trades from a window's equity curve."""
    trades = []
    pos = window_df["position"]
    in_trade = False
    entry_date = None
    side = 0
    entry_equity = 1.0

    for date, row in window_df.iterrows():
        current_pos = row["position"]
        if not in_trade and current_pos != 0:
            in_trade = True
            entry_date = date
            side = int(current_pos)
            entry_equity = row["equity"]
        elif in_trade and current_pos == 0:
            in_trade = False
            gross_pnl = float(row["equity"] - entry_equity)
            cost = float(window_df.loc[entry_date:date, "cost"].sum())
            holding = (date - entry_date).days
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "side": side,
                "gross_pnl": gross_pnl,
                "costs": cost,
                "net_pnl": gross_pnl - cost,
                "holding_days": holding,
            })

    return trades
