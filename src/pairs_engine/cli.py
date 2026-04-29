from __future__ import annotations
import json
from pathlib import Path
import argparse
import yaml
import pandas as pd
from pairs_engine.data import load_or_fetch
from pairs_engine.backtest import walk_forward
from pairs_engine.metrics import summary
from pairs_engine.plotting import plot_equity, plot_spread_z, plot_beta


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairs Trading Engine")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tickers = [cfg["pair"]["y"], cfg["pair"]["x"]]
    log_prices = load_or_fetch(
        tickers,
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        cache_dir=cfg["data"]["cache_dir"],
    )

    result = walk_forward(
        log_prices,
        formation_min=cfg["walk_forward"]["formation_min_bars"],
        trading_len=cfg["walk_forward"]["trading_bars"],
        delta=cfg["kalman"]["delta"],
        R=cfg["kalman"]["R"],
        entry_z=cfg["signals"]["entry_z"],
        exit_z=cfg["signals"]["exit_z"],
        cost_bps=cfg["costs"]["bps"],
    )

    ec = result["equity_curve"]
    ledger = result["trade_ledger"]
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    ec.to_csv(results_dir / "equity_curve.csv")
    if len(ledger) > 0:
        ledger.to_csv(results_dir / "trades.csv", index=False)

    net_rets = ec["net_ret"] if len(ec) > 0 else pd.Series(dtype=float)
    equity = ec["equity"] if len(ec) > 0 else pd.Series(dtype=float)
    perf = summary(
        net_rets, equity, ledger,
        windows_total=result["windows_total"],
        windows_skipped=result["windows_skipped"],
    )
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(perf, f, indent=2)

    if len(ec) > 0:
        plot_equity(ec["equity"], str(results_dir / "equity.png"))
        plot_spread_z(ec, str(results_dir / "spread_z.png"))
        plot_beta(ec, str(results_dir / "beta.png"))

    pair_str = f"{tickers[0]}/{tickers[1]}"
    date_range = f"{ec.index[0].date()} → {ec.index[-1].date()}" if len(ec) > 0 else "N/A"
    print(f"\n{'='*50}")
    print(f"  {pair_str} Pairs Backtest (out-of-sample)")
    print(f"{'='*50}")
    print(f"  Period:              {date_range}")
    print(f"  WF windows:          {result['windows_total']} "
          f"({result['windows_skipped']} skipped — no cointegration)")
    print(f"  Trades:              {perf['n_trades']}")
    print(f"  Win rate:            {perf['win_rate']:.1%}")
    print(f"  Avg holding:         {perf['avg_holding_days']:.1f} days")
    print(f"  Total return:        {perf['total_return']:.2%}")
    print(f"  Sharpe (ann.):       {perf['sharpe']:.3f}")
    print(f"  Max drawdown:        {perf['max_drawdown']:.2%}")
    print(f"  Windows skipped:     {perf['pct_windows_skipped']:.1%}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
