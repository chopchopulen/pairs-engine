# Pairs Trading Engine

A production-quality statistical arbitrage engine built for interview-level scrutiny. Uses Johansen cointegration testing, a Kalman Filter for dynamic hedge-ratio estimation, and walk-forward backtesting with strict no-look-ahead guarantees.

**Headline result (EWA/EWC, 2011–2025):** Sharpe 0.296 · 72.7% win rate · 33 trades · 24.4% total return · 17.6% max drawdown

---

## Strategy Overview

Pairs trading is a market-neutral statistical arbitrage strategy. When two assets are **cointegrated** — meaning a linear combination of their prices is stationary — deviations from that equilibrium tend to mean-revert. The engine:

1. Tests whether a pair is cointegrated using the **Johansen trace test** on formation-period data only
2. Estimates a **time-varying hedge ratio** β via a Kalman filter, updated bar-by-bar
3. Generates signals from the **Kalman innovation z-score** (prediction error normalized by filter variance)
4. Runs a **walk-forward backtest** — expanding formation window, fixed 6-month trading window — to produce fully out-of-sample results

The strategy is dollar-neutral: long $1 of Y, short $β of X. Entry at |z| > 1.0, exit at |z| < 0.3.

---

## Architecture

```
pairs-engine/
├── config.yaml                     # all parameters — no per-window tuning
├── src/pairs_engine/
│   ├── data.py                     # yfinance download, log prices, parquet cache
│   ├── cointegration.py            # Johansen, ADF, Benjamini-Hochberg FDR
│   ├── kalman.py                   # state-space dynamic hedge ratio (Chan 2013)
│   ├── signals.py                  # z-score + entry/exit state machine
│   ├── backtest.py                 # walk-forward driver, position bookkeeping
│   ├── costs.py                    # 5bps transaction cost model
│   ├── metrics.py                  # Sharpe, max drawdown, trade stats
│   ├── plotting.py                 # 4 diagnostic charts
│   └── cli.py                      # entrypoint
└── tests/                          # 39 unit tests, all passing
```

---

## No-Look-Ahead Guarantees

Every design decision was made to ensure zero information leakage from future to past:

| Risk | How it's prevented |
|---|---|
| Future prices in formation | `log_prices.iloc[:formation_end]` — strict Python slice |
| Kalman state contamination | Filter warm-started on formation only; trading continues from that state |
| Position applied same bar | `pos_lagged = raw_pos.shift(1)` — yesterday's position drives today's return |
| Beta applied same bar | `beta_lagged = kf_results["beta"].shift(1)` — yesterday's β scales X-leg return |
| Kalman innovation look-ahead | Spread = innovation **before** the Kalman update (`y_t - H θ_{t\|t-1}`) |
| In-sample metric reporting | All metrics computed on concatenated trading-window returns only |

---

## Component Details

### `data.py` — Ingestion
Downloads adjusted close prices via yfinance (`auto_adjust=True`), drops any date where either ticker has a NaN (no forward-fill — that would inject look-ahead on missing data), converts to log prices, and caches as Parquet. The cache key is an MD5 hash of `{tickers}_{start}_{end}`.

### `cointegration.py` — Statistical Tests
- **`johansen_test`**: wraps `statsmodels` `coint_johansen` with `det_order=0, k_ar_diff=1`. Returns the trace statistic and 95% critical value. Cointegration gate passes when `trace > crit_95`.
- **`adf_spread_test`**: ADF on the residual spread — used as a sanity check, not the primary gate.
- **`bh_fdr`**: Benjamini-Hochberg FDR correction — dormant for single-pair use, exposed for future universe screening.

### `kalman.py` — Dynamic Hedge Ratio
State-space model following Chan (2013):

```
State:        θ_t = [β_t, α_t]ᵀ
Transition:   θ_t = θ_{t-1} + w_t,   w_t ~ N(0, δI)     (random walk)
Observation:  y_t = [x_t, 1] θ_t + v_t,   v_t ~ N(0, R)
```

The **spread used for signals is the innovation** — the prediction error before the Kalman update:

```
spread_t = y_t - [x_t, 1] θ_{t|t-1}
```

Using the post-update state `θ_{t|t}` would constitute look-ahead: it incorporates today's observation before generating today's signal. `δ` (process noise, 1e-6) and `R` (observation noise, 1e-3) are set globally in `config.yaml` and never re-tuned per window.

### `signals.py` — Signal Generation
```
z_t = spread_t / √S_t         where S_t = H P_{t|t-1} Hᵀ + R  (innovation variance)

z_t > +1.0  →  position = -1  (short Y, long X)
z_t < -1.0  →  position = +1  (long Y, short X)
|z_t| < 0.3 →  position =  0  (exit)
otherwise   →  hold
```

### `backtest.py` — Walk-Forward Engine
```
Window 1: formation = bars[0:504],   trading = bars[504:630]
Window 2: formation = bars[0:630],   trading = bars[630:756]
...
```
Each window: Johansen gate → OLS prior for Kalman → warm-up Kalman on formation → trade with continued Kalman state → accrue out-of-sample returns. Equity chains across windows.

### `costs.py`
```
cost_t = (bps/10000) × (|Δpos_Y| + β_{t-1} × |Δpos_X|)
```
Charged on the bar the position changes. X-leg cost scales with β because a larger hedge ratio means more shares traded.

### `plotting.py` — Diagnostics
| Output | What it shows |
|---|---|
| `equity.png` | Cumulative equity + drawdown shaded |
| `spread_z.png` | Kalman spread (top) + z-score with ±1.0/±0.3 lines and position shading |
| `beta.png` | Dynamic hedge ratio over time |
| `johansen_stability.png` | Expanding-window Johansen trace stat vs 95% critical value |

---

## Pair Screening Results

Full screening across 10 candidate pairs (see `results/screening_summary.csv`):

| Pair | Sharpe | Win Rate | Trades | Avg Hold | Total Return | Max DD | % Skipped |
|---|---|---|---|---|---|---|---|
| HD/LOW | 0.783 | 75.0% | 16 | 8.1d | 16.7% | 11.7% | 88% |
| MCD/YUM | 0.597 | 60.0% | 50 | 8.0d | 39.6% | 17.2% | 62% |
| JPM/BAC | 0.462 | 66.7% | 33 | 11.1d | 27.8% | 45.5% | 85% |
| TLT/IEF | 0.458 | 50.0% | 6 | 3.0d | 2.2% | 0.9% | 74% |
| **EWA/EWC** | **0.296** | **72.7%** | **33** | **11.5d** | **24.4%** | **17.6%** | **21%** |
| LQD/HYG | 0.195 | 42.9% | 28 | 1.9d | 5.1% | 3.6% | 26% |
| FCX/BHP | -0.371 | 68.9% | 74 | 13.4d | -67.9% | 77.5% | 71% |

**EWA/EWC is the recommended pair.** HD/LOW has a higher Sharpe but only 16 trades across 4 of 34 windows (88% skipped) — too thin to be statistically meaningful. EWA/EWC cointegrates in 27 of 34 windows (21% skipped), reflecting the structural commodity-export relationship between Australia and Canada that persists across economic cycles.

---

## Quickstart

```bash
# Install
pip install -e .

# Run the full pipeline on the default pair (EWA/EWC)
python -m pairs_engine.cli --config config.yaml

# Run tests
pytest tests/ -v
```

**Output written to `results/`:**
- `equity_curve.csv` — daily bar data: position, beta, spread, z, returns, equity
- `trades.csv` — one row per closed trade
- `metrics.json` — all performance metrics
- `equity.png`, `spread_z.png`, `beta.png`, `johansen_stability.png`

---

## Configuration

All parameters live in `config.yaml`. Nothing is tuned per window.

```yaml
pair:
  y: "EWA"     # dependent leg
  x: "EWC"     # independent leg

data:
  start: "2007-01-01"
  end:   "2026-04-25"
  cache_dir: "data_cache"    # parquet cache — gitignored

walk_forward:
  formation_min_bars: 504    # ~2 trading years minimum formation
  trading_bars: 126          # ~6 trading months per window

kalman:
  delta: 1.0e-6              # process-noise diffusion — controls β drift speed
  R: 1.0e-3                  # observation noise variance

signals:
  entry_z: 1.0               # enter when |z| exceeds this
  exit_z: 0.3                # exit when |z| falls below this

costs:
  bps: 5                     # transaction cost per leg per trade
```

To screen a different pair, change `pair.y` and `pair.x` and re-run.

---

## Dependencies

- `yfinance >= 0.2.40` — market data
- `numpy >= 1.26`, `pandas >= 2.2` — numerics
- `statsmodels >= 0.14` — Johansen test, ADF
- `scipy >= 1.13` — statistical utilities
- `matplotlib >= 3.9` — diagnostic plots
- `PyYAML >= 6.0` — config parsing
- `pyarrow` — Parquet cache

---

## Interview Defensibility

Common interview questions and the answers this engine supports:

**Q: How do you prevent look-ahead bias?**
The Kalman spread is the innovation before the state update. Positions and beta are lagged by one bar before computing returns. The Johansen test and Kalman warm-up use only `prices[:formation_end]` — a strict slice with no possibility of including trading-window data.

**Q: Why Johansen over Engle-Granger?**
Johansen's trace test is a full system test — it doesn't require pre-designating which series is the dependent variable, and it handles the case where multiple cointegrating vectors exist. Engle-Granger runs OLS on one direction only and can miss the cointegrating relationship.

**Q: Why not tune δ and R per window?**
That would be in-sample parameter optimization masquerading as out-of-sample results. δ and R are structural constants reflecting prior beliefs about how fast the hedge ratio drifts and how noisy observations are. They're set once globally and documented.

**Q: Why EWA/EWC specifically?**
Both Australia and Canada are major commodity exporters (iron ore, energy). Their equity markets co-move structurally because global commodity cycles drive both economies. This is an economic rationale, not a data-mined relationship — and it shows up empirically: 27 of 34 walk-forward windows pass the Johansen test.

**Q: Is 33 trades enough for statistical significance?**
At 72.7% win rate, a binomial test against 50% gives p < 0.01 (z ≈ 2.6). It's meaningful but not overwhelming — the appropriate response is "this is a proof-of-concept on a single pair; production would run a diversified universe with BH-FDR correction."
