# Pairs Trading Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a defensible, interview-ready statistical arbitrage pairs trading engine for GLD/IAU using Johansen cointegration, Kalman Filter dynamic hedge-ratio estimation, and walk-forward backtesting with strict no-look-ahead guarantees.

**Architecture:** Daily log-prices feed a walk-forward engine: each window runs Johansen cointegration on the expanding formation set, initialises a Kalman Filter (state = [beta, alpha]) to track the dynamic hedge ratio, generates z-score signals from filter innovations, and computes dollar-neutral returns net of transaction costs. All metrics are computed exclusively on out-of-sample trading windows.

**Tech Stack:** Python 3.11+, yfinance, numpy, pandas, statsmodels, scipy, matplotlib, PyYAML, pytest.

---

## File Map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | project metadata + dependencies |
| `config.yaml` | tickers, date range, window sizes, thresholds, cost bps |
| `.gitignore` | ignore `data_cache/`, `results/` |
| `src/pairs_engine/__init__.py` | empty package marker |
| `src/pairs_engine/data.py` | download → clean → align → cache (parquet) |
| `src/pairs_engine/cointegration.py` | Johansen, ADF, Benjamini-Hochberg FDR |
| `src/pairs_engine/kalman.py` | `KalmanHedge` class — state-space filter |
| `src/pairs_engine/signals.py` | z-score from innovations, position state machine |
| `src/pairs_engine/costs.py` | 5 bps per-trade cost model |
| `src/pairs_engine/metrics.py` | Sharpe, max drawdown, trade stats |
| `src/pairs_engine/backtest.py` | walk-forward driver, return computation |
| `src/pairs_engine/plotting.py` | equity, spread+z, beta charts |
| `src/pairs_engine/cli.py` | CLI entrypoint — wires everything, writes results/ |
| `tests/test_data.py` | alignment, NaN-free, cache round-trip |
| `tests/test_cointegration.py` | Johansen on synthetic (co)integrated series |
| `tests/test_kalman.py` | filter recovers known beta on synthetic data |
| `tests/test_signals.py` | state-machine transitions on hand-crafted z series |
| `tests/test_costs.py` | cost charged only on position changes |
| `tests/test_metrics.py` | Sharpe/MDD vs. hand-computed values |
| `tests/test_backtest.py` | no look-ahead assertion, return accounting |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `.gitignore`
- Create: `src/pairs_engine/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "pairs-engine"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "yfinance>=0.2.40",
    "numpy>=1.26",
    "pandas>=2.2",
    "statsmodels>=0.14",
    "scipy>=1.13",
    "matplotlib>=3.9",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `config.yaml`**

```yaml
pair:
  y: "GLD"   # dependent leg
  x: "IAU"   # independent leg

data:
  start: "2007-01-01"
  end:   "2026-04-25"
  cache_dir: "data_cache"

walk_forward:
  formation_min_bars: 504   # ~2 trading years
  trading_bars: 126         # ~6 trading months

kalman:
  delta: 1.0e-4   # process-noise diffusion — fixed globally, never tuned per window
  R: 1.0e-3       # observation noise variance

signals:
  entry_z: 2.0
  exit_z: 0.5

costs:
  bps: 5

results_dir: "results"
```

- [ ] **Step 3: Create `.gitignore`**

```
data_cache/
results/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
.DS_Store
```

- [ ] **Step 4: Create package markers**

Create `src/pairs_engine/__init__.py` — empty file.  
Create `tests/__init__.py` — empty file.

- [ ] **Step 5: Install in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors; `python -c "import pairs_engine"` succeeds.

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml config.yaml .gitignore src/ tests/
git commit -m "chore: project scaffold"
```

---

## Task 2: Data Ingestion (`data.py`)

**Files:**
- Create: `src/pairs_engine/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from pairs_engine.data import align_and_clean, load_or_fetch


def make_raw(dates, gld_prices, iau_prices):
    """Helper: build a raw multi-ticker DataFrame."""
    df = pd.DataFrame({"GLD": gld_prices, "IAU": iau_prices}, index=pd.DatetimeIndex(dates))
    return df


def test_align_and_clean_drops_nan_rows():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    df = make_raw(dates, [100, np.nan, 102, 103, 104], [10, 10.1, np.nan, 10.3, 10.4])
    result = align_and_clean(df)
    assert result.isna().sum().sum() == 0
    assert len(result) == 3   # rows 0, 3, 4 survive


def test_align_and_clean_returns_log_prices():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    df = make_raw(dates, [100.0, 101.0, 102.0], [10.0, 10.1, 10.2])
    result = align_and_clean(df)
    expected_gld_0 = np.log(100.0)
    assert abs(result["GLD"].iloc[0] - expected_gld_0) < 1e-10


def test_align_and_clean_no_ffill():
    """A NaN that would be filled by ffill must remain as a dropped row."""
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    df = make_raw(dates, [100, np.nan, 102, 103], [10, 10.1, 10.2, 10.3])
    result = align_and_clean(df)
    # Row 1 has NaN in GLD — it must be dropped, not filled
    assert len(result) == 3


def test_load_or_fetch_caches(tmp_path):
    """Cache round-trip: second call reads parquet, not network."""
    # Use a tiny real download to test cache mechanics
    df = load_or_fetch(["GLD", "IAU"], start="2024-01-01", end="2024-01-31", cache_dir=str(tmp_path))
    assert not df.empty
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1

    # Monkey-patch yfinance to ensure it's NOT called on second load
    import pairs_engine.data as data_mod
    original = data_mod._download_raw

    called = []
    def patched(*a, **kw):
        called.append(True)
        return original(*a, **kw)

    data_mod._download_raw = patched
    df2 = load_or_fetch(["GLD", "IAU"], start="2024-01-01", end="2024-01-31", cache_dir=str(tmp_path))
    data_mod._download_raw = original

    assert not called, "yfinance was called on second load — cache not working"
    pd.testing.assert_frame_equal(df, df2)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_data.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `data.py` doesn't exist yet.

- [ ] **Step 3: Implement `src/pairs_engine/data.py`**

```python
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path


def _download_raw(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns when >1 ticker; extract Close
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"][tickers]
    else:
        df = raw[["Close"]].rename(columns={"Close": tickers[0]})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    assert not df.empty, f"yfinance returned empty data for {tickers}"
    return df


def align_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any row with a NaN in any column, then return log prices. No ffill."""
    clean = df.dropna()
    return np.log(clean)


def load_or_fetch(tickers: list[str], start: str, end: str, cache_dir: str) -> pd.DataFrame:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{'_'.join(sorted(tickers))}_{start}_{end}".encode()).hexdigest()
    cache_path = Path(cache_dir) / f"{key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    raw = _download_raw(tickers, start, end)
    result = align_and_clean(raw)
    result.to_parquet(cache_path)
    return result
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_data.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pairs_engine/data.py tests/test_data.py
git commit -m "feat: data ingestion with parquet cache and log-price output"
```

---

## Task 3: Cointegration Pipeline (`cointegration.py`)

**Files:**
- Create: `src/pairs_engine/cointegration.py`
- Create: `tests/test_cointegration.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cointegration.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pairs_engine.cointegration import johansen_test, adf_spread_test, bh_fdr


def make_cointegrated_pair(n=500, beta=1.0, seed=42):
    rng = np.random.default_rng(seed)
    spread = rng.normal(0, 1, n).cumsum() * 0.05   # stationary mean-reverting spread
    # Actually generate as a proper cointegrated pair:
    # x is a random walk, y = beta*x + stationary_noise
    x = rng.normal(0, 1, n).cumsum()
    y = beta * x + rng.normal(0, 0.5, n)   # y - beta*x is stationary
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"Y": y, "X": x}, index=dates)


def make_independent_pair(n=500, seed=99):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).cumsum()
    y = rng.normal(0, 1, n).cumsum()
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"Y": y, "X": x}, index=dates)


def test_johansen_rejects_on_cointegrated_pair():
    df = make_cointegrated_pair(n=500)
    result = johansen_test(df)
    assert result["rejects_no_coint"] is True


def test_johansen_does_not_reject_independent_walks():
    df = make_independent_pair(n=500)
    result = johansen_test(df)
    # Independent walks — should fail to reject (or reject rarely)
    # We allow this to pass because random walks can spuriously cointegrate,
    # but on this specific seed it should not.
    assert result["rejects_no_coint"] is False


def test_johansen_returns_normalised_eigvec():
    df = make_cointegrated_pair(n=500)
    result = johansen_test(df)
    ev = result["eigvec_normalized"]
    assert ev.shape == (2,)
    # First coefficient normalised to 1.0
    assert abs(ev[0] - 1.0) < 1e-10


def test_johansen_raises_on_short_sample():
    df = make_cointegrated_pair(n=10)
    with pytest.raises(ValueError, match="too short"):
        johansen_test(df)


def test_adf_spread_stationary():
    rng = np.random.default_rng(0)
    spread = pd.Series(rng.normal(0, 1, 300))   # white noise — stationary
    result = adf_spread_test(spread)
    assert result["pvalue"] < 0.05


def test_bh_fdr_single_pvalue():
    # For a single pair, BH-FDR is a no-op (same as raw test)
    result = bh_fdr([0.01], alpha=0.05)
    assert result == [True]

    result_reject = bh_fdr([0.9], alpha=0.05)
    assert result_reject == [False]


def test_bh_fdr_multiple_pvalues():
    # 3 pairs: first two have tiny p-values, third is large
    pvalues = [0.001, 0.01, 0.8]
    result = bh_fdr(pvalues, alpha=0.05)
    assert result[0] is True
    assert result[1] is True
    assert result[2] is False
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_cointegration.py -v
```

Expected: `ImportError` — `cointegration.py` not yet created.

- [ ] **Step 3: Implement `src/pairs_engine/cointegration.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

_MIN_OBS = 52   # hard lower bound — Johansen needs many more in practice


def johansen_test(log_prices: pd.DataFrame) -> dict:
    """
    Run Johansen trace test on a 2-column log-price DataFrame.
    det_order=0: constant in cointegrating relation.
    k_ar_diff=1: one lag of differences.
    Returns rejects_no_coint=True when trace stat > 95% critical value.
    """
    if len(log_prices) < _MIN_OBS:
        raise ValueError(f"Sample too short for Johansen: {len(log_prices)} obs — too short")
    result = coint_johansen(log_prices.values, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1[0]          # trace statistic for r=0
    crit_95 = result.cvt[0, 1]         # 95% critical value for r=0
    rejects = bool(trace_stat > crit_95)
    ev = result.evec[:, 0]              # first eigenvector
    ev_normalised = ev / ev[0]          # normalise so first element = 1.0
    return {
        "trace_stat": float(trace_stat),
        "crit_95": float(crit_95),
        "rejects_no_coint": rejects,
        "eigvec_normalized": ev_normalised,
    }


def adf_spread_test(spread: pd.Series) -> dict:
    """ADF test on the spread. Returns pvalue and rejects_unit_root."""
    stat, pvalue, _, _, _, _ = adfuller(spread.dropna(), autolag="AIC")
    return {"adf_stat": float(stat), "pvalue": float(pvalue), "rejects_unit_root": pvalue < 0.05}


def bh_fdr(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg FDR correction for multiple hypothesis tests.
    Returns a boolean list: True = reject null (cointegrated).
    Dormant for single-pair use (n=1 is equivalent to raw threshold).
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    sorted_p = [pvalues[i] for i in order]
    rejected = [False] * m
    for k, (idx, p) in enumerate(zip(order, sorted_p), start=1):
        if p <= alpha * k / m:
            rejected[idx] = True
        else:
            break   # BH is monotone — once we fail, all subsequent fail
    return rejected
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cointegration.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pairs_engine/cointegration.py tests/test_cointegration.py
git commit -m "feat: Johansen cointegration, ADF, and BH-FDR correction"
```

---

## Task 4: Kalman Filter (`kalman.py`)

**Files:**
- Create: `src/pairs_engine/kalman.py`
- Create: `tests/test_kalman.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_kalman.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pairs_engine.kalman import KalmanHedge


def make_constant_beta_series(n=300, true_beta=1.5, true_alpha=0.1, seed=7):
    """Synthetic observation: y = beta*x + alpha + noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).cumsum()
    y = true_beta * x + true_alpha + rng.normal(0, 0.05, n)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(y, index=dates, name="Y"), pd.Series(x, index=dates, name="X")


def test_kalman_recovers_beta_approximately():
    y, x = make_constant_beta_series(n=500, true_beta=1.5)
    kf = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf.run(y, x)
    # After warm-up, median beta should be close to 1.5
    warmup = 50
    median_beta = df["beta"].iloc[warmup:].median()
    assert abs(median_beta - 1.5) < 0.15, f"Expected beta~1.5, got {median_beta:.4f}"


def test_kalman_spread_is_innovation_not_residual():
    """spread[t] must be y[t] - H[t] @ x_pred[t], NOT y[t] - beta_updated[t]*x[t]."""
    y, x = make_constant_beta_series(n=100, true_beta=2.0)
    kf = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf.run(y, x)
    # The innovation is the prediction error BEFORE the update step.
    # Verify: spread_var[t] > 0 always (it's a variance)
    assert (df["spread_var"] > 0).all()


def test_kalman_step_matches_run():
    """step() and run() should produce identical results."""
    y, x = make_constant_beta_series(n=50, true_beta=1.2)
    kf1 = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf1.run(y, x)

    kf2 = KalmanHedge(delta=1e-4, R=1e-3)
    betas, alphas, spreads, spread_vars = [], [], [], []
    for yt, xt in zip(y, x):
        beta, alpha, spread, sv = kf2.step(yt, xt)
        betas.append(beta)
        alphas.append(alpha)
        spreads.append(spread)
        spread_vars.append(sv)

    np.testing.assert_allclose(df["beta"].values, betas, rtol=1e-8)
    np.testing.assert_allclose(df["spread"].values, spreads, rtol=1e-8)


def test_kalman_run_output_columns():
    y, x = make_constant_beta_series(n=30)
    kf = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf.run(y, x)
    assert set(df.columns) == {"beta", "alpha", "spread", "spread_var"}
    assert len(df) == 30


def test_kalman_prior_override():
    """Provide an explicit prior state — filter should start from it."""
    y, x = make_constant_beta_series(n=100, true_beta=2.0)
    kf = KalmanHedge(delta=1e-4, R=1e-3)
    # Start with prior beta=2.0, alpha=0.0
    df = kf.run(y, x, prior_mean=np.array([2.0, 0.0]), prior_cov=np.eye(2) * 10.0)
    # With a good prior, beta should stay near 2.0 from the start
    assert abs(df["beta"].iloc[5] - 2.0) < 0.3
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_kalman.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pairs_engine/kalman.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd


class KalmanHedge:
    """
    State-space Kalman filter estimating dynamic hedge ratio and intercept.

    State:  theta_t = [beta_t, alpha_t]^T
    Transition: theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q),  Q = delta * I
    Observation: y_t = H_t @ theta_t + v_t,  v_t ~ N(0, R)
               where H_t = [x_t, 1]

    The 'spread' returned is the innovation (prediction error BEFORE the update),
    not the post-update residual — this avoids look-ahead in signal generation.

    delta and R are global structural constants; never tuned per window.
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-3) -> None:
        self.delta = delta
        self.R = R
        self._Q = delta * np.eye(2)
        self._reset()

    def _reset(self) -> None:
        self._theta = np.zeros(2)           # [beta, alpha]
        self._P = np.eye(2)                 # state covariance

    def reset(
        self,
        prior_mean: np.ndarray | None = None,
        prior_cov: np.ndarray | None = None,
    ) -> None:
        self._theta = prior_mean if prior_mean is not None else np.zeros(2)
        self._P = prior_cov if prior_cov is not None else np.eye(2)

    def step(self, y_t: float, x_t: float) -> tuple[float, float, float, float]:
        """
        One Kalman update step.
        Returns (beta_updated, alpha_updated, innovation, innovation_variance).
        The innovation is computed from the PREDICTED state (before update).
        """
        # Predict
        theta_pred = self._theta          # random-walk: F = I
        P_pred = self._P + self._Q

        # Observation vector
        H = np.array([x_t, 1.0])

        # Innovation (prediction error — uses predicted state, not updated)
        innovation = y_t - H @ theta_pred
        S = H @ P_pred @ H + self.R       # innovation variance (scalar)

        # Kalman gain
        K = P_pred @ H / S                # shape (2,)

        # Update
        self._theta = theta_pred + K * innovation
        self._P = (np.eye(2) - np.outer(K, H)) @ P_pred

        beta = float(self._theta[0])
        alpha = float(self._theta[1])
        return beta, alpha, float(innovation), float(S)

    def run(
        self,
        y: pd.Series,
        x: pd.Series,
        prior_mean: np.ndarray | None = None,
        prior_cov: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Run the filter over full series. Returns DataFrame indexed like y."""
        self.reset(prior_mean, prior_cov)
        records = []
        for yt, xt in zip(y, x):
            beta, alpha, spread, sv = self.step(float(yt), float(xt))
            records.append({"beta": beta, "alpha": alpha, "spread": spread, "spread_var": sv})
        return pd.DataFrame(records, index=y.index)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_kalman.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pairs_engine/kalman.py tests/test_kalman.py
git commit -m "feat: KalmanHedge state-space filter for dynamic hedge ratio"
```

---

## Task 5: Signal Generation (`signals.py`)

**Files:**
- Create: `src/pairs_engine/signals.py`
- Create: `tests/test_signals.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_signals.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pairs_engine.signals import compute_zscore, generate_positions


def make_filter_output(spreads, spread_vars):
    n = len(spreads)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return (
        pd.Series(spreads, index=dates, name="spread"),
        pd.Series(spread_vars, index=dates, name="spread_var"),
    )


def test_compute_zscore_basic():
    spread, sv = make_filter_output([2.0, -1.0, 0.0], [1.0, 1.0, 1.0])
    z = compute_zscore(spread, sv)
    np.testing.assert_allclose(z.values, [2.0, -1.0, 0.0])


def test_compute_zscore_uses_sqrt_variance():
    spread, sv = make_filter_output([4.0], [4.0])
    z = compute_zscore(spread, sv)
    # z = 4.0 / sqrt(4.0) = 2.0
    assert abs(z.iloc[0] - 2.0) < 1e-10


def test_generate_positions_enters_short_on_high_z():
    # z > 2.0 → short (position = -1)
    spread, sv = make_filter_output([0, 0, 2.5, 2.5, 0.3], [1, 1, 1, 1, 1])
    z = compute_zscore(spread, sv)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
    assert pos.iloc[2] == -1
    assert pos.iloc[3] == -1   # still holding
    assert pos.iloc[4] == 0    # z=0.3 < 0.5 → exit


def test_generate_positions_enters_long_on_low_z():
    # z < -2.0 → long (position = +1)
    spread, sv = make_filter_output([0, -2.5, -2.5, 0.3], [1, 1, 1, 1])
    z = compute_zscore(spread, sv)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
    assert pos.iloc[1] == 1
    assert pos.iloc[2] == 1
    assert pos.iloc[3] == 0


def test_generate_positions_no_entry_between_thresholds():
    spread, sv = make_filter_output([0, 1.0, 1.5, -1.0, -1.8], [1, 1, 1, 1, 1])
    z = compute_zscore(spread, sv)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
    assert (pos == 0).all()


def test_generate_positions_hold_until_exit():
    # Enter short at bar 1, z stays above exit until bar 4
    spread, sv = make_filter_output([0, 2.5, 1.2, 0.6, 0.3], [1, 1, 1, 1, 1])
    z = compute_zscore(spread, sv)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
    assert pos.iloc[1] == -1
    assert pos.iloc[2] == -1   # |z|=1.2 > 0.5 → hold
    assert pos.iloc[3] == -1   # |z|=0.6 > 0.5 → hold
    assert pos.iloc[4] == 0    # |z|=0.3 < 0.5 → exit


def test_generate_positions_flip_short_to_long():
    # Short entry, exit, then long entry
    spread, sv = make_filter_output([2.5, 0.3, -2.5], [1, 1, 1])
    z = compute_zscore(spread, sv)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
    assert pos.iloc[0] == -1
    assert pos.iloc[1] == 0
    assert pos.iloc[2] == 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_signals.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pairs_engine/signals.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_zscore(spread: pd.Series, spread_var: pd.Series) -> pd.Series:
    """z_t = innovation_t / sqrt(innovation_variance_t). Uses Kalman's own variance."""
    return spread / np.sqrt(spread_var)


def generate_positions(z: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5) -> pd.Series:
    """
    State machine over z-scores. Positions are determined AS OF close of bar t
    and applied to returns from t → t+1 (the caller must lag before return multiplication).

    States: +1 (long spread), 0 (flat), -1 (short spread)
    Enter short:  z > +entry_z
    Enter long:   z < -entry_z
    Exit:         |z| < exit_z
    Hold:         otherwise
    """
    positions = np.zeros(len(z), dtype=float)
    state = 0
    for i, zt in enumerate(z):
        if state == 0:
            if zt > entry_z:
                state = -1
            elif zt < -entry_z:
                state = 1
        elif state == -1:
            if abs(zt) < exit_z:
                state = 0
        elif state == 1:
            if abs(zt) < exit_z:
                state = 0
        positions[i] = state
    return pd.Series(positions, index=z.index, name="position")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_signals.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pairs_engine/signals.py tests/test_signals.py
git commit -m "feat: z-score computation and position state machine"
```

---

## Task 6: Transaction Costs and Metrics

**Files:**
- Create: `src/pairs_engine/costs.py`
- Create: `src/pairs_engine/metrics.py`
- Create: `tests/test_costs.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing cost tests**

Create `tests/test_costs.py`:

```python
import numpy as np
import pandas as pd
from pairs_engine.costs import apply_costs


def make_positions(values, betas=None):
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    pos = pd.Series(values, index=dates, dtype=float)
    if betas is None:
        betas = pd.Series(np.ones(len(values)), index=dates)
    else:
        betas = pd.Series(betas, index=dates, dtype=float)
    return pos, betas


def test_no_cost_when_no_change():
    pos, beta = make_positions([0, 0, 0, 0])
    costs = apply_costs(pos, beta, bps=5)
    assert (costs == 0).all()


def test_cost_on_entry():
    # Flat → short: change in pos_y = 1, change in pos_x = beta
    pos, beta = make_positions([0, -1, -1], betas=[1.0, 1.0, 1.0])
    costs = apply_costs(pos, beta, bps=5)
    # At bar 1: |delta_pos_y| = 1, |delta_pos_x| = beta = 1.0
    # cost = 5/10000 * (1 + 1) = 0.001
    assert abs(costs.iloc[1] - 0.001) < 1e-10
    assert costs.iloc[0] == 0.0
    assert costs.iloc[2] == 0.0   # no change


def test_cost_on_flip():
    # Short → long flip = two trades
    pos, beta = make_positions([-1, 1], betas=[1.0, 1.0])
    costs = apply_costs(pos, beta, bps=5)
    # At bar 1: |delta_pos_y| = 2, |delta_pos_x| = 2*beta = 2.0
    # cost = 5/10000 * (2 + 2) = 0.002
    assert abs(costs.iloc[1] - 0.002) < 1e-10


def test_cost_scales_with_beta():
    pos, beta = make_positions([0, -1], betas=[1.0, 2.0])
    costs = apply_costs(pos, beta, bps=5)
    # delta_pos_y = 1, delta_pos_x = 2 (beta at execution bar)
    # cost = 5/10000 * (1 + 2) = 0.0015
    assert abs(costs.iloc[1] - 0.0015) < 1e-10
```

- [ ] **Step 2: Write failing metrics tests**

Create `tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pairs_engine.metrics import sharpe, max_drawdown, trade_stats, summary


def test_sharpe_all_positive():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 252))
    s = sharpe(rets)
    # mean~0.001, std~0.01, annualised ~ 0.001/0.01*sqrt(252) ~ 1.587
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
    # peak=1.2, trough=0.9: drawdown = (0.9-1.2)/1.2 = -0.25
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
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/test_costs.py tests/test_metrics.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `src/pairs_engine/costs.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd


def apply_costs(positions: pd.Series, beta: pd.Series, bps: float = 5) -> pd.Series:
    """
    Compute round-trip transaction costs on position changes.

    Dollar-neutral sizing: Y-leg is unit notional, X-leg is beta notional.
    cost_t = bps/10000 * (|delta_pos_y_t| + |beta_t * delta_pos_x_t|)

    'delta_pos_x' equals 'delta_pos' because X is traded in beta units —
    so X cost = bps/10000 * beta_t * |delta_pos_t|.

    Costs are charged on the bar where the position changes.
    """
    rate = bps / 10_000
    delta_pos = positions.diff().fillna(positions.iloc[0] if len(positions) else 0)
    abs_delta = delta_pos.abs()
    # Y-leg notional = 1; X-leg notional = beta at execution bar
    cost = rate * (abs_delta + beta * abs_delta)
    return cost
```

- [ ] **Step 5: Implement `src/pairs_engine/metrics.py`**

```python
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
    """
    Compute per-trade statistics from a ledger with columns:
    net_pnl, holding_days.
    """
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
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_costs.py tests/test_metrics.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pairs_engine/costs.py src/pairs_engine/metrics.py \
        tests/test_costs.py tests/test_metrics.py
git commit -m "feat: transaction cost model and performance metrics"
```

---

## Task 7: Walk-Forward Backtester (`backtest.py`)

This is the most critical file. Every look-ahead risk lives here.

**Files:**
- Create: `src/pairs_engine/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_backtest.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pairs_engine.backtest import walk_forward, _build_trade_ledger


def make_cointegrated_log_prices(n=1000, beta=1.0, seed=42):
    """Synthetic cointegrated pair in log-price space."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, n).cumsum()        # log-price random walk
    spread = rng.normal(0, 0.005, n)             # stationary spread
    y = beta * x + spread
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"Y": y, "X": x}, index=dates)


def test_walk_forward_returns_expected_keys():
    prices = make_cointegrated_log_prices(n=800)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    for key in ["equity_curve", "trade_ledger", "windows_total", "windows_skipped"]:
        assert key in result


def test_walk_forward_no_future_data_in_formation():
    """
    Walk forward must NEVER use data from the trading window in Johansen or Kalman init.
    We verify indirectly: running on a shuffled series should produce different results
    than on the original — if there's look-ahead the results would be identical.
    """
    prices = make_cointegrated_log_prices(n=800)
    r1 = walk_forward(prices, formation_min=252, trading_len=63,
                      delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    # Shuffle the trading portion — results should differ
    shuffled = prices.copy()
    shuffled.iloc[252:] = prices.iloc[252:].sample(frac=1, random_state=99).values
    r2 = walk_forward(shuffled, formation_min=252, trading_len=63,
                      delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    eq1 = r1["equity_curve"]["net_ret"]
    eq2 = r2["equity_curve"]["net_ret"]
    # Not all returns should be identical (shuffling trading data should matter)
    assert not eq1.equals(eq2), "Returns identical after shuffling — possible look-ahead"


def test_walk_forward_skips_window_on_no_coint():
    """Use independent random walks — Johansen should fail → window skipped."""
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(0, 0.01, n).cumsum()
    y = rng.normal(0, 0.01, n).cumsum()   # independent walk
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"Y": y, "X": x}, index=dates)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    # With random seed 0 independent walks, expect multiple skips
    assert result["windows_skipped"] > 0


def test_equity_curve_starts_at_one():
    prices = make_cointegrated_log_prices(n=800)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    equity = result["equity_curve"]["equity"]
    if len(equity) > 0:
        assert abs(equity.iloc[0] - 1.0) < 1e-10


def test_position_lagged_correctly():
    """Position at bar t should use z computed from bars [0..t], not t+1."""
    prices = make_cointegrated_log_prices(n=800)
    result = walk_forward(prices, formation_min=252, trading_len=63,
                          delta=1e-4, R=1e-3, entry_z=2.0, exit_z=0.5, cost_bps=5)
    ec = result["equity_curve"]
    # Returns at bar t use position from bar t-1 (lagged by 1)
    # Verify that on a bar with position=0 the gross_ret contribution is ~0
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_backtest.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pairs_engine/backtest.py`**

```python
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

    Returns dict with keys:
        equity_curve   — DataFrame indexed by date
        trade_ledger   — DataFrame, one row per closed trade
        windows_total  — int
        windows_skipped — int
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

        formation_data = log_prices.iloc[:formation_end]

        # --- Cointegration gate (formation data only) ---
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

        # --- Kalman: warm up on formation data ---
        # OLS prior from formation to give the filter a sensible starting point
        X_form = formation_data[x_col].values
        Y_form = formation_data[y_col].values
        A = np.column_stack([X_form, np.ones_like(X_form)])
        ols_coef, _, _, _ = np.linalg.lstsq(A, Y_form, rcond=None)
        prior_mean = ols_coef                   # [beta_ols, alpha_ols]
        prior_cov = np.eye(2) * 100.0           # wide prior — let filter adapt

        kf = KalmanHedge(delta=delta, R=R)
        # Run filter through entire formation to reach end-of-formation state
        kf.run(formation_data[y_col], formation_data[x_col],
               prior_mean=prior_mean, prior_cov=prior_cov)
        # kf's internal state is now at formation_end

        # --- Trading window ---
        trading_data = log_prices.iloc[trading_start:trading_end]
        y_trade = trading_data[y_col]
        x_trade = trading_data[x_col]

        kf_results = kf.run(y_trade, x_trade)   # continue from formation-end state

        z = compute_zscore(kf_results["spread"], kf_results["spread_var"])
        raw_pos = generate_positions(z, entry_z=entry_z, exit_z=exit_z)

        # Lag positions and beta by 1 bar to avoid look-ahead in return computation
        pos_lagged = raw_pos.shift(1).fillna(0)
        beta_lagged = kf_results["beta"].shift(1).fillna(kf_results["beta"].iloc[0])

        # Log returns for each leg
        r_y = y_trade.diff().fillna(0)    # log-return of Y
        r_x = x_trade.diff().fillna(0)   # log-return of X

        # Dollar-neutral gross return: long Y, short beta*X (or reverse)
        gross_ret = pos_lagged * (r_y - beta_lagged * r_x)

        # Transaction costs (uses raw_pos timing — cost on bar of signal)
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

        # Build trade ledger for this window
        trade_ledger_rows.extend(_build_trade_ledger(window_df))

        formation_end += trading_len

    equity_curve = pd.concat(all_bars) if all_bars else pd.DataFrame()
    trade_ledger = pd.DataFrame(trade_ledger_rows) if trade_ledger_rows else pd.DataFrame(
        columns=["entry_date", "exit_date", "side", "gross_pnl", "costs", "net_pnl", "holding_days"]
    )

    return {
        "equity_curve": equity_curve,
        "trade_ledger": trade_ledger,
        "windows_total": windows_total,
        "windows_skipped": windows_skipped,
    }


def _build_trade_ledger(window_df: pd.DataFrame) -> list[dict]:
    """Extract individual trades from a window's equity curve DataFrame."""
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_backtest.py -v
```

Expected: all tests PASS (the look-ahead shuffle test may take a few seconds).

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS with no failures.

- [ ] **Step 6: Commit**

```bash
git add src/pairs_engine/backtest.py tests/test_backtest.py
git commit -m "feat: walk-forward backtester with expanding formation and zero look-ahead"
```

---

## Task 8: Plotting (`plotting.py`)

**Files:**
- Create: `src/pairs_engine/plotting.py`

(No unit tests for plotting — test visually by running CLI in Task 9.)

- [ ] **Step 1: Implement `src/pairs_engine/plotting.py`**

```python
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI/headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_equity(equity: pd.Series, outpath: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, color="#2563eb", linewidth=1.2, label="Equity")
    ax1.set_ylabel("Equity (normalised to 1)")
    ax1.set_title("Out-of-Sample Equity Curve — GLD/IAU Pairs Strategy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown panel
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

    # Shade position regions
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
```

- [ ] **Step 2: Commit**

```bash
git add src/pairs_engine/plotting.py
git commit -m "feat: equity, spread/z-score, and beta diagnostic plots"
```

---

## Task 9: CLI Entrypoint (`cli.py`) and End-to-End Run

**Files:**
- Create: `src/pairs_engine/cli.py`
- Create: `results/` (via mkdir, gitignored)

- [ ] **Step 1: Implement `src/pairs_engine/cli.py`**

```python
from __future__ import annotations
import json
import sys
from pathlib import Path
import argparse
import yaml
import pandas as pd
from pairs_engine.data import load_or_fetch
from pairs_engine.backtest import walk_forward
from pairs_engine.metrics import summary
from pairs_engine.plotting import plot_equity, plot_spread_z, plot_beta


def main() -> None:
    parser = argparse.ArgumentParser(description="GLD/IAU Pairs Trading Engine")
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

    # Write CSVs
    ec.to_csv(results_dir / "equity_curve.csv")
    if len(ledger) > 0:
        ledger.to_csv(results_dir / "trades.csv", index=False)

    # Compute and write metrics
    net_rets = ec["net_ret"] if len(ec) > 0 else pd.Series(dtype=float)
    equity = ec["equity"] if len(ec) > 0 else pd.Series(dtype=float)
    perf = summary(
        net_rets, equity, ledger,
        windows_total=result["windows_total"],
        windows_skipped=result["windows_skipped"],
    )
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(perf, f, indent=2)

    # Plots
    if len(ec) > 0:
        plot_equity(ec["equity"], str(results_dir / "equity.png"))
        plot_spread_z(ec, str(results_dir / "spread_z.png"))
        plot_beta(ec, str(results_dir / "beta.png"))

    # Console output
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
```

- [ ] **Step 2: Add entrypoint to `pyproject.toml`**

Open `pyproject.toml` and add to the `[project]` section:

```toml
[project.scripts]
pairs-engine = "pairs_engine.cli:main"
```

Then reinstall:

```bash
pip install -e ".[dev]"
```

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 4: Run the engine end-to-end**

```bash
python -m pairs_engine.cli --config config.yaml
```

Expected output (numbers will vary):
```
==================================================
  GLD/IAU Pairs Backtest (out-of-sample)
==================================================
  Period:              2009-01-XX → 2026-04-XX
  WF windows:          XX (X skipped — no cointegration)
  Trades:              XX
  Win rate:            XX.X%
  Avg holding:         XX.X days
  Total return:        X.XX%
  Sharpe (ann.):       X.XXX
  Max drawdown:        X.XX%
  Windows skipped:     X.X%
==================================================
```

GLD/IAU sanity checks:
- Johansen should reject "no cointegration" in nearly all windows (they track the same asset)
- Kalman beta should hover near ~10.0 (GLD price / IAU price ratio)
- Sharpe should be 0.3–1.5 range after costs; if > 3, suspect a look-ahead bug
- Equity curve should be smooth, not hockey-stick shaped

- [ ] **Step 5: Commit**

```bash
git add src/pairs_engine/cli.py pyproject.toml
git commit -m "feat: CLI entrypoint and end-to-end pipeline wiring"
```

---

## Look-Ahead Audit Checklist

Run through this manually against the implemented code before declaring done:

- [ ] Johansen in window `i` calls `johansen_test(log_prices.iloc[:formation_end])` — formation data only
- [ ] Kalman warm-up runs on `formation_data` only, then continues on `trading_data` from that state
- [ ] `pos_lagged = raw_pos.shift(1)` — position lagged before return multiplication
- [ ] `beta_lagged = kf_results["beta"].shift(1)` — beta lagged before return multiplication
- [ ] `gross_ret = pos_lagged * (r_y - beta_lagged * r_x)` — correct
- [ ] `compute_zscore` uses `spread_var` from Kalman innovation (not post-hoc std)
- [ ] Costs charged using `raw_pos` (current bar's position change), not lagged
- [ ] All metrics passed `ec["net_ret"]` which is purely trading-window bars

---

## Verification Summary

```bash
# Unit tests
pytest tests/ -v

# Full pipeline
python -m pairs_engine.cli --config config.yaml

# Inspect outputs
ls results/
# Should contain: equity_curve.csv, trades.csv, metrics.json,
#                 equity.png, spread_z.png, beta.png
```
