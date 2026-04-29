import numpy as np
import pandas as pd
import pytest
from pairs_engine.kalman import KalmanHedge


def make_constant_beta_series(n=300, true_beta=1.5, true_alpha=0.1, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).cumsum()
    y = true_beta * x + true_alpha + rng.normal(0, 0.05, n)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(y, index=dates, name="Y"), pd.Series(x, index=dates, name="X")


def test_kalman_recovers_beta_approximately():
    y, x = make_constant_beta_series(n=500, true_beta=1.5)
    kf = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf.run(y, x)
    warmup = 50
    median_beta = df["beta"].iloc[warmup:].median()
    assert abs(median_beta - 1.5) < 0.15, f"Expected beta~1.5, got {median_beta:.4f}"


def test_kalman_spread_var_always_positive():
    y, x = make_constant_beta_series(n=100, true_beta=2.0)
    kf = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf.run(y, x)
    assert (df["spread_var"] > 0).all()


def test_kalman_step_matches_run():
    y, x = make_constant_beta_series(n=50, true_beta=1.2)
    kf1 = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf1.run(y, x)

    kf2 = KalmanHedge(delta=1e-4, R=1e-3)
    betas, spreads = [], []
    for yt, xt in zip(y, x):
        beta, alpha, spread, sv = kf2.step(yt, xt)
        betas.append(beta)
        spreads.append(spread)

    np.testing.assert_allclose(df["beta"].values, betas, rtol=1e-8)
    np.testing.assert_allclose(df["spread"].values, spreads, rtol=1e-8)


def test_kalman_run_output_columns():
    y, x = make_constant_beta_series(n=30)
    kf = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf.run(y, x)
    assert set(df.columns) == {"beta", "alpha", "spread", "spread_var"}
    assert len(df) == 30


def test_kalman_prior_override():
    y, x = make_constant_beta_series(n=100, true_beta=2.0)
    kf = KalmanHedge(delta=1e-4, R=1e-3)
    df = kf.run(y, x, prior_mean=np.array([2.0, 0.0]), prior_cov=np.eye(2) * 10.0)
    assert abs(df["beta"].iloc[5] - 2.0) < 0.3
