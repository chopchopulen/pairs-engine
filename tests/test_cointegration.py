import numpy as np
import pandas as pd
import pytest
from pairs_engine.cointegration import johansen_test, adf_spread_test, bh_fdr


def make_cointegrated_pair(n=500, beta=1.0, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).cumsum()
    y = beta * x + rng.normal(0, 0.5, n)
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
    assert result["rejects_no_coint"] is False


def test_johansen_returns_normalised_eigvec():
    df = make_cointegrated_pair(n=500)
    result = johansen_test(df)
    ev = result["eigvec_normalized"]
    assert ev.shape == (2,)
    assert abs(ev[0] - 1.0) < 1e-10


def test_johansen_raises_on_short_sample():
    df = make_cointegrated_pair(n=10)
    with pytest.raises(ValueError, match="too short"):
        johansen_test(df)


def test_adf_spread_stationary():
    rng = np.random.default_rng(0)
    spread = pd.Series(rng.normal(0, 1, 300))
    result = adf_spread_test(spread)
    assert result["pvalue"] < 0.05


def test_bh_fdr_single_pvalue():
    result = bh_fdr([0.01], alpha=0.05)
    assert result == [True]
    result_reject = bh_fdr([0.9], alpha=0.05)
    assert result_reject == [False]


def test_bh_fdr_multiple_pvalues():
    pvalues = [0.001, 0.01, 0.8]
    result = bh_fdr(pvalues, alpha=0.05)
    assert result[0] is True
    assert result[1] is True
    assert result[2] is False
