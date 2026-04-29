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
    assert abs(z.iloc[0] - 2.0) < 1e-10


def test_generate_positions_enters_short_on_high_z():
    spread, sv = make_filter_output([0, 0, 2.5, 2.5, 0.3], [1, 1, 1, 1, 1])
    z = compute_zscore(spread, sv)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
    assert pos.iloc[2] == -1
    assert pos.iloc[3] == -1
    assert pos.iloc[4] == 0


def test_generate_positions_enters_long_on_low_z():
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
    spread, sv = make_filter_output([0, 2.5, 1.2, 0.6, 0.3], [1, 1, 1, 1, 1])
    z = compute_zscore(spread, sv)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
    assert pos.iloc[1] == -1
    assert pos.iloc[2] == -1
    assert pos.iloc[3] == -1
    assert pos.iloc[4] == 0


def test_generate_positions_flip_short_to_long():
    spread, sv = make_filter_output([2.5, 0.3, -2.5], [1, 1, 1])
    z = compute_zscore(spread, sv)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
    assert pos.iloc[0] == -1
    assert pos.iloc[1] == 0
    assert pos.iloc[2] == 1
