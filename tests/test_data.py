import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from pairs_engine.data import align_and_clean, load_or_fetch


def make_raw(dates, gld_prices, iau_prices):
    df = pd.DataFrame({"GLD": gld_prices, "IAU": iau_prices}, index=pd.DatetimeIndex(dates))
    return df


def test_align_and_clean_drops_nan_rows():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    df = make_raw(dates, [100, np.nan, 102, 103, 104], [10, 10.1, np.nan, 10.3, 10.4])
    result = align_and_clean(df)
    assert result.isna().sum().sum() == 0
    assert len(result) == 3


def test_align_and_clean_returns_log_prices():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    df = make_raw(dates, [100.0, 101.0, 102.0], [10.0, 10.1, 10.2])
    result = align_and_clean(df)
    expected_gld_0 = np.log(100.0)
    assert abs(result["GLD"].iloc[0] - expected_gld_0) < 1e-10


def test_align_and_clean_no_ffill():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    df = make_raw(dates, [100, np.nan, 102, 103], [10, 10.1, 10.2, 10.3])
    result = align_and_clean(df)
    assert len(result) == 3


def test_load_or_fetch_caches(tmp_path):
    df = load_or_fetch(["GLD", "IAU"], start="2024-01-01", end="2024-01-31", cache_dir=str(tmp_path))
    assert not df.empty
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1

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
