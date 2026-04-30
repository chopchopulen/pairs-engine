from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path


def _download_raw(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
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
        df = pd.read_parquet(cache_path)
        df.index = df.index.astype("datetime64[s]")
        return df
    raw = _download_raw(tickers, start, end)
    result = align_and_clean(raw)
    result.index = result.index.astype("datetime64[s]")
    result.to_parquet(cache_path)
    return result
