from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

_MIN_OBS = 52


def johansen_test(log_prices: pd.DataFrame) -> dict:
    """
    Johansen trace test on a 2-column log-price DataFrame.
    det_order=0: constant in cointegrating relation.
    k_ar_diff=1: one lag of differences.
    rejects_no_coint=True when trace stat > 95% critical value.
    """
    if len(log_prices) < _MIN_OBS:
        raise ValueError(f"Sample too short for Johansen: {len(log_prices)} obs — too short")
    result = coint_johansen(log_prices.values, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1[0]
    crit_95 = result.cvt[0, 1]
    rejects = bool(trace_stat > crit_95)
    ev = result.evec[:, 0]
    ev_normalised = ev / ev[0]
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
    Benjamini-Hochberg FDR correction.
    Returns True = reject null (cointegrated).
    Dormant for single-pair use.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    sorted_p = [pvalues[i] for i in order]
    rejected = [False] * m
    for k, (idx, p) in enumerate(zip(order, sorted_p), start=1):
        if p <= alpha * k / m:
            rejected[idx] = True
        else:
            break
    return rejected
