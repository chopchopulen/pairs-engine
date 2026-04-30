from __future__ import annotations
import numpy as np
import pandas as pd


def compute_zscore(spread: pd.Series, spread_var: pd.Series) -> pd.Series:
    """z_t = innovation_t / sqrt(innovation_variance_t). Uses Kalman's own variance."""
    return spread / np.sqrt(spread_var)


def generate_positions(z: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5) -> pd.Series:
    """
    State machine over z-scores. Position determined at close of bar t,
    applied to returns t → t+1 (caller must lag before return multiplication).

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
