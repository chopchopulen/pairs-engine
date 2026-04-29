from __future__ import annotations
import numpy as np
import pandas as pd


def apply_costs(positions: pd.Series, beta: pd.Series, bps: float = 5) -> pd.Series:
    """
    Compute transaction costs on position changes.

    Dollar-neutral: Y-leg unit notional, X-leg beta notional.
    cost_t = bps/10000 * (|delta_pos_y_t| + beta_t * |delta_pos_x_t|)

    Since X is traded in beta units, X cost = bps/10000 * beta_t * |delta_pos_t|.
    Charged on the bar where the position changes.
    """
    rate = bps / 10_000
    delta_pos = positions.diff().fillna(positions.iloc[0] if len(positions) else 0)
    abs_delta = delta_pos.abs()
    return rate * (abs_delta + beta * abs_delta)
