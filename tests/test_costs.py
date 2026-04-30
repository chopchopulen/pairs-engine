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
    pos, beta = make_positions([0, -1, -1], betas=[1.0, 1.0, 1.0])
    costs = apply_costs(pos, beta, bps=5)
    # delta_pos=1, beta=1.0 → cost = 5/10000 * (1 + 1) = 0.001
    assert abs(costs.iloc[1] - 0.001) < 1e-10
    assert costs.iloc[0] == 0.0
    assert costs.iloc[2] == 0.0


def test_cost_on_flip():
    pos, beta = make_positions([-1, 1], betas=[1.0, 1.0])
    costs = apply_costs(pos, beta, bps=5)
    # delta=2, beta=1 → cost = 5/10000 * (2 + 2) = 0.002
    assert abs(costs.iloc[1] - 0.002) < 1e-10


def test_cost_scales_with_beta():
    pos, beta = make_positions([0, -1], betas=[1.0, 2.0])
    costs = apply_costs(pos, beta, bps=5)
    # delta=1, beta=2 → cost = 5/10000 * (1 + 2) = 0.0015
    assert abs(costs.iloc[1] - 0.0015) < 1e-10
