from __future__ import annotations
import numpy as np
import pandas as pd


class KalmanHedge:
    """
    State-space Kalman filter estimating dynamic hedge ratio and intercept.

    State:  theta_t = [beta_t, alpha_t]^T
    Transition: theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q),  Q = delta * I
    Observation: y_t = H_t @ theta_t + v_t,  v_t ~ N(0, R),  H_t = [x_t, 1]

    The spread returned is the innovation (prediction error BEFORE update),
    not the post-update residual — avoids look-ahead in signal generation.

    delta and R are global structural constants; never tuned per window.
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-3) -> None:
        self.delta = delta
        self.R = R
        self._Q = delta * np.eye(2)
        self._reset()

    def _reset(self) -> None:
        self._theta = np.zeros(2)
        self._P = np.eye(2)

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
        Innovation uses the predicted state (before update) — no look-ahead.
        """
        theta_pred = self._theta
        P_pred = self._P + self._Q

        H = np.array([x_t, 1.0])

        innovation = y_t - H @ theta_pred
        S = float(H @ P_pred @ H + self.R)

        K = P_pred @ H / S

        self._theta = theta_pred + K * innovation
        self._P = (np.eye(2) - np.outer(K, H)) @ P_pred

        return float(self._theta[0]), float(self._theta[1]), float(innovation), S

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
