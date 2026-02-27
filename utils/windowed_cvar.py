"""
Windowed CVaR Computation Utility (no external RL dependencies).

Computes CVaR of blocking rates over sliding time windows.
Used for post-hoc evaluation of agent performance during bursts.
"""
import numpy as np
from typing import Dict


def compute_windowed_cvar(
    timestamps: np.ndarray,
    blocked: np.ndarray,
    window_s: float = 900.0,
    stride_s: float = 300.0,
    alpha: float = 0.1,
) -> Dict[str, float]:
    """
    Compute CVaR of blocking rate over sliding windows.

    Parameters
    ----------
    timestamps : array, shape (N,)
        Arrival times of each service.
    blocked : array, shape (N,)
        1 if blocked, 0 if accepted.
    window_s : float
        Window size in seconds.
    stride_s : float
        Stride between windows.
    alpha : float
        CVaR risk level (e.g., 0.1 → worst 10 % of windows).

    Returns
    -------
    dict with keys: window_mean, window_var, window_cvar, n_windows,
                    window_rates (array of per-window blocking rates).
    """
    if len(timestamps) < 2:
        return {'window_mean': 0, 'window_var': 0, 'window_cvar': 0,
                'n_windows': 0, 'window_rates': np.array([])}

    t_start = timestamps.min()
    t_end = timestamps.max()
    window_rates = []

    t = t_start
    while t + window_s <= t_end:
        mask = (timestamps >= t) & (timestamps < t + window_s)
        n = mask.sum()
        if n > 0:
            window_rates.append(float(blocked[mask].sum()) / n)
        t += stride_s

    if len(window_rates) == 0:
        return {'window_mean': 0, 'window_var': 0, 'window_cvar': 0,
                'n_windows': 0, 'window_rates': np.array([])}

    rates = np.array(window_rates)
    var_val = float(np.quantile(rates, 1 - alpha))  # upper tail for blocking
    tail = rates[rates >= var_val]
    cvar_val = float(tail.mean()) if len(tail) else var_val

    return {
        'window_mean': float(rates.mean()),
        'window_var': var_val,
        'window_cvar': cvar_val,
        'n_windows': len(rates),
        'window_rates': rates,
    }
