"""Risk/return metrics computed from an equity curve."""
from __future__ import annotations
import numpy as np

CANDLES_PER_YEAR = 365 * 24  # 8760 — hourly candles


def log_returns(equity: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.maximum(equity.astype(float), 1e-6)))


def sharpe_ratio(equity: np.ndarray, periods_per_year: int = CANDLES_PER_YEAR,
                 eps: float = 1e-8) -> float:
    if len(equity) < 2:
        return 0.0
    rets = log_returns(equity)
    if rets.std() < eps:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


def sortino_ratio(equity: np.ndarray, periods_per_year: int = CANDLES_PER_YEAR,
                  eps: float = 1e-8) -> float:
    if len(equity) < 2:
        return 0.0
    rets = log_returns(equity)
    downside = rets[rets < 0]
    if len(downside) == 0 or downside.std() < eps:
        return 0.0
    return float(rets.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max())


def total_return(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity[-1] / equity[0] - 1.0)
