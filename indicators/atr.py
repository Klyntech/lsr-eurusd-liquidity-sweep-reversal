"""
ATR (Average True Range) indicator.

Implements Wilder's smoothing method for computing Average True Range,
which measures market volatility by decomposing the true range of each bar.

TrueRange(t) = max( high(t)-low(t), |high(t)-close(t-1)|, |low(t)-close(t-1)| )
ATR(t) = (ATR(t-1) * (period-1) + TrueRange(t)) / period

First ATR value is seeded as the simple moving average of the first `period`
true ranges, then subsequent values use Wilder's recursive formula.
"""

from typing import Optional

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Compute the True Range component for each bar.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices (shifted internally for previous close).

    Returns
    -------
    pd.Series
        True Range series. First value is NaN (no prior close available).
    """
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    close = close.astype(np.float64)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # First row has no prior close → NaN
    tr.iloc[0] = np.nan
    return tr


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR) using Wilder's smoothing.

    The first ``period`` true ranges are averaged to seed the initial ATR value.
    From index ``period`` onward, the recursive Wilder formula is used:
        ATR(t) = (ATR(t-1) * (period - 1) + TR(t)) / period

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, optional
        Smoothing period (default 14).

    Returns
    -------
    pd.Series
        ATR series, same length as input. The first ``period - 1`` values
        are NaN.
    """
    if period < 1:
        raise ValueError("period must be a positive integer")
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have the same length")

    tr = true_range(high, low, close)

    # Use numpy for the computation, then wrap in Series
    tr_arr = tr.values.astype(np.float64)
    n = len(tr_arr)

    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        # Not enough data — return all NaN
        return pd.Series(result, index=high.index, name="atr")

    # Seed: SMA of first `period` true ranges (indices 1..period inclusive)
    seed_start = 1  # tr[0] is NaN
    seed_end = seed_start + period
    seed_slice = tr_arr[seed_start:seed_end]
    if np.any(np.isnan(seed_slice)):
        return pd.Series(result, index=high.index, name="atr")

    result[seed_end - 1] = seed_slice.mean()

    # Wilder smoothing from seed_end onward
    alpha = 1.0 / period
    for i in range(seed_end, n):
        prev = result[i - 1]
        curr_tr = tr_arr[i]
        if np.isnan(prev) or np.isnan(curr_tr):
            result[i] = np.nan
        else:
            result[i] = prev * (1.0 - alpha) + curr_tr * alpha

    return pd.Series(result, index=high.index, name="atr")
