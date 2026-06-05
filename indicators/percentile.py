"""
Rolling Percentile Rank indicator.

Computes the percentile rank of the current value within a rolling window.
Used (for example) to determine where the current ATR falls relative to
recent history — a key filter in the LSR-EH1 strategy.

percentile = (count of values in window strictly less than current) / (window - 1) * 100

Pure numpy/pandas implementation — no scipy dependency.
"""

import numpy as np
import pandas as pd


def rolling_percentile(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute the rolling percentile rank of each value in the series.

    For each bar, the percentile is the proportion of values in the
    preceding ``window``-length window that are **strictly less** than the
    current value, scaled to 0–100.

        percentile(t) = count(x < x(t) for x in window(t)) / (window - 1) * 100

    Parameters
    ----------
    series : pd.Series
        Input series (e.g. ATR values).
    window : int, optional
        Rolling window size (default 20).

    Returns
    -------
    pd.Series
        Percentile rank (0.0 – 100.0). The first ``window`` values are NaN.

    Notes
    -----
    - If all values in the window are identical, the percentile is 0
      (nothing is *strictly* less).
    - If the current value is the maximum in the window, the percentile
      approaches 100 (specifically ``(window - 1) / (window - 1) * 100``).
    - NaN values in the input are treated as missing; if the current value
      is NaN the result is NaN.

    Examples
    --------
    >>> s = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    >>> rolling_percentile(s, window=5)
    0     NaN
    1     NaN
    2     NaN
    3     NaN
    4    100.0   # 20 is the max of [2,4,6,8,20]
    5    100.0
    6    100.0
    7    100.0
    8    100.0
    9    100.0
    dtype: float64
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    arr = series.values.astype(np.float64)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        # Not enough data — return all NaN
        return pd.Series(result, index=series.index, name="percentile")

    # ---- Vectorised approach using rolling + rank ----
    # pandas .rolling().rank() with method='min' gives integer ranks.
    # We want: count(strictly less) = rank - 1  (for min-rank, ties get same rank).
    # percentile = (rank - 1) / (window - 1) * 100
    s = series.astype(np.float64)
    ranks = s.rolling(window=window, min_periods=window).rank(method="min")

    with np.errstate(divide="ignore", invalid="ignore"):
        result = ((ranks - 1) / (window - 1)) * 100.0

    # Values before the first complete window stay NaN (rolling with
    # min_periods=window already produces NaN for those positions).
    return pd.Series(result.values.astype(np.float64), index=series.index, name="percentile")
