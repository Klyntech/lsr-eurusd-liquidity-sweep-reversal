"""
ADX (Average Directional Index) indicator.

Implements Wilder's ADX with directional movement (+DM / -DM), directional
indicators (+DI / -DI), and the ADX itself — all using Wilder's exponential
smoothing (alpha = 1/period).

Reference: J. Welles Wilder, *New Concepts in Technical Trading Systems* (1978).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: Wilder's EMA
# ---------------------------------------------------------------------------

def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Apply Wilder's smoothing (an EMA with alpha = 1/period) to a series.

    The first valid output is the simple moving average of the first ``period``
    values. Subsequent values are computed recursively:
        S(t) = S(t-1) * (period - 1)/period + value(t) / period

    Parameters
    ----------
    series : pd.Series
        Input series (may contain NaN).
    period : int
        Smoothing period; alpha is fixed at 1/period.

    Returns
    -------
    pd.Series
        Smoothed series. The first ``period - 1`` values are NaN.
    """
    if period < 1:
        raise ValueError("period must be a positive integer")

    arr = series.values.astype(np.float64)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    # Find the first `period` non-NaN values to use as seed.
    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) < period:
        # Not enough valid data — return all NaN
        return pd.Series(result, index=series.index)

    seed_positions = valid_idx[:period]
    # Check that the seed positions form a contiguous block starting at 0.
    # Wilder's method assumes seeding from the very beginning of the data.
    # If the first period values (positions 0..period-1) are all valid, use them.
    if np.all(valid_idx[:period] == np.arange(period)):
        seed_values = arr[:period]
        result[period - 1] = seed_values.mean()
        start_i = period
    else:
        # Fallback: use the first `period` valid values wherever they are.
        # This is a lenient approach to handle sparse data gracefully.
        seed_end_pos = seed_positions[-1]
        seed_values = arr[seed_positions[:period]]
        result[seed_end_pos] = seed_values.mean()
        start_i = seed_end_pos + 1

    # Recursive Wilder smoothing
    alpha = 1.0 / period
    for i in range(start_i, n):
        prev = result[i - 1]
        curr = arr[i]
        if np.isnan(prev) or np.isnan(curr):
            result[i] = np.nan
        else:
            result[i] = prev * (1.0 - alpha) + curr * alpha

    return pd.Series(result, index=series.index)


# ---------------------------------------------------------------------------
# Internal helpers for directional movement
# ---------------------------------------------------------------------------

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute True Range (see atr.py for details)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    return tr


def _directional_movement(
    high: pd.Series, low: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute +DM and -DM (directional movement).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (plus_dm, minus_dm) — first value is 0 for both.
    """
    up_move = high.diff()       # high(t) - high(t-1)
    down_move = -low.diff()    # low(t-1) - low(t)

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
        dtype=np.float64,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
        dtype=np.float64,
    )
    return plus_dm, minus_dm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    return_di: bool = False,
) -> pd.Series | Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute the Average Directional Index (ADX).

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
    return_di : bool, optional
        If True, return a tuple ``(adx, plus_di, minus_di)``.
        If False (default), return only the ADX series.

    Returns
    -------
    pd.Series or Tuple[pd.Series, pd.Series, pd.Series]
        ADX values (and optionally +DI, -DI). NaN for the first
        ``period * 2 - 2`` values.
    """
    if period < 1:
        raise ValueError("period must be a positive integer")
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have the same length")

    # --- Directional Movement ---
    plus_dm, minus_dm = _directional_movement(high, low)

    # --- True Range ---
    tr = _true_range(high, low, close)

    # --- Wilder-smoothed DM and TR ---
    smooth_plus_dm = _wilder_ema(plus_dm, period)
    smooth_minus_dm = _wilder_ema(minus_dm, period)
    smooth_tr = _wilder_ema(tr, period)

    # --- Directional Indicators (+DI / -DI) ---
    # Guard against division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = pd.Series(
            100.0 * smooth_plus_dm / smooth_tr,
            index=high.index,
            dtype=np.float64,
        )
        minus_di = pd.Series(
            100.0 * smooth_minus_dm / smooth_tr,
            index=high.index,
            dtype=np.float64,
        )

    # --- DX ---
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = pd.Series(
            100.0 * di_diff / di_sum,
            index=high.index,
            dtype=np.float64,
        )

    # --- ADX: Wilder-smoothed DX ---
    adx_series = _wilder_ema(dx, period)

    # Name the output series
    adx_series.name = "adx"
    plus_di.name = "plus_di"
    minus_di.name = "minus_di"

    if return_di:
        return adx_series, plus_di, minus_di
    return adx_series
