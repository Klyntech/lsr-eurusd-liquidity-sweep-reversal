"""
Liquidity Sweep Reversal Strategy (LSR-EH1)

Core signal generation for a 1-hour EURUSD mean-reversion system that
targets liquidity sweeps below swing lows / above swing highs, confirmed
by elevated volatility (high ATR percentile) and low trend strength
(low ADX).

Design principles
-----------------
* Zero look-ahead bias — every calculation uses only current and past data.
* Raw signal only — this module does NOT track position state, cooling
  periods, or risk limits.  Those are the backtest engine's responsibility.
"""

from typing import List

import numpy as np
import pandas as pd

from config import (
    CostParams,
    ExitParams,
    FilterParams,
    IndicatorParams,
    NewsBlackout,
    RiskParams,
    StrategyConfig,
    SwingParams,
)
from indicators.adx import adx
from indicators.atr import atr
from indicators.percentile import rolling_percentile


# ---------------------------------------------------------------------------
# Swing Detection
# ---------------------------------------------------------------------------

def detect_swing_high(high: pd.Series, lookback: int = 5) -> pd.Series:
    """
    Identify swing highs using a rolling-maximum criterion with no look-ahead.

    A bar *t* is a swing high when:

        1. ``high[t]`` is the **maximum** of ``high[t-lookback : t+1]``
           (i.e. the highest of the last ``lookback + 1`` bars).
        2. ``high[t] > high[t-1]``
        3. ``high[t] > high[t-2]``

    Parameters
    ----------
    high : pd.Series
        Bar high prices.
    lookback : int, optional
        Number of bars to look back for the rolling-max check (default 5).

    Returns
    -------
    pd.Series
        Boolean series — ``True`` where a swing high is detected.
    """
    rolling_max = high.rolling(window=lookback + 1, min_periods=lookback + 1).max()
    return (
        (high == rolling_max)
        & (high > high.shift(1))
        & (high > high.shift(2))
    )


def detect_swing_low(low: pd.Series, lookback: int = 5) -> pd.Series:
    """
    Identify swing lows using a rolling-minimum criterion with no look-ahead.

    A bar *t* is a swing low when:

        1. ``low[t]`` is the **minimum** of ``low[t-lookback : t+1]``
        2. ``low[t] < low[t-1]``
        3. ``low[t] < low[t-2]``

    Parameters
    ----------
    low : pd.Series
        Bar low prices.
    lookback : int, optional
        Number of bars to look back for the rolling-min check (default 5).

    Returns
    -------
    pd.Series
        Boolean series — ``True`` where a swing low is detected.
    """
    rolling_min = low.rolling(window=lookback + 1, min_periods=lookback + 1).min()
    return (
        (low == rolling_min)
        & (low < low.shift(1))
        & (low < low.shift(2))
    )


def most_recent_swing_value(
    series: pd.Series,
    swing_mask: pd.Series,
    window: int = 20,
    mode: str = "high",
) -> pd.Series:
    """
    For each bar, find the value of the most recent swing point within a
    rolling look-back window **excluding the current bar**.

    The current bar is excluded because including it creates a circular
    dependency: a bar that sweeps below the liquidity level would also
    register as a new swing low, moving the liquidity level below the
    sweep and making the signal condition self-defeating.

    By looking only at bars [t-window, t-1], the liquidity level is
    determined from prior data, and the current bar can legitimately
    sweep below it.

    If **no** swing point exists inside the window, a fallback is used:

    * ``mode="high"`` → fallback is ``series.max()`` over the window
    * ``mode="low"``  → fallback is ``series.min()`` over the window

    Parameters
    ----------
    series : pd.Series
        The price series (``high`` for swing-highs, ``low`` for swing-lows).
    swing_mask : pd.Series
        Boolean mask that is ``True`` at swing-point locations.
    window : int, optional
        Look-back window in bars (default 20).
    mode : str, optional
        ``"high"`` or ``"low"`` — determines the fallback behaviour.

    Returns
    -------
    pd.Series
        Most-recent swing value (or fallback) for every bar.
    """
    n = len(series)
    result = np.full(n, np.nan, dtype=np.float64)

    vals = series.values.astype(np.float64)
    mask = swing_mask.values.astype(bool)

    for i in range(n):
        # Exclude current bar: look at [i-window, i-1]
        end = i  # exclusive, so this goes up to i-1
        start = max(0, end - window)
        if start >= end:
            continue

        segment_mask = mask[start : end]
        segment_vals = vals[start : end]

        # Find positions of swing points within the window
        swing_positions = np.where(segment_mask)[0]

        if len(swing_positions) > 0:
            # Most recent swing = the one closest to i (last in segment)
            most_recent_pos = swing_positions[-1]
            result[i] = segment_vals[most_recent_pos]
        else:
            # Fallback: window extreme (excluding current bar)
            if mode == "high":
                result[i] = np.nanmax(segment_vals)
            elif mode == "low":
                result[i] = np.nanmin(segment_vals)
            else:
                raise ValueError(f"mode must be 'high' or 'low', got {mode!r}")

    return pd.Series(result, index=series.index, name=f"recent_swing_{mode}")


# ---------------------------------------------------------------------------
# ADX alignment helper
# ---------------------------------------------------------------------------

def _compute_adx_aligned_to_1h(
    df: pd.DataFrame,
    adx_period: int,
) -> pd.Series:
    """
    Compute ADX on 4-hour bars, then forward-fill align to the 1-hour index.

    Steps
    -----
    1. Resample the 1h DataFrame to 4h using last-close / max-high / min-low.
    2. Run ``adx()`` on the 4h bars.
    3. Reindex the ADX series back to the original 1h index and forward-fill.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close`` columns with a DatetimeIndex.
    adx_period : int
        ADX period (default from config = 14).

    Returns
    -------
    pd.Series
        ADX values forward-filled to the 1h frequency, same index as ``df``.
    """
    # Build a temporary 4h DataFrame
    h4 = pd.DataFrame(
        {
            "high": df["high"].resample("4h").max(),
            "low": df["low"].resample("4h").min(),
            "close": df["close"].resample("4h").last(),
        },
    )
    # Drop rows that became NaN after resampling (e.g. gaps)
    h4 = h4.dropna(subset=["high", "low", "close"])

    adx_4h = adx(h4["high"], h4["low"], h4["close"], period=adx_period)

    # Reindex to 1h and forward-fill so every 1h bar gets the latest 4h ADX
    adx_1h = adx_4h.reindex(df.index).ffill()
    adx_1h.name = "adx_4h"
    return adx_1h


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Compute raw entry signals for the LSR-EH1 strategy.

    This function adds the following columns to ``df`` **in-place and returns
    the same DataFrame**:

    +---------------------+---------------------------------------------------+
    | Column              | Description                                       |
    +=====================+===================================================+
    | ``swing_high_mask`` | Boolean — swing high detected                     |
    | ``swing_low_mask``  | Boolean — swing low detected                      |
    | ``recent_swing_high``| Float — most recent swing high price (or window   |
    |                      | max fallback)                                      |
    | ``recent_swing_low`` | Float — most recent swing low price (or window    |
    |                      | min fallback)                                      |
    | ``atr``              | Float — ATR(14) on 1h bars                        |
    | ``atr_percentile``   | Float — rolling percentile of ATR                 |
    | ``adx_4h``          | Float — ADX(14) on 4h bars, ffill-aligned to 1h  |
    | ``liquidity_long``  | Float — recent_swing_low - atr_offset * ATR      |
    | ``liquidity_short`` | Float — recent_swing_high + atr_offset * ATR     |
    | ``regime_ok``       | Boolean — trend filter (low ADX) AND vol filter  |
    |                      | (high ATR percentile)                             |
    | ``long_signal``     | Boolean — raw long entry condition                 |
    | ``short_signal``    | Boolean — raw short entry condition                |
    +---------------------+---------------------------------------------------+

    **Responsibility boundary:** This function computes *only* the raw boolean
    signals.  Position tracking, cooling-off periods, risk-limit checks, and
    news-blackout filtering are handled by the backtest engine.

    Parameters
    ----------
    df : pd.DataFrame
        1h OHLC data with columns ``open``, ``high``, ``low``, ``close`` and a
        :class:`pandas.DatetimeIndex`.
    config : StrategyConfig
        Full strategy configuration.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with signal columns appended.
    """
    ind: IndicatorParams = config.indicator
    swg: SwingParams = config.swing
    flt: FilterParams = config.filter

    # ------------------------------------------------------------------
    # 1. ATR on 1h bars
    # ------------------------------------------------------------------
    df["atr"] = atr(df["high"], df["low"], df["close"], period=ind.atr_period)

    # ------------------------------------------------------------------
    # 2. ADX on 4h bars, aligned to 1h
    # ------------------------------------------------------------------
    df["adx_4h"] = _compute_adx_aligned_to_1h(df, adx_period=ind.adx_period)

    # ------------------------------------------------------------------
    # 3. ATR rolling percentile
    # ------------------------------------------------------------------
    df["atr_percentile"] = rolling_percentile(
        df["atr"], window=ind.atr_percentile_window
    )

    # ------------------------------------------------------------------
    # 4. Swing detection
    # ------------------------------------------------------------------
    df["swing_high_mask"] = detect_swing_high(df["high"], lookback=swg.lookback)
    df["swing_low_mask"] = detect_swing_low(df["low"], lookback=swg.lookback)

    # ------------------------------------------------------------------
    # 5. Most recent swing values
    # ------------------------------------------------------------------
    df["recent_swing_high"] = most_recent_swing_value(
        df["high"], df["swing_high_mask"], window=swg.swing_window, mode="high"
    )
    df["recent_swing_low"] = most_recent_swing_value(
        df["low"], df["swing_low_mask"], window=swg.swing_window, mode="low"
    )

    # ------------------------------------------------------------------
    # 6. Liquidity levels
    # ------------------------------------------------------------------
    df["liquidity_long"] = df["recent_swing_low"] - swg.atr_offset * df["atr"]
    df["liquidity_short"] = df["recent_swing_high"] + swg.atr_offset * df["atr"]

    # ------------------------------------------------------------------
    # 7. Regime filter
    # ------------------------------------------------------------------
    #   trend_ok  : ADX < threshold  (weak / no trend)
    #   vol_ok    : ATR percentile > minimum  (elevated volatility)
    trend_ok = df["adx_4h"] < flt.adx_threshold
    vol_ok = df["atr_percentile"] > ind.atr_percentile_min
    df["regime_ok"] = trend_ok & vol_ok

    # ------------------------------------------------------------------
    # 8. Raw signals
    # ------------------------------------------------------------------
    # Long:  price sweeps *below* the liquidity level but closes *above* it
    # Short: price sweeps *above* the liquidity level but closes *below* it
    df["long_signal"] = (
        (df["low"] < df["liquidity_long"])
        & (df["close"] > df["liquidity_long"])
        & df["regime_ok"]
    )
    df["short_signal"] = (
        (df["high"] > df["liquidity_short"])
        & (df["close"] < df["liquidity_short"])
        & df["regime_ok"]
    )

    return df


# ---------------------------------------------------------------------------
# News Blackout
# ---------------------------------------------------------------------------

def is_news_blackout(
    timestamp: pd.Timestamp,
    blacklist: List[NewsBlackout],
) -> bool:
    """
    Check whether a bar's open time falls within any scheduled news blackout
    window.

    Each blackout window is defined as::

        [release_utc_hour - blackout_minutes_before,
         release_utc_hour + blackout_minutes_after]

    The check uses the bar's **hour + minute fraction** on the UTC clock.

    Parameters
    ----------
    timestamp : pd.Timestamp
        The bar's open time (should be tz-aware UTC or naive treated as UTC).
    blacklist : list[NewsBlackout]
        Pre-configured list of blackout events.

    Returns
    -------
    bool
        ``True`` if the timestamp is inside *any* blackout window.
    """
    # Hour as a float: e.g. 12:30 → 12.5
    bar_hour = timestamp.hour + timestamp.minute / 60.0

    for event in blacklist:
        start = event.blackout_start_utc_hour()
        end = event.blackout_end_utc_hour()
        if start <= bar_hour < end:
            return True

    return False
