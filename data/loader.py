"""
Dukascopy Tick Data Loader for LSR-EH1.

Loads Dukascopy CSV tick data and aggregates to OHLC bars, with support
for pre-aggregated OHLC CSV files.  Handles both bid+ask and bid-only
formats, estimates spread when unavailable, and validates output.

Typical usage::

    from data.loader import load_dukascopy_ticks, load_ohlc_csv

    # From raw ticks
    df = load_dukascopy_ticks("EURUSD_ticks.csv", timeframe="1H",
                             start_date="2019-01-01", end_date="2024-01-01")

    # From pre-aggregated OHLC
    df = load_ohlc_csv("EURUSD_1H.csv", start_date="2019-01-01")
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Pandas resampling alias mapping
# ---------------------------------------------------------------------------

_TIMEFRAME_MAP: dict = {
    "1T": "1min",
    "5T": "5min",
    "15T": "15min",
    "30T": "30min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1D",
    "1W": "1W",
    "1M": "1ME",
}

_VALID_TIMEFRAMES = set(_TIMEFRAME_MAP.keys())


# ---------------------------------------------------------------------------
# Public API — tick loader
# ---------------------------------------------------------------------------

def load_dukascopy_ticks(
    filepath: str,
    timeframe: str = "1H",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load Dukascopy tick data and aggregate to OHLC.

    Dukascopy CSV format::

        timestamp,bid,ask
        2019.01.02 00:00:00.000,1.13004,1.13006

    Also supports bid-only format (no ``ask`` column).

    Parameters
    ----------
    filepath : str
        Path to the Dukascopy CSV file.
    timeframe : str
        Aggregation timeframe.  Supported: ``1T``, ``5T``, ``15T``,
        ``30T``, ``1H``, ``4H``, ``1D``, ``1W``, ``1M``.
    start_date, end_date : str, optional
        Date filters in ``YYYY-MM-DD`` format (inclusive).

    Returns
    -------
    pd.DataFrame
        OHLC DataFrame with a :class:`pandas.DatetimeIndex` and columns:
        ``open, high, low, close, spread``.

    Raises
    ------
    ValueError
        If the CSV cannot be parsed or *timeframe* is unsupported.
    """
    # ---- Validate timeframe ----
    if timeframe not in _VALID_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported: {sorted(_VALID_TIMEFRAMES)}"
        )

    # ---- Read CSV ----
    raw = pd.read_csv(filepath, parse_dates=["timestamp"])

    if "timestamp" not in raw.columns:
        raise ValueError(
            "CSV must contain a 'timestamp' column. "
            "Found columns: " + ", ".join(raw.columns)
        )

    # Ensure timestamp is tz-naive DatetimeIndex
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=False)
    raw = raw.set_index("timestamp")

    # Remove rows with NaT index
    raw = raw[raw.index.notna()]

    if len(raw) == 0:
        raise ValueError("No valid rows found after parsing timestamps.")

    # ---- Detect columns ----
    has_bid = "bid" in raw.columns
    has_ask = "ask" in raw.columns

    if not has_bid:
        raise ValueError(
            "CSV must contain at least a 'bid' column. "
            "Found columns: " + ", ".join(raw.columns)
        )

    # Convert to float64
    for col in ("bid", "ask"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["bid"])

    if len(raw) == 0:
        raise ValueError("No valid bid values found in tick data.")

    # ---- Date filter ----
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        raw = raw[raw.index >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        raw = raw[raw.index < end_ts]

    if len(raw) == 0:
        raise ValueError(
            "No data remaining after date filtering. "
            f"start={start_date}, end={end_date}"
        )

    # ---- Sort index ----
    raw = raw.sort_index()

    # ---- Aggregate ----
    df = _aggregate_ticks_to_ohlc(raw, timeframe=timeframe)

    # ---- Estimate spread if all NaN ----
    if "spread" in df.columns and df["spread"].isna().all():
        df = estimate_spread(df)

    # ---- Validate ----
    is_valid, msg = validate_ohlc(df)
    if not is_valid:
        raise ValueError(f"OHLC validation failed: {msg}")

    return df


# ---------------------------------------------------------------------------
# Public API — OHLC loader
# ---------------------------------------------------------------------------

def load_ohlc_csv(
    filepath: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load pre-aggregated OHLC CSV.

    Supports flexible column naming for the timestamp/date column
    (``timestamp``, ``datetime``, ``date``) and optional ``spread``, ``bid``,
    ``ask`` columns.

    Parameters
    ----------
    filepath : str
        Path to the OHLC CSV file.
    start_date, end_date : str, optional
        Date filters in ``YYYY-MM-DD`` format (inclusive).

    Returns
    -------
    pd.DataFrame
        OHLC DataFrame with :class:`pandas.DatetimeIndex` and columns
        ``open, high, low, close, spread``.
    """
    # ---- Read ----
    raw = pd.read_csv(filepath)

    if len(raw) == 0:
        raise ValueError("CSV file is empty.")

    # ---- Detect timestamp column ----
    ts_col: Optional[str] = None
    for candidate in ("timestamp", "datetime", "date"):
        if candidate in raw.columns:
            ts_col = candidate
            break

    if ts_col is None:
        # Maybe the first column is the index (unnamed: 0)
        if raw.columns[0].startswith("Unnamed"):
            ts_col = raw.columns[0]
        else:
            raise ValueError(
                "Could not find timestamp column. "
                "Expected 'timestamp', 'datetime', or 'date'. "
                "Found columns: " + ", ".join(raw.columns)
            )

    raw[ts_col] = pd.to_datetime(raw[ts_col], utc=False)
    raw = raw.set_index(ts_col)
    raw = raw[raw.index.notna()]

    if len(raw) == 0:
        raise ValueError("No valid rows after parsing timestamps.")

    # ---- Standardise column names to lowercase ----
    raw.columns = [c.strip().lower() for c in raw.columns]

    # ---- Check required OHLC columns ----
    required = {"open", "high", "low", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Missing required OHLC columns: {missing}. "
            f"Found: {sorted(raw.columns)}"
        )

    # ---- Convert OHLC to float64 ----
    for col in ("open", "high", "low", "close", "bid", "ask", "spread"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # ---- Drop rows with NaN OHLC ----
    raw = raw.dropna(subset=["open", "high", "low", "close"])

    # ---- Build spread column ----
    if "spread" not in raw.columns:
        if "bid" in raw.columns and "ask" in raw.columns:
            raw["spread"] = raw["ask"] - raw["bid"]
        else:
            raw["spread"] = np.nan

    # ---- Keep only target columns ----
    cols_out = ["open", "high", "low", "close", "spread"]
    df = raw[cols_out].copy()

    # ---- Date filter ----
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        df = df[df.index >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df = df[df.index < end_ts]

    if len(df) == 0:
        raise ValueError(
            "No data remaining after date filtering. "
            f"start={start_date}, end={end_date}"
        )

    # ---- Sort index ----
    df = df.sort_index()

    # ---- Estimate spread if all NaN ----
    if df["spread"].isna().all():
        df = estimate_spread(df)

    # ---- Validate ----
    is_valid, msg = validate_ohlc(df)
    if not is_valid:
        raise ValueError(f"OHLC validation failed: {msg}")

    return df


# ---------------------------------------------------------------------------
# Tick aggregation
# ---------------------------------------------------------------------------

def _aggregate_ticks_to_ohlc(
    ticks: pd.DataFrame,
    timeframe: str = "1H",
) -> pd.DataFrame:
    """Aggregate tick data to OHLC bars.

    For each bar:
    - ``open``  = first tick's mid price (or bid if no ask)
    - ``high``  = max tick's mid price
    - ``low``   = min tick's mid price
    - ``close`` = last tick's mid price
    - ``spread`` = mean ask-bid spread across ticks in bar

    If only ``bid`` is available, OHLC uses bid directly and spread is NaN.

    Parameters
    ----------
    ticks : pd.DataFrame
        Tick data with DatetimeIndex and at minimum a ``bid`` column.
        May also have an ``ask`` column.
    timeframe : str
        Pandas resampling rule (e.g. ``"1H"``, ``"4H"``).

    Returns
    -------
    pd.DataFrame
        OHLC DataFrame with DatetimeIndex and columns
        ``open, high, low, close, spread``.
    """
    has_ask = "ask" in ticks.columns

    # Compute mid price (or use bid if no ask)
    if has_ask:
        mid = (ticks["bid"] + ticks["ask"]) / 2.0
    else:
        mid = ticks["bid"]

    # Compute per-tick spread
    if has_ask:
        tick_spread = ticks["ask"] - ticks["bid"]
    else:
        tick_spread = pd.Series(np.nan, index=ticks.index)

    # Build a temporary DataFrame for resampling
    tmp = pd.DataFrame({"mid": mid, "spread": tick_spread}, index=ticks.index)

    # Map to a valid pandas resampling rule
    rule = _TIMEFRAME_MAP.get(timeframe, timeframe)

    ohlc = tmp.resample(rule).agg(
        open=("mid", "first"),
        high=("mid", "max"),
        low=("mid", "min"),
        close=("mid", "last"),
        spread=("spread", "mean"),
    )

    # Drop bars where open is NaN (empty periods)
    ohlc = ohlc.dropna(subset=["open"])

    # Flatten MultiIndex columns if pandas created them
    if isinstance(ohlc.columns, pd.MultiIndex):
        ohlc.columns = ohlc.columns.get_level_values(1)

    return ohlc


# ---------------------------------------------------------------------------
# Spread estimation (last resort)
# ---------------------------------------------------------------------------

def estimate_spread(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Estimate spread from bar data using ATR as a rough proxy.

    Uses ``spread_estimate = 0.1 * ATR(14)`` as a fallback when real spread
    data is unavailable.  This is a **LAST RESORT** — real spread data is
    always preferred.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with columns ``open, high, low, close``.
    window : int
        Rolling window for ATR calculation (default 20).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with a ``spread`` column added/overwritten.
    """
    # Compute ATR(14)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )

    # Rolling mean for ATR (simple, no EMA — avoids needing warm-up logic)
    atr_series = pd.Series(tr, index=df.index).rolling(
        window=14, min_periods=14
    ).mean()

    # Spread estimate = 0.1 × ATR
    spread_est = 0.1 * atr_series

    df = df.copy()
    df["spread"] = spread_est
    return df


# ---------------------------------------------------------------------------
# OHLC validation
# ---------------------------------------------------------------------------

def validate_ohlc(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate an OHLC DataFrame for backtesting readiness.

    Checks:
    1. Required columns exist (``open, high, low, close``).
    2. :class:`pandas.DatetimeIndex` present and sorted ascending.
    3. No NaN values in OHLC columns.
    4. ``high >= max(open, close)`` for all bars.
    5. ``low <= min(open, close)`` for all bars.
    6. ``high >= low`` for all bars.
    7. All prices positive.
    8. Minimum 500 bars.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[bool, str]
        ``(is_valid, message)`` — *is_valid* is ``True`` when all checks pass.
    """
    # 1. Required columns
    required = {"open", "high", "low", "close"}
    cols = set(df.columns)
    missing = required - cols
    if missing:
        return False, f"Missing required columns: {missing}"

    # 2. DatetimeIndex and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "DataFrame must have a DatetimeIndex."
    if not df.index.is_monotonic_increasing:
        return False, "DatetimeIndex is not sorted in ascending order."

    # 3. No NaN in OHLC
    for col in ("open", "high", "low", "close"):
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            return False, f"Column '{col}' contains {n_nan} NaN value(s)."

    # Convert to numpy for fast vectorised checks
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)

    # 4. high >= max(open, close)
    bar_max = np.maximum(o, c)
    violations_hc = np.sum(h < bar_max)
    if violations_hc > 0:
        return False, (
            f"{violations_hc} bar(s) where high < max(open, close). "
            f"First violation at index {np.argmax(h < bar_max)}."
        )

    # 5. low <= min(open, close)
    bar_min = np.minimum(o, c)
    violations_lc = np.sum(l > bar_min)
    if violations_lc > 0:
        return False, (
            f"{violations_lc} bar(s) where low > min(open, close). "
            f"First violation at index {np.argmax(l > bar_min)}."
        )

    # 6. high >= low
    violations_hl = np.sum(h < l)
    if violations_hl > 0:
        return False, (
            f"{violations_hl} bar(s) where high < low. "
            f"First violation at index {np.argmax(h < l)}."
        )

    # 7. All prices positive
    any_neg = np.any((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0))
    if any_neg:
        return False, "One or more non-positive prices detected."

    # 8. Minimum 500 bars
    n_bars = len(df)
    if n_bars < 500:
        return False, f"Only {n_bars} bars (minimum 500 required)."

    return True, "OK — all checks passed."
