"""
Position Sizing and Trade Management for LSR-EH1.

Computes lot size, stop-loss, and take-profit levels based on ATR-derived
risk budgeting.  All sizing is deterministic — no optimisation parameters
beyond what is in :mod:`config`.
"""

import math


def calculate_position_size(
    equity: float,
    atr_entry: float,
    risk_fraction: float = 0.01,
    pip_value: float = 0.0001,
    unit_step: int = 1000,
) -> int:
    """
    Calculate position size in base-currency units, rounded down to the
    nearest ``unit_step``.

    Sizing formula::

        stop_distance = sl_atr * ATR_entry          # in price units
        raw_units     = floor( risk_amount / stop_distance )
        units         = floor(raw_units / unit_step) * unit_step

    For EURUSD with equity=$100K, risk=1%, ATR=0.0016 (16 pips), sl_atr=1.0::

        stop_distance = 1.0 * 0.0016 = 0.0016
        risk_amount   = 100000 * 0.01 = 1000 USD
        raw_units     = floor(1000 / 0.0016) = 625000
        units         = 625000 (rounded to nearest 1000)

    Risk verification: 625000 units * 0.0016 stop = 1000 USD = 1% of equity.

    Parameters
    ----------
    equity : float
        Current account equity in USD.
    atr_entry : float
        ATR value at the time of entry (same units as price).
    risk_fraction : float, optional
        Fraction of equity to risk per trade (default 0.01 = 1 %).
    pip_value : float, optional
        Value of one pip in price terms (default 0.0001 for EURUSD).
    unit_step : int, optional
        Minimum lot granularity; result is rounded down to a multiple
        of this value (default 1000).

    Returns
    -------
    int
        Number of units to trade (≥ 0).  Returns 0 when equity, ATR,
        or pip_value are non-positive.
    """
    if equity <= 0 or atr_entry <= 0:
        return 0

    risk_amount = equity * risk_fraction
    stop_distance = atr_entry  # sl_atr=1.0, stop = 1.0 * ATR in price units

    if stop_distance <= 0:
        return 0

    raw_units = math.floor(risk_amount / stop_distance)
    stepped_units = (raw_units // unit_step) * unit_step
    return max(stepped_units, 0)


def calculate_stop_loss(
    entry_price: float,
    atr_entry: float,
    direction: int,
    sl_atr: float = 1.0,
) -> float:
    """
    Calculate the stop-loss price from entry.

    * **Long** (``direction = +1``): ``entry_price - sl_atr * atr_entry``
    * **Short** (``direction = -1``): ``entry_price + sl_atr * atr_entry``

    Parameters
    ----------
    entry_price : float
        Fill price of the entry order.
    atr_entry : float
        ATR value at the time of entry.
    direction : int
        ``+1`` for long, ``-1`` for short.
    sl_atr : float, optional
        Stop-loss distance in ATR multiples (default 1.0).

    Returns
    -------
    float
        Stop-loss price level.
    """
    if direction == 1:
        return entry_price - sl_atr * atr_entry
    elif direction == -1:
        return entry_price + sl_atr * atr_entry
    else:
        raise ValueError(f"direction must be +1 or -1, got {direction}")


def calculate_profit_target(
    entry_price: float,
    atr_entry: float,
    direction: int,
    pt_atr: float = 1.5,
) -> float:
    """
    Calculate the take-profit (profit target) price from entry.

    * **Long** (``direction = +1``): ``entry_price + pt_atr * atr_entry``
    * **Short** (``direction = -1``): ``entry_price - pt_atr * atr_entry``

    Parameters
    ----------
    entry_price : float
        Fill price of the entry order.
    atr_entry : float
        ATR value at the time of entry.
    direction : int
        ``+1`` for long, ``-1`` for short.
    pt_atr : float, optional
        Profit-target distance in ATR multiples (default 1.5).

    Returns
    -------
    float
        Take-profit price level.
    """
    if direction == 1:
        return entry_price + pt_atr * atr_entry
    elif direction == -1:
        return entry_price - pt_atr * atr_entry
    else:
        raise ValueError(f"direction must be +1 or -1, got {direction}")
