"""
Trade Cost Estimation for LSR-EH1.

Models the realistic cost of executing a round-trip trade including
half-spread, slippage, and market-impact components.  Used by the
backtest engine to deduct costs from equity / PnL.
"""


def calculate_trade_cost(
    notional: float,
    pip_value: float = 0.0001,
    half_spread_pips: float = 0.2,
    slippage_pips: float = 0.1,
    market_impact_pips: float = 0.05,
) -> float:
    """
    Estimate total round-trip cost in USD for a trade of a given notional size.

    The cost model sums three components (all in pips) and converts to USD::

        cost_pips = half_spread_pips + slippage_pips + market_impact_pips
        cost_usd  = notional * cost_pips * pip_value

    Parameters
    ----------
    notional : float
        Trade notional in base-currency units (e.g. 100,000 for 100 K EURUSD).
    pip_value : float, optional
        Value of one pip in price terms (default 0.0001 for EURUSD).
    half_spread_pips : float, optional
        One-way half-spread cost in pips (default 0.2).
    slippage_pips : float, optional
        Market-order slippage in pips (default 0.1).
    market_impact_pips : float, optional
        Market-impact cost in pips (default 0.05).

    Returns
    -------
    float
        Estimated round-trip cost in USD.  Returns 0.0 for non-positive
        notional.
    """
    if notional <= 0:
        return 0.0

    cost_pips = half_spread_pips + slippage_pips + market_impact_pips
    cost_usd = notional * cost_pips * pip_value
    return cost_usd


def apply_slippage(
    entry_price: float,
    direction: int,
    slippage_pips: float = 0.1,
    pip_value: float = 0.0001,
) -> float:
    """
    Simulate execution slippage on a market order entry.

    Slippage always worsens the fill:

    * **Long** (``direction = +1``): ``entry_price + slippage``
      (you buy at a higher price)
    * **Short** (``direction = -1``): ``entry_price - slippage``
      (you sell at a lower price)

    where ``slippage = slippage_pips * pip_value``.

    Parameters
    ----------
    entry_price : float
        Theoretical / signal entry price.
    direction : int
        ``+1`` for long, ``-1`` for short.
    slippage_pips : float, optional
        Slippage amount in pips (default 0.1).
    pip_value : float, optional
        Value of one pip in price terms (default 0.0001).

    Returns
    -------
    float
        Actual fill price after slippage.
    """
    slippage = slippage_pips * pip_value

    if direction == 1:
        return entry_price + slippage
    elif direction == -1:
        return entry_price - slippage
    else:
        raise ValueError(f"direction must be +1 or -1, got {direction}")
