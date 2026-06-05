"""
Backtest Engine for LSR-EH1 Strategy.

Implements a bar-by-bar event-driven backtester with:
- Next-bar execution (no look-ahead bias)
- Intra-bar profit target / stop loss simulation
- Time-based exits
- Risk limit enforcement (daily / weekly / monthly / hard drawdown)
- Cooling-period management
- News-blackout filtering
- Spread-filter protective exits
- Full equity-curve tracking
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import StrategyConfig
from execution.costs import apply_slippage, calculate_trade_cost
from execution.position_sizing import (
    calculate_position_size,
    calculate_profit_target,
    calculate_stop_loss,
)
from strategy.liquidity_sweep import generate_signals, is_news_blackout


# ---------------------------------------------------------------------------
# Trade Record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Immutable record of a single round-trip trade."""

    trade_id: int
    direction: int                       # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float                   # actual fill price (with slippage)
    atr_entry: float                     # ATR at entry
    stop_loss: float
    profit_target: float
    size: int                            # position size in base-currency units
    notional: float                      # size * entry_price
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pips: Optional[float] = None     # signed PnL in pips
    pnl_usd: Optional[float] = None     # signed PnL in USD (net of round-trip cost)
    exit_reason: Optional[str] = None   # "profit_target" | "stop_loss" | "time_exit" | "spread_filter" | "end_of_data"
    bars_held: int = 0
    reversal_distance: Optional[float] = None  # abs(exit-entry)/atr_entry in ATR units


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Core bar-by-bar backtesting engine for the LSR-EH1 strategy.

    The engine is deliberately *stateless* between calls to :meth:`run` — all
    mutable state (trade list, equity curve) is reset on each invocation.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config: StrategyConfig = config
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        initial_equity: float = 100_000.0,
    ) -> pd.DataFrame:
        """Run a full backtest on the supplied OHLC DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            1 h OHLC data with columns ``open``, ``high``, ``low``, ``close``
            and a :class:`pandas.DatetimeIndex`.  A copy is made so the
            caller's DataFrame is never mutated.
        initial_equity : float
            Starting account equity in USD.

        Returns
        -------
        pd.DataFrame
            Equity curve with columns:

            * ``timestamp``    – bar open time
            * ``equity``       – mark-to-market equity (realized + unrealized)
            * ``drawdown_pct`` – peak-to-current drawdown as a percentage
            * ``position``     – 0 (flat), +1 (long), -1 (short)
            * ``unrealized_pnl`` – unrealized P&L in USD
        """
        # Step 1: Generate strategy signals (operates on a copy)
        df = generate_signals(df.copy(), self.config)

        # -- Shortcut: no valid ATR means no signals possible --
        atr_first_valid = df["atr"].first_valid_index()
        if atr_first_valid is None:
            return self._flat_equity_curve(df, initial_equity)

        start_idx: int = df.index.get_loc(atr_first_valid)

        # -- Configuration aliases (avoid repeated dot-lookups in tight loop) --
        cfg_exit = self.config.exit
        cfg_risk = self.config.risk
        cfg_cost = self.config.cost
        cfg_filter = self.config.filter
        pip_value: float = cfg_risk.pip_value
        max_bars_held: int = cfg_exit.max_bars_held

        # -- Optional spread filter --
        spread_filter_enabled: bool = "spread" in df.columns
        if spread_filter_enabled:
            spread_arr = df["spread"].values.astype(np.float64)
            spread_median_arr = (
                df["spread"].rolling(window=20, min_periods=10).median().values
            )

        # -- Pre-extract numpy arrays for fast iloc-free access --
        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        atrs = df["atr"].values.astype(np.float64)
        timestamps = df.index
        long_signals = df["long_signal"].values.astype(bool)
        short_signals = df["short_signal"].values.astype(bool)

        n_bars: int = len(df)

        # -- Mutable state --
        equity: float = initial_equity
        peak_equity: float = initial_equity          # peak of *realized* equity (for risk limits)
        curve_peak: float = initial_equity            # peak of *total* equity (for drawdown curve)
        trade_id_counter: int = 0

        current_position: Optional[Trade] = None
        pending_entry: Optional[Tuple[int, int]] = None  # (direction, signal_bar_index)
        bars_held_counter: int = 0                       # counts up from 0 while in position

        last_long_entry_bar: int = -100
        last_short_entry_bar: int = -100

        # Period PnL accumulators (reset at each boundary)
        daily_pnl: float = 0.0
        weekly_pnl: float = 0.0
        monthly_pnl: float = 0.0
        prev_date_key = None
        prev_week_key = None
        prev_month_key = None

        curve_records: List[dict] = []

        # ==============================================================
        # Main bar-by-bar loop
        # ==============================================================
        for i in range(n_bars):
            ts: pd.Timestamp = timestamps[i]
            o: float = opens[i]
            h: float = highs[i]
            l: float = lows[i]
            c: float = closes[i]
            cur_atr: float = atrs[i]

            # --- Period boundary resets ---
            date_key = ts.date()
            week_key = (ts.isocalendar()[0], ts.isocalendar()[1])
            month_key = (ts.year, ts.month)

            if prev_date_key is None or date_key != prev_date_key:
                daily_pnl = 0.0
                prev_date_key = date_key
            if prev_week_key is None or week_key != prev_week_key:
                weekly_pnl = 0.0
                prev_week_key = week_key
            if prev_month_key is None or month_key != prev_month_key:
                monthly_pnl = 0.0
                prev_month_key = month_key

            # ----------------------------------------------------------
            # Phase 1 — Execute pending entry (signal at bar i-1,
            #           fill at open[i] with slippage)
            # ----------------------------------------------------------
            if pending_entry is not None and current_position is None:
                direction, _sig_bar = pending_entry
                pending_entry = None

                if not np.isnan(cur_atr) and cur_atr > 0.0:
                    raw_price = o
                    fill_price = apply_slippage(
                        raw_price,
                        direction,
                        slippage_pips=cfg_cost.slippage_pips,
                        pip_value=pip_value,
                    )
                    size = calculate_position_size(
                        equity=equity,
                        atr_entry=cur_atr,
                        risk_fraction=cfg_risk.risk_per_trade,
                        pip_value=pip_value,
                        unit_step=cfg_risk.position_unit_step,
                    )

                    if size > 0:
                        sl = calculate_stop_loss(
                            fill_price, cur_atr, direction,
                            sl_atr=cfg_exit.stop_loss_atr,
                        )
                        pt = calculate_profit_target(
                            fill_price, cur_atr, direction,
                            pt_atr=cfg_exit.profit_target_atr,
                        )
                        current_position = Trade(
                            trade_id=trade_id_counter,
                            direction=direction,
                            entry_time=ts,
                            entry_price=fill_price,
                            atr_entry=cur_atr,
                            stop_loss=sl,
                            profit_target=pt,
                            size=size,
                            notional=size * fill_price,
                        )
                        trade_id_counter += 1
                        bars_held_counter = 0

                        # Update cooling-period tracker
                        if direction == 1:
                            last_long_entry_bar = i
                        else:
                            last_short_entry_bar = i

            # ----------------------------------------------------------
            # Phase 2 — Exit checks (BEFORE any new entries)
            # ----------------------------------------------------------
            if current_position is not None:
                pos: Trade = current_position
                d: int = pos.direction
                exited: bool = False

                # Increment bars held *before* checking time exit so that
                # bar 0 of entry becomes bars_held == 1.
                bars_held_counter += 1

                # (a) Spread filter — exit at open if spread > 1.5× median
                if not exited and spread_filter_enabled:
                    cur_spread = spread_arr[i]
                    med = spread_median_arr[i]
                    if not np.isnan(cur_spread) and not np.isnan(med):
                        if cur_spread > 1.5 * med:
                            exited = self._close_trade(
                                pos=pos,
                                exit_price=o,
                                exit_time=ts,
                                exit_reason="spread_filter",
                                pip_value=pip_value,
                                cfg_cost=cfg_cost,
                            )
                            if exited:
                                equity += pos.pnl_usd
                                daily_pnl += pos.pnl_usd
                                weekly_pnl += pos.pnl_usd
                                monthly_pnl += pos.pnl_usd
                                peak_equity = max(peak_equity, equity)
                                self.trades.append(pos)
                                current_position = None

                # (b) Profit target (checked before stop loss per spec)
                if not exited and current_position is not None:
                    pos = current_position
                    d = pos.direction
                    if d == 1 and h >= pos.profit_target:
                        exited = self._close_trade(
                            pos=pos,
                            exit_price=pos.profit_target,
                            exit_time=ts,
                            exit_reason="profit_target",
                            pip_value=pip_value,
                            cfg_cost=cfg_cost,
                        )
                    elif d == -1 and l <= pos.profit_target:
                        exited = self._close_trade(
                            pos=pos,
                            exit_price=pos.profit_target,
                            exit_time=ts,
                            exit_reason="profit_target",
                            pip_value=pip_value,
                            cfg_cost=cfg_cost,
                        )

                # (c) Stop loss
                if not exited and current_position is not None:
                    pos = current_position
                    d = pos.direction
                    if d == 1 and l <= pos.stop_loss:
                        exited = self._close_trade(
                            pos=pos,
                            exit_price=pos.stop_loss,
                            exit_time=ts,
                            exit_reason="stop_loss",
                            pip_value=pip_value,
                            cfg_cost=cfg_cost,
                        )
                    elif d == -1 and h >= pos.stop_loss:
                        exited = self._close_trade(
                            pos=pos,
                            exit_price=pos.stop_loss,
                            exit_time=ts,
                            exit_reason="stop_loss",
                            pip_value=pip_value,
                            cfg_cost=cfg_cost,
                        )

                # (d) Time exit (at bar close)
                if not exited and current_position is not None:
                    if bars_held_counter >= max_bars_held:
                        exited = self._close_trade(
                            pos=current_position,
                            exit_price=c,
                            exit_time=ts,
                            exit_reason="time_exit",
                            pip_value=pip_value,
                            cfg_cost=cfg_cost,
                        )

                # Record the exit
                if exited and current_position is not None:
                    equity += current_position.pnl_usd
                    daily_pnl += current_position.pnl_usd
                    weekly_pnl += current_position.pnl_usd
                    monthly_pnl += current_position.pnl_usd
                    peak_equity = max(peak_equity, equity)
                    current_position.bars_held = bars_held_counter
                    self.trades.append(current_position)
                    current_position = None

            # ----------------------------------------------------------
            # Phase 3 — New entry checks (only when flat)
            # ----------------------------------------------------------
            if (current_position is None
                    and pending_entry is None
                    and i >= start_idx):

                # (a) News blackout
                if is_news_blackout(ts, self.config.news_blacklist):
                    pass
                # (b) Risk limits
                elif self._check_risk_limits(
                    current_equity=equity,
                    peak_equity=peak_equity,
                    daily_pnl=daily_pnl,
                    weekly_pnl=weekly_pnl,
                    monthly_pnl=monthly_pnl,
                ):
                    pass
                # (c) Cooling period + raw signal
                else:
                    if (long_signals[i]
                            and (i - last_long_entry_bar)
                            >= cfg_filter.min_bars_between_same_dir):
                        pending_entry = (1, i)
                    elif (short_signals[i]
                          and (i - last_short_entry_bar)
                          >= cfg_filter.min_bars_between_same_dir):
                        pending_entry = (-1, i)

            # ----------------------------------------------------------
            # Phase 4 — Equity curve snapshot
            # ----------------------------------------------------------
            position_val: int = 0
            unrealized_pnl: float = 0.0

            if current_position is not None:
                pos = current_position
                position_val = pos.direction
                if pos.direction == 1:
                    unrealized_pnl = (c - pos.entry_price) * pos.size
                else:
                    unrealized_pnl = (pos.entry_price - c) * pos.size

            total_equity = equity + unrealized_pnl
            curve_peak = max(curve_peak, total_equity)
            drawdown_pct: float = 0.0
            if curve_peak > 0.0:
                drawdown_pct = (curve_peak - total_equity) / curve_peak * 100.0

            curve_records.append({
                "timestamp": ts,
                "equity": total_equity,
                "drawdown_pct": drawdown_pct,
                "position": position_val,
                "unrealized_pnl": unrealized_pnl,
            })

        # ==============================================================
        # Post-loop: close any position still open at end of data
        # ==============================================================
        if current_position is not None:
            pos = current_position
            last_ts = timestamps[-1]
            last_close = float(closes[-1])
            self._close_trade(
                pos=pos,
                exit_price=last_close,
                exit_time=last_ts,
                exit_reason="end_of_data",
                pip_value=pip_value,
                cfg_cost=cfg_cost,
            )
            pos.bars_held = bars_held_counter
            equity += pos.pnl_usd
            peak_equity = max(peak_equity, equity)
            self.trades.append(pos)

        # -- Build and store equity curve DataFrame --
        equity_curve_df = pd.DataFrame(curve_records, columns=[
            "timestamp", "equity", "drawdown_pct", "position", "unrealized_pnl",
        ])
        self.equity_curve = equity_curve_df
        return equity_curve_df

    # ------------------------------------------------------------------
    # Risk-limit enforcement
    # ------------------------------------------------------------------

    def _check_risk_limits(
        self,
        current_equity: float,
        peak_equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
    ) -> bool:
        """Return ``True`` if trading must be halted.

        Checks hard drawdown, daily, weekly, and monthly loss limits
        against the *peak* equity.

        Parameters
        ----------
        current_equity : float
            Realized equity (before unrealized PnL).
        peak_equity : float
            Highest realized equity observed so far.
        daily_pnl : float
            Cumulative realized PnL for the current calendar day (negative = loss).
        weekly_pnl : float
            Cumulative realized PnL for the current ISO week.
        monthly_pnl : float
            Cumulative realized PnL for the current calendar month.

        Returns
        -------
        bool
            ``True`` if any risk limit has been breached.
        """
        if peak_equity <= 0.0:
            return True

        risk = self.config.risk

        # Hard drawdown
        dd = (peak_equity - current_equity) / peak_equity
        if dd >= risk.max_drawdown_hard:
            return True

        # Periodic loss limits (relative to peak equity)
        if daily_pnl < -risk.daily_loss_limit * peak_equity:
            return True
        if weekly_pnl < -risk.weekly_loss_limit * peak_equity:
            return True
        if monthly_pnl < -risk.monthly_loss_limit * peak_equity:
            return True

        return False

    # ------------------------------------------------------------------
    # Trade export
    # ------------------------------------------------------------------

    def _export_trades(self, filepath: str) -> None:
        """Write the trade log to a CSV file.

        Parameters
        ----------
        filepath : str
            Destination path (any extension; ``.csv`` is recommended).
        """
        if not self.trades:
            pd.DataFrame(columns=[
                "trade_id", "direction", "entry_time", "entry_price",
                "atr_entry", "stop_loss", "profit_target", "size",
                "notional", "exit_time", "exit_price", "pnl_pips",
                "pnl_usd", "exit_reason", "bars_held", "reversal_distance",
            ]).to_csv(filepath, index=False)
            return

        records = [
            {
                "trade_id": t.trade_id,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_time": t.entry_time.isoformat(),
                "entry_price": round(t.entry_price, 6),
                "atr_entry": round(t.atr_entry, 6),
                "stop_loss": round(t.stop_loss, 6),
                "profit_target": round(t.profit_target, 6),
                "size": t.size,
                "notional": round(t.notional, 2),
                "exit_time": t.exit_time.isoformat() if t.exit_time else "",
                "exit_price": round(t.exit_price, 6) if t.exit_price is not None else "",
                "pnl_pips": round(t.pnl_pips, 1) if t.pnl_pips is not None else "",
                "pnl_usd": round(t.pnl_usd, 2) if t.pnl_usd is not None else "",
                "exit_reason": t.exit_reason or "",
                "bars_held": t.bars_held,
                "reversal_distance": round(t.reversal_distance, 3) if t.reversal_distance is not None else "",
            }
            for t in self.trades
        ]
        pd.DataFrame(records).to_csv(filepath, index=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_pnl_pips(
        direction: int,
        entry_price: float,
        exit_price: float,
        pip_value: float,
    ) -> float:
        """Signed PnL in pips.

        * Long  ``(+1)``: ``(exit - entry) / pip_value``
        * Short ``(-1)``: ``(entry - exit) / pip_value``
        """
        if direction == 1:
            return (exit_price - entry_price) / pip_value
        else:
            return (entry_price - exit_price) / pip_value

    def _close_trade(
        self,
        pos: Trade,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
        pip_value: float,
        cfg_cost,  # CostParams — typed loosely to avoid importing in hot path
    ) -> bool:
        """Populate exit fields on a *Trade* in-place.

        Returns ``True`` on success.  Returns ``False`` if ``exit_price`` is
        not a finite positive number (should never happen in practice).
        """
        if exit_price <= 0 or not np.isfinite(exit_price):
            return False

        pnl_pips = self._calc_pnl_pips(pos.direction, pos.entry_price, exit_price, pip_value)
        cost = calculate_trade_cost(
            notional=pos.notional,
            pip_value=pip_value,
            half_spread_pips=cfg_cost.half_spread_pips,
            slippage_pips=cfg_cost.slippage_pips,
            market_impact_pips=cfg_cost.market_impact_pips,
        )
        pnl_usd = pnl_pips * pip_value * pos.size - cost

        # Reversal distance: how far price moved in ATR units
        reversal_distance = (
            abs(exit_price - pos.entry_price) / pos.atr_entry
            if pos.atr_entry > 0 else np.nan
        )

        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.pnl_pips = pnl_pips
        pos.pnl_usd = pnl_usd
        pos.exit_reason = exit_reason
        pos.reversal_distance = reversal_distance
        return True

    @staticmethod
    def _flat_equity_curve(
        df: pd.DataFrame,
        initial_equity: float,
    ) -> pd.DataFrame:
        """Return a flat equity curve (no trades possible)."""
        return pd.DataFrame({
            "timestamp": df.index,
            "equity": initial_equity,
            "drawdown_pct": 0.0,
            "position": 0,
            "unrealized_pnl": 0.0,
        })
