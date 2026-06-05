"""
LSR-EH1 Edge Characterization Analysis
=======================================

Four hypothesis tests. NO parameter optimization.

Test 1: Continuous sweep depth buckets
    - Thresholds: 0.50, 0.75, 1.00, 1.25 ATR
    - Question: Does PF improve monotonically as sweep depth increases?
    - Verifies the regime analysis finding directly as a forward filter.

Test 2: Profit by sweep-depth decile
    - D1 through D10 (equal-frequency bins)
    - Question: Is the sweep-depth → PF relationship structural (monotonic)?
    - Structural = stronger evidence than a single binary threshold.

Test 3: Time exit duration sweep
    - MAX_BARS_HELD: 4, 6, 8, 10, 12
    - All other parameters frozen. sweep >= 0.5 ATR filter applied.
    - Question: How long does the reversal actually take?

Test 4: MAE/MFE analysis
    - Maximum Adverse Excursion / Maximum Favorable Excursion per trade
    - Question: Is the stop too tight? Target too ambitious? How much
      reversal actually occurs before the trade resolves?

This uses the SAME experiment runner from sweep_depth_experiment.py
extended with MAE/MFE tracking and variable max_bars_held.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import StrategyConfig
from strategy.liquidity_sweep import generate_signals, is_news_blackout
from execution.costs import apply_slippage, calculate_trade_cost
from execution.position_sizing import (
    calculate_position_size,
    calculate_stop_loss,
    calculate_profit_target,
)


# ---------------------------------------------------------------------------
# Unified Experiment Runner with MAE/MFE + Variable Time Exit
# ---------------------------------------------------------------------------

def run_experiment_mae_mfe(
    df: pd.DataFrame,
    config: StrategyConfig,
    min_sweep_depth: float = 0.0,
    exit_mode: str = "standard",
    trail_atr: float = 1.0,
    max_bars_held_override: int = None,
    initial_equity: float = 100_000.0,
    label: str = "",
    disable_risk_limits: bool = True,
) -> dict:
    """Run a single backtest experiment with MAE/MFE tracking.

    Parameters
    ----------
    exit_mode : str
        "standard"     - 1.5 ATR target, 1.0 ATR stop, configurable time
        "target_1atr"  - 1.0 ATR target, 1.0 ATR stop, configurable time
        "trailing"     - 1.0 ATR trailing from HWM, 1.0 ATR initial stop, no target
        "time_only"    - No target, 1.0 ATR stop, configurable time
    max_bars_held_override : int or None
        If provided, overrides config.exit.max_bars_held.
    """
    # Generate signals with optional sweep depth filter
    df = generate_signals(df.copy(), config)

    # ALWAYS compute sweep depth columns for per-trade metadata
    with np.errstate(divide="ignore", invalid="ignore"):
        df["sweep_depth_long"] = np.where(
            df["low"] < df["liquidity_long"],
            np.abs(df["liquidity_long"] - df["low"]) / df["atr"],
            0.0,
        )
        df["sweep_depth_short"] = np.where(
            df["high"] > df["liquidity_short"],
            np.abs(df["high"] - df["liquidity_short"]) / df["atr"],
            0.0,
        )

    if min_sweep_depth > 0:
        df["long_signal"] = (
            df["long_signal"] & (df["sweep_depth_long"] >= min_sweep_depth)
        )
        df["short_signal"] = (
            df["short_signal"] & (df["sweep_depth_short"] >= min_sweep_depth)
        )

    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    atrs = df["atr"].values.astype(np.float64)
    timestamps = df.index
    long_signals = df["long_signal"].values.astype(bool)
    short_signals = df["short_signal"].values.astype(bool)

    cfg_exit = config.exit
    cfg_risk = config.risk
    cfg_cost = config.cost
    cfg_filter = config.filter
    pip_value = cfg_risk.pip_value
    max_bars_held = max_bars_held_override if max_bars_held_override is not None else cfg_exit.max_bars_held

    atr_first_valid = df["atr"].first_valid_index()
    if atr_first_valid is None:
        return {"label": label, "trades": pd.DataFrame(), "total_pnl": 0.0}
    start_idx = df.index.get_loc(atr_first_valid)

    n_bars = len(df)
    equity = initial_equity
    peak_equity = initial_equity
    trade_id_counter = 0

    current_position = None
    pending_entry = None
    bars_held_counter = 0
    hwm = 0.0
    trailing_stop_price = 0.0

    last_long_entry_bar = -100
    last_short_entry_bar = -100

    daily_pnl = weekly_pnl = monthly_pnl = 0.0
    prev_date_key = prev_week_key = prev_month_key = None

    trades = []

    for i in range(n_bars):
        ts = timestamps[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        cur_atr = atrs[i]

        # Period resets
        date_key = ts.date()
        week_key = (ts.isocalendar()[0], ts.isocalendar()[1])
        month_key = (ts.year, ts.month)
        if prev_date_key is None or date_key != prev_date_key:
            daily_pnl = 0.0; prev_date_key = date_key
        if prev_week_key is None or week_key != prev_week_key:
            weekly_pnl = 0.0; prev_week_key = week_key
        if prev_month_key is None or month_key != prev_month_key:
            monthly_pnl = 0.0; prev_month_key = month_key

        # --- Execute pending entry ---
        if pending_entry is not None and current_position is None:
            direction, _sig_bar = pending_entry
            pending_entry = None

            if not np.isnan(cur_atr) and cur_atr > 0.0:
                fill_price = apply_slippage(
                    o, direction,
                    slippage_pips=cfg_cost.slippage_pips,
                    pip_value=pip_value,
                )
                size = calculate_position_size(
                    equity, cur_atr, cfg_risk.risk_per_trade,
                    pip_value, cfg_risk.position_unit_step,
                )

                if size > 0:
                    sl = calculate_stop_loss(
                        fill_price, cur_atr, direction,
                        sl_atr=cfg_exit.stop_loss_atr,
                    )

                    if exit_mode in ("standard",):
                        pt = calculate_profit_target(
                            fill_price, cur_atr, direction,
                            pt_atr=cfg_exit.profit_target_atr,
                        )
                    elif exit_mode == "target_1atr":
                        pt = calculate_profit_target(
                            fill_price, cur_atr, direction, pt_atr=1.0,
                        )
                    else:
                        pt = None  # trailing or time_only: no target

                    # Compute sweep metrics at the SIGNAL bar (not entry bar)
                    # _sig_bar is the index where the signal fired; entry is at i (next bar)
                    sb = _sig_bar
                    sig_atr = float(atrs[sb])

                    # Metric 1: WICK PENETRATION (how deep the wick went below/above liquidity)
                    # This is the TRUE sweep depth
                    if direction == 1:
                        sweep_depth = float(df["sweep_depth_long"].iloc[sb]) if sb < len(df) else 0.0
                    else:
                        sweep_depth = float(df["sweep_depth_short"].iloc[sb]) if sb < len(df) else 0.0

                    # Metric 2: ENTRY DISTANCE (abs(entry - liquidity) / atr at signal bar)
                    # This matches the enrichment's sweep_size_atr and was the metric
                    # that showed the monotonic PF relationship in regime analysis
                    if direction == 1:
                        liq_sig = float(df["liquidity_long"].iloc[sb]) if sb < len(df) else np.nan
                    else:
                        liq_sig = float(df["liquidity_short"].iloc[sb]) if sb < len(df) else np.nan
                    if not np.isnan(liq_sig) and sig_atr > 0:
                        sweep_entry_dist = abs(fill_price - liq_sig) / sig_atr
                    else:
                        sweep_entry_dist = 0.0

                    current_position = {
                        "trade_id": trade_id_counter,
                        "direction": direction,
                        "entry_time": ts,
                        "entry_price": fill_price,
                        "atr_entry": cur_atr,
                        "stop_loss": sl,
                        "profit_target": pt,
                        "size": size,
                        "notional": size * fill_price,
                        "sweep_depth_atr": sweep_depth,
                        "sweep_entry_dist_atr": sweep_entry_dist,
                        # MAE/MFE tracking
                        "mae_price": 0.0,   # max adverse excursion in price
                        "mfe_price": 0.0,   # max favorable excursion in price
                    }
                    trade_id_counter += 1
                    bars_held_counter = 0
                    hwm = fill_price
                    trailing_stop_price = sl

                    if direction == 1:
                        last_long_entry_bar = i
                    else:
                        last_short_entry_bar = i

        # --- Exit checks + MAE/MFE tracking ---
        if current_position is not None:
            pos = current_position
            d = pos["direction"]
            bars_held_counter += 1

            # Track MAE/MFE before any exit checks
            if d == 1:
                # Long: adverse = price dropping, favorable = price rising
                adverse = pos["entry_price"] - l  # how far below entry
                favorable = h - pos["entry_price"]  # how far above entry
            else:
                # Short: adverse = price rising, favorable = price dropping
                adverse = h - pos["entry_price"]
                favorable = pos["entry_price"] - l

            pos["mae_price"] = max(pos["mae_price"], adverse)
            pos["mfe_price"] = max(pos["mfe_price"], favorable)

            exit_price = None
            exit_reason = None

            # Update trailing stop
            if exit_mode == "trailing":
                if d == 1:
                    hwm = max(hwm, h)
                    trailing_stop_price = hwm - trail_atr * pos["atr_entry"]
                    trailing_stop_price = max(trailing_stop_price, pos["stop_loss"])
                else:
                    hwm = min(hwm, l)
                    trailing_stop_price = hwm + trail_atr * pos["atr_entry"]
                    trailing_stop_price = min(trailing_stop_price, pos["stop_loss"])

            # (a) Profit target
            if exit_price is None and pos["profit_target"] is not None:
                if d == 1 and h >= pos["profit_target"]:
                    exit_price = pos["profit_target"]
                    exit_reason = "profit_target"
                elif d == -1 and l <= pos["profit_target"]:
                    exit_price = pos["profit_target"]
                    exit_reason = "profit_target"

            # (b) Trailing stop
            if exit_price is None and exit_mode == "trailing":
                if d == 1 and l <= trailing_stop_price:
                    exit_price = trailing_stop_price
                    exit_reason = "trailing_stop"
                elif d == -1 and h >= trailing_stop_price:
                    exit_price = trailing_stop_price
                    exit_reason = "trailing_stop"

            # (c) Fixed stop loss
            if exit_price is None:
                if d == 1 and l <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"
                elif d == -1 and h >= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"

            # (d) Time exit
            if exit_price is None and bars_held_counter >= max_bars_held:
                exit_price = c
                exit_reason = "time_exit"

            # Close trade
            if exit_price is not None:
                pnl_pips = (
                    (exit_price - pos["entry_price"]) / pip_value if d == 1
                    else (pos["entry_price"] - exit_price) / pip_value
                )
                cost = calculate_trade_cost(
                    notional=pos["notional"], pip_value=pip_value,
                    half_spread_pips=cfg_cost.half_spread_pips,
                    slippage_pips=cfg_cost.slippage_pips,
                    market_impact_pips=cfg_cost.market_impact_pips,
                )
                pnl_usd = pnl_pips * pip_value * pos["size"] - cost
                rev_dist = (
                    abs(exit_price - pos["entry_price"]) / pos["atr_entry"]
                    if pos["atr_entry"] > 0 else np.nan
                )

                # Convert MAE/MFE to ATR units and pips
                mae_atr = pos["mae_price"] / pos["atr_entry"] if pos["atr_entry"] > 0 else np.nan
                mfe_atr = pos["mfe_price"] / pos["atr_entry"] if pos["atr_entry"] > 0 else np.nan
                mae_pips = pos["mae_price"] / pip_value
                mfe_pips = pos["mfe_price"] / pip_value

                trades.append({
                    "trade_id": pos["trade_id"],
                    "direction": "LONG" if d == 1 else "SHORT",
                    "entry_time": pos["entry_time"],
                    "entry_price": pos["entry_price"],
                    "atr_entry": pos["atr_entry"],
                    "stop_loss": pos["stop_loss"],
                    "profit_target": pos["profit_target"],
                    "size": pos["size"],
                    "notional": pos["notional"],
                    "exit_time": ts,
                    "exit_price": exit_price,
                    "pnl_pips": pnl_pips,
                    "pnl_usd": pnl_usd,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held_counter,
                    "reversal_distance": rev_dist,
                    "sweep_depth_atr": pos["sweep_depth_atr"],
                    "sweep_entry_dist_atr": pos["sweep_entry_dist_atr"],
                    "mae_price": pos["mae_price"],
                    "mfe_price": pos["mfe_price"],
                    "mae_atr": mae_atr,
                    "mfe_atr": mfe_atr,
                    "mae_pips": mae_pips,
                    "mfe_pips": mfe_pips,
                })

                equity += pnl_usd
                daily_pnl += pnl_usd
                weekly_pnl += pnl_usd
                monthly_pnl += pnl_usd
                peak_equity = max(peak_equity, equity)
                current_position = None

        # --- New entry checks ---
        if current_position is None and pending_entry is None and i >= start_idx:
            if is_news_blackout(ts, config.news_blacklist):
                pass
            elif not disable_risk_limits and peak_equity <= 0.0:
                pass
            elif not disable_risk_limits and (peak_equity - equity) / peak_equity >= cfg_risk.max_drawdown_hard:
                pass
            elif not disable_risk_limits and daily_pnl < -cfg_risk.daily_loss_limit * peak_equity:
                pass
            elif not disable_risk_limits and weekly_pnl < -cfg_risk.weekly_loss_limit * peak_equity:
                pass
            elif not disable_risk_limits and monthly_pnl < -cfg_risk.monthly_loss_limit * peak_equity:
                pass
            else:
                if (long_signals[i]
                        and (i - last_long_entry_bar)
                        >= cfg_filter.min_bars_between_same_dir):
                    pending_entry = (1, i)
                elif (short_signals[i]
                      and (i - last_short_entry_bar)
                      >= cfg_filter.min_bars_between_same_dir):
                    pending_entry = (-1, i)

    # --- Compute summary ---
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    summary = {"label": label}

    if len(trades_df) > 0:
        winners = trades_df[trades_df["pnl_usd"] > 0]
        losers = trades_df[trades_df["pnl_usd"] < 0]
        summary["trades"] = len(trades_df)
        summary["long"] = len(trades_df[trades_df["direction"] == "LONG"])
        summary["short"] = len(trades_df[trades_df["direction"] == "SHORT"])
        summary["win_rate"] = len(winners) / len(trades_df)
        summary["profit_factor"] = (
            abs(winners["pnl_usd"].sum() / losers["pnl_usd"].sum())
            if len(losers) > 0 else np.nan
        )
        summary["total_pnl_usd"] = trades_df["pnl_usd"].sum()
        summary["avg_pnl_pips"] = trades_df["pnl_pips"].mean()
        summary["expectancy"] = trades_df["pnl_usd"].mean()

        for reason in ["profit_target", "stop_loss", "time_exit", "trailing_stop"]:
            subset = trades_df[trades_df["exit_reason"] == reason]
            if len(subset) > 0:
                summary[f"{reason}_count"] = len(subset)
                summary[f"{reason}_pct"] = len(subset) / len(trades_df)
                summary[f"{reason}_avg_pnl"] = subset["pnl_pips"].mean()
                summary[f"{reason}_total_pnl"] = subset["pnl_usd"].sum()
            else:
                summary[f"{reason}_count"] = 0
                summary[f"{reason}_pct"] = 0.0
                summary[f"{reason}_avg_pnl"] = 0.0
                summary[f"{reason}_total_pnl"] = 0.0

        summary["max_dd_pct"] = _max_drawdown(trades_df, initial_equity)

        # MAE/MFE summary
        summary["avg_mae_atr"] = trades_df["mae_atr"].mean()
        summary["avg_mfe_atr"] = trades_df["mfe_atr"].mean()
        summary["median_mae_atr"] = trades_df["mae_atr"].median()
        summary["median_mfe_atr"] = trades_df["mfe_atr"].median()
        summary["pct_mae_gt_stop"] = (trades_df["mae_atr"] >= cfg_exit.stop_loss_atr).mean()
        summary["pct_mfe_gt_target"] = (trades_df["mfe_atr"] >= cfg_exit.profit_target_atr).mean() if cfg_exit.profit_target_atr > 0 else np.nan
    else:
        summary["trades"] = 0
        summary["win_rate"] = np.nan
        summary["profit_factor"] = np.nan
        summary["total_pnl_usd"] = 0.0

    summary["trades_df"] = trades_df
    return summary


def _max_drawdown(trades_df: pd.DataFrame, initial_equity: float) -> float:
    """Compute max drawdown from trade P&L."""
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    for _, row in trades_df.iterrows():
        equity += row["pnl_usd"]
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
    return max_dd * 100


# ---------------------------------------------------------------------------
# Report Formatters
# ---------------------------------------------------------------------------

def format_test1_report(results: list) -> str:
    """Test 1: Continuous sweep depth buckets."""
    lines = []
    def w(s=""): lines.append(s)

    w("=" * 76)
    w("  TEST 1: CONTINUOUS SWEEP DEPTH BUCKETS")
    w("  Question: Does PF improve monotonically as sweep depth increases?")
    w("=" * 76)
    w()
    w(f"  {'Threshold':<16} {'Trades':>6} {'Long':>5} {'Short':>5} "
      f"{'PF':>7} {'WR':>6} {'Avg Pips':>9} {'Total $':>12} {'MaxDD%':>7}")
    w(f"  {'-'*74}")

    for r in results:
        label = r["label"]
        trades = int(r.get("trades", 0))
        longs = int(r.get("long", 0))
        shorts = int(r.get("short", 0))
        pf = r.get("profit_factor", np.nan)
        wr = r.get("win_rate", np.nan)
        avg = r.get("avg_pnl_pips", np.nan)
        total = r.get("total_pnl_usd", 0)
        mdd = r.get("max_dd_pct", 0)

        if not np.isnan(pf):
            w(f"  {label:<16} {trades:>6} {longs:>5} {shorts:>5} "
              f"{pf:>7.2f} {wr:>5.0%} {avg:>+9.2f} {total:>+12,.0f} {mdd:>6.1f}%")
        else:
            w(f"  {label:<16} {trades:>6} {longs:>5} {shorts:>5} "
              f"{'N/A':>7} {'N/A':>6} {'N/A':>9} {total:>+12,.0f} {'N/A':>7}")

    w()
    w("  Exit Reason Breakdown:")
    w()

    for r in results:
        w(f"    {r['label']}:")
        for reason in ["profit_target", "stop_loss", "time_exit"]:
            count = r.get(f"{reason}_count", 0)
            if count > 0:
                pct = r.get(f"{reason}_pct", 0.0) * 100
                avg_pnl = r.get(f"{reason}_avg_pnl", 0.0)
                total_pnl = r.get(f"{reason}_total_pnl", 0.0)
                w(f"      {reason:<16}: {count:>3} ({pct:>5.1f}%)  "
                  f"avg {avg_pnl:>+7.2f} pips  total ${total_pnl:>+12,.0f}")
        w()

    # Monotonicity check
    pfs = [(r["label"], r.get("profit_factor", 0)) for r in results
           if not np.isnan(r.get("profit_factor", 0))]
    if len(pfs) >= 2:
        monotonic_up = all(pfs[i][1] <= pfs[i+1][1] for i in range(len(pfs)-1))
        monotonic_down = all(pfs[i][1] >= pfs[i+1][1] for i in range(len(pfs)-1))
        w("  MONOTONICITY CHECK:")
        if monotonic_up:
            w("  YES -- PF increases monotonically with sweep depth.")
            w("  This is strong structural evidence.")
        elif monotonic_down:
            w("  PF decreases with sweep depth -- unexpected.")
        else:
            w("  NO -- PF is non-monotonic across buckets.")
            w(f"  Values: {', '.join(f'{label}={pf:.2f}' for label, pf in pfs)}")
        w()

    return "\n".join(lines)


def format_test2_report(decile_results: list) -> str:
    """Test 2: Sweep-depth decile analysis."""
    lines = []
    def w(s=""): lines.append(s)

    w("=" * 76)
    w("  TEST 2: PROFIT BY SWEEP-DEPTH DECILE")
    w("  Metric: abs(entry - liquidity) / atr (enrichment-compatible)")
    w("  Also shows wick penetration depth for cross-reference.")
    w("  Question: Is the sweep-depth to PF relationship structural?")
    w("=" * 76)
    w()
    w(f"  {'Decile':<10} {'Trades':>6} {'PF':>7} {'WR':>6} "
      f"{'Avg Pips':>9} {'Total $':>12} {'Entry Dist':>10} {'Wick Depth':>10} {'Avg MAE':>8} {'Avg MFE':>8}")
    w(f"  {'-'*92}")

    for r in decile_results:
        decile = r["label"]
        trades = r["trades"]
        pf = r.get("profit_factor", np.nan)
        wr = r.get("win_rate", np.nan)
        avg = r.get("avg_pnl_pips", np.nan)
        total = r.get("total_pnl_usd", 0)
        avg_sweep = r.get("avg_sweep", np.nan)
        avg_wick = r.get("avg_wick_depth", np.nan)
        avg_mae = r.get("avg_mae", np.nan)
        avg_mfe = r.get("avg_mfe", np.nan)

        if trades > 0 and not np.isnan(pf):
            w(f"  {decile:<10} {trades:>6} {pf:>7.2f} {wr:>5.0%} "
              f"{avg:>+9.2f} {total:>+12,.0f} {avg_sweep:>9.3f} {avg_wick:>9.3f} {avg_mae:>7.3f} {avg_mfe:>7.3f}")
        elif trades > 0:
            w(f"  {decile:<10} {trades:>6} {'N/A':>7} {wr:>5.0%} "
              f"{avg:>+9.2f} {total:>+12,.0f} {avg_sweep:>9.3f} {avg_wick:>9.3f} {avg_mae:>7.3f} {avg_mfe:>7.3f}")
        else:
            w(f"  {decile:<10} {trades:>6} {'---':>7} {'---':>6} "
              f"{'---':>9} {'---':>12} {'---':>10} {'---':>10} {'---':>8} {'---':>8}")

    w()

    # Spearman rank correlation check
    valid = [r for r in decile_results if r["trades"] > 0 and not np.isnan(r.get("profit_factor", np.nan))]
    if len(valid) >= 3:
        ranks_pf = sorted(range(len(valid)), key=lambda i: valid[i].get("profit_factor", 0))
        ranks_sweep = sorted(range(len(valid)), key=lambda i: valid[i].get("avg_sweep", 0))
        # Simple Spearman: count concordant pairs
        n = len(ranks_pf)
        concordant = sum(1 for i in range(n) for j in range(i+1, n)
                        if (ranks_pf[i] - ranks_pf[j]) * (ranks_sweep[i] - ranks_sweep[j]) > 0)
        total_pairs = n * (n - 1) / 2
        spearman = (2 * concordant / total_pairs - 1) if total_pairs > 0 else 0  # normalized to [-1, 1]

        w(f"  SPEARMAN RANK CORRELATION (Decile Rank vs Sweep Depth Rank):")
        w(f"    rho = {spearman:+.3f}")
        if spearman > 0.7:
            w("    STRONG positive monotonic relationship.")
        elif spearman > 0.4:
            w("    MODERATE positive monotonic relationship.")
        elif spearman > 0:
            w("    WEAK positive relationship.")
        else:
            w("    NO monotonic relationship detected.")
        w()

    # Exit reason by decile
    w("  Exit Reason by Decile:")
    w()
    for r in decile_results:
        if r["trades"] == 0:
            continue
        w(f"    {r['label']} ({r['trades']} trades):")
        for reason, label_display in [("profit_target", "Target"), ("stop_loss", "Stop"),
                                       ("time_exit", "Time"), ("trailing_stop", "Trail")]:
            count = r.get(f"{reason}_count", 0)
            if count > 0:
                pct = count / r["trades"] * 100
                avg_pnl = r.get(f"{reason}_avg_pnl", 0.0)
                w(f"      {label_display:<8}: {count:>3} ({pct:>5.1f}%)  avg {avg_pnl:>+7.2f} pips")
        w()

    return "\n".join(lines)


def format_test3_report(results: list) -> str:
    """Test 3: Time exit duration sweep."""
    lines = []
    def w(s=""): lines.append(s)

    w("=" * 76)
    w("  TEST 3: TIME EXIT DURATION SWEEP")
    w("  Question: How long does the reversal actually take?")
    w("  All tests use sweep >= 0.5 ATR filter, standard exit (1.5 ATR target)")
    w("=" * 76)
    w()
    w(f"  {'Max Bars':<12} {'Trades':>6} {'PF':>7} {'WR':>6} "
      f"{'Avg Pips':>9} {'Total $':>12} {'MaxDD%':>7} "
      f"{'Target%':>7} {'Stop%':>7} {'Time%':>7}")
    w(f"  {'-'*86}")

    for r in results:
        label = r["label"]
        trades = int(r.get("trades", 0))
        pf = r.get("profit_factor", np.nan)
        wr = r.get("win_rate", np.nan)
        avg = r.get("avg_pnl_pips", np.nan)
        total = r.get("total_pnl_usd", 0)
        mdd = r.get("max_dd_pct", 0)
        tgt_pct = r.get("profit_target_pct", 0) * 100
        stp_pct = r.get("stop_loss_pct", 0) * 100
        tim_pct = r.get("time_exit_pct", 0) * 100

        if not np.isnan(pf):
            w(f"  {label:<12} {trades:>6} {pf:>7.2f} {wr:>5.0%} "
              f"{avg:>+9.2f} {total:>+12,.0f} {mdd:>6.1f}% "
              f"{tgt_pct:>6.1f}% {stp_pct:>6.1f}% {tim_pct:>6.1f}%")
        else:
            w(f"  {label:<12} {trades:>6} {'N/A':>7} {'N/A':>6} "
              f"{'N/A':>9} {total:>+12,.0f} {'N/A':>7} "
              f"{'---':>7} {'---':>7} {'---':>7}")

    w()

    # Find optimal
    valid = [(r["label"], r.get("profit_factor", 0), r.get("total_pnl_usd", 0))
             for r in results if not np.isnan(r.get("profit_factor", 0))]
    if valid:
        best_label, best_pf, best_pnl = max(valid, key=lambda x: x[1])
        w(f"  BEST PF: {best_label} (PF {best_pf:.2f}, P&L ${best_pnl:>+,.0f})")
        best_pnl_label = max(valid, key=lambda x: x[2])
        if best_pnl_label[0] != best_label:
            w(f"  BEST P&L: {best_pnl_label[0]} (${best_pnl_label[2]:>+,.0f})")
    w()

    # Time exit P&L by duration
    w("  Time Exit P&L by Duration:")
    w()
    for r in results:
        te_count = r.get("time_exit_count", 0)
        if te_count > 0:
            te_pnl = r.get("time_exit_avg_pnl", 0.0)
            te_total = r.get("time_exit_total_pnl", 0.0)
            w(f"    {r['label']:<12}: {te_count:>3} time exits, "
              f"avg {te_pnl:>+7.2f} pips, total ${te_total:>+12,.0f}")

    w()
    return "\n".join(lines)


def format_test4_report(baseline_mae: dict, filtered_mae: dict,
                         winner_mae: dict, loser_mae: dict) -> str:
    """Test 4: MAE/MFE analysis."""
    lines = []
    def w(s=""): lines.append(s)

    w("=" * 76)
    w("  TEST 4: MAE/MFE ANALYSIS")
    w("  Question: Is the stop too tight? Target too ambitious?")
    w("  How much reversal actually occurs before the trade resolves?")
    w("=" * 76)
    w()

    # --- Overall MAE/MFE statistics ---
    for label, data in [("ALL 400 TRADES (no filter)", baseline_mae),
                        ("FILTERED (sweep >= 0.5)", filtered_mae)]:
        trades_df = data.get("trades_df", pd.DataFrame())
        if len(trades_df) == 0:
            continue

        w(f"  --- {label} ({len(trades_df)} trades) ---")
        w()

        # MAE/MFE summary stats
        w(f"  {'Statistic':<28} {'MAE (ATR)':>10} {'MFE (ATR)':>10} {'MAE (pips)':>11} {'MFE (pips)':>11}")
        w(f"  {'-'*72}")

        for stat_name, stat_func in [("Mean", "mean"), ("Median", "median"),
                                      ("Std Dev", "std"),
                                      ("25th pctile", lambda x: x.quantile(0.25)),
                                      ("75th pctile", lambda x: x.quantile(0.75)),
                                      ("90th pctile", lambda x: x.quantile(0.90)),
                                      ("95th pctile", lambda x: x.quantile(0.95)),
                                      ("Max", "max")]:
            mae = trades_df["mae_atr"].agg(stat_func)
            mfe = trades_df["mfe_atr"].agg(stat_func)
            mae_p = trades_df["mae_pips"].agg(stat_func)
            mfe_p = trades_df["mfe_pips"].agg(stat_func)

            if stat_func == "std":
                w(f"  {stat_name:<28} {mae:>10.3f} {mfe:>10.3f} {mae_p:>+10.1f} {mfe_p:>+10.1f}")
            elif stat_func in ("mean", "median"):
                w(f"  {stat_name:<28} {mae:>10.3f} {mfe:>10.3f} {mae_p:>+10.1f} {mfe_p:>+10.1f}")
            else:
                w(f"  {stat_name:<28} {mae:>10.3f} {mfe:>10.3f} {mae_p:>+10.1f} {mfe_p:>+10.1f}")

        w()

        # Key questions
        stop_atr = 1.0
        target_atr = 1.5
        pct_mae_ge_stop = (trades_df["mae_atr"] >= stop_atr).mean() * 100
        pct_mfe_ge_target = (trades_df["mfe_atr"] >= target_atr).mean() * 100
        pct_mfe_ge_1atr = (trades_df["mfe_atr"] >= 1.0).mean() * 100
        avg_mfe_atr = trades_df["mfe_atr"].mean()

        w(f"  DIAGNOSTIC QUESTIONS:")
        w(f"    Q: How often did price move >= 1.0 ATR against us (touch stop zone)?")
        w(f"    A: {pct_mae_ge_stop:.1f}% of trades ({int(pct_mae_ge_stop * len(trades_df) / 100)} / {len(trades_df)})")
        w()
        w(f"    Q: How often did price move >= 1.5 ATR in our favor (reach target zone)?")
        w(f"    A: {pct_mfe_ge_target:.1f}% of trades ({int(pct_mfe_ge_target * len(trades_df) / 100)} / {len(trades_df)})")
        w()
        w(f"    Q: How often did price move >= 1.0 ATR in our favor?")
        w(f"    A: {pct_mfe_ge_1atr:.1f}% of trades ({int(pct_mfe_ge_1atr * len(trades_df) / 100)} / {len(trades_df)})")
        w()
        w(f"    Q: Average MFE (how far does price actually go in our favor)?")
        w(f"    A: {avg_mfe_atr:.3f} ATR")
        w()

        # MAE/MFE by exit reason
        w(f"  MAE/MFE BY EXIT REASON:")
        w()
        w(f"  {'Exit Reason':<18} {'Count':>5} {'Avg MAE':>8} {'Avg MFE':>8} {'Avg PnL':>9} {'WR':>6}")
        w(f"  {'-'*56}")

        for reason in ["profit_target", "stop_loss", "time_exit", "trailing_stop"]:
            subset = trades_df[trades_df["exit_reason"] == reason]
            if len(subset) == 0:
                continue
            avg_mae = subset["mae_atr"].mean()
            avg_mfe = subset["mfe_atr"].mean()
            avg_pnl = subset["pnl_pips"].mean()
            wr = (subset["pnl_usd"] > 0).mean()
            w(f"  {reason:<18} {len(subset):>5} {avg_mae:>7.3f} {avg_mfe:>7.3f} "
              f"{avg_pnl:>+8.2f} {wr:>5.0%}")

        w()

    # --- Winner vs Loser comparison ---
    w(f"  --- WINNER vs LOSER CHARACTERISTICS ---")
    w()

    for label, data in [("ALL TRADES", baseline_mae), ("FILTERED", filtered_mae)]:
        trades_df = data.get("trades_df", pd.DataFrame())
        if len(trades_df) == 0:
            continue

        winners = trades_df[trades_df["pnl_usd"] > 0]
        losers = trades_df[trades_df["pnl_usd"] <= 0]

        if len(winners) == 0 or len(losers) == 0:
            continue

        w(f"  {label} ({len(winners)} winners / {len(losers)} losers):")
        w(f"  {'Characteristic':<30} {'Winners':>12} {'Losers':>12} {'Delta':>12}")
        w(f"  {'-'*68}")

        comparisons = [
            ("Avg Entry Dist (ATR)", "sweep_entry_dist_atr"),
            ("Avg Wick Depth (ATR)", "sweep_depth_atr"),
            ("Avg MAE (ATR)", "mae_atr"),
            ("Avg MFE (ATR)", "mfe_atr"),
            ("Avg MAE (pips)", "mae_pips"),
            ("Avg MFE (pips)", "mfe_pips"),
            ("Avg Bars Held", "bars_held"),
            ("Avg Reversal Distance", "reversal_distance"),
        ]

        for comp_label, col in comparisons:
            if col in winners.columns and col in losers.columns:
                w_val = winners[col].mean()
                l_val = losers[col].mean()
                delta = w_val - l_val
                if col.endswith("_atr") or col.endswith("depth"):
                    w(f"  {comp_label:<30} {w_val:>11.3f} {l_val:>11.3f} {delta:>+11.3f}")
                else:
                    w(f"  {comp_label:<30} {w_val:>11.1f} {l_val:>11.1f} {delta:>+11.1f}")

        # MAE/MFE ratio
        if "mae_atr" in winners.columns and "mfe_atr" in winners.columns:
            w_ratio_w = winners["mfe_atr"].mean() / winners["mae_atr"].mean() if winners["mae_atr"].mean() > 0 else np.nan
            w_ratio_l = losers["mfe_atr"].mean() / losers["mae_atr"].mean() if losers["mae_atr"].mean() > 0 else np.nan
            w(f"  {'MFE/MAE Ratio':<30} {w_ratio_w:>11.3f} {w_ratio_l:>11.3f} {w_ratio_w - w_ratio_l:>+11.3f}")

        w()

    # --- MAE/MFE heatmap by sweep depth bucket ---
    w(f"  --- MAE/MFE BY SWEEP DEPTH BUCKET (all 400 trades) ---")
    w()
    w(f"  {'Sweep Bucket':<18} {'N':>5} {'Avg MAE':>8} {'Med MAE':>8} "
      f"{'Avg MFE':>8} {'Med MFE':>8} {'MFE/MAE':>8} {'PF':>7}")
    w(f"  {'-'*66}")

    buckets = [
        ("0.00-0.25", 0.0, 0.25),
        ("0.25-0.50", 0.25, 0.50),
        ("0.50-0.75", 0.50, 0.75),
        ("0.75-1.00", 0.75, 1.00),
        ("1.00-1.50", 1.00, 1.50),
        ("1.50-2.00", 1.50, 2.00),
        ("2.00+",      2.00, 999.0),
    ]

    trades_df = baseline_mae.get("trades_df", pd.DataFrame())
    for bucket_label, lo, hi in buckets:
        if "sweep_depth_atr" not in trades_df.columns:
            break
        subset = trades_df[(trades_df["sweep_depth_atr"] >= lo) & (trades_df["sweep_depth_atr"] < hi)]
        if len(subset) == 0:
            w(f"  {bucket_label:<18} {0:>5} {'---':>8} {'---':>8} {'---':>8} {'---':>8} {'---':>8} {'---':>7}")
            continue
        n = len(subset)
        avg_mae = subset["mae_atr"].mean()
        med_mae = subset["mae_atr"].median()
        avg_mfe = subset["mfe_atr"].mean()
        med_mfe = subset["mfe_atr"].median()
        ratio = avg_mfe / avg_mae if avg_mae > 0 else np.nan
        wins = subset[subset["pnl_usd"] > 0]
        loss = subset[subset["pnl_usd"] <= 0]
        pf = abs(wins["pnl_usd"].sum() / loss["pnl_usd"].sum()) if len(loss) > 0 and loss["pnl_usd"].sum() != 0 else np.nan

        pf_str = f"{pf:.2f}" if not np.isnan(pf) else "---"
        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "---"
        w(f"  {bucket_label:<18} {n:>5} {avg_mae:>7.3f} {med_mae:>7.3f} "
          f"{avg_mfe:>7.3f} {med_mfe:>7.3f} {ratio_str:>8} {pf_str:>7}")

    w()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = Path("/home/z/my-project/download")

    print("Loading data...")
    ohlc_df = pd.read_csv(data_dir / "eurusd_1h_research.csv", parse_dates=["timestamp"])
    ohlc_df.set_index("timestamp", inplace=True)
    ohlc_df.sort_index(inplace=True)

    config = StrategyConfig()
    initial_equity = 100_000.0

    print(f"Bars: {len(ohlc_df):,}")
    print(f"Range: {ohlc_df.index[0]} to {ohlc_df.index[-1]}")
    print()

    report_sections = []

    # ================================================================
    # TEST 1: Continuous Sweep Depth Buckets
    # ================================================================
    print("=" * 60)
    print("  TEST 1: Continuous Sweep Depth Buckets")
    print("=" * 60)

    thresholds = [0.0, 0.50, 0.75, 1.00, 1.25]
    test1_results = []

    for thresh in thresholds:
        label = f"sweep >= {thresh:.2f}" if thresh > 0 else "No filter"
        print(f"  Running: {label}...", end=" ", flush=True)
        result = run_experiment_mae_mfe(
            ohlc_df, config,
            min_sweep_depth=thresh,
            exit_mode="standard",
            max_bars_held_override=6,
            initial_equity=initial_equity,
            label=label,
            disable_risk_limits=True,
        )
        pf = result.get("profit_factor", np.nan)
        n = result.get("trades", 0)
        pnl = result.get("total_pnl_usd", 0)
        print(f"PF={pf:.2f}, {n} trades, P&L ${pnl:+,.0f}")
        test1_results.append(result)

    # Save individual trade CSVs
    for r in test1_results:
        trades_df = r.get("trades_df", pd.DataFrame())
        if len(trades_df) > 0:
            safe_name = r["label"].replace(" ", "_").replace(">=", "gte").replace(".", "")
            trades_df.to_csv(output_dir / f"test1_{safe_name}_trades.csv", index=False)

    report1 = format_test1_report(test1_results)
    print(report1)
    report_sections.append(report1)

    # ================================================================
    # TEST 2: Sweep-Depth Decile Analysis
    # ================================================================
    print("=" * 60)
    print("  TEST 2: Sweep-Depth Decile Analysis")
    print("=" * 60)

    # Use the no-filter baseline (400 trades) with MAE/MFE data
    all_trades = test1_results[0].get("trades_df", pd.DataFrame())

    # Run decile analysis on BOTH sweep metrics:
    # 1. sweep_entry_dist_atr: abs(entry - liquidity)/atr (matches enrichment, showed monotonic PF)
    # 2. sweep_depth_atr: abs(liquidity - low)/atr (true wick penetration)

    decile_results = []
    if len(all_trades) > 0 and "sweep_entry_dist_atr" in all_trades.columns:
        all_trades["decile"] = pd.qcut(
            all_trades["sweep_entry_dist_atr"],
            q=10, labels=False, duplicates="drop"
        )

        for d in range(10):
            subset = all_trades[all_trades["decile"] == d]
            if len(subset) == 0:
                decile_results.append({
                    "label": f"D{d+1}",
                    "trades": 0,
                    "profit_factor": np.nan,
                    "win_rate": np.nan,
                    "avg_pnl_pips": np.nan,
                    "total_pnl_usd": 0,
                    "avg_sweep": np.nan,
                    "avg_wick_depth": np.nan,
                    "avg_mae": np.nan,
                    "avg_mfe": np.nan,
                })
                continue

            winners = subset[subset["pnl_usd"] > 0]
            losers = subset[subset["pnl_usd"] <= 0]

            decile_data = {
                "label": f"D{d+1}",
                "trades": len(subset),
                "profit_factor": (
                    abs(winners["pnl_usd"].sum() / losers["pnl_usd"].sum())
                    if len(losers) > 0 and losers["pnl_usd"].sum() != 0 else np.nan
                ),
                "win_rate": (subset["pnl_usd"] > 0).mean(),
                "avg_pnl_pips": subset["pnl_pips"].mean(),
                "total_pnl_usd": subset["pnl_usd"].sum(),
                "avg_sweep": subset["sweep_entry_dist_atr"].mean(),
                "avg_wick_depth": subset["sweep_depth_atr"].mean(),
                "avg_mae": subset["mae_atr"].mean(),
                "avg_mfe": subset["mfe_atr"].mean(),
            }

            for reason in ["profit_target", "stop_loss", "time_exit", "trailing_stop"]:
                r_subset = subset[subset["exit_reason"] == reason]
                decile_data[f"{reason}_count"] = len(r_subset)
                decile_data[f"{reason}_avg_pnl"] = r_subset["pnl_pips"].mean() if len(r_subset) > 0 else 0.0

            decile_results.append(decile_data)

        # Print decile ranges for both metrics
        print("  Decile ranges (sweep_entry_dist_atr = enrichment-style metric):")
        for d in range(10):
            subset = all_trades[all_trades["decile"] == d]
            if len(subset) > 0:
                print(f"    D{d+1}: entry_dist {subset['sweep_entry_dist_atr'].min():.3f} - {subset['sweep_entry_dist_atr'].max():.3f} ATR "
                      f"| wick_depth {subset['sweep_depth_atr'].mean():.3f} avg "
                      f"({len(subset)} trades)")
        print()

    else:
        print("  ERROR: No trades or sweep columns available for decile analysis.")

    report2 = format_test2_report(decile_results)
    print(report2)
    report_sections.append(report2)

    # ================================================================
    # TEST 3: Time Exit Duration Sweep
    # ================================================================
    print("=" * 60)
    print("  TEST 3: Time Exit Duration Sweep")
    print("=" * 60)

    durations = [4, 6, 8, 10, 12]
    test3_results = []

    for dur in durations:
        label = f"{dur} bars"
        print(f"  Running: {label}...", end=" ", flush=True)
        result = run_experiment_mae_mfe(
            ohlc_df, config,
            min_sweep_depth=0.5,  # Apply filter from Experiment A
            exit_mode="standard",  # Keep 1.5 ATR target
            max_bars_held_override=dur,
            initial_equity=initial_equity,
            label=label,
            disable_risk_limits=True,
        )
        pf = result.get("profit_factor", np.nan)
        n = result.get("trades", 0)
        pnl = result.get("total_pnl_usd", 0)

        # Add percentage breakdowns
        if n > 0:
            result["profit_target_pct"] = result.get("profit_target_pct", result.get("profit_target_count", 0) / n)
            result["stop_loss_pct"] = result.get("stop_loss_pct", result.get("stop_loss_count", 0) / n)
            result["time_exit_pct"] = result.get("time_exit_pct", result.get("time_exit_count", 0) / n)

        print(f"PF={pf:.2f}, {n} trades, P&L ${pnl:+,.0f}")
        test3_results.append(result)

    # Also run time-only exit variants at each duration
    print()
    print("  Also testing time-only exit at each duration...")
    for dur in durations:
        label = f"{dur} bars (no target)"
        print(f"  Running: {label}...", end=" ", flush=True)
        result = run_experiment_mae_mfe(
            ohlc_df, config,
            min_sweep_depth=0.5,
            exit_mode="time_only",
            max_bars_held_override=dur,
            initial_equity=initial_equity,
            label=label,
            disable_risk_limits=True,
        )
        pf = result.get("profit_factor", np.nan)
        n = result.get("trades", 0)
        pnl = result.get("total_pnl_usd", 0)

        if n > 0:
            result["profit_target_pct"] = result.get("profit_target_pct", result.get("profit_target_count", 0) / n)
            result["stop_loss_pct"] = result.get("stop_loss_pct", result.get("stop_loss_count", 0) / n)
            result["time_exit_pct"] = result.get("time_exit_pct", result.get("time_exit_count", 0) / n)

        print(f"PF={pf:.2f}, {n} trades, P&L ${pnl:+,.0f}")
        test3_results.append(result)

    report3 = format_test3_report(test3_results)
    print(report3)
    report_sections.append(report3)

    # ================================================================
    # TEST 4: MAE/MFE Analysis
    # ================================================================
    print("=" * 60)
    print("  TEST 4: MAE/MFE Analysis")
    print("=" * 60)

    # We already have MAE/MFE data from Test 1 (baseline = all 400, filtered = sweep >= 0.5)
    baseline_mae = test1_results[0]
    filtered_mae = test1_results[1]  # sweep >= 0.5

    # Winner/loser split on filtered
    filtered_df = filtered_mae.get("trades_df", pd.DataFrame())
    if len(filtered_df) > 0:
        winners_df = filtered_df[filtered_df["pnl_usd"] > 0]
        losers_df = filtered_df[filtered_df["pnl_usd"] <= 0]

        winner_summary = {"label": "Winners", "trades_df": winners_df}
        loser_summary = {"label": "Losers", "trades_df": losers_df}
    else:
        winner_summary = {"label": "Winners", "trades_df": pd.DataFrame()}
        loser_summary = {"label": "Losers", "trades_df": pd.DataFrame()}

    # Save MAE/MFE enriched trades
    all_mae_df = baseline_mae.get("trades_df", pd.DataFrame())
    if len(all_mae_df) > 0:
        all_mae_df.to_csv(output_dir / "test4_mae_mfe_all_trades.csv", index=False)
        print(f"  Saved: test4_mae_mfe_all_trades.csv ({len(all_mae_df)} trades)")

    filtered_mae_df = filtered_mae.get("trades_df", pd.DataFrame())
    if len(filtered_mae_df) > 0:
        filtered_mae_df.to_csv(output_dir / "test4_mae_mfe_filtered_trades.csv", index=False)
        print(f"  Saved: test4_mae_mfe_filtered_trades.csv ({len(filtered_mae_df)} trades)")

    report4 = format_test4_report(baseline_mae, filtered_mae,
                                   winner_summary, loser_summary)
    print(report4)
    report_sections.append(report4)

    # ================================================================
    # Combined Report
    # ================================================================
    combined = []
    combined.append("=" * 76)
    combined.append("  LSR-EH1 EDGE CHARACTERIZATION — COMPREHENSIVE REPORT")
    combined.append("  Four hypothesis tests. NO parameter optimization.")
    combined.append("=" * 76)
    combined.append("")
    combined.append("  Test 1: Continuous sweep depth (0.50, 0.75, 1.00, 1.25)")
    combined.append("    Verifies monotonic PF improvement with deeper sweeps.")
    combined.append("")
    combined.append("  Test 2: Sweep-depth deciles (D1-D10)")
    combined.append("    Structural relationship test — stronger than single threshold.")
    combined.append("")
    combined.append("  Test 3: Time exit duration (4, 6, 8, 10, 12 bars)")
    combined.append("    With and without profit target. Finds optimal holding period.")
    combined.append("")
    combined.append("  Test 4: MAE/MFE analysis")
    combined.append("    Maximum adverse/favorable excursion — the highest-value")
    combined.append("    remaining analysis for stop/target calibration.")
    combined.append("")
    combined.append("  All tests use sweep >= 0.5 ATR filter except Test 1 and Test 2")
    combined.append("  which sweep across multiple thresholds on all 400 trades.")
    combined.append("")

    for section in report_sections:
        combined.append(section)
        combined.append("")

    full_report = "\n".join(combined)

    report_path = output_dir / "edge_characterization_report.txt"
    report_path.write_text(full_report)
    print(f"Full report saved: {report_path}")


if __name__ == "__main__":
    main()
