"""
LSR-EH1 Sweep Depth Filter Experiments
========================================

Two clean hypothesis tests. NO parameter optimization.

Experiment A: Does sweep_depth >= 0.5 ATR filter improve PF?
    - Everything identical except one additional signal filter
    - Objective: "Are shallow sweeps responsible for most losses?"

Experiment B: On filtered signals, which exit strategy works best?
    - 1.5 ATR target  (current baseline)
    - 1.0 ATR target  (lower bar)
    - Trailing stop    (1.0 ATR trail from high water mark, no target)
    - Time-only exit   (no target, just stop + 6-bar time)

These are regime FILTERS, not parameter optimizations.
The sweep depth evidence from the regime investigation was monotonic:
    0-0.25 ATR: PF 0.10
    0.25-0.5:   PF 0.49
    0.5-1.0:    PF 1.08
    1.0-2.0:    PF 1.29
    2.0+:       PF 4.66

This test asks whether that relationship holds as a forward-looking filter.
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
# Signal Generation with Optional Sweep Depth Filter
# ---------------------------------------------------------------------------

def generate_signals_filtered(
    df: pd.DataFrame,
    config: StrategyConfig,
    min_sweep_depth: float = 0.0,
) -> pd.DataFrame:
    """Generate signals with optional sweep depth filter.

    Adds sweep_depth columns and filters signals that don't meet minimum depth.
    """
    df = generate_signals(df, config)

    if min_sweep_depth > 0:
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

        df["long_signal"] = (
            df["long_signal"] & (df["sweep_depth_long"] >= min_sweep_depth)
        )
        df["short_signal"] = (
            df["short_signal"] & (df["sweep_depth_short"] >= min_sweep_depth)
        )

    return df


# ---------------------------------------------------------------------------
# Experiment Backtest Runner
# ---------------------------------------------------------------------------

def run_experiment(
    df: pd.DataFrame,
    config: StrategyConfig,
    min_sweep_depth: float = 0.0,
    exit_mode: str = "standard",
    trail_atr: float = 1.0,
    initial_equity: float = 100_000.0,
    label: str = "",
    disable_risk_limits: bool = True,
) -> dict:
    """Run a single backtest experiment.

    Parameters
    ----------
    exit_mode : str
        "standard"     - 1.5 ATR target, 1.0 ATR stop, 6-bar time
        "target_1atr"  - 1.0 ATR target, 1.0 ATR stop, 6-bar time
        "trailing"     - 1.0 ATR trailing from HWM, 1.0 ATR initial stop, no target
        "time_only"    - No target, 1.0 ATR stop, 6-bar time

    Returns
    -------
    dict with trade-level and summary statistics
    """
    df = generate_signals_filtered(df.copy(), config, min_sweep_depth)

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
    max_bars_held = cfg_exit.max_bars_held

    atr_first_valid = df["atr"].first_valid_index()
    if atr_first_valid is None:
        return {"label": label, "trades": pd.DataFrame(), "total_pnl": 0.0}
    start_idx = df.index.get_loc(atr_first_valid)

    n_bars = len(df)
    equity = initial_equity
    peak_equity = initial_equity
    curve_peak = initial_equity
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
                    }
                    trade_id_counter += 1
                    bars_held_counter = 0
                    hwm = fill_price
                    trailing_stop_price = sl

                    if direction == 1:
                        last_long_entry_bar = i
                    else:
                        last_short_entry_bar = i

        # --- Exit checks ---
        if current_position is not None:
            pos = current_position
            d = pos["direction"]
            bars_held_counter += 1

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

            # (a) Profit target (standard and target_1atr only)
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

            # (c) Fixed stop loss (all modes)
            if exit_price is None:
                if d == 1 and l <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"
                elif d == -1 and h >= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"

            # (d) Time exit (all modes)
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
                    "hwm": hwm if exit_mode == "trailing" else np.nan,
                    "trailing_stop_exit": trailing_stop_price if exit_mode == "trailing" else np.nan,
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
# Report Generation
# ---------------------------------------------------------------------------

def print_experiment_a(baseline: dict, filtered: dict) -> str:
    """Experiment A report."""
    lines = []
    def w(s=""): lines.append(s)

    w("=" * 72)
    w("  EXPERIMENT A: Sweep Depth Filter")
    w("  Question: Are shallow sweeps responsible for most losses?")
    w("=" * 72)
    w()
    w(f"  {'Metric':<22} {'No Filter':>14} {'sweep >= 0.5':>14} {'Delta':>14}")
    w(f"  {'-'*64}")

    metrics = [
        ("Trades", "trades", True),
        ("Profit Factor", "profit_factor", False),
        ("Win Rate", "win_rate", False),
        ("Avg P&L (pips)", "avg_pnl_pips", False),
        ("Total P&L ($)", "total_pnl_usd", False),
        ("Max Drawdown (%)", "max_dd_pct", False),
        ("Expectancy ($)", "expectancy", False),
    ]

    for label, key, is_int in metrics:
        v1 = baseline.get(key, np.nan)
        v2 = filtered.get(key, np.nan)
        delta = v2 - v1 if not np.isnan(v1) and not np.isnan(v2) else np.nan

        if is_int:
            w(f"  {label:<22} {int(v1):>14} {int(v2):>14} {int(delta):>+14}")
        else:
            w(f"  {label:<22} {v1:>14.2f} {v2:>14.2f} {delta:>+14.2f}")

    w()
    w("  Exit Reason Breakdown:")
    w()

    for label, data in [("No Filter", baseline), ("sweep >= 0.5", filtered)]:
        w(f"    {label}:")
        for reason in ["profit_target", "stop_loss", "time_exit"]:
            count = data.get(f"{reason}_count", 0)
            pct = data.get(f"{reason}_pct", 0.0) * 100
            avg_pnl = data.get(f"{reason}_avg_pnl", 0.0)
            total_pnl = data.get(f"{reason}_total_pnl", 0.0)
            w(f"      {reason:<16}: {count:>3} ({pct:>5.1f}%)  "
              f"avg {avg_pnl:>+7.2f} pips  total ${total_pnl:>+12,.0f}")
        w()

    w("  CONCLUSION:")
    if filtered.get("profit_factor", 0) > 1.0:
        w("  YES -- shallow sweeps are responsible for most losses.")
        w(f"  Removing trades with sweep < 0.5 ATR improved PF from "
          f"{baseline.get('profit_factor', 0):.2f} to {filtered.get('profit_factor', 0):.2f}")
        w(f"  and eliminated {baseline.get('trades', 0) - filtered.get('trades', 0)} unprofitable trades.")
    else:
        w("  INCONCLUSIVE -- filter did not push PF above 1.0.")
        w("  The monotonic sweep-depth relationship may not be exploitable as a filter.")

    w()
    return "\n".join(lines)


def print_experiment_b(results: list) -> str:
    """Experiment B report."""
    lines = []
    def w(s=""): lines.append(s)

    w("=" * 72)
    w("  EXPERIMENT B: Exit Strategy Comparison (on filtered signals)")
    w("  Question: Which exit strategy captures the reversal edge best?")
    w("=" * 72)
    w()
    w(f"  {'Variant':<22} {'Trades':>6} {'PF':>7} {'WR':>6} {'Avg Pips':>9} {'Total $':>12} {'MaxDD%':>7}")
    w(f"  {'-'*72}")

    for r in results:
        label = r["label"]
        trades = int(r.get("trades", 0))
        pf = r.get("profit_factor", np.nan)
        wr = r.get("win_rate", np.nan)
        avg = r.get("avg_pnl_pips", np.nan)
        total = r.get("total_pnl_usd", 0)
        mdd = r.get("max_dd_pct", 0)
        marker = " <<<" if pf == max(r2.get("profit_factor", 0) for r2 in results if not np.isnan(r2.get("profit_factor", 0))) else ""
        w(f"  {label:<22} {trades:>6} {pf:>7.2f} {wr:>5.0%} {avg:>+9.2f} {total:>+12,.0f} {mdd:>6.1f}%{marker}")

    w()
    w("  Exit Type Breakdown per Variant:")
    w()

    for r in results:
        w(f"  {r['label']}:")
        for reason in ["profit_target", "stop_loss", "time_exit", "trailing_stop"]:
            count = r.get(f"{reason}_count", 0)
            if count > 0:
                pct = r.get(f"{reason}_pct", 0.0) * 100
                avg_pnl = r.get(f"{reason}_avg_pnl", 0.0)
                total_pnl = r.get(f"{reason}_total_pnl", 0.0)
                w(f"    {reason:<16}: {count:>3} ({pct:>5.1f}%)  "
                  f"avg {avg_pnl:>+7.2f} pips  total ${total_pnl:>+12,.0f}")
        w()

    # Find best PF
    valid = [(r["label"], r.get("profit_factor", 0)) for r in results
             if not np.isnan(r.get("profit_factor", 0)) and r.get("profit_factor", 0) > 0]
    if valid:
        best_label, best_pf = max(valid, key=lambda x: x[1])
        w(f"  BEST EXIT: {best_label} (PF {best_pf:.2f})")
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

    # ================================================================
    # EXPERIMENT A: Sweep Depth Filter
    # ================================================================
    print("Running Experiment A...")

    baseline = run_experiment(
        ohlc_df, config,
        min_sweep_depth=0.0,
        exit_mode="standard",
        initial_equity=initial_equity,
        label="No Filter (baseline)",
        disable_risk_limits=True,
    )

    filtered = run_experiment(
        ohlc_df, config,
        min_sweep_depth=0.5,
        exit_mode="standard",
        initial_equity=initial_equity,
        label="sweep >= 0.5 ATR",
        disable_risk_limits=True,
    )

    report_a = print_experiment_a(baseline, filtered)
    print(report_a)

    # Save filtered trades
    trades_out = filtered.get("trades_df", pd.DataFrame())
    if len(trades_out) > 0:
        trades_out.to_csv(output_dir / "exp_a_filtered_trades.csv", index=False)
        print(f"  Saved: {output_dir / 'exp_a_filtered_trades.csv'}")
    print()

    # ================================================================
    # EXPERIMENT B: Exit Strategy Variants (on filtered signals)
    # ================================================================
    print("Running Experiment B (4 variants)...")

    variants = [
        ("1.5 ATR target", "standard", 0.5),
        ("1.0 ATR target", "target_1atr", 0.5),
        ("Trailing 1.0 ATR", "trailing", 0.5),
        ("Time-only exit", "time_only", 0.5),
    ]

    results_b = []
    for label, exit_mode, sweep_filter in variants:
        print(f"  Running: {label}...", end=" ", flush=True)
        result = run_experiment(
            ohlc_df, config,
            min_sweep_depth=sweep_filter,
            exit_mode=exit_mode,
            trail_atr=1.0,
            initial_equity=initial_equity,
            label=label,
            disable_risk_limits=True,
        )
        print(f"PF={result.get('profit_factor', 0):.2f}, "
              f"{result.get('trades', 0)} trades, "
              f"P&L ${result.get('total_pnl_usd', 0):+,.0f}")
        results_b.append(result)

    # Save variant trades
    for r in results_b:
        trades_df = r.get("trades_df", pd.DataFrame())
        if len(trades_df) > 0:
            safe_name = r["label"].replace(" ", "_").replace(".", "")
            trades_df.to_csv(output_dir / f"exp_b_{safe_name}_trades.csv", index=False)

    report_b = print_experiment_b(results_b)
    print(report_b)

    # ================================================================
    # Combined Report
    # ================================================================
    combined = []
    combined.append(report_a)
    combined.append(report_b)
    combined.append("\n")
    combined.append("=" * 72)
    combined.append("  METHODOLOGY NOTE")
    combined.append("=" * 72)
    combined.append("")
    combined.append("  These experiments test a single causal variable (sweep depth)")
    combined.append("  identified from regime analysis. They do NOT optimize parameters.")
    combined.append("")
    combined.append("  Experiment A asks: are shallow sweeps noise, not signal?")
    combined.append("    - If PF crosses 1.0 with filter, the hypothesis is supported.")
    combined.append("    - If PF stays below 1.0, the sweep-depth relationship may be")
    combined.append("      a descriptive artifact, not an exploitable filter.")
    combined.append("")
    combined.append("  Experiment B asks: given filtered signals, what's the right")
    combined.append("    way to harvest the reversal? This tests exit logic,")
    combined.append("    not entry logic. The time-exit data (+2.64 pips avg)")
    combined.append("    suggests the current 1.5 ATR target may be too ambitious.")
    combined.append("")

    full_report = "\n".join(combined)

    # Save full report
    report_path = output_dir / "sweep_depth_experiment_report.txt"
    report_path.write_text(full_report)
    print(f"Full report saved: {report_path}")


if __name__ == "__main__":
    main()
