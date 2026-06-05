"""
LSR-EH1 — Liquidity Sweep Reversal: EURUSD Hourly

Main entry point. Orchestrates:
1. Data loading (Dukascopy ticks, OHLC CSV, or synthetic)
2. Signal frequency analysis
3. Backtest execution
4. Performance reporting with reversal distance distribution
5. Walk-forward analysis
6. Cost stress testing
7. Equity curve + trade log export
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import StrategyConfig
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, print_report
from validation.walk_forward import WalkForwardRunner
from backtest.cost_stress import CostStressRunner, COST_SCENARIOS
from data.loader import load_dukascopy_ticks, load_ohlc_csv, validate_ohlc


# ---------------------------------------------------------------------------
# Synthetic Data Generator (for integration testing only)
# ---------------------------------------------------------------------------

def generate_synthetic_eurusd(
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic EURUSD hourly OHLC for integration testing.

    NOT suitable for alpha validation. Use real data for that.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    all_hours = pd.date_range(start, end, freq="1h")
    timestamps = all_hours[all_hours.dayofweek < 5]
    n = len(timestamps)

    dt = 1.0 / (252 * 24)
    theta = 3.0
    long_term_mean = 1.1200

    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    closes = np.empty(n)

    opens[0] = 1.1200
    closes[0] = 1.1200 + rng.normal(0, 0.0003)
    highs[0] = max(opens[0], closes[0]) + abs(rng.normal(0, 0.0004))
    lows[0] = min(opens[0], closes[0]) - abs(rng.normal(0, 0.0004))

    vol = 0.0008
    vol_mean = 0.0008
    kappa = 0.002

    for t in range(1, n):
        vol += kappa * (vol_mean - vol) * dt + vol * 0.3 * abs(rng.standard_normal()) * np.sqrt(dt)
        vol = np.clip(vol, 0.0002, 0.005)
        if rng.random() < 0.003:
            vol *= rng.uniform(2.0, 5.0)
        bar_vol = vol * rng.uniform(0.5, 3.0)
        opens[t] = closes[t-1] + rng.normal(0, vol * 0.3)
        bar_type = rng.random()
        if bar_type < 0.50:
            body = rng.normal(0, bar_vol * 0.5)
            closes[t] = opens[t] + body
            wick = bar_vol * rng.uniform(0.2, 0.8)
            highs[t] = max(opens[t], closes[t]) + wick
            lows[t] = min(opens[t], closes[t]) - wick
        elif bar_type < 0.75:
            direction = 1 if rng.random() > 0.5 else -1
            body = direction * bar_vol * rng.uniform(0.3, 1.0)
            closes[t] = opens[t] + body
            wick_ext = bar_vol * rng.uniform(0.1, 0.5)
            if direction > 0:
                highs[t] = closes[t] + wick_ext
                lows[t] = opens[t] - wick_ext
            else:
                highs[t] = opens[t] + wick_ext
                lows[t] = closes[t] - wick_ext
        else:
            sweep_depth = bar_vol * rng.uniform(1.5, 4.0)
            if rng.random() > 0.5:
                lows[t] = opens[t] - sweep_depth
                closes[t] = lows[t] + sweep_depth * rng.uniform(0.5, 0.95)
                highs[t] = max(opens[t], closes[t]) + abs(rng.normal(0, bar_vol * 0.3))
            else:
                highs[t] = opens[t] + sweep_depth
                closes[t] = highs[t] - sweep_depth * rng.uniform(0.5, 0.95)
                lows[t] = min(opens[t], closes[t]) - abs(rng.normal(0, bar_vol * 0.3))
        closes[t] += theta * (long_term_mean - closes[t]) * dt
        highs[t] = max(highs[t], opens[t], closes[t])
        lows[t] = min(lows[t], opens[t], closes[t])
        highs[t] = max(highs[t], lows[t] + 1e-6)

    df = pd.DataFrame(
        {"open": np.round(opens, 5), "high": np.round(highs, 5),
         "low": np.round(lows, 5), "close": np.round(closes, 5)},
        index=timestamps,
    )
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# Reversal Distance Analysis
# ---------------------------------------------------------------------------

def analyze_reversal_distance(trades) -> None:
    """Print reversal distance distribution for all closed trades.

    reversal_distance = abs(exit_price - entry_price) / atr_entry

    This tells us how far price typically reverses after a sweep,
    which is the key input for setting profit target and stop loss.
    """
    rd = [t.reversal_distance for t in trades
          if t.reversal_distance is not None and np.isfinite(t.reversal_distance)]
    if not rd:
        print("\n  ── Reversal Distance ─────────────────────────")
    print("  ── Reversal Distance Distribution ────────────")
    print(f"    Trades with data:  {len(rd)}")
    print(f"    Mean:              {np.mean(rd):.3f} ATR")
    print(f"    Std:               {np.std(rd):.3f} ATR")
    print(f"    10th percentile:   {np.percentile(rd, 10):.3f} ATR")
    print(f"    25th percentile:   {np.percentile(rd, 25):.3f} ATR")
    print(f"    50th percentile:   {np.percentile(rd, 50):.3f} ATR")
    print(f"    75th percentile:   {np.percentile(rd, 75):.3f} ATR")
    print(f"    90th percentile:   {np.percentile(rd, 90):.3f} ATR")
    print(f"    95th percentile:   {np.percentile(rd, 95):.3f} ATR")
    print(f"    Max:               {np.max(rd):.3f} ATR")
    print()

    # Contextual guidance
    median_rd = np.median(rd)
    p75_rd = np.percentile(rd, 75)
    p90_rd = np.percentile(rd, 90)

    if median_rd < 0.6:
        print("  >> WARNING: Median reversal is < 0.6 ATR.")
        print("     The 1.5 ATR profit target may be too aggressive.")
        print("     Consider reducing to 0.8-1.0 ATR.")
    elif median_rd < 1.0:
        print("  >> NOTE: Median reversal is < 1.0 ATR.")
        print("     The 1.5 ATR target captures < 50% of reversals.")
        print("     Consider 1.0-1.2 ATR for higher hit rate.")
    elif median_rd > 1.5:
        print("  >> NOTE: Median reversal exceeds 1.5 ATR.")
        print("     The target may be conservative. Consider 2.0 ATR.")
    else:
        print("  >> OK: Median reversal is within 1.0-1.5 ATR range.")
        print("     Current 1.5 ATR target is reasonable.")

    print()
    print(f"  Current settings: PT = 1.5 ATR, SL = 1.0 ATR")
    print(f"    Trades reaching PT (>={config.exit.profit_target_atr} ATR): "
          f"{sum(1 for r in rd if r >= config.exit.profit_target_atr)} / {len(rd)} "
          f"({sum(1 for r in rd if r >= config.exit.profit_target_atr)/len(rd)*100:.0f}%)")
    print(f"    Trades hitting SL (>={config.exit.stop_loss_atr} ATR adverse): "
          f"{sum(1 for r in rd if r >= config.exit.stop_loss_atr)} / {len(rd)} "
          f"(direction-dependent)")


# ---------------------------------------------------------------------------
# Signal Frequency Analysis
# ---------------------------------------------------------------------------

def analyze_signal_frequency(df: pd.DataFrame, config: StrategyConfig) -> None:
    """Analyze raw signal frequency before running backtest."""
    from strategy.liquidity_sweep import generate_signals

    df_sig = generate_signals(df.copy(), config)
    valid = df_sig.dropna(subset=["long_signal", "short_signal"])

    long_count = valid["long_signal"].sum()
    short_count = valid["short_signal"].sum()
    total_signals = long_count + short_count

    # Weekly signal count
    weekly = valid.resample("W").agg(
        long_sig=("long_signal", "sum"),
        short_sig=("short_signal", "sum"),
    )
    weekly["total"] = weekly["long_sig"] + weekly["short_sig"]
    avg_per_week = weekly["total"].mean() if len(weekly) > 0 else 0

    # Consecutive signal gaps
    signal_mask = valid["long_signal"] | valid["short_signal"]
    signal_indices = signal_mask[signal_mask].index
    if len(signal_indices) > 1:
        gaps = signal_indices.to_series().diff().dt.total_seconds() / 3600  # in hours
        avg_gap_hours = gaps.mean()
        median_gap_hours = gaps.median()
    else:
        avg_gap_hours = median_gap_hours = 0

    n_bars = len(valid)
    n_weeks = (valid.index[-1] - valid.index[0]).total_seconds() / (7 * 86400)

    print("  ── Signal Frequency ─────────────────────────")
    print(f"    Total bars analyzed: {n_bars:,}")
    print(f"    Total raw signals:   {total_signals}")
    print(f"      Long signals:      {long_count}")
    print(f"      Short signals:     {short_count}")
    print(f"    Signal rate:         {total_signals/n_bars*100:.2f}% of bars")
    print(f"    Avg per week:        {avg_per_week:.1f} signals")
    print(f"    Avg gap between signals: {avg_gap_hours:.1f} hours")
    print(f"    Median gap:          {median_gap_hours:.1f} hours")
    print()

    if avg_per_week < 1:
        print("  >> WARNING: < 1 signal/week. Very low frequency.")
    elif avg_per_week < 3:
        print("  >> NOTE: 1-3 signals/week. Consistent with research spec (~3/week).")
    elif avg_per_week > 10:
        print("  >> WARNING: > 10 signals/week. Filters may be too loose.")
    else:
        print("  >> OK: Reasonable signal frequency.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Module-level reference for reversal_distance analysis
config = None  # set in main()


def main():
    global config

    parser = argparse.ArgumentParser(
        description="LSR-EH1 Backtest Engine — Liquidity Sweep Reversal EURUSD Hourly"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to OHLC CSV or Dukascopy tick CSV. If not provided, synthetic data is used.",
    )
    parser.add_argument(
        "--dukascopy", action="store_true",
        help="Interpret --data as Dukascopy tick data (aggregate to 1H).",
    )
    parser.add_argument(
        "--equity", type=float, default=100_000.0,
        help="Initial equity in USD (default: 100000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--start", type=str, default="2019-01-01",
        help="Start date for synthetic data (default: 2019-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default="2024-12-31",
        help="End date for synthetic data (default: 2024-12-31)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic data (default: 42)",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward analysis (24m train, 6m test).",
    )
    parser.add_argument(
        "--cost-stress", action="store_true",
        help="Run cost stress test (0.20 to 1.00 pip scenarios).",
    )
    parser.add_argument(
        "--signals-only", action="store_true",
        help="Only analyze signal frequency, skip backtest.",
    )

    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  LSR-EH1 — Liquidity Sweep Reversal: EURUSD Hourly")
    print("=" * 62)
    print()

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    if args.data:
        print(f"  Loading data from: {args.data}")
        if args.dukascopy:
            df = load_dukascopy_ticks(args.data, timeframe="1H")
        else:
            df = load_ohlc_csv(args.data)

        # Validate
        is_valid, msg = validate_ohlc(df)
        if not is_valid:
            print(f"  ERROR: Data validation failed: {msg}")
            return
        print(f"  Data validation: PASSED")
    else:
        print(f"  [SYNTHETIC DATA] No real data provided. Using synthetic generator.")
        print(f"  This is for integration testing only — not alpha validation.")
        print()
        df = generate_synthetic_eurusd(
            start_date=args.start,
            end_date=args.end,
            seed=args.seed,
        )

    print(f"  Bars loaded: {len(df):,}")
    print(f"  Date range:  {df.index[0]} → {df.index[-1]}")
    print(f"  Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
    print()

    # ---------------------------------------------------------------
    # 2. Initialize
    # ---------------------------------------------------------------
    config = StrategyConfig()
    engine = BacktestEngine(config)

    print(f"  Strategy: {config.name}")
    print(f"    ATR(14), ADX(14)<25, ATR_pctl>70")
    print(f"    PT={config.exit.profit_target_atr}x ATR, SL={config.exit.stop_loss_atr}x ATR")
    print(f"    Max hold={config.exit.max_bars_held} bars, Risk={config.risk.risk_per_trade:.1%}")
    print(f"    Equity: ${args.equity:,.0f}")
    print()

    # ---------------------------------------------------------------
    # 3. Signal frequency analysis
    # ---------------------------------------------------------------
    print("  Analyzing signals...")
    analyze_signal_frequency(df, config)

    if args.signals_only:
        print("  Done (signals-only mode).")
        return

    # ---------------------------------------------------------------
    # 4. Run backtest
    # ---------------------------------------------------------------
    print("  Running backtest...")
    equity_curve = engine.run(df, initial_equity=args.equity)
    trades = engine.trades

    print(f"  Trades executed: {len(trades)}")
    print()

    # ---------------------------------------------------------------
    # 5. Performance report
    # ---------------------------------------------------------------
    report = compute_metrics(trades, equity_curve, args.equity)
    print(print_report(report))

    # ---------------------------------------------------------------
    # 6. Reversal distance distribution
    # ---------------------------------------------------------------
    if trades:
        analyze_reversal_distance(trades)

    # ---------------------------------------------------------------
    # 7. Trade breakdown
    # ---------------------------------------------------------------
    if trades:
        print("  ── Trade Breakdown ──────────────────────────")
        long_t = [t for t in trades if t.direction == 1]
        short_t = [t for t in trades if t.direction == -1]
        print(f"    Long:  {len(long_t)}  |  Short:  {len(short_t)}")

        reasons = {}
        for t in trades:
            r = t.exit_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"      {reason:<20} {count:>4} ({count/len(trades):.0%})")

        pnl_arr = np.array([t.pnl_pips for t in trades if t.pnl_pips is not None])
        if len(pnl_arr) > 0:
            print(f"\n    PnL (pips): total={pnl_arr.sum():+.1f}  avg={pnl_arr.mean():+.1f}  "
                  f"std={pnl_arr.std():.1f}  best={pnl_arr.max():+.1f}  worst={pnl_arr.min():+.1f}")
        print()

    # ---------------------------------------------------------------
    # 8. Export results
    # ---------------------------------------------------------------
    trades_csv = output_dir / "trades.csv"
    engine._export_trades(str(trades_csv))
    print(f"  Trade log:    {trades_csv}")

    eq_csv = output_dir / "equity_curve.csv"
    equity_curve.to_csv(eq_csv, index=False)
    print(f"  Equity curve: {eq_csv}")

    # Reversal distance distribution CSV
    if trades:
        rd_data = []
        for t in trades:
            rd_data.append({
                "trade_id": t.trade_id,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "exit_reason": t.exit_reason,
                "reversal_distance": t.reversal_distance,
                "pnl_pips": t.pnl_pips,
                "pnl_usd": t.pnl_usd,
                "bars_held": t.bars_held,
            })
        rd_csv = output_dir / "reversal_distances.csv"
        pd.DataFrame(rd_data).to_csv(rd_csv, index=False)
        print(f"  Reversal data: {rd_csv}")

    # Performance summary CSV
    summary_csv = output_dir / "performance_summary.csv"
    pd.DataFrame({
        "Metric": [
            "Net Profit", "CAGR", "Max Drawdown (%)", "Annualized Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Total Trades", "Win Rate", "Profit Factor", "Expectancy",
            "Avg Winner", "Avg Loser", "Time in Market (%)",
        ],
        "Value": [
            report.net_profit, report.cagr, report.max_drawdown_pct,
            report.annualized_volatility, report.sharpe_ratio,
            report.sortino_ratio, report.calmar_ratio,
            report.total_trades, report.win_rate, report.profit_factor,
            report.expectancy, report.avg_winner_usd, report.avg_loser_usd,
            report.time_in_market_pct,
        ],
    }).to_csv(summary_csv, index=False)
    print(f"  Summary:      {summary_csv}")
    print()

    # ---------------------------------------------------------------
    # 9. Walk-Forward (optional)
    # ---------------------------------------------------------------
    if args.walk_forward:
        print("  ── Walk-Forward Analysis ────────────────────")
        wf_runner = WalkForwardRunner(config)
        folds = wf_runner.run(
            df,
            train_months=24,
            test_months=6,
            initial_equity=args.equity,
        )
        if folds:
            print(wf_runner.summary(folds))
            wf_csv = output_dir / "walk_forward_results.csv"
            wf_runner.export(folds, str(wf_csv))
            print(f"  Walk-forward: {wf_csv}")
        else:
            print("  Insufficient data for walk-forward analysis.")
        print()

    # ---------------------------------------------------------------
    # 10. Cost Stress (optional)
    # ---------------------------------------------------------------
    if args.cost_stress:
        print("  ── Cost Stress Test ──────────────────────────")
        cs_runner = CostStressRunner(config)
        cs_results = cs_runner.run(df, initial_equity=args.equity)
        if cs_results:
            print(cs_runner.summary(cs_results))
            cs_csv = output_dir / "cost_stress_results.csv"
            cs_runner.export(cs_results, str(cs_csv))
            print(f"  Cost stress:  {cs_csv}")
        else:
            print("  No trades to stress test.")
        print()

    print("  Done.")


if __name__ == "__main__":
    main()
