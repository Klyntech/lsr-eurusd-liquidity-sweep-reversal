"""
generate_test_data.py — Create a controlled EURUSD test dataset
with explicit sweep events for validating the LSR-EH1 backtest engine.

This is NOT for production backtesting. It's for integration testing:
verify that the engine generates trades, exits correctly, and
produces valid equity curves.

For real backtesting, use actual EURUSD tick/1H data from a broker.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from indicators.atr import atr
from strategy.liquidity_sweep import (
    detect_swing_high, detect_swing_low, most_recent_swing_value,
)


def generate_test_dataset(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a controlled dataset with periodic sweep events.

    Strategy:
    1. Generate a base ranging price series around 1.1200
    2. Every ~50-100 bars, create a deliberate sweep pattern:
       - Several bars declining to form a swing low
       - Then a bar that breaks below the swing low and closes back above
    3. Intersperse with normal ranging bars
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2019-01-01", periods=n_bars, freq="1h")
    # Remove weekends
    timestamps = timestamps[timestamps.dayofweek < 5]
    n = len(timestamps)

    base_price = 1.1200
    prices = np.empty(n)
    prices[0] = base_price

    # Generate base price with moderate mean reversion
    for t in range(1, n):
        vol = 0.0015 + 0.001 * np.sin(t / 200)  # slowly varying volatility
        prices[t] = prices[t-1] + rng.normal(0, vol)
        # Strong mean reversion to keep price in range
        prices[t] += 0.05 * (base_price - prices[t]) * 0.01
    prices = np.clip(prices, 1.08, 1.16)

    # Build OHLC
    opens = prices.copy()
    bar_range = 0.001 + np.abs(rng.normal(0, 0.0005, n))
    highs = opens + bar_range * rng.uniform(0.2, 0.8, n)
    lows = opens - bar_range * rng.uniform(0.2, 0.8, n)
    closes = lows + (highs - lows) * rng.uniform(0.3, 0.7, n)

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    highs = np.maximum(highs, lows + 1e-6)

    df = pd.DataFrame({
        "open": np.round(opens, 5),
        "high": np.round(highs, 5),
        "low": np.round(lows, 5),
        "close": np.round(closes, 5),
    }, index=timestamps)

    # ---- Post-process: compute indicators and inject sweeps ----
    # We do this iteratively until sweeps align with computed levels
    for iteration in range(3):
        # Recompute indicators on current data
        df_copy = df.copy()
        atr_series = atr(df_copy["high"], df_copy["low"], df_copy["close"], period=14)
        sh = detect_swing_high(df_copy["high"], lookback=5)
        sl = detect_swing_low(df_copy["low"], lookback=5)
        rsh = most_recent_swing_value(df_copy["high"], sh, window=20, mode="high")
        rsl = most_recent_swing_value(df_copy["low"], sl, window=20, mode="low")
        liq_long = rsl - 0.5 * atr_series
        liq_short = rsh + 0.5 * atr_series

        valid_idx = atr_series.notna()
        valid_positions = np.where(valid_idx.values)[0]
        valid_positions = valid_positions[40:]  # warmup

        # Find bars that are NOT currently sweeps but could be turned into ones
        sweeps_injected = 0
        # Target: ~1 sweep per 50 bars (realistic for the strategy)
        target_sweeps = n // 50
        spacing = max(30, n // (target_sweeps * 2))  # minimum spacing between injections

        last_inject_pos = -spacing - 1
        for pos in valid_positions:
            if pos - last_inject_pos < spacing:
                continue

            atr_val = atr_series.iloc[pos]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            liq_l = liq_long.iloc[pos]
            liq_s = liq_short.iloc[pos]

            cur_low = df.iloc[pos]["low"]
            cur_high = df.iloc[pos]["high"]
            cur_open = df.iloc[pos]["open"]
            cur_close = df.iloc[pos]["close"]

            injected = False

            # Try downward sweep: low < liq_long, close > liq_long
            if rng.random() < 0.5 and not np.isnan(liq_l):
                penetration = atr_val * rng.uniform(0.1, 0.5)
                new_low = liq_l - penetration
                new_close = liq_l + atr_val * rng.uniform(0.05, 0.4)

                # Check it's actually a sweep (low goes below, close above)
                if new_low < cur_low:
                    df.at[df.index[pos], "low"] = round(new_low, 5)
                    df.at[df.index[pos], "close"] = round(new_close, 5)
                    df.at[df.index[pos], "high"] = round(
                        max(cur_high, max(cur_open, new_close)), 5)
                    df.at[df.index[pos], "low"] = round(
                        min(new_low, min(cur_open, df.at[df.index[pos], "close"])), 5)
                    injected = True

            # Try upward sweep: high > liq_short, close < liq_short
            if not injected and rng.random() < 1.0 and not np.isnan(liq_s):
                penetration = atr_val * rng.uniform(0.1, 0.5)
                new_high = liq_s + penetration
                new_close = liq_s - atr_val * rng.uniform(0.05, 0.4)

                if new_high > cur_high:
                    df.at[df.index[pos], "high"] = round(new_high, 5)
                    df.at[df.index[pos], "close"] = round(new_close, 5)
                    df.at[df.index[pos], "low"] = round(
                        min(cur_low, min(cur_open, new_close)), 5)
                    df.at[df.index[pos], "high"] = round(
                        max(new_high, max(cur_open, df.at[df.index[pos], "close"])), 5)
                    injected = True

            if injected:
                sweeps_injected += 1
                last_inject_pos = pos

        print(f"  [data] Iteration {iteration+1}: injected {sweeps_injected} sweeps")

    # Final validation: count how many bars actually produce sweeps with fresh indicators
    df_final = df.copy()
    atr_final = atr(df_final["high"], df_final["low"], df_final["close"], period=14)
    sh_final = detect_swing_high(df_final["high"], lookback=5)
    sl_final = detect_swing_low(df_final["low"], lookback=5)
    rsh_final = most_recent_swing_value(df_final["high"], sh_final, window=20, mode="high")
    rsl_final = most_recent_swing_value(df_final["low"], sl_final, window=20, mode="low")
    liq_long_final = rsl_final - 0.5 * atr_final
    liq_short_final = rsh_final + 0.5 * atr_final

    valid = atr_final.notna()
    low_sweeps = valid & (df_final["low"] < liq_long_final) & (df_final["close"] > liq_long_final)
    high_sweeps = valid & (df_final["high"] > liq_short_final) & (df_final["close"] < liq_short_final)

    print(f"  [data] Validation: {low_sweeps.sum()} long sweeps, {high_sweeps.sum()} short sweeps")

    return df


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_test_dataset(n_bars=5000, seed=42)
    output_path = output_dir / "eurusd_test_1h.csv"
    df.to_csv(output_path)

    print(f"\n  Saved to: {output_path}")
    print(f"  Bars: {len(df)}")
    print(f"  Range: {df.index[0]} → {df.index[-1]}")
    print(f"  Price: {df['low'].min():.5f} – {df['high'].max():.5f}")
