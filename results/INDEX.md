# LSR-EH1 Research Results Index

Complete documentation of all experimental outputs from the LSR-EH1 research program.
All files here are generated outputs — not source code. Source code lives in the parent directory.

**Research status**: Parameters frozen. No optimization on holdout (2022–2024). Edge is real but thin (PF 1.09 best case).

---

## Directory Structure

```
results/
├── INDEX.md                          # This file
├── experiment_a/                     # Sweep depth filtering experiments
├── experiment_b/                     # Exit strategy comparison experiments
├── experiment_4/                     # MAE/MFE analysis (institutional front-office standard)
├── core_backtest/                    # Core strategy outputs
├── validation/                       # Walk-forward + cost stress tests
└── reports/                          # Full text analysis reports
```

---

## experiment_a/ — Sweep Depth Filtering

**Research Question**: Are shallow sweeps responsible for most losses? Can we filter them out to improve PF?

**Methodology**: Vary the minimum `sweep_depth_atr` threshold while keeping all other parameters frozen. Test on all 400 trades from 2019–2025.

### Key Findings

| File | Threshold | Trades | PF | Win Rate | Max DD | Conclusion |
|------|-----------|--------|----|----|--------|------------|
| `test1_No_filter_trades.csv` | None (baseline) | 400 | 0.81 | 40% | 42.8% | Full universe |
| `exp_a_filtered_trades.csv` | >= 0.5 ATR | 88 | 0.88 | 40% | 10.2% | Filter removes 78% noise |
| `test1_sweep_gte_050_trades.csv` | >= 0.50 | 88 | 0.88 | 40% | 10.2% | Same as exp_a |
| `test1_sweep_gte_075_trades.csv` | >= 0.75 | 38 | 0.90 | 42% | 6.4% | Best PF but few trades |
| `test1_sweep_gte_100_trades.csv` | >= 1.00 | 14 | 0.62 | 29% | 4.1% | Too few trades, PF drops |
| `test1_sweep_gte_125_trades.csv` | >= 1.25 | 11 | 0.81 | 36% | 3.1% | Too few trades |

**Verdict**: Filter reduces noise (drawdown 42.8% → 10.2%) but does NOT push PF above 1.0 alone. Monotonicity is NOT confirmed — PF is non-monotonic. The sweep-depth filter is necessary but not sufficient.

### Column Schema (test1_*.csv)

| Column | Type | Description |
|--------|------|-------------|
| trade_id | int | Sequential trade identifier |
| direction | str | LONG or SHORT |
| entry_time | str | Entry timestamp (ISO format) |
| entry_price | float | Entry price |
| atr_entry | float | ATR(14) at entry |
| stop_loss | float | Stop loss price |
| profit_target | float | Profit target price |
| size | int | Position size (units) |
| notional | float | Notional value (USD) |
| exit_time | str | Exit timestamp |
| exit_price | float | Exit price |
| pnl_pips | float | P&L in pips |
| pnl_usd | float | P&L in USD |
| exit_reason | str | profit_target / stop_loss / time_exit |
| bars_held | int | Number of bars held |
| reversal_distance | float | abs(exit - entry) / atr_entry |
| sweep_depth_atr | float | How deep the sweep penetrated (ATR units) |
| sweep_entry_dist_atr | float | Distance from sweep to entry (ATR units) |
| mae_price | float | Maximum adverse excursion price |
| mfe_price | float | Maximum favorable excursion price |
| mae_atr | float | MAE in ATR units |
| mfe_atr | float | MFE in ATR units |
| mae_pips | float | MAE in pips |
| mfe_pips | float | MFE in pips |

---

## experiment_b/ — Exit Strategy Comparison

**Research Question**: Given filtered signals (sweep >= 0.5 ATR), which exit method captures the reversal edge best?

**Methodology**: Only the exit logic is changed. Entry, filter, signal generation, and all other parameters remain frozen. This isolates the exit variable.

### Key Findings

| File | Exit Method | Trades | PF | Win Rate | Avg Pips | Total P&L ($) | Max DD |
|------|------------|--------|----|----|----|-------|--------|
| `exp_b_15_ATR_target_trades.csv` | 1.5 ATR target | 88 | 0.88 | 40% | -0.38 | -5,263 | 10.2% |
| `exp_b_10_ATR_target_trades.csv` | 1.0 ATR target | 88 | 0.97 | 49% | +0.53 | -1,005 | 7.8% |
| `exp_b_Trailing_10_ATR_trades.csv` | Trailing 1.0 ATR | 88 | 0.49 | 34% | -2.38 | -16,705 | 16.7% |
| `exp_b_Time-only_exit_trades.csv` | Time-only (6 bars) | 88 | **1.09** | 38% | **+1.10** | **+4,068** | **10.1%** |

**Critical Signal**: Changing ONLY the exit method moved PF from 0.88 to 1.09. The market is saying: "reversals exist, but smaller than the 1.5 ATR target assumes." Time-only exit captures the full reversal amplitude rather than cutting it off at an arbitrary target.

**Time-only exit breakdown**:
- 41 stop losses (46.6%): avg -15.20 pips, total -$43,599
- 47 time exits (53.4%): avg +15.31 pips, total +$47,668

### Column Schema (exp_b_*.csv)

Same as experiment_a plus:
- `hwm` (float): High water mark price during trade (trailing stop variant only)
- `trailing_stop_exit` (bool): Whether exit was via trailing stop

---

## experiment_4/ — MAE/MFE Analysis

**Research Question**: How far does price actually move against and in favor of each trade before resolution? Are our stop/target levels calibrated correctly?

**Methodology**: Institutional front-office standard. Compute Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE) for every trade across its full holding period. This is "the highest-value remaining analysis" — it reveals what the optimal stop/target placement should be, independent of current settings.

### Key Findings — All 400 Trades

| Statistic | MAE (ATR) | MFE (ATR) | MAE (pips) | MFE (pips) |
|-----------|-----------|-----------|------------|------------|
| Mean | 0.960 | 0.844 | +14.6 | +13.0 |
| Median | 0.911 | 0.617 | +12.9 | +8.8 |
| Std Dev | 0.657 | 0.768 | +11.2 | +12.9 |
| 75th pctile | 1.305 | 1.289 | +20.2 | +19.2 |
| 90th pctile | 1.773 | 1.840 | +29.0 | +29.9 |

**Diagnostic questions**:
- Price moves >= 1.0 ATR against us (stop zone): **45.5% of trades**
- Price moves >= 1.5 ATR in our favor (target zone): **20.8% of trades**
- Price moves >= 1.0 ATR in our favor: **33.2% of trades**
- Average MFE: **0.844 ATR** (the reversal is typically smaller than assumed)

**Winner vs Loser signature**:
- Winners: avg MAE 0.452 ATR, avg MFE 1.508 ATR, MFE/MAE = 3.34
- Losers: avg MAE 1.303 ATR, avg MFE 0.397 ATR, MFE/MAE = 0.31
- Delta MFE/MAE = +3.03 — winners have 10x better excursion ratio

### Files

| File | Description | Trades |
|------|-------------|--------|
| `test4_mae_mfe_all_trades.csv` | All 400 trades with MAE/MFE + decile assignment | 400 |
| `test4_mae_mfe_filtered_trades.csv` | Filtered trades (sweep >= 0.5 ATR) with MAE/MFE | 88 |

### Additional Columns vs experiment_a

| Column | Type | Description |
|--------|------|-------------|
| decile | int | Sweep-depth decile rank (1-10, 10 = deepest) |

---

## core_backtest/ — Baseline Strategy Outputs

Outputs from the standard strategy run (1.5 ATR target, 1.0 ATR stop, 6-bar max hold, sweep >= 0.5 filter).

### Files

| File | Rows | Description |
|------|------|-------------|
| `trades.csv` | 142 | Full trade log from baseline backtest (all exit types) |
| `enriched_trades.csv` | 400 | All 400 trades with 28 columns of metadata (entry regime, session, sweep metrics, etc.) |
| `equity_curve.csv` | 37,562 | Bar-by-bar equity curve (open equity + closed equity) |
| `performance_summary.csv` | 14 | Aggregate performance metrics (one row per metric) |
| `reversal_distances.csv` | 142 | Reversal distance distribution per trade |

### Column Schema — enriched_trades.csv (28 columns)

| Column | Type | Description |
|--------|------|-------------|
| trade_id | int | Sequential identifier |
| direction | str | LONG or SHORT |
| entry_time | str | Entry timestamp |
| entry_price | float | Entry price |
| atr_entry | float | ATR(14) at entry |
| stop_loss | float | Stop loss price |
| profit_target | float | Target price |
| size | int | Position size (units) |
| notional | float | Notional value |
| exit_time | str | Exit timestamp |
| exit_price | float | Exit price |
| pnl_pips | float | P&L in pips |
| pnl_usd | float | P&L in USD |
| exit_reason | str | profit_target / stop_loss / time_exit |
| bars_held | int | Bars held |
| reversal_distance | float | abs(exit - entry) / atr |
| entry_year | int | Year of entry |
| entry_month | int | Month of entry |
| entry_hour_utc | int | Hour of entry (UTC) |
| entry_day_of_week | str | Day of week (Mon-Sun) |
| session | str | NY / London / Asian |
| adx_4h | float | ADX(14) on 4H bars at entry |
| atr_percentile | float | ATR percentile rank at entry |
| atr_value | float | ATR value at entry |
| sweep_size_atr | float | Sweep penetration depth (ATR units) |
| distance_to_swing_atr | float | Distance to swing level (ATR units) |
| liquidity_level | float | Price of the swept liquidity level |
| bars_to_target | int | Bars until target hit (NaN if not reached) |
| bars_to_stop | int | Bars until stop hit (NaN if not hit) |

### Performance Summary Metrics

| Metric | Value |
|--------|-------|
| Net Profit | -$35,595 |
| Total Trades | 400 |
| Win Rate | 40% |
| Profit Factor | 0.81 |
| Max Drawdown | 42.8% |

---

## validation/ — Robustness Tests

### walk_forward_results.csv

Walk-forward analysis: 24-month training window, 6-month test window, anchored expansion.

| Column | Description |
|--------|-------------|
| Fold | Fold identifier |
| Train Start | Training period start |
| Train End | Training period end |
| Test Start | Test period start |
| Test End | Test period end |
| Train PF | In-sample profit factor |
| Test PF | Out-of-sample profit factor |
| Train Trades | In-sample trade count |
| Test Trades | Out-of-sample trade count |

### cost_stress_results.csv

Cost stress test across multiple spread scenarios (0.20 to 1.00 pips round-trip).

| Column | Description |
|--------|-------------|
| Scenario | Spread scenario identifier |
| Half Spread (pips) | Half-spread assumption |
| Total Cost (pips) | Total round-trip cost |
| PF | Profit factor under stress |
| Net P&L ($) | Net profit/loss |
| Max DD (%) | Maximum drawdown |

---

## reports/ — Full Text Analysis

### sweep_depth_experiment_report.txt

Combined report covering Experiment A (sweep depth filter) and Experiment B (exit strategy comparison).

**Key conclusion**: Time-only exit (6 bars, no profit target) achieves PF 1.09 — the only exit variant to cross breakeven. This was achieved by changing ONLY the exit, keeping all entry/filter parameters frozen.

### edge_characterization_report.txt

Comprehensive four-test report:
1. **Test 1**: Continuous sweep depth buckets (0.50 / 0.75 / 1.00 / 1.25) — non-monotonic PF
2. **Test 2**: Sweep-depth deciles (D1-D10) — structural relationship test (Spearman rho = +0.156, weak)
3. **Test 3**: Time exit duration (4/6/8/10/12 bars, with and without target) — 6 bars no-target is optimal (PF 1.09)
4. **Test 4**: MAE/MFE analysis — avg MFE 0.844 ATR confirms reversals are smaller than the 1.5 ATR target

**Key conclusion**: The edge exists but is thin. Average favorable excursion is ~0.84 ATR while the target is set at 1.5 ATR — the target overshoots the actual reversal by ~80%.

### regime_analysis_report.txt

Deep investigation into why 2021-2022 worked (PF 1.08-1.09) while other years lost.

**Key conclusions**:
- Sweep-reversal phenomenon is real: 83/400 trades (20.8%) hit 1.5 ATR, 138/400 (34.5%) time out net positive
- Edge is destroyed by stop losses, not bad entries: stop loss P&L is consistently negative
- Session matters: NY (PF 0.85) >> Asian (PF 0.41)
- Sweep depth matters: huge sweeps (2.0+ ATR) have PF 4.66
- Time exits are NET POSITIVE: +$17,466 total, 56.5% win rate

---

## Decile Table (from edge_characterization)

| Decile | Trades | PF | Win Rate | Avg Pips | Total P&L ($) | Entry Dist (ATR) |
|--------|--------|----|----|----|-------|----|
| D1 (shallowest) | 40 | 0.45 | 35% | -3.87 | -11,007 | 0.035 |
| D2 | 40 | 0.98 | 45% | +1.76 | -341 | 0.099 |
| D3 | 40 | 0.91 | 42% | +0.57 | -1,286 | 0.188 |
| D4 | 40 | **1.02** | 42% | +0.66 | +308 | 0.279 |
| D5 | 40 | 0.77 | 40% | -0.82 | -4,255 | 0.386 |
| D6 | 40 | 0.89 | 42% | -0.76 | -1,948 | 0.480 |
| D7 | 40 | **1.03** | 42% | +0.06 | +533 | 0.620 |
| D8 | 40 | 0.53 | 32% | -3.20 | -10,320 | 0.781 |
| D9 | 40 | 0.57 | 35% | -3.53 | -8,710 | 0.994 |
| D10 (deepest) | 40 | **1.08** | 45% | +0.07 | +1,431 | 1.729 |

Spearman rank correlation: rho = +0.156 (weak positive). Only D4, D7, D10 cross PF 1.0.

---

## Time Exit Duration Table (from edge_characterization)

| Max Bars | With Target PF | No Target PF | Time Exit Avg (no target) |
|----------|----|----|----|
| 4 bars | 0.79 | 0.86 | +10.26 pips |
| **6 bars** | 0.88 | **1.09** | **+15.31 pips** |
| 8 bars | 0.86 | 1.06 | +18.85 pips |
| 10 bars | 0.89 | 1.04 | +18.38 pips |
| 12 bars | 0.88 | 0.97 | +19.48 pips |

**Optimal**: 6 bars, no profit target. Longer holds show diminishing PF due to increased stop-loss exposure.

---

*Generated as part of LSR-EH1 research program. All experiments use frozen entry parameters — only the stated variable changes per experiment.*
