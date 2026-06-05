# LSR-EH1 — Liquidity Sweep Reversal: EURUSD Hourly

Institutional-grade quantitative research system for detecting and exploiting liquidity sweep reversals on EURUSD H1. Built as a disciplined research framework with frozen parameters, cost stress testing, and walk-forward validation.

## Research Status

| Component | Status |
|-----------|--------|
| Data Quality | ✅ Dukascopy H1 tick data |
| Signal Generation | ✅ 400 trades over 6+ years |
| Backtest Engine | ✅ Full event-driven engine |
| Cost Modeling | ⚠️ Basic (spread + slippage) |
| Deployable Strategy | ❌ Not yet — edge is real but thin |
| Core Hypothesis | ✅ Still alive — PF crosses 1.0 with time-exit |

### Key Finding (Research Cycle 2)

Changing *only* the exit method (no parameter tuning) moved Profit Factor from 0.88 to 1.09. The market signal says: **reversals exist, but smaller than assumed**. Next research cycle focuses on sweep-depth bucketing, time-exit optimization, and MAE/MFE characterization.

## Architecture

```
lsr_eh1/
├── config.py              # All strategy parameters (FROZEN after IS2 validation)
├── main.py                # Entry point — orchestrates full pipeline
├── generate_test_data.py  # Synthetic OHLC generator for integration testing
│
├── indicators/
│   ├── atr.py             # Average True Range
│   ├── adx.py             # Average Directional Index
│   └── percentile.py      # Rolling ATR percentile rank
│
├── strategy/
│   └── liquidity_sweep.py # Core signal generation: swing detection + sweep logic
│
├── execution/
│   ├── position_sizing.py # Kelly-based fractional position sizing
│   └── costs.py           # Transaction cost application (spread + slippage)
│
├── backtest/
│   ├── engine.py          # Event-driven backtest engine
│   ├── metrics.py         # Performance metrics (Sharpe, Sortino, Calmar, etc.)
│   └── cost_stress.py     # Cost scenario stress testing
│
├── validation/
│   └── walk_forward.py    # Anchored walk-forward analysis
│
├── analysis/              # Research experiment scripts
│   ├── sweep_depth_experiment.py
│   ├── edge_characterization.py
│   └── regime_investigation.py
│
├── data/
│   ├── loader.py          # Dukascopy tick + OHLC loader with validation
│   ├── raw/               # (gitignored) Dukascopy CSV downloads
│   └── processed/         # (gitignored) Generated trade logs & equity curves
│
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with synthetic data (integration test)
python main.py

# Run with real data
python main.py --data data/processed/eurusd_1h_research.csv

# Walk-forward analysis
python main.py --data data/processed/eurusd_1h_research.csv --walk-forward

# Cost stress test
python main.py --data data/processed/eurusd_1h_research.csv --cost-stress

# Full analysis pipeline
python main.py --data data/processed/eurusd_1h_research.csv --walk-forward --cost-stress
```

## Data Acquisition

The primary dataset is EURUSD H1 from Dukascopy (2019–2025). Download scripts are in `data/raw/`. The research CSV (`eurusd_1h_research.csv`, ~44K bars) must be obtained separately due to size constraints.

To rebuild:
1. Download quarterly Dukascopy H1 CSVs into `data/raw/download/`
2. Combine into single OHLC file with consistent timestamp format
3. Save as `data/processed/eurusd_1h_research.csv`

## Strategy Logic

1. **Swing Detection**: Identify recent swing highs/lows using a 5-bar lookback window within a 20-bar search window
2. **ATR Percentile**: Compute 20-bar rolling ATR percentile; only trade when ATR is in top 30% (high volatility)
3. **Liquidity Level**: Offset swing high/low by 0.5× ATR to create false-breakout threshold
4. **Sweep Signal**: Price breaks the liquidity level then reverses — enters on the first retracement bar
5. **ADX Filter**: ADX(14) < 25 — exclude trending markets (mean-reversion requires range conditions)
6. **Same-Direction Cooldown**: Minimum 12 bars between same-direction entries

## Risk Framework

- 1% equity risk per trade
- 1.5 ATR profit target / 1.0 ATR stop loss
- 6-bar maximum holding period
- Daily loss limit: 3% | Weekly: 5% | Monthly: 8% | Hard drawdown: 15%
- News blackout windows: NFP, FOMC, ECB rate decisions, US CPI

## Research Discipline

- All entry parameters **frozen** after IS2 validation (2019–2021)
- No optimization on holdout period (2022–2024)
- Walk-forward validation required before any parameter change
- Cost stress testing across 0.2–1.0 pip scenarios

## License

Proprietary research code. Not for production deployment without further validation.
