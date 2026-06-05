"""
Data loading package for LSR-EH1.

Provides Dukascopy tick data loading with OHLC aggregation, pre-aggregated
OHLC loading, spread estimation, and OHLC validation.
"""

from data.loader import (
    load_dukascopy_ticks,
    load_ohlc_csv,
    estimate_spread,
    validate_ohlc,
)

__all__ = [
    "load_dukascopy_ticks",
    "load_ohlc_csv",
    "estimate_spread",
    "validate_ohlc",
]
