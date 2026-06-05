"""
Backtest package for LSR-EH1.

Exposes the core engine, trade data model, and performance metrics.
"""

from backtest.engine import BacktestEngine, Trade
from backtest.metrics import PerformanceReport, compute_metrics, print_report
from backtest.cost_stress import CostStressRunner, CostStressResult, COST_SCENARIOS

__all__ = [
    "BacktestEngine",
    "Trade",
    "PerformanceReport",
    "compute_metrics",
    "print_report",
    "CostStressRunner",
    "CostStressResult",
    "COST_SCENARIOS",
]
