"""
Cost Stress Testing Framework for LSR-EH1.

Runs the same backtest at multiple cost levels and compares key metrics
to assess strategy robustness to transaction-cost degradation.

Usage::

    from config import StrategyConfig
    from backtest.cost_stress import CostStressRunner

    runner = CostStressRunner(base_config)
    results = runner.run(df, initial_equity=100_000)
    print(runner.summary(results))
    runner.export(results, "cost_stress.csv")
"""

import copy
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from config import StrategyConfig, CostParams
from backtest.engine import BacktestEngine, Trade
from backtest.metrics import compute_metrics, PerformanceReport

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

COST_SCENARIOS: List[float] = [0.20, 0.35, 0.50, 0.75, 1.00]  # round-trip pips


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CostStressResult:
    """Results for a single cost scenario."""

    total_cost_pips: float                  # the cost level used
    net_profit: float
    cagr: float
    sharpe: float
    sortino: float
    profit_factor: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float
    # Sensitivity
    sharpe_vs_baseline: float               # change from baseline
    profit_factor_vs_baseline: float
    # Per-trade cost as % of avg winner
    cost_as_pct_of_avg_winner: float
    avg_winner_usd: float = 0.0              # stored for downstream use


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class CostStressRunner:
    """Run the same backtest at escalating cost levels and compare metrics.

    Each scenario deep-copies the base :class:`StrategyConfig`, overrides
    :attr:`CostParams` components, instantiates a fresh
    :class:`BacktestEngine`, and collects the resulting performance metrics.
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Store base config for cloning with different costs.

        Parameters
        ----------
        config : StrategyConfig
            The strategy configuration whose *non-cost* parameters are kept
            constant across all scenarios.
        """
        self._base_config: StrategyConfig = config

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        cost_scenarios: Optional[List[float]] = None,
        initial_equity: float = 100_000.0,
        baseline_pips: float = 0.35,
    ) -> List[CostStressResult]:
        """Run backtest at each cost level.

        Parameters
        ----------
        df : pd.DataFrame
            OHLC DataFrame with DatetimeIndex and columns
            ``open, high, low, close`` (and optionally ``spread``).
        cost_scenarios : list[float], optional
            Total round-trip cost levels (in pips) to test.
            Defaults to :data:`COST_SCENARIOS`.
        initial_equity : float
            Starting equity in USD.
        baseline_pips : float
            Reference cost level for sensitivity deltas (default 0.35).

        Returns
        -------
        list[CostStressResult]
            Sorted ascending by cost level.
        """
        if cost_scenarios is None:
            cost_scenarios = COST_SCENARIOS

        results: List[CostStressResult] = []
        baseline_report: Optional[PerformanceReport] = None

        for scenario_pips in cost_scenarios:
            # 1. Deep-copy base config
            cfg = copy.deepcopy(self._base_config)

            # 2. Set total round-trip cost
            cfg.cost.total_round_trip_pips = scenario_pips

            # 3. Distribute cost across components
            cfg.cost.half_spread_pips = round(0.5 * scenario_pips, 6)
            cfg.cost.slippage_pips = round(0.3 * scenario_pips, 6)
            cfg.cost.market_impact_pips = round(0.2 * scenario_pips, 6)

            # 4. Fresh engine + run
            engine = BacktestEngine(cfg)
            equity_curve = engine.run(df, initial_equity=initial_equity)

            # 5. Compute metrics
            report = compute_metrics(
                trades=engine.trades,
                equity_curve=equity_curve,
                initial_equity=initial_equity,
            )

            # Capture baseline when it matches
            if abs(scenario_pips - baseline_pips) < 1e-9:
                baseline_report = report

            # 7. Cost as % of avg winner
            cost_as_pct: float = 0.0
            if report.avg_winner_usd > 0:
                cost_usd_per_trade = scenario_pips * cfg.risk.pip_value
                cost_as_pct = (cost_usd_per_trade / report.avg_winner_usd) * 100.0

            # 8. Sensitivity vs baseline
            sharpe_vs_baseline: float = 0.0
            pf_vs_baseline: float = 0.0
            if baseline_report is not None and abs(scenario_pips - baseline_pips) > 1e-9:
                sharpe_vs_baseline = report.sharpe_ratio - baseline_report.sharpe_ratio
                pf_raw = report.profit_factor
                pf_base = baseline_report.profit_factor
                if np.isfinite(pf_base) and np.isfinite(pf_raw):
                    pf_vs_baseline = pf_raw - pf_base
                elif np.isfinite(pf_raw):
                    pf_vs_baseline = pf_raw
                # When both are inf, delta is 0

            avg_trade_pnl = report.expectancy

            result = CostStressResult(
                total_cost_pips=scenario_pips,
                net_profit=report.net_profit,
                cagr=report.cagr,
                sharpe=report.sharpe_ratio,
                sortino=report.sortino_ratio,
                profit_factor=report.profit_factor,
                max_drawdown=report.max_drawdown_pct,
                win_rate=report.win_rate,
                total_trades=report.total_trades,
                avg_trade_pnl=avg_trade_pnl,
                sharpe_vs_baseline=sharpe_vs_baseline,
                profit_factor_vs_baseline=pf_vs_baseline,
                cost_as_pct_of_avg_winner=cost_as_pct,
                avg_winner_usd=report.avg_winner_usd,
            )
            results.append(result)

        # Sort by cost level
        results.sort(key=lambda r: r.total_cost_pips)
        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, results: List[CostStressResult]) -> str:
        """Format a comparison table followed by sensitivity analysis.

        Parameters
        ----------
        results : list[CostStressResult]
            Output of :meth:`run`.

        Returns
        -------
        str
            Multi-line text summary.
        """
        if not results:
            return "  (no results)"

        lines: List[str] = []

        # ---- Header ----
        lines.append("")
        lines.append("=" * 76)
        lines.append("  COST STRESS TEST RESULTS")
        lines.append("=" * 76)
        lines.append("")

        # ---- Table ----
        header = (
            f"  {'Cost(pips)':>10s} │ {'Net Profit':>12s} │ {'CAGR':>7s} │ "
            f"{'Sharpe':>7s} │ {'PF':>6s} │ {'Max DD':>7s} │ "
            f"{'Win Rate':>8s} │ {'Trades':>7s}"
        )
        sep = "  " + "─" * 10 + "┼" + "─" * 14 + "┼" + "─" * 9 + "┼" + "─" * 9 + "┼" + "─" * 8 + "┼" + "─" * 9 + "┼" + "─" * 10 + "┼" + "─" * 9

        lines.append(header)
        lines.append(sep)

        for r in results:
            pf_str = f"{r.profit_factor:.2f}" if np.isfinite(r.profit_factor) else "  ∞  "
            line = (
                f"  {r.total_cost_pips:>10.2f} │ ${r.net_profit:>11,.2f} │ "
                f"{r.cagr:>6.2f}% │ {r.sharpe:>7.2f} │ {pf_str:>6s} │ "
                f"{r.max_drawdown:>6.2f}% │ {r.win_rate * 100:>7.1f}% │ "
                f"{r.total_trades:>7d}"
            )
            lines.append(line)

        lines.append("=" * 76)

        # ---- Sensitivity analysis ----
        lines.append("")
        lines.append("  ── Sensitivity Analysis ──────────────────────────────────────")
        lines.append("")

        # Linear degradation rate per 0.10 pip
        if len(results) >= 2:
            # Use the cost range from lowest to highest
            x_lo = results[0].total_cost_pips
            x_hi = results[-1].total_cost_pips
            dx = x_hi - x_lo
            if dx > 0:
                sharpe_delta = results[-1].sharpe - results[0].sharpe
                pf_delta_lo = results[0].profit_factor
                pf_delta_hi = results[-1].profit_factor
                pf_delta = (
                    pf_delta_hi - pf_delta_lo
                    if np.isfinite(pf_delta_lo) and np.isfinite(pf_delta_hi)
                    else 0.0
                )
                sharpe_per_010 = (sharpe_delta / dx) * 0.10
                pf_per_010 = (pf_delta / dx) * 0.10

                lines.append(f"    Strategy degradation per 0.10 pip cost increase:")
                lines.append(f"      Sharpe change:    {sharpe_per_010:+.4f}")
                lines.append(f"      PF change:        {pf_per_010:+.4f}")
                lines.append("")

        # Threshold crossings
        sharpe_thresh = self._find_threshold(results, "sharpe", 1.0)
        pf_thresh = self._find_threshold(results, "profit_factor", 1.2)
        net_neg_thresh = self._find_threshold(results, "net_profit", 0.0)

        lines.append(f"    Cost at which Sharpe drops below 1.0:   {sharpe_thresh:.2f} pips")
        lines.append(f"    Cost at which PF drops below 1.2:        {pf_thresh:.2f} pips")
        lines.append(f"    Cost at which strategy becomes net negative: {net_neg_thresh:.2f} pips")

        # ---- Verdict ----
        lines.append("")
        lines.append("  ── Verdict ──────────────────────────────────────────────────")
        lines.append("")
        if sharpe_thresh == float("inf"):
            lines.append("    Strategy remains profitable across ALL tested cost levels.")
        else:
            lines.append(
                f"    Strategy tolerates up to {sharpe_thresh:.2f} pips in round-trip costs"
            )
            lines.append("    before Sharpe falls below 1.0.")
        lines.append("")
        lines.append("=" * 76)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, results: List[CostStressResult], filepath: str) -> None:
        """Export results to CSV.

        Parameters
        ----------
        results : list[CostStressResult]
            Output of :meth:`run`.
        filepath : str
            Destination CSV path.
        """
        if not results:
            empty = pd.DataFrame(columns=[
                "total_cost_pips", "net_profit", "cagr", "sharpe", "sortino",
                "profit_factor", "max_drawdown", "win_rate", "total_trades",
                "avg_trade_pnl", "sharpe_vs_baseline",
                "profit_factor_vs_baseline", "cost_as_pct_of_avg_winner",
            ])
            empty.to_csv(filepath, index=False)
            return

        records = [
            {
                "total_cost_pips": r.total_cost_pips,
                "net_profit": r.net_profit,
                "cagr": r.cagr,
                "sharpe": r.sharpe,
                "sortino": r.sortino,
                "profit_factor": (
                    r.profit_factor if np.isfinite(r.profit_factor) else None
                ),
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
                "avg_trade_pnl": r.avg_trade_pnl,
                "sharpe_vs_baseline": r.sharpe_vs_baseline,
                "profit_factor_vs_baseline": (
                    r.profit_factor_vs_baseline
                    if np.isfinite(r.profit_factor_vs_baseline)
                    else None
                ),
                "cost_as_pct_of_avg_winner": r.cost_as_pct_of_avg_winner,
            }
            for r in results
        ]
        pd.DataFrame(records).to_csv(filepath, index=False)

    # ------------------------------------------------------------------
    # Threshold interpolation
    # ------------------------------------------------------------------

    def _find_threshold(
        self,
        results: List[CostStressResult],
        field: str,
        threshold: float,
    ) -> float:
        """Interpolate the cost level at which *field* crosses *threshold*.

        Uses linear interpolation between the two bounding scenarios.
        If the metric never crosses the threshold, returns ``float('inf')``.

        Parameters
        ----------
        results : list[CostStressResult]
            Must be sorted by ``total_cost_pips`` ascending.
        field : str
            Attribute name on :class:`CostStressResult` (e.g. ``"sharpe"``,
            ``"profit_factor"``, ``"net_profit"``).
        threshold : float
            The crossing level to search for.

        Returns
        -------
        float
            Interpolated cost level, or ``float('inf')`` if never crossed.
        """
        if not results or len(results) < 2:
            return float("inf")

        # Determine the direction of crossing.
        # For "sharpe", "profit_factor", "net_profit" we expect values
        # to *decrease* as cost increases.  We want the cost at which the
        # metric falls *below* the threshold (descending crossing).
        #
        # However, we generalise: look for a sign change in
        # (value - threshold) as cost increases.

        for i in range(len(results) - 1):
            val_a = getattr(results[i], field)
            val_b = getattr(results[i + 1], field)

            # Handle non-finite values gracefully
            if not np.isfinite(val_a):
                val_a = float("inf") if threshold > 0 else float("-inf")
            if not np.isfinite(val_b):
                val_b = float("inf") if threshold > 0 else float("-inf")

            diff_a = val_a - threshold
            diff_b = val_b - threshold

            # Sign change → crossing between i and i+1
            if diff_a * diff_b < 0:
                x_a = results[i].total_cost_pips
                x_b = results[i + 1].total_cost_pips
                # Linear interpolation: threshold = val_a + t*(val_b - val_a)
                denominator = val_b - val_a
                if abs(denominator) < 1e-12:
                    return x_a  # degenerate — both nearly equal
                t = (threshold - val_a) / denominator
                return x_a + t * (x_b - x_a)

        # Check if already past threshold at the lowest cost
        first_val = getattr(results[0], field)
        if not np.isfinite(first_val):
            first_val = float("inf") if threshold > 0 else float("-inf")
        if first_val < threshold:
            return results[0].total_cost_pips

        return float("inf")
