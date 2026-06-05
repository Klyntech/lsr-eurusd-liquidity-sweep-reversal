"""
Performance Metrics for LSR-EH1 Backtest.

Computes a comprehensive set of return, risk, risk-adjusted, trade-level,
and exposure metrics from the trade list and equity curve produced by the
backtest engine.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from backtest.engine import Trade


# ---------------------------------------------------------------------------
# Performance Report
# ---------------------------------------------------------------------------

@dataclass
class PerformanceReport:
    """Minimal but complete performance metrics for Phase 1.

    Attributes
    ----------
    net_profit : float
        Total realised P&L in USD (final equity − initial equity).
    cagr : float
        Compound annual growth rate (annualised return).
    max_drawdown_pct : float
        Worst peak-to-trough drawdown as a percentage (0–100).
    annualized_volatility : float
        Annualised standard deviation of daily returns.
    sharpe_ratio : float
        Annualised Sharpe ratio (excess over risk-free).
    sortino_ratio : float
        Annualised Sortino ratio (downside deviation denominator).
    calmar_ratio : float
        CAGR divided by max drawdown (in decimal form).
    total_trades : int
        Total number of round-trip trades.
    win_rate : float
        Fraction of trades that were profitable (0–1).
    profit_factor : float
        Sum of gross wins ÷ |sum of gross losses|.
    expectancy : float
        Average trade P&L in USD.
    avg_winner_usd : float
        Average USD P&L of winning trades.
    avg_loser_usd : float
        Average USD P&L of losing trades (always ≤ 0).
    time_in_market_pct : float
        Percentage of bars with an open position (0–100).
    """

    # Return metrics
    net_profit: float
    cagr: float

    # Risk metrics
    max_drawdown_pct: float
    annualized_volatility: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_winner_usd: float
    avg_loser_usd: float

    # Exposure
    time_in_market_pct: float


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    trades: List[Trade],
    equity_curve: pd.DataFrame,
    initial_equity: float,
    risk_free_rate: float = 0.0,
) -> PerformanceReport:
    """Compute all performance metrics from trade list and equity curve.

    Parameters
    ----------
    trades : list[Trade]
        List of closed trade records produced by :class:`BacktestEngine`.
    equity_curve : pd.DataFrame
        Per-bar equity curve returned by :meth:`BacktestEngine.run`.
        Must contain columns ``timestamp``, ``equity``, ``drawdown_pct``,
        ``position``, ``unrealized_pnl``.
    initial_equity : float
        Starting account equity in USD.
    risk_free_rate : float
        Annualised risk-free rate (default 0.0).

    Returns
    -------
    PerformanceReport
        Fully populated metrics dataclass.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
        """Return numerator / denominator, or *fallback* when denominator ≈ 0."""
        if np.isnan(denominator) or abs(denominator) < 1e-12:
            return fallback
        if np.isnan(numerator):
            return fallback
        return numerator / denominator

    # ------------------------------------------------------------------
    # Return metrics (from equity curve)
    # ------------------------------------------------------------------
    if equity_curve is not None and len(equity_curve) > 0:
        final_equity: float = float(equity_curve["equity"].iloc[-1])
    else:
        final_equity = initial_equity

    net_profit: float = final_equity - initial_equity

    # CAGR
    if (
        equity_curve is not None
        and len(equity_curve) > 1
        and final_equity > 0.0
    ):
        t_start = equity_curve["timestamp"].iloc[0]
        t_end = equity_curve["timestamp"].iloc[-1]
        n_days: float = (t_end - t_start).total_seconds() / 86400.0
        n_years: float = n_days / 365.25
        cagr: float = (final_equity / initial_equity) ** (1.0 / n_years) - 1.0
    else:
        cagr = 0.0

    # ------------------------------------------------------------------
    # Risk metrics (from equity curve)
    # ------------------------------------------------------------------
    max_drawdown_pct: float = 0.0
    if equity_curve is not None and len(equity_curve) > 0:
        max_drawdown_pct = float(equity_curve["drawdown_pct"].max())

    # Daily returns for volatility / Sharpe / Sortino
    if equity_curve is not None and len(equity_curve) > 1:
        ec = equity_curve.set_index("timestamp")
        # Resample to daily frequency (last equity value of each day)
        daily_eq = ec["equity"].resample("D").last().dropna()
        daily_returns = daily_eq.pct_change().dropna()
    else:
        daily_returns = pd.Series(dtype=np.float64)

    if len(daily_returns) > 1:
        # Annualised volatility
        ann_vol: float = float(daily_returns.std()) * np.sqrt(252.0)

        # Excess returns
        daily_rf = risk_free_rate / 252.0
        excess_returns = daily_returns.values - daily_rf

        mean_excess: float = float(np.mean(excess_returns))
        std_excess: float = float(np.std(excess_returns, ddof=1))

        sharpe_ratio: float = _safe_div(
            mean_excess * np.sqrt(252.0), std_excess
        )

        # Sortino — downside deviation (only negative excess returns)
        neg_excess = excess_returns[excess_returns < 0.0]
        if len(neg_excess) > 1:
            downside_std: float = float(np.std(neg_excess, ddof=1))
            sortino_ratio: float = _safe_div(
                mean_excess * np.sqrt(252.0), downside_std
            )
        else:
            sortino_ratio = 0.0
    else:
        ann_vol = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0

    # Calmar ratio
    calmar_ratio: float = _safe_div(cagr, max_drawdown_pct / 100.0)

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------
    total_trades: int = len(trades)

    if total_trades == 0:
        return PerformanceReport(
            net_profit=net_profit,
            cagr=cagr,
            max_drawdown_pct=max_drawdown_pct,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_winner_usd=0.0,
            avg_loser_usd=0.0,
            time_in_market_pct=0.0,
        )

    pnls = np.array([t.pnl_usd for t in trades if t.pnl_usd is not None], dtype=np.float64)

    if len(pnls) == 0:
        pnls = np.array([0.0])

    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]
    n_winners: int = int(len(wins))
    n_losers: int = int(len(losses))

    win_rate: float = n_winners / total_trades

    gross_wins: float = float(wins.sum()) if n_winners > 0 else 0.0
    gross_losses_abs: float = float(np.abs(losses).sum()) if n_losers > 0 else 0.0

    if gross_losses_abs > 0.0:
        profit_factor: float = gross_wins / gross_losses_abs
    elif n_winners > 0 and n_losers == 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_winner: float = float(wins.mean()) if n_winners > 0 else 0.0
    avg_loser: float = float(losses.mean()) if n_losers > 0 else 0.0

    expectancy: float = avg_winner * win_rate + avg_loser * (1.0 - win_rate)
    avg_winner_usd: float = avg_winner
    avg_loser_usd: float = avg_loser

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------
    time_in_market_pct: float = 0.0
    if equity_curve is not None and len(equity_curve) > 0:
        in_market = equity_curve["position"] != 0
        time_in_market_pct = float(in_market.sum()) / len(equity_curve) * 100.0

    return PerformanceReport(
        net_profit=net_profit,
        cagr=cagr,
        max_drawdown_pct=max_drawdown_pct,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_winner_usd=avg_winner_usd,
        avg_loser_usd=avg_loser_usd,
        time_in_market_pct=time_in_market_pct,
    )


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_report(report: PerformanceReport) -> str:
    """Format a human-readable summary of the performance report.

    Parameters
    ----------
    report : PerformanceReport
        Metrics dataclass to format.

    Returns
    -------
    str
        Multi-line text summary suitable for console output.
    """
    lines: List[str] = []
    w = 38  # width of label field

    def row(label: str, value: str) -> None:
        lines.append(f"  {label:<{w}}{value}")

    lines.append("=" * 62)
    lines.append("  PERFORMANCE REPORT — LSR-EH1 Backtest")
    lines.append("=" * 62)

    lines.append("")
    lines.append("  ── Returns ────────────────────────────────")
    row("Net Profit:", f"${report.net_profit:,.2f}")
    row("CAGR:", f"{report.cagr:,.2%}")

    lines.append("")
    lines.append("  ── Risk ───────────────────────────────────")
    row("Max Drawdown:", f"{report.max_drawdown_pct:.2f}%")
    row("Annualised Volatility:", f"{report.annualized_volatility:,.2%}")

    lines.append("")
    lines.append("  ── Risk-Adjusted ──────────────────────────")
    row("Sharpe Ratio:", f"{report.sharpe_ratio:.2f}")
    row("Sortino Ratio:", f"{report.sortino_ratio:.2f}")
    row("Calmar Ratio:", f"{report.calmar_ratio:.2f}")

    lines.append("")
    lines.append("  ── Trade Statistics ────────────────────────")
    row("Total Trades:", f"{report.total_trades}")
    row("Win Rate:", f"{report.win_rate:.1%}")
    pf_str = f"{report.profit_factor:.2f}" if np.isfinite(report.profit_factor) else "∞"
    row("Profit Factor:", pf_str)
    row("Expectancy:", f"${report.expectancy:,.2f}")
    row("Avg Winner:", f"${report.avg_winner_usd:,.2f}")
    row("Avg Loser:", f"${report.avg_loser_usd:,.2f}")

    lines.append("")
    lines.append("  ── Exposure ───────────────────────────────")
    row("Time in Market:", f"{report.time_in_market_pct:.1f}%")

    lines.append("")
    lines.append("=" * 62)

    return "\n".join(lines)
