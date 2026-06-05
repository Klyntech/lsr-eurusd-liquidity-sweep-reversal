"""
Walk-Forward Validation Framework for LSR-EH1 Strategy.

Implements anchored walk-forward testing with rolling out-of-sample (OOS)
windows. Each fold isolates an OOS test period and optionally a training
period, producing per-fold performance metrics and walk-forward efficiency
(WFE) to assess strategy robustness and detect overfitting.

Usage::

    from config import StrategyConfig
    from validation.walk_forward import WalkForwardRunner

    config = StrategyConfig()
    runner = WalkForwardRunner(config)
    folds = runner.run(df, train_months=24, test_months=6)
    print(runner.summary(folds))
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from config import StrategyConfig
from backtest.engine import BacktestEngine, Trade
from backtest.metrics import PerformanceReport, compute_metrics


# ---------------------------------------------------------------------------
# Walk-Forward Fold Result
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold.

    Attributes
    ----------
    fold : int
        1-based fold index.
    train_start : pd.Timestamp
        First bar timestamp of the training window.
    train_end : pd.Timestamp
        Last bar timestamp of the training window.
    test_start : pd.Timestamp
        First bar timestamp of the out-of-sample test window.
    test_end : pd.Timestamp
        Last bar timestamp of the out-of-sample test window.
    train_trades : int
        Number of trades generated during the training period (0 if
        ``run_train=False``).
    test_trades : int
        Number of trades generated during the test period.
    test_net_profit : float
        Net profit (USD) on the test window.
    test_cagr : float
        Compound annual growth rate on the test window.
    test_sharpe : float
        Annualised Sharpe ratio on the test window.
    test_profit_factor : float
        Profit factor on the test window.
    test_max_drawdown : float
        Maximum drawdown (%) on the test window.
    test_win_rate : float
        Win rate (0-1) on the test window.
    test_avg_reversal_distance : float
        Mean ``|exit_price - entry_price| / atr_entry`` across test trades,
        measuring the typical price reversal magnitude in ATR units.
    walk_forward_efficiency : float
        ``test_sharpe / train_sharpe``. Returns 0.0 when train_sharpe <= 0.
        Values near 1.0 indicate the strategy generalises well; values << 1.0
        suggest overfitting.
    """

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_trades: int
    test_trades: int
    test_net_profit: float
    test_cagr: float
    test_sharpe: float
    test_profit_factor: float
    test_max_drawdown: float
    test_win_rate: float
    test_avg_reversal_distance: float
    walk_forward_efficiency: float


# ---------------------------------------------------------------------------
# Walk-Forward Runner
# ---------------------------------------------------------------------------

class WalkForwardRunner:
    """Walk-forward validation runner.

    Splits historical OHLC data into rolling anchored train/test folds and
    evaluates strategy performance on each OOS window independently. Each fold
    receives a fresh :class:`BacktestEngine` instance (clean state reset) and
    starts with ``initial_equity`` (no compounding between folds) for strict
    isolation.

    Parameters
    ----------
    config : StrategyConfig
        Strategy configuration used to create BacktestEngine instances for
        each fold.
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize with strategy config.

        Parameters
        ----------
        config : StrategyConfig
            The configuration is used to create BacktestEngine instances for
            each fold.
        """
        self._config: StrategyConfig = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        train_months: int = 24,
        test_months: int = 6,
        initial_equity: float = 100_000.0,
        run_train: bool = True,
    ) -> List[WalkForwardFold]:
        """Run walk-forward analysis.

        The runner splits data into rolling anchored folds:

        - Fold 1: train [0 : train_months],       test [train_months : train_months+test_months]
        - Fold 2: train [test_months : +train_months], test [train_months+test_months : +test_months]
        - ...

        Each fold steps forward by ``test_months`` (anchored walk-forward).

        Steps for each fold:

        1. Slice train and test DataFrames by month boundaries.
        2. Create a fresh ``BacktestEngine(config)`` for each period.
        3. Run backtest on the test period (required).
        4. Optionally run backtest on the train period (for walk-forward
           efficiency).
        5. Compute metrics on test trades via :func:`compute_metrics`.
        6. Calculate walk-forward efficiency = test_sharpe / train_sharpe.
        7. Store results in a :class:`WalkForwardFold`.

        **IMPORTANT**: Each fold gets a fresh engine (reset state). The test
        period starts with ``initial_equity`` (no compounding between folds)
        for clean isolation.

        Parameters
        ----------
        df : pd.DataFrame
            OHLC data with a :class:`pandas.DatetimeIndex`. Must contain at
            least 2 rows.
        train_months : int
            Width of the training window in months. Must be >= 1.
        test_months : int
            Width of the out-of-sample test window in months. Must be >= 1.
        initial_equity : float
            Starting equity for each fold's backtest (no compounding between
            folds).
        run_train : bool
            Whether to also backtest the training period (needed for
            walk-forward efficiency computation).

        Returns
        -------
        list[WalkForwardFold]
            One :class:`WalkForwardFold` per completed fold, in chronological
            order. Returns an empty list if the data is insufficient for even
            a single fold.
        """
        # -- Guard clauses --
        if len(df) < 2 or train_months < 1 or test_months < 1:
            return []

        anchor: pd.Timestamp = pd.Timestamp(df.index[0])
        total_window_months: int = train_months + test_months
        step: int = test_months  # fold advances by test_months

        folds: List[WalkForwardFold] = []
        fold_idx: int = 0

        while True:
            # -- Compute month-based boundaries --
            fold_month_offset: int = fold_idx * step

            train_start_ts: pd.Timestamp = anchor + pd.DateOffset(
                months=fold_month_offset
            )
            test_start_ts: pd.Timestamp = anchor + pd.DateOffset(
                months=fold_month_offset + train_months
            )
            test_end_ts: pd.Timestamp = (
                anchor
                + pd.DateOffset(months=fold_month_offset + total_window_months)
                - pd.Timedelta(days=1)
            )

            # Stop if the test window starts at or beyond the last bar
            if test_start_ts >= df.index[-1]:
                break

            # Clamp test_end to the last available bar
            if test_end_ts > df.index[-1]:
                test_end_ts = pd.Timestamp(df.index[-1])

            train_end_ts: pd.Timestamp = test_start_ts - pd.Timedelta(days=1)

            # -- Slice test data --
            test_mask = (df.index >= test_start_ts) & (df.index <= test_end_ts)
            test_df: pd.DataFrame = df.loc[test_mask]

            if len(test_df) < 2:
                # Not enough bars in this test window — advance and retry
                fold_idx += 1
                continue

            # -- Run test backtest (fresh engine per fold) --
            test_engine: BacktestEngine = BacktestEngine(self._config)
            test_curve: pd.DataFrame = test_engine.run(
                test_df, initial_equity=initial_equity
            )
            test_trades: List[Trade] = list(test_engine.trades)

            test_report: PerformanceReport = compute_metrics(
                trades=test_trades,
                equity_curve=test_curve,
                initial_equity=initial_equity,
            )

            # -- Optionally run train backtest for WFE --
            train_report: Optional[PerformanceReport] = None
            train_trade_count: int = 0

            if run_train and train_start_ts < test_start_ts:
                train_mask = (
                    (df.index >= train_start_ts) & (df.index <= train_end_ts)
                )
                train_df: pd.DataFrame = df.loc[train_mask]

                if len(train_df) >= 2:
                    train_engine: BacktestEngine = BacktestEngine(self._config)
                    train_curve: pd.DataFrame = train_engine.run(
                        train_df, initial_equity=initial_equity
                    )
                    train_trade_count = len(train_engine.trades)
                    train_report = compute_metrics(
                        trades=train_engine.trades,
                        equity_curve=train_curve,
                        initial_equity=initial_equity,
                    )

            # -- Walk-forward efficiency --
            train_sharpe: float = (
                train_report.sharpe_ratio if train_report is not None else 0.0
            )
            test_sharpe: float = test_report.sharpe_ratio
            wfe: float = 0.0
            if train_sharpe > 0.0:
                wfe = test_sharpe / train_sharpe

            # -- Average reversal distance --
            avg_rev_dist: float = _compute_avg_reversal_distance(test_trades)

            # -- Handle infinite profit factor --
            pf: float = test_report.profit_factor
            if not np.isfinite(pf):
                pf = 0.0

            folds.append(
                WalkForwardFold(
                    fold=fold_idx + 1,
                    train_start=train_start_ts,
                    train_end=train_end_ts,
                    test_start=test_start_ts,
                    test_end=test_end_ts,
                    train_trades=train_trade_count,
                    test_trades=len(test_trades),
                    test_net_profit=test_report.net_profit,
                    test_cagr=test_report.cagr,
                    test_sharpe=test_sharpe,
                    test_profit_factor=pf,
                    test_max_drawdown=test_report.max_drawdown_pct,
                    test_win_rate=test_report.win_rate,
                    test_avg_reversal_distance=avg_rev_dist,
                    walk_forward_efficiency=wfe,
                )
            )

            fold_idx += 1

        return folds

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, folds: List[WalkForwardFold]) -> str:
        """Format a summary table of all folds.

        Produces a box-drawing table with per-fold OOS metrics followed by
        aggregate statistics across all folds.

        Parameters
        ----------
        folds : list[WalkForwardFold]
            Fold results produced by :meth:`run`.

        Returns
        -------
        str
            Multi-line text summary suitable for console output.
        """
        if not folds:
            return "No walk-forward folds to summarise."

        # Column definitions: (header, width, format_spec)
        col_defs: List[tuple] = [
            ("Fold", 6, None),
            ("Train Start", 14, None),
            ("Train End", 14, None),
            ("Test Start", 14, None),
            ("Trades", 8, None),
            ("Sharpe", 8, None),
            ("PF", 10, None),
            ("Max DD", 10, None),
            ("Win Rate", 10, None),
            ("WFE", 10, None),
        ]
        widths = [w for _, w, _ in col_defs]
        headers = [h for h, _, _ in col_defs]

        lines: List[str] = []

        # Top border
        lines.append(_separator(widths, "┌", "┬", "┐"))

        # Header row
        header_cells = [_center(h, w) for h, w in zip(headers, widths)]
        lines.append("│" + "│".join(header_cells) + "│")

        # Header separator
        lines.append(_separator(widths, "├", "┼", "┤"))

        # Data rows
        for f in folds:
            row_values = [
                str(f.fold),
                f.train_start.strftime("%Y-%m-%d"),
                f.train_end.strftime("%Y-%m-%d"),
                f.test_start.strftime("%Y-%m-%d"),
                f"{f.test_trades}",
                f"{f.test_sharpe:.2f}",
                f"{f.test_profit_factor:.2f}",
                f"{f.test_max_drawdown:.2f}%",
                f"{f.test_win_rate:.1%}",
                f"{f.walk_forward_efficiency:.2f}",
            ]
            cells = [_center(v, w) for v, w in zip(row_values, widths)]
            lines.append("│" + "│".join(cells) + "│")

        # Bottom border
        lines.append(_separator(widths, "└", "┴", "┘"))

        # -- Aggregate statistics --
        n_folds: int = len(folds)
        sharpes = np.array([f.test_sharpe for f in folds])
        pfs = np.array([f.test_profit_factor for f in folds])
        win_rates = np.array([f.test_win_rate for f in folds])
        max_dds = np.array([f.test_max_drawdown for f in folds])
        wfes = np.array([f.walk_forward_efficiency for f in folds])

        avg_sharpe = float(np.mean(sharpes))
        avg_pf = float(np.mean(pfs))
        avg_wr = float(np.mean(win_rates))
        avg_dd = float(np.mean(max_dds))
        avg_wfe = float(np.mean(wfes))
        worst_sharpe = float(np.min(sharpes))
        best_sharpe = float(np.max(sharpes))
        sharpe_positive = int(np.sum(sharpes > 0))
        sharpe_above_1 = int(np.sum(sharpes > 1.0))

        lines.append("")
        lines.append("  Aggregate Statistics (OOS)")
        lines.append(f"  Average OOS Sharpe:      {avg_sharpe:.2f}")
        lines.append(f"  Average OOS PF:          {avg_pf:.2f}")
        lines.append(f"  Average OOS Win Rate:    {avg_wr:.1%}")
        lines.append(f"  Average OOS Max DD:      {avg_dd:.2f}%")
        lines.append(f"  Average WFE:             {avg_wfe:.2f}")
        lines.append(f"  Worst Fold Sharpe:       {worst_sharpe:.2f}")
        lines.append(f"  Best Fold Sharpe:        {best_sharpe:.2f}")
        lines.append(
            f"  Folds with Sharpe > 0:   {sharpe_positive} / {n_folds}"
            f" ({sharpe_positive / n_folds:.0%})"
        )
        lines.append(
            f"  Folds with Sharpe > 1:   {sharpe_above_1} / {n_folds}"
            f" ({sharpe_above_1 / n_folds:.0%})"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, folds: List[WalkForwardFold], filepath: str) -> None:
        """Export fold results to CSV.

        Each row corresponds to one :class:`WalkForwardFold`. All fields are
        written as columns.

        Parameters
        ----------
        folds : list[WalkForwardFold]
            Fold results to export.
        filepath : str
            Destination CSV file path.
        """
        if not folds:
            # Write empty file with headers
            records: List[dict] = []
        else:
            records = [
                {
                    "fold": f.fold,
                    "train_start": f.train_start.isoformat(),
                    "train_end": f.train_end.isoformat(),
                    "test_start": f.test_start.isoformat(),
                    "test_end": f.test_end.isoformat(),
                    "train_trades": f.train_trades,
                    "test_trades": f.test_trades,
                    "test_net_profit": f.test_net_profit,
                    "test_cagr": f.test_cagr,
                    "test_sharpe": f.test_sharpe,
                    "test_profit_factor": f.test_profit_factor,
                    "test_max_drawdown": f.test_max_drawdown,
                    "test_win_rate": f.test_win_rate,
                    "test_avg_reversal_distance": f.test_avg_reversal_distance,
                    "walk_forward_efficiency": f.walk_forward_efficiency,
                }
                for f in folds
            ]

        df_out = pd.DataFrame(records)
        df_out.to_csv(filepath, index=False)


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------

def _compute_avg_reversal_distance(trades: List[Trade]) -> float:
    """Compute mean ``|exit_price - entry_price| / atr_entry``.

    Parameters
    ----------
    trades : list[Trade]
        Trade records to compute the metric over.

    Returns
    -------
    float
        Average reversal distance in ATR units. Returns 0.0 if no valid
        trades (i.e. no trades with non-zero ATR at entry and a filled exit
        price).
    """
    if not trades:
        return 0.0

    distances: List[float] = []
    for t in trades:
        if (
            t.exit_price is not None
            and t.atr_entry is not None
            and t.atr_entry > 0.0
        ):
            distances.append(abs(t.exit_price - t.entry_price) / t.atr_entry)

    return float(np.mean(distances)) if distances else 0.0


def _center(text: str, width: int) -> str:
    """Center *text* within a fixed-width cell, padded with spaces.

    Parameters
    ----------
    text : str
        Cell content.
    width : int
        Total cell width (in characters).

    Returns
    -------
    str
        Centered text padded to exactly *width* characters.
    """
    # Handle multibyte / wide characters conservatively by using ASCII len
    visible_len = len(text)
    if visible_len >= width:
        return text[:width]
    total_pad = width - visible_len
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return " " * left_pad + text + " " * right_pad


def _separator(
    widths: List[int],
    left: str,
    mid: str,
    right: str,
    fill: str = "\u2500",
) -> str:
    """Build a box-drawing separator line.

    Parameters
    ----------
    widths : list[int]
        Column widths in characters.
    left : str
        Left corner character (e.g. ``┌``, ``├``, ``└``).
    mid : str
        Column separator character (e.g. ``┬``, ``┼``, ``┴``).
    right : str
        Right corner character (e.g. ``┐``, ``┤``, ``┘``).
    fill : str
        Fill character within each column (default box-drawing horizontal ``─``).

    Returns
    -------
    str
        A complete separator line, e.g. ``┌──────┬──────────┐``.
    """
    segments = [fill * w for w in widths]
    return left + mid.join(segments) + right
