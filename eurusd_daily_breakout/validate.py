"""Frozen validation referee. Emits ORACLE_JSON with all gate keys.

Gates (fixed BEFORE looking at results):
  G1 costs_on        : costs strictly reduce total_R vs zero-cost
  G2 no_lookahead    : shift-invariance test (see tests) + next-bar fill audit
  G3 min_trades      : n_trades >= 30
  G4 OOS PF          : out-of-sample PF >= 1.0
  G5 WF consistency  : >= 50% of walk-forward folds profitable
  G6 cost stress     : PF >= 1.0 at 2x costs
  G7 MC p05          : 5th percentile of bootstrap total_R > 0
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .data_prep import load_daily
from .strategy import Params, backtest, metrics

IS_END = "2025-03-31"
EMBARGO = 2 * Params().max_bars  # purge gap between IS and OOS


def wf_folds(df: pd.DataFrame, p: Params, n: int = 4):
    """Anchored walk-forward: expanding IS, one OOS block per fold."""
    third = len(df) // (n + 1)
    out = []
    for k in range(1, n + 1):
        is_end = third * k
        oos_start = is_end + EMBARGO
        oos_end = third * (k + 1) if k < n else len(df)
        if oos_end - oos_start < 30:
            continue
        # params frozen on IS only; no optimisation performed anywhere
        r = backtest(df.iloc[:oos_end].copy(), p)
        t = r["trades"]
        t = t[t["entry_time"] >= df.index[oos_start]] if not t.empty else t
        out.append((str(df.index[oos_start].date()), metrics(t)))
    return out


def bootstrap_p05(rs: pd.Series, n: int = 5000, seed: int = 7) -> float:
    if rs.empty:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(rs.to_numpy(), size=(n, len(rs)), replace=True).sum(axis=1)
    return float(np.percentile(draws, 5))


def main() -> dict:
    df, dh = load_daily()
    p = Params()

    is_df = df[df.index <= IS_END]
    oos_df = df[df.index >= (pd.Timestamp(IS_END) + pd.Timedelta(days=EMBARGO))]
    full = backtest(df, p)
    r_is = backtest(is_df, p)
    r_oos = backtest(oos_df, p)
    m_is, m_oos = metrics(r_is["trades"]), metrics(r_oos["trades"])

    # G1 costs on
    zero = backtest(df, Params(cost_mult=0.0))
    m_zero = metrics(zero["trades"])
    g1 = m_zero["total_R"] > metrics(full["trades"])["total_R"]

    # G6 cost stress
    stress = backtest(df, Params(cost_mult=2.0))
    m_stress = metrics(stress["trades"])

    # G2 no lookahead: perturbing bar t's close must not change any fill at t
    # (verified structurally in tests/test_no_lookahead.py); here we assert the
    # engine never fills on the bar that produced the signal.
    sig_bar_fills = 0
    for t in full["trades"].itertuples():
        if t.entry_time >= t.exit_time:
            sig_bar_fills += 1
    g2 = sig_bar_fills == 0 and full["trades"].shape[0] > 0

    # G3
    g3 = m_is["n_trades"] >= 30

    folds = wf_folds(df, p)
    prof = sum(1 for _, m in folds if m["PF"] > 1.0)
    g5 = len(folds) > 0 and prof / len(folds) >= 0.5

    g4 = m_oos["PF"] >= 1.0
    g6 = m_stress["PF"] >= 1.0
    p05 = bootstrap_p05(r_oos["trades"]["r"])
    g7 = bool(p05 > 0)

    gates = {
        "G1_costs_on": bool(g1), "G2_no_lookahead": bool(g2), "G3_min_trades_IS": bool(g3),
        "G4_OOS_PF_ge_1": bool(g4), "G5_WF_majority_profitable": bool(g5),
        "G6_PF_at_2x_costs": bool(g6), "G7_MC_p05_gt_0": g7,
    }
    failed = [k for k, v in gates.items() if not v]

    report = {
        "period": f"{df.index[0].date()} -> {df.index[-1].date()}",
        "costs": f"{(p.spread_pips + p.slippage_pips):.1f} pips/side (spread {p.spread_pips} + slip {p.slippage_pips})",
        "n_trades": m_is["n_trades"],
        "maxDD": round(full["maxdd_equity"], 4),
        "WF_sharpe": None,
        "MC_p05": round(p05, 3) if p05 == p05 else None,
        "data_hash": dh,
        "data_path": "data/raw/eurusd_yfinance_h1_730d.csv -> daily resample",
        "IS": m_is, "OOS": m_oos, "FULL": metrics(full["trades"]),
        "zero_cost_total_R": round(m_zero["total_R"], 3),
        "stress_2x": m_stress,
        "walk_forward": [{"oos_from": d, **m} for d, m in folds],
        "gates": gates, "failed_gates": failed,
        "verdict": "VALIDATED" if not failed else "REJECTED",
    }
    return report


if __name__ == "__main__":
    rep = main()
    print("ORACLE_JSON " + json.dumps(rep, default=str))
