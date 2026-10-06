"""EUR/USD daily Donchian breakout + ATR risk. Explicit, auditable execution.

RULES (frozen; all decision inputs come from bar t's CLOSE, fills at t+1 OPEN):
  * ENTRY LONG  : close[t] > max(high[t-LB .. t-1])  AND ATR%[t] >= vol_min
  * ENTRY SHORT : close[t] < min(low [t-LB .. t-1])  AND ATR%[t] >= vol_min
  * STOP        : entry -/+ stop_mult*ATR[t], set at fill, never widened
  * EXIT        : stop hit intrabar, else EMA(ema_exit) close-cross, else
                  max_bars time stop (close)
  * COSTS       : (spread_pips + slippage_pips)*cost_mult pips PER SIDE on
                  every fill (entry and exit)

NO-LOOKAHEAD CONTROLS
  1. Donchian window is .rolling(LB).max().shift(1) -> bar t can never see itself.
  2. ATR is a causal ewm of lagged true range.
  3. Signal on close[t] -> fill on open[t+1]. Never same-bar entry.
  4. Stop is checked before other exits (pessimistic on intrabar collisions).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

PIP = 0.0001


@dataclass(frozen=True)
class Params:
    lookback: int = 20
    atr_period: int = 14
    stop_mult: float = 2.0
    ema_exit: int = 10
    vol_min: float = 0.0035
    max_bars: int = 10
    risk_frac: float = 0.01
    spread_pips: float = 0.8
    slippage_pips: float = 0.4
    cost_mult: float = 1.0

    def cost_per_side(self) -> float:
        return (self.spread_pips + self.slippage_pips) * self.cost_mult * PIP


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(), (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Signal frame. long_sig/short_sig on row t are ACTED ON at row t+1."""
    s = df.copy()
    s["atr"] = _atr(df, p.atr_period)
    s["hh"] = df["high"].rolling(p.lookback).max().shift(1)
    s["ll"] = df["low"].rolling(p.lookback).min().shift(1)
    s["ema"] = df["close"].ewm(span=p.ema_exit, adjust=False).mean()
    s["atrpct"] = s["atr"] / df["close"]
    ready = s["atr"].notna() & s["hh"].notna() & s["ll"].notna() & (s["atrpct"] >= p.vol_min)
    s["long_sig"] = (ready & (df["close"] > s["hh"])).to_numpy()
    s["short_sig"] = (ready & (df["close"] < s["ll"])).to_numpy()
    return s


def backtest(df: pd.DataFrame, p: Params) -> dict:
    s = signals(df, p)
    cost = p.cost_per_side()
    idx = s.index
    o, h, l, c = s["open"].to_numpy(), s["high"].to_numpy(), s["low"].to_numpy(), s["close"].to_numpy()
    ema = s["ema"].to_numpy()
    atr = s["atr"].to_numpy()
    lsig, ssig = s["long_sig"].to_numpy(), s["short_sig"].to_numpy()

    trades = []
    equity, peak, maxdd = 1.0, 1.0, 0.0
    pos, entry, stop, units, entry_eq, bars = 0, 0.0, 0.0, 0.0, 1.0, 0
    entry_i = 0

    for i in range(1, len(s)):
        # ---- FLAT: act on the signal generated at the previous close ----
        if pos == 0:
            d = 1 if lsig[i - 1] else (-1 if ssig[i - 1] else 0)
            if d != 0 and atr[i - 1] and atr[i - 1] > 0:
                fill = o[i] + d * cost            # buyer pays up, seller hits bid
                stop = fill - d * p.stop_mult * atr[i - 1]
                entry, pos, entry_i, bars = fill, d, i, 0
                entry_eq = equity
                risk_px = abs(fill - stop)
                units = (equity * p.risk_frac) / risk_px if risk_px > 0 else 0.0
            continue

        # ---- IN POSITION: stop first (pessimistic), then ema, then time ----
        bars += 1
        exit_px, reason = None, ""
        if pos == 1 and l[i] <= stop:
            exit_px, reason = stop - cost, "stop"
        elif pos == -1 and h[i] >= stop:
            exit_px, reason = stop + cost, "stop"
        elif not np.isnan(ema[i - 1]) and pos == 1 and c[i] < ema[i - 1]:
            exit_px, reason = c[i] - cost, "ema"
        elif not np.isnan(ema[i - 1]) and pos == -1 and c[i] > ema[i - 1]:
            exit_px, reason = c[i] + cost, "ema"
        elif bars >= p.max_bars:
            exit_px, reason = c[i] - pos * cost, "time"

        if exit_px is not None:
            pnl = pos * (exit_px - entry) * units
            equity += pnl
            risk_amt = abs(entry - stop) * units
            trades.append(
                {
                    "entry_time": idx[entry_i], "exit_time": idx[i],
                    "side": "long" if pos == 1 else "short", "exit_reason": reason,
                    "bars_held": bars, "units": units, "pnl": pnl,
                    "r": pnl / risk_amt if risk_amt > 0 else 0.0,
                    "entry_px": entry, "exit_px": exit_px,
                }
            )
            pos, units, bars = 0, 0.0, 0
        else:
            peak = max(peak, equity)
            maxdd = max(maxdd, (peak - equity) / peak)
    return {
        "trades": pd.DataFrame(trades),
        "params": asdict(p),
        "final_equity": equity,
        "cost_per_side": cost,
        "maxdd_equity": maxdd,
    }


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0, "PF": float("nan"), "avg_R": float("nan"),
                "win_rate": float("nan"), "total_R": 0.0, "avg_bars": float("nan")}
    r = trades["r"]
    wins, losses = r[r > 0], r[r <= 0]
    gw, gl = wins.sum(), -losses.sum()
    return {
        "n_trades": int(len(r)),
        "PF": float(gw / gl) if gl > 0 else float("inf"),
        "avg_R": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "total_R": float(r.sum()),
        "gross_win_R": float(gw),
        "gross_loss_R": float(gl),
        "avg_bars": float(trades["bars_held"].mean()),
        "net_pnl": float(trades["pnl"].sum()),
    }
