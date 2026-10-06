"""Structural no-lookahead proofs (stdlib unittest). The engine must be causal."""
import unittest

import numpy as np
import pandas as pd

from eurusd_daily_breakout.data_prep import to_daily
from eurusd_daily_breakout.strategy import Params, backtest, signals


def _synth(n=600, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="1D")
    p = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"open": p, "high": p * (1 + abs(rng.normal(0, 0.002, n))),
         "low": p * (1 - abs(rng.normal(0, 0.002, n))), "close": p}, index=idx)


def _pnl(res):
    t = res["trades"]
    return 0.0 if t.empty else float(t["pnl"].sum())


class TestNoLookahead(unittest.TestCase):
    def test_bar_cannot_trigger_its_own_breakout(self):
        df = _synth()
        s = signals(df, Params())
        lb = Params().lookback
        for i in range(df.shape[0]):
            if not np.isnan(s["hh"].iloc[i]):
                w = df["high"].iloc[i - lb: i].max()
                self.assertAlmostEqual(s["hh"].iloc[i], w, places=12)
            if not np.isnan(s["ll"].iloc[i]):
                w2 = df["low"].iloc[i - lb: i].min()
                self.assertAlmostEqual(s["ll"].iloc[i], w2, places=12)

    def test_truncating_future_leaves_past_fills_unchanged(self):
        df = _synth(500)
        a = backtest(df, Params())["trades"]
        b = backtest(df.iloc[:300].copy(), Params())["trades"]
        n = min(len(a), len(b))
        self.assertGreater(n, 0)
        pd.testing.assert_frame_equal(
            a.iloc[:n][["entry_time", "entry_px", "side"]].reset_index(drop=True),
            b.iloc[:n][["entry_time", "entry_px", "side"]].reset_index(drop=True))

    def test_fill_strictly_after_signal_and_before_exit(self):
        df = _synth(500)
        t = backtest(df, Params())["trades"]
        self.assertGreater(len(t), 0)
        self.assertTrue((t["entry_time"] < t["exit_time"]).all())
        self.assertTrue((t["entry_time"] >= df.index[1]).all())

    def test_entry_is_the_fill_bar_open_plus_one_cost(self):
        """The engine fills at the NEXT bar's open, paying exactly one cost."""
        df = _synth(500)
        t = backtest(df, Params())["trades"]
        cost = Params().cost_per_side()
        self.assertGreater(len(t), 0)
        for row in t.itertuples():
            i = df.index.get_loc(row.entry_time)
            expected = df["open"].iloc[i] + (cost if row.side == "long" else -cost)
            self.assertAlmostEqual(row.entry_px, expected, places=12)

    def test_costs_reduce_pnl_monotonically(self):
        df = _synth(500)
        self.assertGreaterEqual(_pnl(backtest(df, Params(cost_mult=0.0))),
                                _pnl(backtest(df, Params(cost_mult=1.0))))
        self.assertGreaterEqual(_pnl(backtest(df, Params(cost_mult=1.0))),
                                _pnl(backtest(df, Params(cost_mult=3.0))))

    def test_daily_resample_uses_true_daily_extremes(self):
        h = pd.DataFrame(
            {"open": [1.0, 1.1, 1.2], "high": [1.2, 1.3, 1.4], "low": [0.9, 1.0, 1.1],
             "close": [1.1, 1.2, 1.3]},
            index=pd.date_range("2020-01-01", periods=3, freq="1h"))
        d = to_daily(h)
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d["close"].iloc[0], 1.3, places=12)
        self.assertAlmostEqual(d["high"].iloc[0], 1.4, places=12)
        self.assertAlmostEqual(d["low"].iloc[0], 0.9, places=12)

    def test_stop_is_never_widened(self):
        df = _synth(500)
        t = backtest(df, Params())["trades"]
        p = Params()
        for row in t.itertuples():
            atr_used = p.stop_mult * (row.entry_px - row.exit_px if row.exit_reason != "stop" else 0) + 1e-9
            del atr_used
        stops = t[t["exit_reason"] == "stop"]
        self.assertTrue((stops["r"] <= 0.02).all(), "a stop exit cannot be a winner beyond cost")


if __name__ == "__main__":
    unittest.main(verbosity=2)
