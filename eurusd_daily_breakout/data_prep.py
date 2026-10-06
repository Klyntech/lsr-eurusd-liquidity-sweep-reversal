"""Resample the 1h EUR/USD series to daily bars (UTC) and hash the result.

No lookahead is introduced here: a daily bar for date D is final at the close
of D, and it is only ever consumed by the strategy at the open of D+1.
"""
from __future__ import annotations

import hashlib
import pathlib

import pandas as pd

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "eurusd_yfinance_h1_730d.csv"


def load_hourly(path: pathlib.Path | str = RAW) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df.set_index("timestamp")[["open", "high", "low", "close"]]


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    agg = hourly.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return agg.dropna()


def data_hash(df: pd.DataFrame) -> str:
    payload = df.to_csv().encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def load_daily(path: pathlib.Path | str = RAW) -> tuple[pd.DataFrame, str]:
    d = to_daily(load_hourly(path))
    return d, data_hash(d)


if __name__ == "__main__":
    d, h = load_daily()
    print(f"daily bars={len(d)} from={d.index[0].date()} to={d.index[-1].date()} hash={h}")
