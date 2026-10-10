from __future__ import annotations
import zlib
import numpy as np
import pandas as pd


def fetch_yahoo(symbol: str, years: int) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance (adjusted)."""
    import yfinance as yf
    df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[~df.index.duplicated()].sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def synthetic(symbol: str, years: int) -> pd.DataFrame:
    """Deterministic fake prices (volatility clustering + a whisper of momentum).
    Used by --demo and tests so the site works with no network access."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    n = int(years * 252)
    vol = np.empty(n); vol[0] = 0.01
    for i in range(1, n):
        vol[i] = np.clip(0.0004 + 0.92 * vol[i - 1] + rng.normal(0, 0.0012), 0.004, 0.04)
    r = np.empty(n); r[0] = 0.0
    for i in range(1, n):
        r[i] = 0.0004 + 0.05 * r[i - 1] + vol[i] * rng.standard_normal()
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volu = (1e6 * np.exp(rng.normal(0, 0.3, n)) * (1 + 20 * np.abs(r))).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volu}, index=idx)
