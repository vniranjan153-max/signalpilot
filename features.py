from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature at row t uses data up to and including t only (no look-ahead)."""
    c, v = df["Close"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    ret1 = np.log(c).diff()
    for k in (1, 3, 5, 10, 20, 60):
        f[f"ret_{k}"] = np.log(c).diff(k)
    f["rsi14"] = rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["sma10_50"] = c.rolling(10).mean() / c.rolling(50).mean() - 1
    f["dist_sma200"] = c / c.rolling(200).mean() - 1
    f["atr_pct"] = atr(df) / c
    f["vol20"] = ret1.rolling(20).std()
    f["vol_ratio"] = ret1.rolling(5).std() / ret1.rolling(60).std()
    f["range_pos"] = (c - c.rolling(20).min()) / (c.rolling(20).max() - c.rolling(20).min())
    f["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    f["dow"] = df.index.dayofweek
    return f.replace([np.inf, -np.inf], np.nan)


def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """1 if close is higher `horizon` trading days from now. NaN for the last `horizon` rows."""
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    y = (fwd > 0).astype(float)
    y[fwd.isna()] = np.nan
    return y
