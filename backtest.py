from __future__ import annotations
import numpy as np
import pandas as pd


def to_signal(p: float, buy: float, sell: float) -> str:
    return "BUY" if p >= buy else "SELL" if p <= sell else "HOLD"


def run_backtest(close: pd.Series, pred: pd.Series, buy: float, sell: float, cost_bps: float) -> dict:
    """Long-only: be long after a BUY, flat after a SELL, keep the previous state on HOLD.
    The position chosen at close t earns the t -> t+1 return. Costs are charged on every position change."""
    px = close.reindex(pred.index)
    nxt = close.pct_change().shift(-1).reindex(pred.index)
    pos = pd.Series(0.0, index=pred.index)
    state = 0.0
    for t, p in pred.items():
        if p >= buy: state = 1.0
        elif p <= sell: state = 0.0
        pos[t] = state
    turn = pos.diff().abs().fillna(pos.abs())
    strat = (pos * nxt - turn * cost_bps / 1e4).dropna()
    bh = nxt.dropna()
    n = max(len(strat), 1)
    eq = (1 + strat).cumprod()
    eq_bh = (1 + bh.reindex(strat.index)).cumprod()
    ann = lambda r: float((1 + r).prod() ** (252 / n) - 1) if n > 20 else float("nan")
    sharpe = float(strat.mean() / strat.std() * np.sqrt(252)) if strat.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    in_mkt = strat[pos.reindex(strat.index) > 0]
    return {
        "oos_days": int(n),
        "strategy_cagr": ann(strat), "buyhold_cagr": ann(bh.reindex(strat.index)),
        "strategy_total": float(eq.iloc[-1] - 1) if len(eq) else 0.0,
        "buyhold_total": float(eq_bh.iloc[-1] - 1) if len(eq_bh) else 0.0,
        "sharpe": sharpe, "max_drawdown": dd,
        "trades": int((turn > 0).sum()), "time_in_market": float(pos.mean()),
        "day_hit_rate": float((in_mkt > 0).mean()) if len(in_mkt) else float("nan"),
    }, eq, eq_bh
