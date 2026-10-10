"""Usage:  python -m signals.run            (live data from Yahoo Finance)
          python -m signals.run --demo     (synthetic data, no network)"""
from __future__ import annotations
import argparse, datetime as dt, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from .data import fetch_yahoo, synthetic
from .features import make_features, make_target, atr
from .model import walk_forward, fit_latest, oos_auc
from .backtest import run_backtest, to_signal

DISCLAIMER = ("Educational project, not financial advice. Model output is a statistical estimate, can be wrong, "
              "and past/backtested performance does not predict future results. Trading involves risk of loss.")


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 5)
    if isinstance(o, (np.integer,)): return int(o)
    return o


def analyse(df: pd.DataFrame, cfg: dict) -> dict:
    h = cfg["horizon_days"]
    X, y = make_features(df), make_target(df, h)
    pred = walk_forward(X, y, h, cfg["min_train_rows"], cfg["retrain_every"])
    if len(pred) < 60:
        raise RuntimeError("not enough history to validate the model")
    metrics, eq, eq_bh = run_backtest(df["Close"], pred, cfg["buy_threshold"], cfg["sell_threshold"], cfg["cost_bps"])
    metrics["auc"] = oos_auc(pred, y)
    metrics["accuracy"] = float(((pred > 0.5).astype(float) == y.reindex(pred.index)).dropna().mean())
    _, p_up = fit_latest(X, y, h)
    close = float(df["Close"].iloc[-1]); a = float(atr(df).iloc[-1])
    sig = to_signal(p_up, cfg["buy_threshold"], cfg["sell_threshold"])
    tail = 120
    return {
        "asof": df.index[-1].strftime("%Y-%m-%d"), "close": close, "signal": sig, "prob_up": p_up,
        "strength": abs(p_up - 0.5) * 2,
        "levels": {"stop": close - 2 * a, "target": close + 3 * a, "atr": a} if sig == "BUY" else None,
        "metrics": metrics,
        "history": {"dates": [d.strftime("%Y-%m-%d") for d in df.index[-tail:]], "close": df["Close"].iloc[-tail:].tolist(),
                    "pred": pred.reindex(df.index[-tail:]).tolist()},
        "equity": {"dates": [d.strftime("%Y-%m-%d") for d in eq.index[-250:]],
                   "strategy": eq.iloc[-250:].tolist(), "buyhold": eq_bh.iloc[-250:].tolist()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="site/data/signals.json")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no network)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    assets, errors = [], []
    for item in cfg["assets"]:
        sym = item["symbol"]
        try:
            df = synthetic(sym, cfg["history_years"]) if a.demo else fetch_yahoo(sym, cfg["history_years"])
            res = analyse(df, cfg)
            assets.append({"symbol": sym, "name": item["name"], **res})
            print(f"{sym:14s} {res['signal']:5s} P(up)={res['prob_up']:.2f}  AUC={res['metrics']['auc']}")
        except Exception as e:  # keep going; show the failure on the page
            errors.append({"symbol": sym, "error": str(e)[:200]}); print(f"{sym}: FAILED {e}", file=sys.stderr)
    if not assets:
        print("no assets produced", file=sys.stderr); return 1
    order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    assets.sort(key=lambda r: (order[r["signal"]], -r["strength"]))
    out = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "mode": "demo" if a.demo else "live", "horizon_days": cfg["horizon_days"],
           "thresholds": {"buy": cfg["buy_threshold"], "sell": cfg["sell_threshold"]}, "cost_bps": cfg["cost_bps"],
           "disclaimer": DISCLAIMER, "assets": assets, "errors": errors}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(clean(out), separators=(",", ":")))
    print(f"wrote {a.out} ({len(assets)} assets, {len(errors)} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
