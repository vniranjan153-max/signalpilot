import json, numpy as np, pandas as pd
from signals.data import synthetic
from signals.features import make_features, make_target
from signals.backtest import to_signal
from signals import run


def test_features_have_no_lookahead():
    df = synthetic("TEST", 3)
    full = make_features(df)
    cut = 600
    part = make_features(df.iloc[:cut])
    pd.testing.assert_series_equal(full.iloc[cut - 1], part.iloc[cut - 1], check_names=False)


def test_target_uses_future_only_for_label():
    df = synthetic("TEST", 2)
    y = make_target(df, 5)
    assert y.iloc[-5:].isna().all() and y.iloc[:-5].notna().all()


def test_signal_thresholds():
    assert to_signal(0.7, 0.58, 0.42) == "BUY"
    assert to_signal(0.3, 0.58, 0.42) == "SELL"
    assert to_signal(0.5, 0.58, 0.42) == "HOLD"


def test_end_to_end_demo(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("history_years: 4\nhorizon_days: 5\nbuy_threshold: 0.58\nsell_threshold: 0.42\ncost_bps: 10\n"
                   "min_train_rows: 400\nretrain_every: 42\nassets:\n  - {symbol: AAA, name: A}\n")
    out = tmp_path / "s.json"
    assert run.main(["--config", str(cfg), "--out", str(out), "--demo"]) == 0
    d = json.loads(out.read_text())
    a = d["assets"][0]
    assert d["mode"] == "demo" and a["signal"] in {"BUY", "SELL", "HOLD"} and 0 <= a["prob_up"] <= 1
    assert "disclaimer" in d and len(a["history"]["close"]) == 120
