from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def new_model() -> HistGradientBoostingClassifier:
    # deliberately small + regularised: markets are noisy, big models just memorise
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                          l2_regularization=2.0, min_samples_leaf=40, random_state=0)


def walk_forward(X: pd.DataFrame, y: pd.Series, horizon: int, min_train: int, step: int) -> pd.Series:
    """Out-of-sample P(up) for every row after `min_train`, using expanding windows.
    Training rows stop `horizon` days before each test block (purge) so labels never overlap the future."""
    data = X.join(y.rename("y")).dropna()
    idx = data.index
    pred = pd.Series(np.nan, index=idx)
    for start in range(min_train, len(idx), step):
        end = min(start + step, len(idx))
        train = data.iloc[: max(0, start - horizon)]
        if train["y"].nunique() < 2:
            continue
        m = new_model().fit(train.drop(columns="y"), train["y"])
        pred.iloc[start:end] = m.predict_proba(data.iloc[start:end].drop(columns="y"))[:, 1]
    return pred.dropna()


def fit_latest(X: pd.DataFrame, y: pd.Series, horizon: int):
    """Final model on everything with a known label; returns (model, P(up) for the newest bar)."""
    data = X.join(y.rename("y")).dropna()
    m = new_model().fit(data.drop(columns="y"), data["y"])
    last = X.iloc[[-1]]
    if last.isna().any(axis=1).iloc[0]:
        last = last.fillna(data.drop(columns="y").median())
    return m, float(m.predict_proba(last)[:, 1][0])


def oos_auc(pred: pd.Series, y: pd.Series) -> float | None:
    yy = y.reindex(pred.index).dropna()
    p = pred.reindex(yy.index)
    return float(roc_auc_score(yy, p)) if yy.nunique() == 2 else None
