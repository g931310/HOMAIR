from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, roc_auc_score
from catboost import CatBoostClassifier, Pool

from .public_core import RANDOM_STATE, CATBOOST_PARAMS, prepare_catboost_pair


def bootstrap_metric_ci(y, pred, metric_fn, n_boot=1000, alpha=0.05, stratified=False):
    """Percentile bootstrap CI. RNG is initialized once with seed 42."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    n = len(y)

    if stratified:
        classes = np.unique(y)
        class_idx = {c: np.where(y == c)[0] for c in classes}
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(class_idx[c], size=len(class_idx[c]), replace=True) for c in classes])
            values.append(metric_fn(y[idx], pred[idx]))
    else:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            values.append(metric_fn(y[idx], pred[idx]))

    lo, hi = np.quantile(values, [alpha/2, 1-alpha/2])
    return float(np.mean(values)), float(lo), float(hi)


def paired_regression_comparison(y, pred_a, pred_b, n_boot=2000):
    """Participant-level squared-error tests plus paired bootstrap ΔRMSE (A-B)."""
    y = np.asarray(y); a = np.asarray(pred_a); b = np.asarray(pred_b)
    err_a = (y-a)**2; err_b = (y-b)**2
    t = stats.ttest_rel(err_a, err_b)
    w = stats.wilcoxon(err_a, err_b, zero_method="wilcox", alternative="two-sided")

    rng = np.random.default_rng(RANDOM_STATE)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        rmse_a = mean_squared_error(y[idx], a[idx])**0.5
        rmse_b = mean_squared_error(y[idx], b[idx])**0.5
        diffs.append(rmse_a-rmse_b)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "delta_rmse": mean_squared_error(y,a)**0.5 - mean_squared_error(y,b)**0.5,
        "bootstrap_ci": (float(lo), float(hi)),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": float(w.pvalue),
    }


def _compute_midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k,m)); ty = np.empty((k,n)); tz = np.empty((k,m+n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1)/(m*n) - (m+1.0)/(2*n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx/m + sy/n
    return aucs, np.atleast_2d(cov)


def delong_roc_test(y_true, prob_a, prob_b):
    """Two-sided DeLong test for two correlated AUROCs."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y)
    m = int(y.sum())
    preds = np.vstack([np.asarray(prob_a), np.asarray(prob_b)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ cov @ contrast.T)
    z = float((aucs[0]-aucs[1]) / np.sqrt(max(var, np.finfo(float).eps)))
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "delta_auc": float(aucs[0]-aucs[1]), "p_value": p}


def fit_catboost_and_global_shap(df: pd.DataFrame, numeric: list[str], categorical: list[str]):
    """Fit the published CatBoost classifier on supplied data and return global native SHAP importance.

    This helper does not save or export the trained model.
    """
    xtr, _ = prepare_catboost_pair(df, df, numeric, categorical)
    model = CatBoostClassifier(**CATBOOST_PARAMS, loss_function="Logloss", verbose=0,
                               random_seed=RANDOM_STATE, thread_count=1)
    model.fit(xtr, df["IR_2p5"].astype(int), cat_features=categorical)
    pool = Pool(xtr, cat_features=categorical)
    values = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    mean_abs = np.mean(np.abs(values), axis=0)
    rel = 100 * mean_abs / mean_abs.sum()
    return pd.DataFrame({"feature": xtr.columns, "mean_abs_shap": mean_abs,
                         "relative_importance_percent": rel}).sort_values(
                             "mean_abs_shap", ascending=False).reset_index(drop=True)
