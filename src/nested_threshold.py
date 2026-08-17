from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

from .public_core import RANDOM_STATE, CATBOOST_PARAMS, prepare_catboost_pair

THRESHOLD_GRID = np.arange(0.01, 0.9901, 0.001)


def _threshold_stats(y, p, t):
    pred = p >= t
    tp = np.sum((y == 1) & pred); fn = np.sum((y == 1) & ~pred)
    tn = np.sum((y == 0) & ~pred); fp = np.sum((y == 0) & pred)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return sens, spec


def select_thresholds(y, p):
    rows = []
    for t in THRESHOLD_GRID:
        sens, spec = _threshold_stats(y, p, t)
        rows.append((t, sens, spec, sens + spec - 1))
    tab = pd.DataFrame(rows, columns=["threshold", "sensitivity", "specificity", "youden"])

    youden = float(tab.loc[tab["youden"].idxmax(), "threshold"])

    def sensitivity_rule(target):
        ok = tab.loc[tab["sensitivity"] >= target]
        if ok.empty:
            return float(tab.loc[tab["sensitivity"].idxmax(), "threshold"])
        # maximal specificity; if tied, use the largest threshold
        max_spec = ok["specificity"].max()
        return float(ok.loc[ok["specificity"].eq(max_spec), "threshold"].max())

    return {
        "Default_0p50": 0.50,
        "Youden": youden,
        "Sensitivity80": sensitivity_rule(0.80),
        "Sensitivity90": sensitivity_rule(0.90),
    }


def _model():
    return CatBoostClassifier(**CATBOOST_PARAMS, loss_function="Logloss", verbose=0,
                              random_seed=RANDOM_STATE, thread_count=1)


def nested_threshold_validation(df: pd.DataFrame, numeric: list[str], categorical: list[str],
                                outer_splits=5, inner_splits=4):
    """Nested threshold validation with seed 42 for all stochastic splits and fits."""
    y = df["IR_2p5"].to_numpy(int)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    nested_prob = np.full(len(df), np.nan)
    pred_by_rule = {k: np.full(len(df), -1, dtype=int) for k in
                    ["Default_0p50", "Youden", "Sensitivity80", "Sensitivity90"]}
    fold_thresholds = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer.split(df, y), start=1):
        outer_train = df.iloc[outer_train_idx].reset_index(drop=True)
        outer_test = df.iloc[outer_test_idx]
        ytr = outer_train["IR_2p5"].to_numpy(int)

        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
        inner_oof = np.full(len(outer_train), np.nan)
        for itrain, ival in inner.split(outer_train, ytr):
            tr, va = outer_train.iloc[itrain], outer_train.iloc[ival]
            xtr, xva = prepare_catboost_pair(tr, va, numeric, categorical)
            model = _model()
            model.fit(xtr, tr["IR_2p5"].astype(int), cat_features=categorical)
            inner_oof[ival] = model.predict_proba(xva)[:, 1]

        thresholds = select_thresholds(ytr, inner_oof)
        xtr, xte = prepare_catboost_pair(outer_train, outer_test, numeric, categorical)
        model = _model()
        model.fit(xtr, ytr, cat_features=categorical)
        prob = model.predict_proba(xte)[:, 1]
        nested_prob[outer_test_idx] = prob

        for rule, t in thresholds.items():
            pred_by_rule[rule][outer_test_idx] = (prob >= t).astype(int)
            fold_thresholds.append({"outer_fold": outer_fold, "rule": rule, "threshold": t})

    summaries = []
    fold_thresholds = pd.DataFrame(fold_thresholds)
    for rule, pred in pred_by_rule.items():
        t = fold_thresholds.loc[fold_thresholds.rule.eq(rule), "threshold"]
        tp = ((y == 1) & (pred == 1)).sum(); tn = ((y == 0) & (pred == 0)).sum()
        fp = ((y == 0) & (pred == 1)).sum(); fn = ((y == 1) & (pred == 0)).sum()
        sens = tp/(tp+fn); spec = tn/(tn+fp)
        summaries.append({
            "rule": rule,
            "threshold_mean": t.mean(), "threshold_median": t.median(),
            "threshold_min": t.min(), "threshold_max": t.max(),
            "AUROC": roc_auc_score(y, nested_prob),
            "AUPRC": average_precision_score(y, nested_prob),
            "Brier": brier_score_loss(y, nested_prob),
            "Sensitivity": sens, "Specificity": spec,
            "PPV": tp/(tp+fp) if tp+fp else np.nan,
            "NPV": tn/(tn+fn) if tn+fn else np.nan,
            "F1": 2*tp/(2*tp+fp+fn),
            "BalancedAccuracy": (sens+spec)/2,
        })
    return pd.DataFrame(summaries), fold_thresholds, nested_prob


def derive_deployment_threshold(df: pd.DataFrame, numeric: list[str], categorical: list[str], rule="Sensitivity80"):
    """Post-validation threshold derivation from full-development five-fold OOF probabilities.

    This returns an operational probability threshold only; it is not a diagnostic HOMA-IR cutoff.
    No trained model object is exported by this public helper.
    """
    y = df["IR_2p5"].to_numpy(int)
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(df), np.nan)
    for tr_idx, va_idx in split.split(df, y):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        xtr, xva = prepare_catboost_pair(tr, va, numeric, categorical)
        model = _model()
        model.fit(xtr, tr["IR_2p5"].astype(int), cat_features=categorical)
        oof[va_idx] = model.predict_proba(xva)[:, 1]
    return select_thresholds(y, oof)[rule]
