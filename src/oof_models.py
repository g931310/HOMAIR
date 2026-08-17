from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from .public_core import RANDOM_STATE, N_SPLITS, CATBOOST_PARAMS, make_preprocessor, prepare_catboost_pair


def _regressor(name: str):
    if name == "MLR":
        return LinearRegression()
    if name == "SGB":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                         subsample=0.8, random_state=RANDOM_STATE)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8,
                            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1)
    if name == "LightGBM":
        return LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)
    if name == "CatBoost":
        return CatBoostRegressor(**CATBOOST_PARAMS, loss_function="RMSE", verbose=0,
                                 random_seed=RANDOM_STATE, thread_count=1)
    raise ValueError(name)


def _classifier(name: str):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)
    if name == "SGB":
        return GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                                          subsample=0.8, random_state=RANDOM_STATE)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8,
                             objective="binary:logistic", eval_metric="logloss",
                             random_state=RANDOM_STATE, n_jobs=1)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)
    if name == "CatBoost":
        return CatBoostClassifier(**CATBOOST_PARAMS, loss_function="Logloss", verbose=0,
                                  random_seed=RANDOM_STATE, thread_count=1)
    raise ValueError(name)


def regression_oof(df: pd.DataFrame, numeric: list[str], categorical: list[str],
                   models=("MLR", "SGB", "XGBoost", "LightGBM", "CatBoost")):
    y = df["HOMA_IR"].to_numpy(float)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    for name in models:
        pred = np.full(len(df), np.nan)
        for train_idx, test_idx in splitter.split(df):
            tr, te = df.iloc[train_idx], df.iloc[test_idx]
            if name == "CatBoost":
                xtr, xte = prepare_catboost_pair(tr, te, numeric, categorical)
                model = _regressor(name)
                model.fit(xtr, tr["HOMA_IR"], cat_features=categorical)
                pred[test_idx] = model.predict(xte)
            else:
                prep = make_preprocessor(numeric, categorical, scale_numeric=False)
                model = Pipeline([("preprocess", prep), ("model", _regressor(name))])
                model.fit(tr[numeric + categorical], tr["HOMA_IR"])
                pred[test_idx] = model.predict(te[numeric + categorical])
        out[name] = {
            "predictions": pred,
            "RMSE": mean_squared_error(y, pred) ** 0.5,
            "R2": r2_score(y, pred),
        }
    return out


def classification_oof(df: pd.DataFrame, numeric: list[str], categorical: list[str],
                        models=("LogisticRegression", "SGB", "XGBoost", "LightGBM", "CatBoost"),
                        threshold=0.50):
    y = df["IR_2p5"].to_numpy(int)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    for name in models:
        prob = np.full(len(df), np.nan)
        for train_idx, test_idx in splitter.split(df, y):
            tr, te = df.iloc[train_idx], df.iloc[test_idx]
            if name == "CatBoost":
                xtr, xte = prepare_catboost_pair(tr, te, numeric, categorical)
                model = _classifier(name)
                model.fit(xtr, tr["IR_2p5"].astype(int), cat_features=categorical)
                prob[test_idx] = model.predict_proba(xte)[:, 1]
            else:
                prep = make_preprocessor(numeric, categorical, scale_numeric=(name == "LogisticRegression"))
                model = Pipeline([("preprocess", prep), ("model", _classifier(name))])
                model.fit(tr[numeric + categorical], tr["IR_2p5"].astype(int))
                prob[test_idx] = model.predict_proba(te[numeric + categorical])[:, 1]
        pred = (prob >= threshold).astype(int)
        tp = ((y == 1) & (pred == 1)).sum(); tn = ((y == 0) & (pred == 0)).sum()
        fp = ((y == 0) & (pred == 1)).sum(); fn = ((y == 1) & (pred == 0)).sum()
        out[name] = {
            "probabilities": prob,
            "AUROC": roc_auc_score(y, prob),
            "AUPRC": average_precision_score(y, prob),
            "Brier": brier_score_loss(y, prob),
            "Accuracy": accuracy_score(y, pred),
            "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp),
            "PPV": tp / (tp + fp) if tp + fp else np.nan,
            "NPV": tn / (tn + fn) if tn + fn else np.nan,
            "F1": f1_score(y, pred),
            "BalancedAccuracy": balanced_accuracy_score(y, pred),
        }
    return out
