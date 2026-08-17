from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

from .public_core import RANDOM_STATE, prepare_catboost_pair

CANDIDATES = {
    1: dict(iterations=500, depth=6, learning_rate=0.030, l2_leaf_reg=3, random_strength=1.0),
    2: dict(iterations=700, depth=7, learning_rate=0.030, l2_leaf_reg=5, random_strength=0.5),
    3: dict(iterations=800, depth=6, learning_rate=0.025, l2_leaf_reg=5, random_strength=0.5),
    4: dict(iterations=500, depth=5, learning_rate=0.040, l2_leaf_reg=5, random_strength=0.5),
}


def nested_catboost_tuning(df: pd.DataFrame, numeric: list[str], categorical: list[str],
                           outer_splits=5, inner_splits=3):
    """Compact nested sensitivity analysis reported in the supplementary material."""
    y = df["HOMA_IR"].to_numpy(float)
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for outer_fold, (otr, ote) in enumerate(outer.split(df), start=1):
        train = df.iloc[otr].reset_index(drop=True)
        inner = KFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
        candidate_scores = {}
        for cid, params in CANDIDATES.items():
            rmses = []
            for itr, iva in inner.split(train):
                tr, va = train.iloc[itr], train.iloc[iva]
                xtr, xva = prepare_catboost_pair(tr, va, numeric, categorical)
                model = CatBoostRegressor(**params, loss_function="RMSE", verbose=0,
                                          random_seed=RANDOM_STATE, thread_count=1, allow_writing_files=False)
                model.fit(xtr, tr["HOMA_IR"], cat_features=categorical)
                pred = model.predict(xva)
                rmses.append(mean_squared_error(va["HOMA_IR"], pred) ** 0.5)
            candidate_scores[cid] = float(np.mean(rmses))

        selected = min(candidate_scores, key=candidate_scores.get)
        params = CANDIDATES[selected]
        xtr, xte = prepare_catboost_pair(df.iloc[otr], df.iloc[ote], numeric, categorical)
        model = CatBoostRegressor(**params, loss_function="RMSE", verbose=0,
                                  random_seed=RANDOM_STATE, thread_count=1, allow_writing_files=False)
        model.fit(xtr, df.iloc[otr]["HOMA_IR"], cat_features=categorical)
        pred = model.predict(xte)
        records.append({
            "outer_fold": outer_fold,
            "selected_candidate": selected,
            "mean_inner_cv_rmse": candidate_scores[selected],
            "outer_rmse": mean_squared_error(y[ote], pred) ** 0.5,
        })
    return pd.DataFrame(records)
