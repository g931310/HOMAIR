from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
HOMA_IR_CUTOFF = 2.5
N_SPLITS = 5

CATBOOST_PARAMS = dict(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=3,
    random_strength=1.0,
    allow_writing_files=False,
)

BASE_NUMERIC = [
    "age", "bmi", "whr", "sbp", "dbp", "fasting_glucose",
    "total_bilirubin", "direct_bilirubin", "alp", "ast", "alt", "ggt",
    "bun", "creatinine", "uric_acid", "triglycerides", "hdl_cholesterol",
    "ldl_cholesterol", "hba1c", "exercise_hours_week",
]

EXTRA_NUMERIC = ["body_fat", "egfr", "total_cholesterol"]
ENGINEERED = [
    "tg_hdl_ratio", "non_hdl_cholesterol", "mean_arterial_pressure",
    "pulse_pressure", "ast_alt_ratio", "tyg_index",
]
CATEGORICAL = [
    "married", "education", "household_income", "current_smoker",
    "current_alcohol_use", "sleep_duration",
]
NONBIO_NUMERIC = [
    "age", "bmi", "whr", "body_fat", "sbp", "dbp",
    "mean_arterial_pressure", "pulse_pressure", "exercise_hours_week",
]


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    ok = num.notna() & den.notna() & den.ne(0)
    out.loc[ok] = num.loc[ok] / den.loc[ok]
    return out


def prepare_analysis_ready(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prepare an already-eligible, analysis-ready cohort using public variable names.

    Source-database extraction, menopause determination, medication exclusions,
    hysterectomy exclusions and questionnaire-version harmonization are deliberately
    outside this public repository.
    """
    df = frame.copy()
    for req in ["fasting_glucose", "fasting_insulin"]:
        if req not in df.columns:
            raise KeyError(f"Missing required outcome field: {req}")

    df["fasting_glucose"] = pd.to_numeric(df["fasting_glucose"], errors="coerce")
    df["fasting_insulin"] = pd.to_numeric(df["fasting_insulin"], errors="coerce")
    df = df.loc[df["fasting_glucose"].gt(0) & df["fasting_insulin"].gt(0)].copy()

    df["HOMA_IR"] = df["fasting_glucose"] * df["fasting_insulin"] / 405.0
    df["IR_2p5"] = (df["HOMA_IR"] >= HOMA_IR_CUTOFF).astype(int)

    if {"triglycerides", "hdl_cholesterol"}.issubset(df.columns):
        df["tg_hdl_ratio"] = _safe_ratio(df["triglycerides"], df["hdl_cholesterol"])
    if {"total_cholesterol", "hdl_cholesterol"}.issubset(df.columns):
        df["non_hdl_cholesterol"] = df["total_cholesterol"] - df["hdl_cholesterol"]
    if {"sbp", "dbp"}.issubset(df.columns):
        df["mean_arterial_pressure"] = (df["sbp"] + 2 * df["dbp"]) / 3.0
        df["pulse_pressure"] = df["sbp"] - df["dbp"]
    if {"ast", "alt"}.issubset(df.columns):
        df["ast_alt_ratio"] = _safe_ratio(df["ast"], df["alt"])
    if {"triglycerides", "fasting_glucose"}.issubset(df.columns):
        product = df["triglycerides"] * df["fasting_glucose"] / 2.0
        df["tyg_index"] = np.where(product > 0, np.log(product), np.nan)

    present = lambda cols: [c for c in cols if c in df.columns]
    full_numeric = list(dict.fromkeys(present(BASE_NUMERIC) + present(EXTRA_NUMERIC) + present(ENGINEERED)))
    categorical = present(CATEGORICAL)

    if full_numeric or categorical:
        missing = df[full_numeric + categorical].isna().mean()
        drop = set(missing[missing > 0.90].index)
    else:
        drop = set()

    full_numeric = [c for c in full_numeric if c not in drop]
    categorical = [c for c in categorical if c not in drop]
    formula_numeric = [c for c in full_numeric if c not in {"fasting_glucose", "tyg_index"}]
    nonbio_numeric = [c for c in present(NONBIO_NUMERIC) if c not in drop]
    nonbio_categorical = [c for c in categorical]

    tiers = {
        "full": {"numeric": full_numeric, "categorical": categorical},
        "formula_independent": {"numeric": formula_numeric, "categorical": categorical},
        "nonbiochemical": {"numeric": nonbio_numeric, "categorical": nonbio_categorical},
    }
    return df.reset_index(drop=True), tiers


def make_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool = False):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_catboost_pair(train: pd.DataFrame, test: pd.DataFrame,
                          numeric: list[str], categorical: list[str]):
    xtr = train[numeric + categorical].copy()
    xte = test[numeric + categorical].copy()

    for c in numeric:
        med = pd.to_numeric(xtr[c], errors="coerce").median()
        xtr[c] = pd.to_numeric(xtr[c], errors="coerce").fillna(med)
        xte[c] = pd.to_numeric(xte[c], errors="coerce").fillna(med)

    for c in categorical:
        xtr[c] = xtr[c].astype("object").where(xtr[c].notna(), "__MISSING__").astype(str)
        xte[c] = xte[c].astype("object").where(xte[c].notna(), "__MISSING__").astype(str)

    return xtr, xte
