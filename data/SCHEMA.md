# Public analysis-ready schema

This repository uses **publication-level variable names only**. It intentionally does not expose MJ source-field names or questionnaire-version mappings.

## Required outcome fields

| Column | Meaning | Unit |
|---|---|---|
| `fasting_glucose` | Fasting plasma glucose | mg/dL |
| `fasting_insulin` | Fasting insulin | µU/mL |

HOMA-IR is calculated as `fasting_glucose × fasting_insulin / 405`.
`fasting_insulin` is used only for outcome construction and is never a predictor.

## Predictor columns used by the public code

Numeric publication-level fields:

`age`, `bmi`, `whr`, `body_fat`, `sbp`, `dbp`, `fasting_glucose`, `total_bilirubin`, `direct_bilirubin`, `alp`, `ast`, `alt`, `ggt`, `bun`, `creatinine`, `uric_acid`, `triglycerides`, `hdl_cholesterol`, `ldl_cholesterol`, `hba1c`, `exercise_hours_week`, `egfr`, `total_cholesterol`

Categorical publication-level fields:

`married`, `education`, `household_income`, `current_smoker`, `current_alcohol_use`, `sleep_duration`

Derived fields are created by `src/public_core.py`:

- triglyceride/HDL ratio
- non-HDL cholesterol
- mean arterial pressure
- pulse pressure
- AST/ALT ratio
- triglyceride-glucose (TyG) index

## Predictor tiers

- **Full enhanced:** anthropometric + biochemical + lifestyle predictors, excluding fasting insulin.
- **Formula-independent:** full enhanced minus fasting glucose and TyG.
- **Non-biochemical:** age, anthropometry, blood pressure and lifestyle only.

## Important scope limitation

The scripts assume that menopause status, hysterectomy exclusions, medication exclusions, duplicate-record handling, and any source-specific questionnaire harmonization have already been completed in the authorized upstream dataset. This repository does not distribute those source-level operations.
