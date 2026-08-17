# HOMA-IR prediction in postmenopausal women — public reproducibility core

This repository is the **minimal public code release** accompanying the manuscript:

**Explainable machine learning for HOMA-IR prediction in postmenopausal women: nested validation and laboratory-free risk stratification**

It is intentionally limited to the analysis methods needed to understand and reproduce the principal modeling workflow with an appropriately licensed, analysis-ready dataset.

## What is included

- five-fold out-of-fold regression and classification
- the published comparator model families and CatBoost benchmark configuration
- three predictor tiers: full enhanced, formula-independent, and non-biochemical
- nested 5 × 4-fold probability-threshold validation
- compact nested CatBoost tuning sensitivity analysis
- bootstrap uncertainty, paired regression comparisons, correlated-AUROC DeLong comparison, and native CatBoost SHAP helpers
- a **fully synthetic** example dataset using publication-level variable names
- a global random seed of **42** for stochastic splitting/model fitting; bootstrap RNGs are initialized once with seed 42 and then advanced sequentially

## What is intentionally NOT included

This public release does **not** contain:

- MJ Health Database participant-level data
- source-database extraction code or internal database field mappings
- source-level eligibility-screening code
- participant IDs, fold assignments, out-of-fold predictions, or intermediate study outputs
- trained model objects or deployment artifacts
- reviewer-response notebooks, development notebooks, diagnostics, or internal pipeline utilities
- manuscript-generation code or private research workflow files

The public code expects an **authorized, analysis-ready cohort** that has already undergone the study's source-level eligibility screening. This is deliberate: the source database is licensed and its extraction/mapping logic is not distributed here.

## Quick start with the synthetic example

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_example.py
```

The synthetic dataset is provided only to demonstrate the code interface. **Its results are not study results and must not be compared with the manuscript.**

## Using an authorized analysis-ready dataset

Use the publication-level column names described in `data/SCHEMA.md`. At minimum, fasting glucose and fasting insulin are required to construct HOMA-IR; fasting insulin is never used as a predictor.

The repository does not attempt to reconstruct source-database menopause, hysterectomy, medication, or questionnaire-version logic. Those steps must be completed under the data provider's authorized data-governance process before using these scripts.

## Random-seed policy

`RANDOM_STATE = 42` is used throughout stochastic splits and model fitting. For bootstrap procedures, the random-number generator is initialized **once** with seed 42 and sequential resamples are drawn from that generator, so bootstrap replicates remain distinct while the full sequence remains reproducible.

## Relationship to the manuscript

- `src/oof_models.py`: five-fold out-of-fold regression/classification
- `src/nested_threshold.py`: nested operating-threshold validation and post-validation deployment-threshold derivation
- `src/catboost_tuning_sensitivity.py`: compact CatBoost nested tuning sensitivity analysis
- `src/statistics_shap.py`: uncertainty, paired comparisons, DeLong AUROC comparison, and SHAP helper functions
- `src/public_core.py`: publication-level feature definitions and leakage-controlled preprocessing utilities

## Data and model availability

No individual-level MJ Health Database data or trained model objects are distributed in this repository. Access to the source database remains subject to the MJ Health Research Foundation's application, review, authorization, and data-governance procedures.

## Citation

Please cite the associated peer-reviewed article once final bibliographic information is available.

## License

This repository is licensed under the Apache License 2.0.
See the LICENSE file for details.

The license applies only to the code and synthetic/example materials
distributed in this repository. It does not grant access to or rights
over the MJ Health Database or any participant-level data.
