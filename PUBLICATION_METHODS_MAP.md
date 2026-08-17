# Manuscript-to-code map

This file helps reviewers locate the public implementation without exposing the private research workflow.

| Manuscript method | Public file |
|---|---|
| HOMA-IR outcome and three predictor tiers | `src/public_core.py` |
| Five-fold OOF continuous regression | `src/oof_models.py` |
| Five-fold OOF HOMA-IR ≥2.5 classification | `src/oof_models.py` |
| Nested 5 × 4-fold threshold validation | `src/nested_threshold.py` |
| Sensitivity ≥80% / ≥90% and Youden operating rules | `src/nested_threshold.py` |
| Post-validation deployment-threshold derivation | `src/nested_threshold.py` |
| Compact nested CatBoost tuning sensitivity analysis | `src/catboost_tuning_sensitivity.py` |
| Bootstrap uncertainty | `src/statistics_shap.py` |
| Paired regression comparisons | `src/statistics_shap.py` |
| Correlated-AUROC DeLong comparison | `src/statistics_shap.py` |
| Native CatBoost global SHAP helper | `src/statistics_shap.py` |

## Deliberately outside the public repository

MJ source-field mapping, source-level eligibility extraction, questionnaire-version recoding, participant-level outputs, trained models, manuscript-generation utilities, internal diagnostics, reviewer-specific scripts and historical development notebooks are not distributed.
