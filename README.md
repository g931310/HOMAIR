# HOMAIR

**Current release:** v1.0.0
**Python:** 3.11+
**License:** Apache-2.0

HOMAIR is an open-source reproducibility framework for explainable machine-learning prediction of insulin resistance using the Homeostatic Model Assessment of Insulin Resistance (HOMA-IR).

The repository provides a reusable research workflow for model development, internal validation, operating-threshold selection, statistical comparison, and model interpretation in clinical machine-learning studies.

The public repository is designed to support methodological transparency and reproducibility without distributing participant-level health data or proprietary database mappings.

---

## Overview

Insulin resistance is a major metabolic abnormality associated with cardiometabolic disease and may be present before overt diabetes develops.

HOMAIR provides a reproducible machine-learning framework for investigating HOMA-IR as both:

* a continuous regression outcome; and
* a binary classification outcome based on a prespecified HOMA-IR threshold.

The framework emphasizes rigorous internal validation, transparent operating-point selection, and interpretable prediction rather than model discrimination alone.

The current public release focuses on the reusable methodological components of the research workflow.

---

## Key Features

HOMAIR includes reusable components for:

* clinical predictor preprocessing
* continuous HOMA-IR prediction
* binary insulin-resistance classification
* CatBoost-based machine learning
* conventional statistical and machine-learning comparators
* out-of-fold prediction
* cross-validation
* nested validation
* operating-threshold optimization
* nested threshold validation
* bootstrap uncertainty estimation
* paired model comparison
* AUROC comparison
* calibration assessment
* SHAP-based model interpretation
* synthetic-data demonstration
* reproducible Python environment specification

---

## Validation Strategy

The framework separates model evaluation from operating-threshold selection.

Model discrimination is evaluated using out-of-fold or nested cross-validation predictions whenever appropriate.

For binary classification, operating thresholds are selected within training data and evaluated on held-out validation data to reduce optimistic bias.

The public workflow therefore distinguishes between:

1. model development,
2. hyperparameter selection,
3. probability prediction,
4. operating-threshold selection, and
5. unbiased internal validation.

This separation is particularly important when sensitivity, specificity, positive predictive value, negative predictive value, or other threshold-dependent measures are reported.

---

## Model Interpretation

HOMAIR supports SHAP-based interpretation for compatible tree-based models.

Interpretation utilities may be used to examine:

* global feature importance
* individual-level model explanations
* feature contribution distributions
* directionality of predictor effects

SHAP results should be interpreted as explanations of model predictions rather than estimates of causal effects.

---

## Repository Scope

### Included

The public repository contains methodological components intended to support reproducibility, including:

* reusable analysis code
* model-development utilities
* validation utilities
* threshold-selection procedures
* statistical comparison procedures
* SHAP interpretation utilities
* environment specifications
* synthetic or example data
* documentation describing the analysis workflow

### Not Included

The following materials are intentionally not distributed:

* participant-level MJ Health Database data
* personally identifiable information
* proprietary database mappings
* participant identifiers
* confidential data-cleaning procedures
* restricted source-database variables
* internal administrative files
* unpublished proprietary research workflows
* private reviewer-response materials
* deployment credentials or access keys

Access to the original health examination database is governed independently by the relevant data provider and institutional authorization procedures.

---

## Synthetic Example Data

The repository includes synthetic or example data solely for demonstrating:

* expected input structure
* variable formatting
* preprocessing interfaces
* model-training workflows
* validation procedures
* output generation

Synthetic data are not derived representations of identifiable participants and must not be interpreted as reproducing the statistical distribution or clinical characteristics of the original research cohort.

Results obtained from synthetic data are provided for software demonstration only and are not clinical study results.

---

## Installation

A dedicated Python environment is recommended.

### Using `requirements.txt`

```bash
python -m venv homair_env
```

Activate the environment.

Windows:

```bash
homair_env\Scripts\activate
```

macOS/Linux:

```bash
source homair_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Using Conda

If an `environment.yml` file is provided:

```bash
conda env create -f environment.yml
conda activate homair
```

Python 3.11 or a compatible environment is recommended for reproducing the current public release.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/g931310/HOMAIR.git
cd HOMAIR
```

Create and activate the environment, then install the required packages.

Example:

```bash
pip install -r requirements.txt
```

Users should first run the synthetic/example-data workflow to confirm that the local environment is configured correctly before adapting the pipeline to an authorized research dataset.

---

## Recommended Reproducibility Workflow

A typical workflow is:

```text
Data preparation
      ↓
Predictor preprocessing
      ↓
Cross-validation split
      ↓
Model fitting
      ↓
Out-of-fold prediction
      ↓
Performance estimation
      ↓
Nested threshold selection
      ↓
Threshold-dependent evaluation
      ↓
Statistical comparison
      ↓
SHAP interpretation
      ↓
Final model development
```

For research use, preprocessing and feature engineering should always be performed within the appropriate training folds whenever information leakage could otherwise occur.

---

## Reproducibility Principles

The public HOMAIR framework follows several methodological principles.

### Separation of training and evaluation

Performance should be assessed on observations not used to fit the corresponding model.

### Threshold selection within training data

Classification thresholds should not be optimized directly on the final evaluation sample.

### Out-of-fold prediction

Cross-validated out-of-fold predictions are preferred for unbiased comparison of candidate models during internal validation.

### Reproducible randomness

Random seeds should be explicitly controlled where supported by the corresponding algorithms.

### Transparent reporting

Model settings, predictor definitions, validation procedures, threshold-selection rules, and uncertainty estimates should be documented sufficiently for independent methodological reproduction.

---

## Statistical Evaluation

Depending on the analysis, performance measures may include:

### Regression

* root mean squared error (RMSE)
* mean absolute error (MAE)
* coefficient of determination (R²)

### Classification

* area under the receiver operating characteristic curve (AUROC)
* area under the precision-recall curve (AUPRC)
* sensitivity
* specificity
* positive predictive value
* negative predictive value
* accuracy
* F1 score
* Brier score

Confidence intervals and paired statistical comparisons should be used when appropriate.

---

## Clinical Use

This repository is intended for research and methodological reproducibility.

The code, models, example outputs, and synthetic data are **not intended for independent clinical diagnosis, treatment decisions, or direct patient management**.

Any future clinical implementation would require additional validation, calibration, governance, regulatory assessment, and evaluation in the intended target population and clinical environment.

---

## Data Availability

Participant-level data used in the associated research are not distributed through this repository.

The original study data originate from an authorized secondary health database and are subject to data-use, privacy, ethical, and institutional restrictions.

Researchers interested in accessing the underlying database should apply through the appropriate data-provider authorization process.

The public synthetic/example dataset is provided only to demonstrate the software interface and analysis workflow.

---

## Reproducibility and the Associated Manuscript

The repository is intended to document the reusable computational methodology underlying the associated HOMA-IR machine-learning study.

Because the original participant-level dataset cannot be publicly redistributed, exact numerical reproduction of the manuscript results requires authorized access to the corresponding analysis-ready research dataset.

The synthetic workflow is instead intended to verify:

* software execution
* pipeline structure
* model interfaces
* validation logic
* expected output formats

The repository may be updated as the associated manuscript undergoes peer review and methodological refinement.

---

## Versioning

HOMAIR follows semantic-style versioning for major public releases.

### v1.0.0

Initial public reproducibility release.

Core components include:

* regression and classification workflows
* out-of-fold validation
* nested operating-threshold validation
* statistical model comparison
* bootstrap uncertainty estimation
* SHAP interpretation
* synthetic-data demonstration
* reproducible environment specifications

See the GitHub Releases page for release-specific information.

---

## Contributing

Contributions that improve the public reproducibility framework are welcome.

Examples include:

* documentation improvements
* bug reports
* reproducibility fixes
* validation utilities
* statistical evaluation utilities
* testing improvements
* synthetic example workflows

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

Issues may also be used to report bugs, suggest enhancements, or discuss future methodological extensions.

---

## Roadmap

Planned development may include:

* automated reproducibility testing
* continuous-integration checks
* additional nested-validation utilities
* expanded threshold-stability analyses
* improved statistical comparison tools
* additional worked examples
* expanded documentation
* broader environment compatibility testing

See [ROADMAP.md](ROADMAP.md) for additional details.

---

## Citation

If you use HOMAIR in academic work, please cite the associated manuscript after publication.

Citation information will be updated when the final bibliographic details become available.

Until then, users may cite the software repository and the corresponding GitHub release.

---

## License

This repository is licensed under the **Apache License 2.0**.

The license applies to the source code and synthetic/example materials distributed in this repository.

It does not provide access to, ownership of, or rights over:

* the MJ Health Database
* participant-level health information
* restricted source data
* proprietary database mappings
* materials not distributed as part of this repository

See [LICENSE](LICENSE) for the complete license terms.

---

## Disclaimer

HOMAIR is provided for research, education, and methodological reproducibility.

The software is provided without warranty of clinical performance, fitness for a particular clinical purpose, or suitability for direct medical decision-making.
