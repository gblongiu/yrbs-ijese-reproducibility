# YRBS article reproducibility materials

This repository reproduces the analyses for **Calibration-Aware Evaluation of Machine Learning Models for Survey-Based Prediction** by Gabriel Long and Sridhar Ramachandran.

The analysis compares a demographic histogram gradient boosting model with an expanded model that adds two bullying exposure indicators. It includes the 2023 internal evaluation, calibration sensitivity, respondent-weight and public-design sensitivities, subgroup diagnostics, and cross-wave evaluation in the 2021 National Youth Risk Behavior Survey (YRBS). The results describe predictive associations for research use. They do not establish causality, clinical validity, fairness, or deployment readiness.

## Data

Download the 2021 and 2023 National YRBS public-use ASCII files from the [CDC YRBS data page](https://www.cdc.gov/yrbs/data/index.html):

- `XXH2021_YRBS_Data.dat`
- `XXH2023_YRBS_Data.dat`

Raw and reconstructed respondent-level data are not redistributed. The workflow verifies the expected source-file hashes before analysis.

## Environment

The verified Windows run used Python 3.13.12 and the package versions in `requirements.txt`.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux or macOS, create a Python 3.13 environment and use its equivalent Python executable.

## Complete reproduction

Run the workflow from the repository root. The work directory must be outside this repository and absent or empty. It will contain reconstructed respondent-level data and all generated outputs, so it must remain private.

```powershell
.\.venv\Scripts\python.exe run_workflow.py `
  --data2023 'data\raw\XXH2023_YRBS_Data.dat' `
  --data2021 'data\raw\XXH2021_YRBS_Data.dat' `
  --work-dir '..\private-reproduction-run'
```

The execution order is source verification and reconstruction, fixed-split candidate regeneration, 2023 analysis, 2021 cross-wave evaluation, public-design sensitivity, evidence tables, and article figures. Seed 2026 controls splitting and model fitting. Inner calibration folds use seeds 2526 through 2530. Paired internal bootstrap intervals use base seed 20260915. Subgroup bootstrap intervals use base seed 20260916 plus deterministic variable/category offsets. External model intervals use seeds 20260917 for the baseline and 20260918 for the expanded model; paired external deltas use seed 20261017. Public-design sensitivity uses seed 20260918.

Generated results appear under `<work-dir>/revision_r1/outputs/`. Key outputs include:

- `metrics/matched_hgb_2023_metrics_seed2026.csv`: matched baseline and expanded HGB metrics;
- `tables/training_side_candidate_selection_seed2026.csv`: retrospective training-side candidate comparison;
- `tables/paired_improvement_ci_2023_seed20260915.csv`: paired internal improvement intervals;
- `tables/survey_weight_sensitivity_2023_seed2026.csv`: respondent-weight sensitivity;
- `tables/public_design_paired_sensitivity.csv`: conditional PSU resampling sensitivity;
- `tables/subgroup_calibration_diagnostics_2023_seed2026.csv`: subgroup counts, eligibility, metrics, and intervals;
- `external/yrbs2021_external_metrics.csv`: 2021 cross-wave performance;
- `figures/manuscript/`: the three article figures.
- `article_tables/`: deterministic, reader-facing exports corresponding to article Tables 1–6.

The `expected/` directory contains aggregate reference results for comparison. It contains no respondent-level records or bootstrap draws. The `models/` directory contains the fitted 2023 baseline and expanded Platt-calibrated pipelines used for the two evaluation waves. Their feature lists and seed are stored in each artifact; `expected/tables/frozen_model_identity.csv` records the artifact digests and prediction-identity check.

Legacy CSV fields named `pr_auc` contain average precision computed with `sklearn.metrics.average_precision_score`; they are not trapezoidal area under an interpolated precision-recall curve.

Run the lightweight repository checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Methodological notes

The 2023 evaluation set was consulted during the earlier model-selection history. The reported training-side selection is therefore a retrospective reanalysis, and 2021 provides independent cross-wave confirmation. HGB settings were selected on the demographic baseline and held fixed for the expanded comparison. Respondent-weighted analyses do not constitute full design-based inference. The public-design analysis is a conditional PSU resampling sensitivity because the public files do not provide all stages and selection probabilities of the CDC sample design.

This article develops work originating in Gabriel Long's 2026 undergraduate senior thesis, *Incremental Predictive Value of Bullying Exposure Indicators for Predicting Persistent Sadness or Hopelessness in the 2023 National Youth Risk Behavior Survey*. Citation details for the article and repository should be added after publication metadata are assigned.
