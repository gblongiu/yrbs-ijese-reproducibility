# Output guide

`expected/` contains the publication reference summaries. A complete run writes the same named summaries and its respondent-level intermediate files beneath the private work directory.

The workflow also writes six reader-facing CSVs to `revision_r1/outputs/article_tables/` in the private run:

| Article table | Generated CSV | Computational sources |
|---|---|---|
| Table 1, study design | `Table_1_study_design.csv` | Fixed split, feature definitions, calibration boundary, and sample counts encoded in the workflow |
| Table 2, internal model comparison | `Table_2_internal_model_comparison.csv` | Six regenerated held-out metric files under `outputs/metrics/` |
| Table 3, paired feature-set improvements | `Table_3_paired_feature_set_improvements.csv` | `paired_improvement_ci_2023_seed20260915.csv` |
| Table 4, expanded-HGB calibration sensitivity | `Table_4_calibration_sensitivity.csv` | Expanded HGB rows from Table 2 |
| Table 5, racial/ethnic calibration diagnostics | `Table_5_race_ethnicity_calibration.csv` | `subgroup_calibration_diagnostics_2023_seed2026.csv` with CDC category labels and adequacy suppression |
| Table 6, matched cross-wave performance | `Table_6_matched_cross_wave_performance.csv` | `matched_hgb_2023_metrics_seed2026.csv` and `yrbs2021_external_metrics.csv` |

These exports retain full computational precision. The article presents most values to three decimal places. The legacy field `pr_auc` is renamed `average_precision` in the reader-facing exports.

The public fitted artifacts are supplied for exact reuse with the pinned software environment. Reconstruct them by rerunning the workflow when using a different environment. Do not interpret their probabilities as clinical scores or individual assessments.
