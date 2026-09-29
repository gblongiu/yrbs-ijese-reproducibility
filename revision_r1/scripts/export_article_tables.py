"""Create reader-facing CSV exports corresponding to article Tables 1–6."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "revision_r1/outputs"
OUT = R1 / "article_tables"
OUT.mkdir(parents=True, exist_ok=True)


def table_1() -> None:
    rows = [
        ("Development data", "2023 National YRBS", "Model development and internal evaluation"),
        ("Outcome", "Persistent sadness or hopelessness", "Binary target"),
        ("Baseline", "Age, sex, grade, race or ethnicity", "Demographic comparator"),
        ("Added feature group", "School-property and electronic bullying", "Researcher-specified comparison"),
        ("2023 split", "80/20, seed 2026", "Fixed respondent-level training and internal evaluation partitions"),
        ("Selection", "Fixed training-side cross-validation", "Brier score, calibration slope, then ROC AUC"),
        ("External validation", "2021 National YRBS", "Frozen-pipeline reverse-time transportability"),
        ("Claim boundary", "Predictive, non-causal, research-only", "Limits interpretation"),
    ]
    pd.DataFrame(rows, columns=["component", "specification", "analytic_role"]).to_csv(OUT / "Table_1_study_design.csv", index=False)


def table_2() -> None:
    matched = pd.read_csv(R1 / "metrics/matched_hgb_2023_metrics_seed2026.csv")
    specs = [
        ("Logistic", "Baseline", "none", "metrics_test_seed2026_logreg_baseline_none.csv"),
        ("HGB", "Baseline", "none", "metrics_test_seed2026_hgb_baseline_none.csv"),
        ("HGB", "Baseline", "Platt", None),
        ("Logistic", "Expanded", "none", "metrics_test_seed2026_logreg_full_none.csv"),
        ("HGB", "Expanded", "none", "metrics_test_seed2026_hgb_full_none.csv"),
        ("HGB", "Expanded", "Platt", "metrics_test_seed2026_hgb_full_platt.csv"),
        ("HGB", "Expanded", "Isotonic", "metrics_test_seed2026_hgb_full_isotonic.csv"),
    ]
    rows = []
    for model, feature_set, calibration, filename in specs:
        if filename is None:
            row = matched.query("feature_set == 'baseline' and calibration == 'platt'").iloc[0]
            values = {"roc_auc": row.roc_auc, "pr_auc": row.average_precision, "brier": row.brier, "calibration_slope": row.calibration_slope, "calibration_intercept": row.calibration_intercept}
        else:
            row = pd.read_csv(ROOT / "outputs/metrics" / filename).iloc[0]
            values = {k: row[k] for k in ["roc_auc", "pr_auc", "brier", "calibration_slope", "calibration_intercept"]}
        rows.append({"model": model, "feature_set": feature_set, "calibration": calibration, **values})
    pd.DataFrame(rows).rename(columns={"pr_auc": "average_precision"}).to_csv(OUT / "Table_2_internal_model_comparison.csv", index=False)


def table_3() -> None:
    source = pd.read_csv(R1 / "tables/paired_improvement_ci_2023_seed20260915.csv")
    source[["comparison", "metric", "baseline", "expanded", "point_difference", "ci95_low", "ci95_high"]].to_csv(OUT / "Table_3_paired_feature_set_improvements.csv", index=False)


def table_4() -> None:
    table2 = pd.read_csv(OUT / "Table_2_internal_model_comparison.csv")
    table2.query("model == 'HGB' and feature_set == 'Expanded'").to_csv(OUT / "Table_4_calibration_sensitivity.csv", index=False)


def table_5() -> None:
    labels = {
        "cat_1": "American Indian/Alaskan Native",
        "cat_2": "Asian",
        "cat_3": "Black or African American",
        "cat_4": "Native Hawaiian or Other Pacific Islander",
        "cat_5": "White",
        "cat_6": "Hispanic/Latino",
        "cat_7": "Multiple–Hispanic/Latino",
        "cat_8": "Multiple–Non-Hispanic/Latino",
    }
    source = pd.read_csv(R1 / "tables/subgroup_calibration_diagnostics_2023_seed2026.csv")
    source = source.query("subgroup_variable == 'raceeth'").copy()
    labels["<missing>"] = "Missing"
    source["subgroup"] = source["subgroup_value"].map(labels)
    source["adequacy_status"] = np.where(source["calibration_adequate"], "Eligible", "Insufficient events or non-events")
    source[["subgroup", "n", "events", "non_events", "calibration_slope", "calibration_intercept", "adequacy_status"]].to_csv(OUT / "Table_5_race_ethnicity_calibration.csv", index=False)


def table_6() -> None:
    internal = pd.read_csv(R1 / "metrics/matched_hgb_2023_metrics_seed2026.csv").query("calibration == 'platt'").copy()
    internal.insert(0, "wave_evaluation", "2023 internal")
    external = pd.read_csv(R1 / "external/yrbs2021_external_metrics.csv").copy()
    external.insert(0, "wave_evaluation", "2021 cross-wave")
    columns = ["wave_evaluation", "feature_set", "roc_auc", "average_precision", "brier", "calibration_slope", "calibration_intercept"]
    pd.concat([internal[columns], external[columns]], ignore_index=True).to_csv(OUT / "Table_6_matched_cross_wave_performance.csv", index=False)


if __name__ == "__main__":
    table_1()
    table_2()
    table_3()
    table_4()
    table_5()
    table_6()
    print(f"Article table exports written to {OUT}")
