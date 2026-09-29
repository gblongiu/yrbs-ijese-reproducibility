from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_expected_sample_counts_and_selected_candidate() -> None:
    counts = pd.read_csv(ROOT / "expected/tables/sample_counts.csv")
    assert set(map(tuple, counts[["wave", "scope", "n"]].to_records(index=False))) == {
        (2023, "analytic", 19863),
        (2023, "training", 15890),
        (2023, "internal", 3973),
        (2021, "external", 16961),
    }
    selected = pd.read_csv(ROOT / "expected/tables/training_side_candidate_selection_seed2026.csv").iloc[0]
    assert selected["model"] == "hgb"
    assert selected["featureset"] == "full"
    assert selected["calibration_method"] == "platt"


def test_distributed_models_have_expected_public_schema() -> None:
    for name, features in {
        "baseline": ["q1", "q2", "q3", "raceeth"],
        "expanded": ["x_qn24", "x_qn25", "q1", "q2", "q3", "raceeth"],
    }.items():
        payload = joblib.load(ROOT / f"models/{name}_platt_2023_frozen.joblib")
        assert set(payload) == {"estimator", "calibrator", "features", "seed"}
        assert payload["features"] == features
        assert payload["seed"] == 2026


def test_article_table_exports_match_manuscript_structure() -> None:
    table1 = pd.read_csv(ROOT / "expected/article_tables/Table_1_study_design.csv")
    assert list(table1.columns) == ["component", "specification", "analytic_role"]
    assert table1["component"].tolist() == [
        "Development data", "Outcome", "Baseline", "Added feature group",
        "2023 split", "Selection", "External validation", "Claim boundary",
    ]

    table2 = pd.read_csv(ROOT / "expected/article_tables/Table_2_internal_model_comparison.csv")
    assert list(zip(table2.model, table2.feature_set, table2.calibration)) == [
        ("Logistic", "Baseline", "none"),
        ("HGB", "Baseline", "none"),
        ("HGB", "Baseline", "Platt"),
        ("Logistic", "Expanded", "none"),
        ("HGB", "Expanded", "none"),
        ("HGB", "Expanded", "Platt"),
        ("HGB", "Expanded", "Isotonic"),
    ]

    table5 = pd.read_csv(ROOT / "expected/article_tables/Table_5_race_ethnicity_calibration.csv")
    assert len(table5) == 9
    missing = table5.loc[table5.subgroup == "Missing"].iloc[0]
    assert (missing["n"], missing["events"], missing["non_events"]) == (71, 23, 48)
    assert pd.isna(missing["calibration_slope"])
