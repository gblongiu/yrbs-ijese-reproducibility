import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
import json

import pandas as pd

from src.config import (
    RAW_FILE_2023,
    PROCESSED_DIR,
    TABLES_DIR,
    LOGS_DIR,
    TARGET_PRIMARY,
    BULLYING_EXPOSURES,
    SECONDARY_TARGETS,
    BASELINE_COVARIATES,
    SURVEY_DESIGN_COLS,
)
from src.data.coding import coerce_categorical, recode_binary_yn, summarize_missingness
from src.data.validate import assert_required_columns


YRBS_YES_VALUES = (1,)
YRBS_NO_VALUES = (2,)
YRBS_MISSING_VALUES = (7, 9, 97, 99)


def _as_int_if_integer_like(x):
    try:
        fx = float(x)
    except Exception:
        return x
    if fx.is_integer():
        return int(fx)
    return x


def _observed_non_null_codes(series: pd.Series) -> list:
    values = [_as_int_if_integer_like(v) for v in series.dropna().unique().tolist()]
    # Deterministic ordering for mixed types: numeric first, then string.
    numeric = sorted([v for v in values if isinstance(v, (int, float))])
    non_numeric = sorted([v for v in values if not isinstance(v, (int, float))], key=lambda v: str(v))
    return numeric + non_numeric


def _sha256_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("||".join(df.columns.astype(str).tolist()).encode("utf-8"))
    h.update("||".join(map(str, df.dtypes.tolist())).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy()
    h.update(row_hashes.tobytes())
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _binary_recoding_decision(series: pd.Series, source_col: str) -> dict:
    observed = _observed_non_null_codes(series)
    observed_set = set(observed)

    allowed_basic = set(YRBS_YES_VALUES + YRBS_NO_VALUES)
    allowed_full = set(YRBS_YES_VALUES + YRBS_NO_VALUES + YRBS_MISSING_VALUES)

    missing_seen = sorted([v for v in observed if v in set(YRBS_MISSING_VALUES)])
    decision = {
        "source_column": source_col,
        "yes_values": list(YRBS_YES_VALUES),
        "no_values": list(YRBS_NO_VALUES),
        "missing_values": list(YRBS_MISSING_VALUES),
        "observed_values": observed,
        "missing_values_seen": missing_seen,
    }

    if observed_set <= allowed_basic:
        decision["rule"] = "observed_values_subset_of_{1,2}: assume 1=Yes, 2=No"
        return decision

    unexpected = sorted([v for v in observed if v not in allowed_full])
    if unexpected:
        raise SystemExit(
            f"Unexpected codes in {source_col}: {unexpected}. "
            f"Expected yes={YRBS_YES_VALUES}, no={YRBS_NO_VALUES}, missing={YRBS_MISSING_VALUES}."
        )

    decision["rule"] = "observed_values_include_special_missing_codes: treat {7,9,97,99} as NA; 1=Yes; 2=No"
    return decision


def build_modeling_table(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = (
        [TARGET_PRIMARY]
        + list(SECONDARY_TARGETS)
        + list(BULLYING_EXPOSURES)
        + list(BASELINE_COVARIATES)
        + list(SURVEY_DESIGN_COLS)
    )
    assert_required_columns(df_raw, required)

    decisions: dict = {
        "input_file": _rel(RAW_FILE_2023),
        "columns": {
            "target_primary": TARGET_PRIMARY,
            "secondary_targets": list(SECONDARY_TARGETS),
            "bullying_exposures": list(BULLYING_EXPOSURES),
            "baseline_covariates": list(BASELINE_COVARIATES),
            "survey_design_cols": list(SURVEY_DESIGN_COLS),
        },
        "binary_recoding": {},
        "row_filters": [],
        "survey_design_handling": {
            "modeling_table": "Keep design fields unchanged; do not impute; do not drop rows for missing design fields.",
            "weighted_descriptives": "Drop rows missing weight/stratum/psu only when producing weighted descriptive tables.",
        },
    }

    # Defensive binary mapping decisions (do not assume directionality without evidence).
    for source_col in [TARGET_PRIMARY] + list(SECONDARY_TARGETS) + list(BULLYING_EXPOSURES):
        decisions["binary_recoding"][source_col] = _binary_recoding_decision(df_raw[source_col], source_col)

    # Recode key dichotomized YRBS variables to explicit binary columns.
    y_qn26 = recode_binary_yn(
        df_raw[TARGET_PRIMARY],
        yes_values=YRBS_YES_VALUES,
        no_values=YRBS_NO_VALUES,
        missing_values=YRBS_MISSING_VALUES,
    )

    # Secondary outcomes are included for appendix-scoped analyses (Week 9).
    # They are not used by the primary modeling pipeline unless explicitly enabled downstream.
    secondary_data = {}
    for source_col in SECONDARY_TARGETS:
        out_col = f"y_{source_col.lower()}"
        secondary_data[out_col] = recode_binary_yn(
            df_raw[source_col],
            yes_values=YRBS_YES_VALUES,
            no_values=YRBS_NO_VALUES,
            missing_values=YRBS_MISSING_VALUES,
        )
    x_qn24 = recode_binary_yn(
        df_raw[BULLYING_EXPOSURES[0]],
        yes_values=YRBS_YES_VALUES,
        no_values=YRBS_NO_VALUES,
        missing_values=YRBS_MISSING_VALUES,
    )
    x_qn25 = recode_binary_yn(
        df_raw[BULLYING_EXPOSURES[1]],
        yes_values=YRBS_YES_VALUES,
        no_values=YRBS_NO_VALUES,
        missing_values=YRBS_MISSING_VALUES,
    )

    modeling = pd.DataFrame({"y_qn26": y_qn26, **secondary_data, "x_qn24": x_qn24, "x_qn25": x_qn25})

    # Baseline covariates: treat as categorical unless ordinality can be justified from the file alone.
    for cov in BASELINE_COVARIATES:
        modeling[cov] = coerce_categorical(df_raw[cov], numeric_to_string=True)

    # Survey design fields: preserve unchanged.
    for col in SURVEY_DESIGN_COLS:
        modeling[col] = df_raw[col]

    # Apply missingness plan: drop rows with missing outcome.
    n_before = len(modeling)
    modeling = modeling.loc[modeling["y_qn26"].notna()].reset_index(drop=True)
    n_after = len(modeling)
    decisions["row_filters"].append(
        {
            "rule": "drop_missing_outcome",
            "column": "y_qn26",
            "dropped_rows": n_before - n_after,
        }
    )

    # Deterministic column order.
    ordered_cols = (
        ["y_qn26"]
        + list(secondary_data.keys())
        + ["x_qn24", "x_qn25"]
        + list(BASELINE_COVARIATES)
        + list(SURVEY_DESIGN_COLS)
    )
    modeling = modeling[ordered_cols]

    return modeling, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis-ready modeling table from YRBS 2023 subset.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional: read only the first N rows (for tests).")
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=PROCESSED_DIR / "yrbs_2023_modeling.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=TABLES_DIR / "modeling_table_audit.csv",
        help="Output audit CSV path.",
    )
    parser.add_argument(
        "--missingness-csv",
        type=Path,
        default=TABLES_DIR / "missingness_modeling.csv",
        help="Output missingness summary CSV path.",
    )
    parser.add_argument(
        "--decisions-json",
        type=Path,
        default=LOGS_DIR / "decisions.json",
        help="Output JSON file for coding/filter decisions.",
    )
    args = parser.parse_args()

    if not RAW_FILE_2023.exists():
        raise SystemExit(f"Input file not found: {RAW_FILE_2023}")

    if not BASELINE_COVARIATES:
        raise SystemExit("BASELINE_COVARIATES is empty. Update src/config.py before building the dataset.")

    if list(BULLYING_EXPOSURES) != ["QN24", "QN25"]:
        raise SystemExit(f"Expected BULLYING_EXPOSURES=['QN24','QN25'] but got: {BULLYING_EXPOSURES}")

    df_raw = pd.read_excel(RAW_FILE_2023, nrows=args.nrows)
    raw_rows = len(df_raw)

    modeling, decisions = build_modeling_table(df_raw)
    modeling_rows, modeling_cols = modeling.shape

    # Missingness summary
    args.missingness_csv.parent.mkdir(parents=True, exist_ok=True)
    miss = summarize_missingness(modeling)
    miss.to_csv(args.missingness_csv, index=False)

    # Class balance (among non-missing outcome rows; outcome missing rows were dropped)
    y_counts = modeling["y_qn26"].value_counts(dropna=False).to_dict()
    n_1 = int(y_counts.get(1, 0))
    n_0 = int(y_counts.get(0, 0))
    y_rate = round(n_1 / (n_0 + n_1), 6) if (n_0 + n_1) else None

    content_hash = _sha256_df(modeling)

    # Write decisions JSON (deterministic ordering).
    args.decisions_json.parent.mkdir(parents=True, exist_ok=True)
    decisions_payload = {
        **decisions,
        "raw_rows": raw_rows,
        "modeling_rows": modeling_rows,
        "modeling_cols": modeling_cols,
        "output_parquet": _rel(args.out_parquet),
        "missingness_csv": _rel(args.missingness_csv),
        "content_hash_sha256": content_hash,
    }
    args.decisions_json.write_text(json.dumps(decisions_payload, indent=2, sort_keys=True), encoding="utf-8")

    # Write modeling table
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    modeling.to_parquet(args.out_parquet, index=False)

    # Write compact audit CSV
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(
        [
            {
                "raw_rows": raw_rows,
                "modeling_rows": modeling_rows,
                "modeling_cols": modeling_cols,
                "y_qn26_n": n_0 + n_1,
                "y_qn26_n1": n_1,
                "y_qn26_n0": n_0,
                "y_qn26_rate": y_rate,
                "missingness_csv": _rel(args.missingness_csv),
                "content_hash_sha256": content_hash,
            }
        ]
    )
    audit.to_csv(args.audit_csv, index=False)

    print(f"Wrote {args.out_parquet}")
    print(f"Wrote {args.audit_csv}")
    print(f"Wrote {args.missingness_csv}")
    print(f"Wrote {args.decisions_json}")


if __name__ == "__main__":
    main()
