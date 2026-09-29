from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

R1_SCRIPT = runpy.run_path(str(PROJECT_ROOT / "revision_r1" / "scripts" / "run_2023_revision_analyses.py"))
EXTERNAL_FILE = PROJECT_ROOT / "revision_r1" / "data" / "external" / "XXH2021_YRBS_Data.dat"
OUT_ROOT = PROJECT_ROOT / "revision_r1" / "outputs" / "external"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260917


def _field(line: str, start: int, end: int) -> str:
    return line[start - 1 : end].strip()


def _as_int(value: str) -> float:
    if value == "":
        return np.nan
    try:
        return int(value)
    except ValueError:
        return np.nan


def _as_float(value: str) -> float:
    if value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _cat(value: float) -> object:
    if pd.isna(value):
        return np.nan
    return f"cat_{int(value)}"


def load_2021_ascii(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            "Missing 2021 National YRBS ASCII file. Place the CDC file "
            "XXH2021_YRBS_Data.dat in revision_r1/data/external/ and rerun."
        )

    rows = []
    with path.open("r", encoding="latin-1", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if len(line) < 415:
                raise RuntimeError(f"Line {line_number} is shorter than the 415-column 2021 codebook layout.")
            rows.append(
                {
                    "source_row": line_number,
                    "q1_raw": _as_int(_field(line, 17, 17)),
                    "q2_raw": _as_int(_field(line, 18, 18)),
                    "q3_raw": _as_int(_field(line, 19, 19)),
                    "qn23_raw": _as_int(_field(line, 200, 200)),
                    "qn24_raw": _as_int(_field(line, 201, 201)),
                    "qn25_raw": _as_int(_field(line, 202, 202)),
                    "weight": _as_float(_field(line, 390, 399)),
                    "stratum": _as_int(_field(line, 400, 402)),
                    "psu": _as_int(_field(line, 403, 408)),
                    "raceeth_raw": _as_int(_field(line, 414, 415)),
                }
            )
    raw = pd.DataFrame(rows)
    if len(raw) != 17232:
        raise RuntimeError(f"Expected 17,232 2021 national records; observed {len(raw):,}.")
    return raw


def yn_to_binary(series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out.loc[series == 1] = 1
    out.loc[series == 2] = 0
    return out


def build_external_table(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "source_row": raw["source_row"],
            # 2021 QN25 is the item equivalent of the 2023 QN26 outcome.
            "y_qn26": yn_to_binary(raw["qn25_raw"]),
            # 2021 QN23/QN24 map to the 2023 QN24/QN25 bullying inputs.
            "x_qn24": yn_to_binary(raw["qn23_raw"]),
            "x_qn25": yn_to_binary(raw["qn24_raw"]),
            "q1": raw["q1_raw"].map(_cat),
            "q2": raw["q2_raw"].map(_cat),
            "q3": raw["q3_raw"].map(_cat),
            "raceeth": raw["raceeth_raw"].map(_cat),
            "weight": raw["weight"],
            "stratum": raw["stratum"],
            "psu": raw["psu"],
        }
    )
    df = df.loc[df["y_qn26"].notna()].reset_index(drop=True)
    if len(df) != 16961:
        raise RuntimeError(f"Expected 16,961 records with usable 2021 QN25 outcome; observed {len(df):,}.")
    return df


def fit_2023_frozen_model(feature_set: str):
    build_ns, week06_ns = R1_SCRIPT["load_project_modules"]()
    df2023 = R1_SCRIPT["load_modeling_table"](build_ns)
    train_idx, _, fold_id = R1_SCRIPT["load_frozen_indices"](df2023, week06_ns)
    params = R1_SCRIPT["load_hgb_params"]()
    fs = week06_ns["derive_feature_sets"](df2023.columns)
    features = fs.baseline_features if feature_set == "baseline" else fs.full_features
    x2023 = week06_ns["prepare_feature_frame"](df2023, features, ["q1", "q2", "q3", "raceeth"])
    x_train = x2023.loc[train_idx]
    y_train = df2023.loc[train_idx, "y_qn26"].astype(int)
    factory = R1_SCRIPT["make_hgb_factory"](week06_ns, features, params)

    estimator = factory()
    R1_SCRIPT["fit_pipeline"](estimator, x_train, y_train, None)
    oof = R1_SCRIPT["oof_predictions"](factory, x_train, y_train, fold_id, None)
    calibrator = R1_SCRIPT["fit_platt"](oof, y_train.to_numpy(dtype=int), None)
    return estimator, calibrator, features, week06_ns


def external_predictions(df2021: pd.DataFrame, feature_set: str) -> np.ndarray:
    import joblib
    payload = joblib.load(PROJECT_ROOT / 'revision_r1' / 'outputs' / 'models' / f'{feature_set}_platt_2023_frozen.joblib')
    estimator, calibrator, features = payload['estimator'], payload['calibrator'], payload['features']
    _, week06_ns = R1_SCRIPT['load_project_modules']()
    x = week06_ns["prepare_feature_frame"](df2021, features, ["q1", "q2", "q3", "raceeth"])
    raw_prob = np.clip(estimator.predict_proba(x)[:, 1], R1_SCRIPT["EPS"], 1.0 - R1_SCRIPT["EPS"])
    return R1_SCRIPT["apply_platt"](calibrator, raw_prob)


def _score_group_index(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, int]:
    """Map observations to descending distinct-score groups once for bootstrap reuse."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(p)])
    sorted_group = np.repeat(np.arange(len(starts), dtype=int), lengths)
    group = np.empty(len(p), dtype=int)
    group[order] = sorted_group
    return group, len(starts)


def _metrics_from_bootstrap_counts(
    y: np.ndarray,
    p: np.ndarray,
    counts: np.ndarray,
    group_info: tuple[np.ndarray, int],
) -> tuple[float, float, float]:
    """Exact ROC AUC, average precision, and Brier for an integer-weight bootstrap sample."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    counts = np.asarray(counts, dtype=float)
    group, n_groups = group_info

    pos_weight = np.bincount(group, weights=counts * y, minlength=n_groups)
    neg_weight = np.bincount(group, weights=counts * (1 - y), minlength=n_groups)
    tp = np.cumsum(pos_weight)
    fp = np.cumsum(neg_weight)
    total_pos = tp[-1]
    total_neg = fp[-1]

    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    recall = tp / total_pos
    previous_recall = np.r_[0.0, recall[:-1]]
    average_precision = float(np.sum((recall - previous_recall) * precision))

    negative_below = total_neg - fp
    roc_auc = float(
        np.sum(pos_weight * (negative_below + 0.5 * neg_weight)) / (total_pos * total_neg)
    )
    brier = float(np.sum(counts * (p - y) ** 2) / np.sum(counts))
    return roc_auc, average_precision, brier


def _stratified_bootstrap_counts(
    rng: np.random.Generator,
    pos: np.ndarray,
    neg: np.ndarray,
    n: int,
) -> np.ndarray:
    idx = np.concatenate(
        [
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ]
    )
    return np.bincount(idx, minlength=n)


def metric_bootstrap_ci(y: np.ndarray, p: np.ndarray, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    group_info = _score_group_index(y, p)
    draws = {
        "roc_auc": np.empty(BOOTSTRAP_REPS, dtype=float),
        "average_precision": np.empty(BOOTSTRAP_REPS, dtype=float),
        "brier": np.empty(BOOTSTRAP_REPS, dtype=float),
    }
    for i in range(BOOTSTRAP_REPS):
        counts = _stratified_bootstrap_counts(rng, pos, neg, len(y))
        roc_auc, average_precision, brier = _metrics_from_bootstrap_counts(
            y, p, counts, group_info
        )
        draws["roc_auc"][i] = roc_auc
        draws["average_precision"][i] = average_precision
        draws["brier"][i] = brier
    return {
        key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for key, values in draws.items()
    }


def paired_delta_ci(y: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 100)
    y = np.asarray(y, dtype=int)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    group0 = _score_group_index(y, p0)
    group1 = _score_group_index(y, p1)
    draws = {
        "roc_auc_change": np.empty(BOOTSTRAP_REPS, dtype=float),
        "average_precision_change": np.empty(BOOTSTRAP_REPS, dtype=float),
        "brier_improvement": np.empty(BOOTSTRAP_REPS, dtype=float),
    }
    for i in range(BOOTSTRAP_REPS):
        counts = _stratified_bootstrap_counts(rng, pos, neg, len(y))
        roc0, ap0, brier0 = _metrics_from_bootstrap_counts(y, p0, counts, group0)
        roc1, ap1, brier1 = _metrics_from_bootstrap_counts(y, p1, counts, group1)
        draws["roc_auc_change"][i] = roc1 - roc0
        draws["average_precision_change"][i] = ap1 - ap0
        draws["brier_improvement"][i] = brier0 - brier1

    point = {
        "roc_auc_change": roc_auc_score(y, p1) - roc_auc_score(y, p0),
        "average_precision_change": average_precision_score(y, p1)
        - average_precision_score(y, p0),
        "brier_improvement": brier_score_loss(y, p0) - brier_score_loss(y, p1),
    }
    rows = []
    for metric, value in point.items():
        values = draws[metric]
        rows.append(
            {
                "metric": metric,
                "point_difference": float(value),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED + 100,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_2021_ascii(EXTERNAL_FILE)
    df = build_external_table(raw)
    y = df["y_qn26"].astype(int).to_numpy()
    weight = df["weight"].to_numpy(dtype=float)

    p_baseline = external_predictions(df, "baseline")
    p_expanded = external_predictions(df, "expanded")

    prediction_table = df.copy()
    prediction_table["baseline_platt_2023_frozen"] = p_baseline
    prediction_table["expanded_platt_2023_frozen"] = p_expanded
    prediction_table.to_csv(OUT_ROOT / "yrbs2021_external_predictions.csv", index=False)

    prevalence_ci = proportion_confint(int(y.sum()), len(y), alpha=0.05, method="wilson")
    prevalence = pd.DataFrame(
        [
            {
                "n": len(y),
                "events": int(y.sum()),
                "prevalence": float(y.mean()),
                "prevalence_ci95_low": float(prevalence_ci[0]),
                "prevalence_ci95_high": float(prevalence_ci[1]),
            }
        ]
    )
    prevalence.to_csv(OUT_ROOT / "yrbs2021_external_outcome_prevalence.csv", index=False)

    rows = []
    for i, (label, p) in enumerate([("baseline", p_baseline), ("expanded", p_expanded)]):
        point = R1_SCRIPT["metric_row"](y, p)
        ci = metric_bootstrap_ci(y, p, BOOTSTRAP_SEED + i)
        cal_ci = R1_SCRIPT["calibration_wald_ci"](y, p)
        row = {"feature_set": label, "calibration": "platt_2023_frozen", **point}
        for metric, bounds in {**ci, **cal_ci}.items():
            row[f"{metric}_ci95_low"] = bounds[0]
            row[f"{metric}_ci95_high"] = bounds[1]
        row["bootstrap_replicates"] = BOOTSTRAP_REPS
        row["bootstrap_seed"] = BOOTSTRAP_SEED + i
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_ROOT / "yrbs2021_external_metrics.csv", index=False)

    weighted_rows = []
    for label, p in [("baseline", p_baseline), ("expanded", p_expanded)]:
        weighted_rows.append(
            {
                "feature_set": label,
                "evaluation": "cdc_weighted",
                **R1_SCRIPT["metric_row"](y, p, weight),
            }
        )
    pd.DataFrame(weighted_rows).to_csv(OUT_ROOT / "yrbs2021_external_metrics_cdc_weighted.csv", index=False)

    paired_delta_ci(y, p_baseline, p_expanded).to_csv(
        OUT_ROOT / "yrbs2021_external_incremental_value_ci.csv", index=False
    )

    print("2021 cross-wave external validation completed.")


if __name__ == "__main__":
    main()
