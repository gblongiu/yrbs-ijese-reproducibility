from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CV_FOLDS, TARGET_COL, TEST_SIZE  # noqa: E402
from src.evaluation.metrics import compute_binary_metrics  # noqa: E402


REQUIRED_METRIC_COLS = [
    "roc_auc",
    "pr_auc",
    "brier",
    "calibration_slope",
    "calibration_intercept",
]

TARGET_REGEX_PATTERNS = [
    re.compile(r"^y_qn26$", flags=re.IGNORECASE),
    re.compile(r"^y_qn27$", flags=re.IGNORECASE),
    re.compile(r"^y_qn28$", flags=re.IGNORECASE),
    re.compile(r"^y_qn29$", flags=re.IGNORECASE),
    re.compile(r"^y_qn30$", flags=re.IGNORECASE),
    re.compile(r"^qn26$", flags=re.IGNORECASE),
    re.compile(r"^qn27$", flags=re.IGNORECASE),
    re.compile(r"^qn28$", flags=re.IGNORECASE),
    re.compile(r"^qn29$", flags=re.IGNORECASE),
    re.compile(r"^qn30$", flags=re.IGNORECASE),
]

KNOWN_TARGET_COLS = {"y_qn26", "qn26"}
KNOWN_SECONDARY_TARGET_COLS = {
    "y_qn27",
    "y_qn28",
    "y_qn29",
    "y_qn30",
    "qn27",
    "qn28",
    "qn29",
    "qn30",
}
KNOWN_DESIGN_COLS = {"weight", "stratum", "psu"}
KNOWN_IDENTIFIER_COLS = {
    "id",
    "record",
    "orig_rec",
    "site",
    "student_id",
    "case_id",
    "caseid",
    "row_id",
    "respondent_id",
}
# Explicit finite list only. Kept intentionally conservative.
KNOWN_POST_EVENT_COLS = {
    "post_event_flag",
    "post_event_indicator",
    "post_qn26_indicator",
}


@dataclass(frozen=True)
class FeatureSets:
    baseline_features: List[str]
    full_features: List[str]
    full_minus_bullying_features: List[str]
    excluded_columns: Dict[str, List[str]]
    equals_baseline: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def require_paths(paths: Sequence[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError("Missing required file(s):\n" + "\n".join(missing))


def snapshot_hashes(paths: Sequence[Path]) -> Dict[str, str]:
    return {str(p): sha256_file(p) for p in paths}


def verify_hashes_unchanged(before: Dict[str, str], after: Dict[str, str]) -> None:
    if set(before.keys()) != set(after.keys()):
        raise RuntimeError("Frozen artifact hash-key set changed unexpectedly.")
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        lines = [f"{k}: {before[k]} -> {after[k]}" for k in changed]
        raise RuntimeError("Frozen artifact hash mismatch detected:\n" + "\n".join(lines))


def array_equal_with_tolerance(a: np.ndarray, b: np.ndarray, atol: float = 1e-12) -> bool:
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return np.array_equal(a, b)
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True)
    return np.array_equal(a, b)


def recompute_split_and_fold_artifacts(
    *,
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    test_size: float,
    cv_folds: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df[target_col].astype(int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pos, test_pos = next(splitter.split(df, y))

    row_index = df.index.to_numpy()
    train_idx = row_index[train_pos]
    test_idx = row_index[test_pos]

    y_train = y.loc[train_idx]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_id = np.full(len(train_idx), fill_value=-1, dtype=int)
    for fold, (_, va_pos) in enumerate(skf.split(np.zeros(len(train_idx)), y_train), start=0):
        fold_id[va_pos] = fold

    if (fold_id < 0).any():
        raise RuntimeError("Failed to assign all training samples to folds.")

    return train_idx, test_idx, fold_id


def verify_frozen_split_and_fold_equality(
    *,
    holdout_npz: Path,
    cvfolds_npz: Path,
    recomputed_train_idx: np.ndarray,
    recomputed_test_idx: np.ndarray,
    recomputed_fold_id: np.ndarray,
    atol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    holdout = np.load(holdout_npz)
    cvfolds = np.load(cvfolds_npz)

    frozen_train_idx = holdout["train_idx"]
    frozen_test_idx = holdout["test_idx"]
    frozen_train_idx_cv = cvfolds["train_idx"]
    frozen_fold_id = cvfolds["fold_id"]

    checks = [
        (
            "holdout train_idx",
            array_equal_with_tolerance(frozen_train_idx, recomputed_train_idx, atol=atol),
        ),
        (
            "holdout test_idx",
            array_equal_with_tolerance(frozen_test_idx, recomputed_test_idx, atol=atol),
        ),
        (
            "cvfolds train_idx",
            array_equal_with_tolerance(frozen_train_idx_cv, recomputed_train_idx, atol=atol),
        ),
        (
            "cvfolds fold_id",
            array_equal_with_tolerance(frozen_fold_id, recomputed_fold_id, atol=atol),
        ),
    ]
    failures = [label for label, ok in checks if not ok]
    if failures:
        raise RuntimeError("Frozen split/fold mismatch for: " + ", ".join(failures))

    return frozen_train_idx, frozen_test_idx, frozen_fold_id


def _is_target_like(col: str) -> bool:
    return any(pattern.match(col) for pattern in TARGET_REGEX_PATTERNS)


def derive_feature_sets(df_columns: Sequence[str]) -> FeatureSets:
    cols = [str(c) for c in df_columns]

    baseline = [c for c in ["q1", "q2", "q3", "raceeth"] if c in cols]

    excluded = {
        "target_cols": sorted([c for c in cols if c in KNOWN_TARGET_COLS]),
        "secondary_target_cols": sorted([c for c in cols if c in KNOWN_SECONDARY_TARGET_COLS]),
        "target_pattern_cols": sorted([c for c in cols if _is_target_like(c)]),
        "design_cols": sorted([c for c in cols if c in KNOWN_DESIGN_COLS]),
        "identifier_cols": sorted([c for c in cols if c in KNOWN_IDENTIFIER_COLS]),
        "post_event_cols": sorted([c for c in cols if c in KNOWN_POST_EVENT_COLS]),
    }

    excluded_union = set().union(*[set(v) for v in excluded.values()])

    full = [
        c
        for c in cols
        if c not in excluded_union
    ]

    # Keep deterministic order based on modeling table column order.
    full_minus_bullying = [c for c in full if c not in {"x_qn24", "x_qn25"}]

    if "x_qn24" not in full or "x_qn25" not in full:
        raise RuntimeError("full_features must include x_qn24 and x_qn25 after exclusions.")

    equals_baseline = full_minus_bullying == baseline

    return FeatureSets(
        baseline_features=baseline,
        full_features=full,
        full_minus_bullying_features=full_minus_bullying,
        excluded_columns=excluded,
        equals_baseline=equals_baseline,
    )


def prepare_feature_frame(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_covariates: Sequence[str],
) -> pd.DataFrame:
    x = df[list(feature_cols)].copy()

    for c in categorical_covariates:
        if c not in x.columns:
            continue
        s = x[c]
        if pd.api.types.is_numeric_dtype(s):

            def _to_cat(v):
                if pd.isna(v):
                    return np.nan
                fv = float(v)
                if fv.is_integer():
                    return f"cat_{int(fv)}"
                return f"cat_{fv}"

            x[c] = s.map(_to_cat).astype(object)
        else:
            x[c] = s.astype(object)

    for c in [col for col in x.columns if col not in set(categorical_covariates)]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace({pd.NA: np.nan})
    return x


def build_preprocessor(feature_cols: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    cat_cols = [c for c in categorical_cols if c in set(feature_cols)]
    num_cols = [c for c in feature_cols if c not in set(cat_cols)]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
            ),
        ]
    )
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    transformers: List[tuple] = []
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))

    if not transformers:
        raise RuntimeError("No feature columns were selected for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def resolved_hgb_params(tuned_best_params: Dict[str, object], seed: int) -> Dict[str, object]:
    params: Dict[str, object] = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 400,
        "early_stopping": True,
    }
    params.update(tuned_best_params)
    params["random_state"] = seed
    return params


def build_estimator(
    *,
    model_name: str,
    preprocessor: ColumnTransformer,
    seed: int,
    tuned_hgb_params: Optional[Dict[str, object]] = None,
) -> Pipeline:
    if model_name == "logreg":
        model = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
    elif model_name == "hgb":
        if tuned_hgb_params is None:
            raise RuntimeError("tuned_hgb_params are required for model_name='hgb'.")
        model = HistGradientBoostingClassifier(**resolved_hgb_params(tuned_hgb_params, seed))
    else:
        raise RuntimeError(f"Unsupported model_name: {model_name}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def clip_probs(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)


def predict_proba_positive(estimator: Pipeline, x: pd.DataFrame) -> np.ndarray:
    prob = estimator.predict_proba(x)
    if prob.ndim != 2 or prob.shape[1] < 2:
        raise RuntimeError("predict_proba output has invalid shape")
    return clip_probs(prob[:, 1])


def _platt_feature(y_prob: np.ndarray) -> np.ndarray:
    p = clip_probs(y_prob)
    logit = np.log(p / (1.0 - p))
    return logit.reshape(-1, 1)


def fit_calibrator(method: str, y_prob: np.ndarray, y_true: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if method == "none":
        return None

    if len(np.unique(y_true)) < 2:
        return None

    if method == "platt":
        model = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
        model.fit(_platt_feature(y_prob), y_true)
        return {"method": "platt", "model": model}

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(clip_probs(y_prob), y_true)
        return {"method": "isotonic", "model": model}

    raise RuntimeError(f"Unsupported calibration method: {method}")


def apply_calibrator(calibrator, y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    if calibrator is None:
        return clip_probs(y_prob)

    method = calibrator.get("method")
    model = calibrator.get("model")
    if method == "platt":
        return clip_probs(model.predict_proba(_platt_feature(y_prob))[:, 1])
    if method == "isotonic":
        return clip_probs(np.asarray(model.predict(clip_probs(y_prob)), dtype=float))
    raise RuntimeError(f"Unsupported calibrator method payload: {method}")


def _choose_inner_splits(y_train: pd.Series) -> int:
    counts = y_train.value_counts(dropna=False)
    if counts.empty:
        return 2
    min_count = int(counts.min())
    return max(2, min(5, min_count))


def build_oof_probs_from_fold_ids(
    *,
    estimator_factory: Callable[[], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fold_id: np.ndarray,
) -> np.ndarray:
    oof = np.full(len(x_train), np.nan, dtype=float)
    unique_folds = sorted(np.unique(fold_id).tolist())

    for fold in unique_folds:
        va_mask = fold_id == fold
        tr_mask = ~va_mask

        est = estimator_factory()
        est.fit(x_train.iloc[tr_mask], y_train.iloc[tr_mask])
        oof[va_mask] = predict_proba_positive(est, x_train.iloc[va_mask])

    if np.isnan(oof).any():
        raise RuntimeError("OOF probability generation failed: NaN values found.")

    return clip_probs(oof)


def fit_fold_safe_calibrator_from_training_fold(
    *,
    estimator_factory: Callable[[], Pipeline],
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    calibration_method: str,
    seed: int,
    outer_fold_id: int,
):
    if calibration_method == "none":
        return None

    n_splits = _choose_inner_splits(y_tr)
    if n_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 500 + outer_fold_id)
    oof = np.full(len(x_tr), np.nan, dtype=float)

    for tr_pos, va_pos in skf.split(x_tr, y_tr):
        est = estimator_factory()
        est.fit(x_tr.iloc[tr_pos], y_tr.iloc[tr_pos])
        oof[va_pos] = predict_proba_positive(est, x_tr.iloc[va_pos])

    if np.isnan(oof).any():
        raise RuntimeError("Inner-fold OOF probabilities contain NaN values.")

    return fit_calibrator(calibration_method, oof, y_tr.to_numpy(dtype=int))


def run_cv_metrics_with_leakage_safe_calibration(
    *,
    estimator_factory: Callable[[], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fold_id: np.ndarray,
    calibration_method: str,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    unique_folds = sorted(np.unique(fold_id).tolist())
    for fold in unique_folds:
        va_mask = fold_id == fold
        tr_mask = ~va_mask

        x_tr, y_tr = x_train.iloc[tr_mask], y_train.iloc[tr_mask]
        x_va, y_va = x_train.iloc[va_mask], y_train.iloc[va_mask]

        # Guardrail: fold separation for leakage prevention.
        if set(x_tr.index.tolist()).intersection(set(x_va.index.tolist())):
            raise RuntimeError(f"Leakage detected in CV split for fold {fold}.")

        est = estimator_factory()
        est.fit(x_tr, y_tr)
        y_prob_raw = predict_proba_positive(est, x_va)

        calibrator = fit_fold_safe_calibrator_from_training_fold(
            estimator_factory=estimator_factory,
            x_tr=x_tr,
            y_tr=y_tr,
            calibration_method=calibration_method,
            seed=seed,
            outer_fold_id=int(fold),
        )
        y_prob = apply_calibrator(calibrator, y_prob_raw)

        metrics = compute_binary_metrics(y_va.to_numpy(dtype=int), y_prob)
        rows.append({"fold": int(fold), **{k: float(v) for k, v in metrics.items()}})

    fold_df = pd.DataFrame(rows).sort_values("fold", kind="mergesort").reset_index(drop=True)
    return fold_df


def aggregate_cv_metrics(fold_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in REQUIRED_METRIC_COLS:
        out[metric] = float(fold_df[metric].mean())
        out[f"{metric}_std"] = float(fold_df[metric].std(ddof=1))
    return out


def run_test_metrics_with_train_only_calibration(
    *,
    estimator_factory: Callable[[], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    fold_id: np.ndarray,
    calibration_method: str,
) -> Dict[str, float]:
    if set(x_train.index.tolist()).intersection(set(x_test.index.tolist())):
        raise RuntimeError("Leakage detected: training and test indices overlap.")

    final_est = estimator_factory()
    final_est.fit(x_train, y_train)
    y_prob_raw_test = predict_proba_positive(final_est, x_test)

    if calibration_method == "none":
        y_prob_test = y_prob_raw_test
    else:
        oof_probs_train = build_oof_probs_from_fold_ids(
            estimator_factory=estimator_factory,
            x_train=x_train,
            y_train=y_train,
            fold_id=fold_id,
        )
        calibrator = fit_calibrator(calibration_method, oof_probs_train, y_train.to_numpy(dtype=int))
        y_prob_test = apply_calibrator(calibrator, y_prob_raw_test)

    metrics = compute_binary_metrics(y_test.to_numpy(dtype=int), y_prob_test)
    return {k: float(v) for k, v in metrics.items()}


def write_row_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def validate_metric_fields(path: Path) -> None:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_METRIC_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Metric contract failure in {path}: missing columns {missing}")


def metric_value_from_csv(path: Path, metric: str, split_scope: str) -> float:
    row = pd.read_csv(path).iloc[0]
    # Prefer direct metric columns.
    if metric in row.index:
        return float(row[metric])
    # Compatibility with legacy summary naming.
    if split_scope == "cv" and f"{metric}_mean" in row.index:
        return float(row[f"{metric}_mean"])
    if f"uncalibrated_{metric}" in row.index:
        return float(row[f"uncalibrated_{metric}"])
    raise RuntimeError(f"Could not resolve metric '{metric}' from {path}")


def _as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def validate_table_schema(
    *,
    df: pd.DataFrame,
    table_name: str,
    required_columns: Sequence[str],
    numeric_columns: Sequence[str],
    unique_key: Sequence[str],
    source_path_columns: Sequence[str],
) -> None:
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"{table_name}: missing required columns {missing_cols}")

    null_cols = [c for c in required_columns if df[c].isna().any()]
    if null_cols:
        raise RuntimeError(f"{table_name}: null values present in required columns {null_cols}")

    for c in numeric_columns:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise RuntimeError(f"{table_name}: column '{c}' must be numeric")
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{table_name}: column '{c}' contains non-finite values")

    if unique_key:
        dup_mask = df.duplicated(subset=list(unique_key), keep=False)
        if dup_mask.any():
            dup_records = df.loc[dup_mask, list(unique_key)]
            raise RuntimeError(
                f"{table_name}: uniqueness violation for key {list(unique_key)}."
                f" Duplicates: {dup_records.to_dict(orient='records')}"
            )

    for source_col in source_path_columns:
        if source_col not in df.columns:
            continue
        for raw in df[source_col].astype(str).tolist():
            candidate = Path(raw)
            candidate_abs = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
            if not candidate_abs.exists():
                raise RuntimeError(f"{table_name}: source path does not exist -> {raw}")


def build_week06_tables(
    *,
    metrics_paths: Dict[Tuple[str, str, str, str], Path],
    baseline_paths: Dict[str, Path],
    out_full_table: Path,
    out_ablation_table: Path,
    out_cal_table: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = ["cv", "test"]

    def m(split: str, model: str, featureset: str, cal: str, metric: str) -> float:
        return metric_value_from_csv(metrics_paths[(split, model, featureset, cal)], metric, split_scope=split)

    baseline_metric: Dict[str, Dict[str, float]] = {s: {} for s in splits}
    for split in splits:
        for metric in REQUIRED_METRIC_COLS:
            baseline_metric[split][metric] = metric_value_from_csv(baseline_paths[split], metric, split_scope=split)

    full_rows = []
    for split in splits:
        split_scope = "cv_mean" if split == "cv" else "heldout_test"
        for model in ["logreg", "hgb"]:
            p = metrics_paths[(split, model, "full", "none")]
            row = {
                "split_scope": split_scope,
                "model": model,
                "featureset": "full",
                "calibration_method": "none",
                "roc_auc": m(split, model, "full", "none", "roc_auc"),
                "pr_auc": m(split, model, "full", "none", "pr_auc"),
                "brier": m(split, model, "full", "none", "brier"),
                "calibration_slope": m(split, model, "full", "none", "calibration_slope"),
                "calibration_intercept": m(split, model, "full", "none", "calibration_intercept"),
                "delta_roc_auc_vs_logreg_baseline": m(split, model, "full", "none", "roc_auc")
                - baseline_metric[split]["roc_auc"],
                "delta_pr_auc_vs_logreg_baseline": m(split, model, "full", "none", "pr_auc")
                - baseline_metric[split]["pr_auc"],
                "delta_brier_vs_logreg_baseline": m(split, model, "full", "none", "brier")
                - baseline_metric[split]["brier"],
                "delta_calibration_slope_vs_logreg_baseline": m(split, model, "full", "none", "calibration_slope")
                - baseline_metric[split]["calibration_slope"],
                "delta_calibration_intercept_vs_logreg_baseline": m(
                    split, model, "full", "none", "calibration_intercept"
                )
                - baseline_metric[split]["calibration_intercept"],
                "source_metrics_path": _as_repo_relative(p),
            }
            full_rows.append(row)

    full_df = pd.DataFrame(full_rows)
    validate_table_schema(
        df=full_df,
        table_name="week06_full_feature_comparison",
        required_columns=[
            "split_scope",
            "model",
            "featureset",
            "calibration_method",
            "roc_auc",
            "pr_auc",
            "brier",
            "calibration_slope",
            "calibration_intercept",
            "delta_roc_auc_vs_logreg_baseline",
            "delta_pr_auc_vs_logreg_baseline",
            "delta_brier_vs_logreg_baseline",
            "delta_calibration_slope_vs_logreg_baseline",
            "delta_calibration_intercept_vs_logreg_baseline",
            "source_metrics_path",
        ],
        numeric_columns=[
            "roc_auc",
            "pr_auc",
            "brier",
            "calibration_slope",
            "calibration_intercept",
            "delta_roc_auc_vs_logreg_baseline",
            "delta_pr_auc_vs_logreg_baseline",
            "delta_brier_vs_logreg_baseline",
            "delta_calibration_slope_vs_logreg_baseline",
            "delta_calibration_intercept_vs_logreg_baseline",
        ],
        unique_key=["split_scope", "model", "featureset", "calibration_method"],
        source_path_columns=["source_metrics_path"],
    )
    out_full_table.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_full_table, index=False)

    ablation_rows = []
    for split in splits:
        split_scope = "cv_mean" if split == "cv" else "heldout_test"
        path_a = metrics_paths[(split, "hgb", "full_minus_bullying", "none")]
        path_b = metrics_paths[(split, "hgb", "full", "none")]
        for metric in REQUIRED_METRIC_COLS:
            a_val = m(split, "hgb", "full_minus_bullying", "none", metric)
            b_val = m(split, "hgb", "full", "none", metric)
            ablation_rows.append(
                {
                    "split_scope": split_scope,
                    "metric": metric,
                    "model_a_name": "hgb_full_minus_bullying_none",
                    "model_b_name": "hgb_full_none",
                    "model_a_value": a_val,
                    "model_b_value": b_val,
                    "delta_b_minus_a": b_val - a_val,
                    "source_metrics_path_a": _as_repo_relative(path_a),
                    "source_metrics_path_b": _as_repo_relative(path_b),
                }
            )

    ablation_df = pd.DataFrame(ablation_rows)
    validate_table_schema(
        df=ablation_df,
        table_name="week06_bullying_ablation_comparison",
        required_columns=[
            "split_scope",
            "metric",
            "model_a_name",
            "model_b_name",
            "model_a_value",
            "model_b_value",
            "delta_b_minus_a",
            "source_metrics_path_a",
            "source_metrics_path_b",
        ],
        numeric_columns=["model_a_value", "model_b_value", "delta_b_minus_a"],
        unique_key=["split_scope", "metric", "model_a_name", "model_b_name"],
        source_path_columns=["source_metrics_path_a", "source_metrics_path_b"],
    )
    out_ablation_table.parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(out_ablation_table, index=False)

    cal_rows = []
    methods = ["none", "platt", "isotonic"]
    for split in splits:
        split_scope = "cv_mean" if split == "cv" else "heldout_test"
        none_vals = {
            metric: m(split, "hgb", "full", "none", metric)
            for metric in REQUIRED_METRIC_COLS
        }
        for method in methods:
            p = metrics_paths[(split, "hgb", "full", method)]
            row = {
                "split_scope": split_scope,
                "calibration_method": method,
                "roc_auc": m(split, "hgb", "full", method, "roc_auc"),
                "pr_auc": m(split, "hgb", "full", method, "pr_auc"),
                "brier": m(split, "hgb", "full", method, "brier"),
                "calibration_slope": m(split, "hgb", "full", method, "calibration_slope"),
                "calibration_intercept": m(split, "hgb", "full", method, "calibration_intercept"),
                "delta_brier_vs_none": m(split, "hgb", "full", method, "brier") - none_vals["brier"],
                "delta_calibration_slope_vs_none": m(split, "hgb", "full", method, "calibration_slope")
                - none_vals["calibration_slope"],
                "delta_calibration_intercept_vs_none": m(split, "hgb", "full", method, "calibration_intercept")
                - none_vals["calibration_intercept"],
                "source_metrics_path": _as_repo_relative(p),
            }
            cal_rows.append(row)

    cal_df = pd.DataFrame(cal_rows)
    validate_table_schema(
        df=cal_df,
        table_name="week06_calibration_sensitivity",
        required_columns=[
            "split_scope",
            "calibration_method",
            "roc_auc",
            "pr_auc",
            "brier",
            "calibration_slope",
            "calibration_intercept",
            "delta_brier_vs_none",
            "delta_calibration_slope_vs_none",
            "delta_calibration_intercept_vs_none",
            "source_metrics_path",
        ],
        numeric_columns=[
            "roc_auc",
            "pr_auc",
            "brier",
            "calibration_slope",
            "calibration_intercept",
            "delta_brier_vs_none",
            "delta_calibration_slope_vs_none",
            "delta_calibration_intercept_vs_none",
        ],
        unique_key=["split_scope", "calibration_method"],
        source_path_columns=["source_metrics_path"],
    )
    out_cal_table.parent.mkdir(parents=True, exist_ok=True)
    cal_df.to_csv(out_cal_table, index=False)

    # Ensure deltas are present and non-null.
    for df_name, df_obj, delta_cols in [
        (
            "week06_full_feature_comparison",
            full_df,
            [
                "delta_roc_auc_vs_logreg_baseline",
                "delta_pr_auc_vs_logreg_baseline",
                "delta_brier_vs_logreg_baseline",
                "delta_calibration_slope_vs_logreg_baseline",
                "delta_calibration_intercept_vs_logreg_baseline",
            ],
        ),
        ("week06_bullying_ablation_comparison", ablation_df, ["delta_b_minus_a"]),
        (
            "week06_calibration_sensitivity",
            cal_df,
            [
                "delta_brier_vs_none",
                "delta_calibration_slope_vs_none",
                "delta_calibration_intercept_vs_none",
            ],
        ),
    ]:
        missing = [c for c in delta_cols if c not in df_obj.columns]
        if missing:
            raise RuntimeError(f"{df_name}: missing delta column(s): {missing}")
        nulls = [c for c in delta_cols if df_obj[c].isna().any()]
        if nulls:
            raise RuntimeError(f"{df_name}: null delta value(s): {nulls}")

    return full_df, ablation_df, cal_df


def build_week06_figures(
    *,
    full_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    out_full_fig: Path,
    out_ablation_fig: Path,
    out_cal_fig: Path,
) -> None:
    out_full_fig.parent.mkdir(parents=True, exist_ok=True)

    # Figure 1: full-feature comparison on held-out metrics.
    heldout = full_df[full_df["split_scope"] == "heldout_test"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["roc_auc", "pr_auc", "brier"]
    for ax, metric in zip(axes, metrics):
        ax.bar(heldout["model"], heldout[metric], color=["#4e79a7", "#f28e2b"])
        ax.set_title(metric)
        ax.set_xlabel("model")
        ax.set_ylabel(metric)
    fig.suptitle("Week 6 Full Feature Comparison (Held-out)")
    fig.tight_layout()
    fig.savefig(out_full_fig, dpi=300)
    plt.close(fig)

    # Figure 2: ablation deltas by split.
    ab = ablation_df.copy()
    pivot = ab.pivot(index="metric", columns="split_scope", values="delta_b_minus_a")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Week 6 Bullying Block Ablation Deltas (B - A)")
    ax.set_ylabel("Delta")
    ax.set_xlabel("Metric")
    fig.tight_layout()
    fig.savefig(out_ablation_fig, dpi=300)
    plt.close(fig)

    # Figure 3: calibration sensitivity on held-out.
    cal_held = cal_df[cal_df["split_scope"] == "heldout_test"].copy()
    method_order = ["none", "platt", "isotonic"]
    cal_held["method_order"] = cal_held["calibration_method"].map({m: i for i, m in enumerate(method_order)})
    cal_held = cal_held.sort_values("method_order", kind="mergesort")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(cal_held["calibration_method"], cal_held["brier"], marker="o")
    axes[0].set_title("Held-out Brier")
    axes[0].set_xlabel("Calibration")
    axes[0].set_ylabel("Brier")

    axes[1].plot(cal_held["calibration_method"], cal_held["calibration_slope"], marker="o")
    axes[1].axhline(1.0, color="gray", linewidth=1, linestyle="--")
    axes[1].set_title("Held-out Calibration Slope")
    axes[1].set_xlabel("Calibration")
    axes[1].set_ylabel("Slope")

    fig.suptitle("Week 6 Calibration Sensitivity")
    fig.tight_layout()
    fig.savefig(out_cal_fig, dpi=300)
    plt.close(fig)


def run_permutation_importance_stability(
    *,
    estimator_factory: Callable[[], Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fold_id: np.ndarray,
    n_repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_repeats < 20:
        raise RuntimeError("n_repeats must be at least 20 for stability analysis.")

    rows: List[Dict[str, object]] = []
    unique_folds = sorted(np.unique(fold_id).tolist())
    for fold in unique_folds:
        va_mask = fold_id == fold
        tr_mask = ~va_mask
        x_tr, y_tr = x_train.iloc[tr_mask], y_train.iloc[tr_mask]
        x_va, y_va = x_train.iloc[va_mask], y_train.iloc[va_mask]

        est = estimator_factory()
        est.fit(x_tr, y_tr)

        perm_brier = permutation_importance(
            estimator=est,
            X=x_va,
            y=y_va,
            scoring="neg_brier_score",
            n_repeats=n_repeats,
            random_state=seed + 1000 + int(fold),
            n_jobs=1,
        )
        perm_roc = permutation_importance(
            estimator=est,
            X=x_va,
            y=y_va,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=seed + 2000 + int(fold),
            n_jobs=1,
        )

        feature_names = x_va.columns.tolist()
        means_brier = pd.Series(perm_brier.importances_mean, index=feature_names)
        stds_brier = pd.Series(perm_brier.importances_std, index=feature_names)
        means_roc = pd.Series(perm_roc.importances_mean, index=feature_names)
        stds_roc = pd.Series(perm_roc.importances_std, index=feature_names)
        rank = means_brier.rank(ascending=False, method="min")

        for name in feature_names:
            rows.append(
                {
                    "fold": int(fold),
                    "feature_name": str(name),
                    "importance_mean_neg_brier_score": float(means_brier.loc[name]),
                    "importance_std_neg_brier_score": float(stds_brier.loc[name]),
                    "importance_mean_roc_auc": float(means_roc.loc[name]),
                    "importance_std_roc_auc": float(stds_roc.loc[name]),
                    "rank_neg_brier_score": float(rank.loc[name]),
                    "n_repeats": int(n_repeats),
                }
            )

    by_fold = pd.DataFrame(rows).sort_values(["fold", "rank_neg_brier_score", "feature_name"], kind="mergesort")

    summary = (
        by_fold.groupby("feature_name", as_index=False)
        .agg(
            mean_importance_neg_brier_score=("importance_mean_neg_brier_score", "mean"),
            sd_importance_neg_brier_score=("importance_mean_neg_brier_score", "std"),
            mean_importance_roc_auc=("importance_mean_roc_auc", "mean"),
            sd_importance_roc_auc=("importance_mean_roc_auc", "std"),
            mean_rank_neg_brier_score=("rank_neg_brier_score", "mean"),
            sd_rank_neg_brier_score=("rank_neg_brier_score", "std"),
            folds_present=("fold", "nunique"),
        )
        .sort_values(["mean_importance_neg_brier_score", "mean_rank_neg_brier_score"], ascending=[False, True])
        .reset_index(drop=True)
    )

    summary["fold_consistency_all_folds"] = summary["folds_present"] == len(unique_folds)
    summary["rank_range_neg_brier_score"] = summary["sd_rank_neg_brier_score"].fillna(0.0)

    for bully in ["x_qn24", "x_qn25"]:
        if bully not in set(summary["feature_name"].tolist()):
            raise RuntimeError(f"Permutation-importance summary missing required bullying row: {bully}")

    return by_fold.reset_index(drop=True), summary


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
