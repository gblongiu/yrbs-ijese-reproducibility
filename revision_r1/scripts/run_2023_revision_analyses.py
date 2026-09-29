from __future__ import annotations

import json
import math
import runpy
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT / "revision_r1" / "outputs"
TABLES_DIR = OUT_ROOT / "tables"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"
for path in [TABLES_DIR, METRICS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

SEED = 2026
PAIRED_BOOTSTRAP_SEED = 20260915
PAIRED_BOOTSTRAP_REPS = 2000
SUBGROUP_BOOTSTRAP_SEED = 20260916
SUBGROUP_BOOTSTRAP_REPS = 500
SUBGROUP_MIN_POS = 100
SUBGROUP_MIN_NEG = 100
EPS = 1e-6


def load_project_modules() -> tuple[dict, dict]:
    build_ns = runpy.run_path(str(PROJECT_ROOT / "scripts" / "01_build_dataset.py"))
    week06_ns = runpy.run_path(str(PROJECT_ROOT / "scripts" / "week06.py"))
    return build_ns, week06_ns


def load_modeling_table(build_ns: dict) -> pd.DataFrame:
    raw = pd.read_excel(PROJECT_ROOT / "data" / "raw" / "YRBS_2023_MH_subset.xlsx")
    modeling, _ = build_ns["build_modeling_table"](raw)
    return modeling


def load_frozen_indices(df: pd.DataFrame, week06_ns: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recomputed = week06_ns["recompute_split_and_fold_artifacts"](
        df=df,
        target_col="y_qn26",
        seed=SEED,
        test_size=0.2,
        cv_folds=5,
    )
    return week06_ns["verify_frozen_split_and_fold_equality"](
        holdout_npz=PROJECT_ROOT / "outputs" / "splits" / "holdout_seed2026.npz",
        cvfolds_npz=PROJECT_ROOT / "outputs" / "splits" / "cvfolds_seed2026.npz",
        recomputed_train_idx=recomputed[0],
        recomputed_test_idx=recomputed[1],
        recomputed_fold_id=recomputed[2],
    )


def load_hgb_params() -> dict:
    payload = json.loads(
        (PROJECT_ROOT / "outputs" / "tuning" / "hgb_seed2026_baseline_best_params.json").read_text(
            encoding="utf-8"
        )
    )
    return payload["best_params"]


def select_from_training_cv() -> pd.DataFrame:
    candidate_files = [
        PROJECT_ROOT / "outputs" / "metrics" / "metrics_cv_seed2026_logreg_full_none.csv",
        PROJECT_ROOT / "outputs" / "metrics" / "metrics_cv_seed2026_hgb_full_none.csv",
        PROJECT_ROOT / "outputs" / "metrics" / "metrics_cv_seed2026_hgb_full_platt.csv",
        PROJECT_ROOT / "outputs" / "metrics" / "metrics_cv_seed2026_hgb_full_isotonic.csv",
    ]
    rows = []
    for path in candidate_files:
        row = pd.read_csv(path).iloc[0].to_dict()
        row["source_file"] = str(path.relative_to(PROJECT_ROOT))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["abs_slope_error"] = (df["calibration_slope"] - 1.0).abs()
    df = df.sort_values(
        ["brier", "abs_slope_error", "roc_auc"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    df["selection_rank"] = np.arange(1, len(df) + 1)
    df["selected_training_cv"] = df["selection_rank"] == 1
    cols = [
        "selection_rank",
        "selected_training_cv",
        "model",
        "featureset",
        "calibration_method",
        "roc_auc",
        "pr_auc",
        "brier",
        "calibration_slope",
        "calibration_intercept",
        "abs_slope_error",
        "source_file",
    ]
    out = df[cols]
    out.to_csv(TABLES_DIR / "training_side_candidate_selection_seed2026.csv", index=False)
    return out


def normalized_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise RuntimeError("Survey weights must be finite and strictly positive.")
    return values / values.mean()


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    if sample_weight is None:
        model.fit(logit, np.asarray(y_true, dtype=int))
    else:
        model.fit(logit, np.asarray(y_true, dtype=int), sample_weight=normalized_weights(sample_weight))
    return float(model.coef_[0][0]), float(model.intercept_[0])


def metric_row(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if sample_weight is None:
        w = None
    else:
        w = np.asarray(sample_weight, dtype=float)
    slope, intercept = calibration_slope_intercept(y, p, w)
    return {
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "average_precision": float(average_precision_score(y, p, sample_weight=w)),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def fit_pipeline(estimator, x: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None):
    if sample_weight is None:
        estimator.fit(x, y)
    else:
        estimator.fit(x, y, model__sample_weight=normalized_weights(sample_weight))
    return estimator


def platt_feature(y_prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def fit_platt(y_prob: np.ndarray, y_true: np.ndarray, sample_weight: np.ndarray | None = None):
    model = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
    if sample_weight is None:
        model.fit(platt_feature(y_prob), np.asarray(y_true, dtype=int))
    else:
        model.fit(
            platt_feature(y_prob),
            np.asarray(y_true, dtype=int),
            sample_weight=normalized_weights(sample_weight),
        )
    return model


def apply_platt(model, y_prob: np.ndarray) -> np.ndarray:
    p = model.predict_proba(platt_feature(y_prob))[:, 1]
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def oof_predictions(
    estimator_factory: Callable,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fold_id: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    oof = np.full(len(x_train), np.nan, dtype=float)
    for fold in sorted(np.unique(fold_id).tolist()):
        valid = fold_id == fold
        train = ~valid
        est = estimator_factory()
        w = None if sample_weight is None else np.asarray(sample_weight)[train]
        fit_pipeline(est, x_train.iloc[train], y_train.iloc[train], w)
        oof[valid] = est.predict_proba(x_train.iloc[valid])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("OOF prediction generation produced missing values.")
    return np.clip(oof, EPS, 1.0 - EPS)


def test_predictions(
    estimator_factory: Callable,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    fold_id: np.ndarray,
    calibration: str,
    train_weight: np.ndarray | None = None,
    artifact_name: str | None = None,
) -> np.ndarray:
    est = estimator_factory()
    fit_pipeline(est, x_train, y_train, train_weight)
    raw_test = np.clip(est.predict_proba(x_test)[:, 1], EPS, 1.0 - EPS)
    if calibration == "none":
        return raw_test
    if calibration != "platt":
        raise RuntimeError(f"Unsupported calibration state: {calibration}")
    oof = oof_predictions(estimator_factory, x_train, y_train, fold_id, train_weight)
    calibrator = fit_platt(oof, y_train.to_numpy(dtype=int), train_weight)
    if artifact_name is not None:
        import joblib
        model_dir = OUT_ROOT / 'models'
        model_dir.mkdir(exist_ok=True)
        joblib.dump({'estimator': est, 'calibrator': calibrator,
                     'features': list(x_train.columns), 'seed': SEED},
                    model_dir / f'{artifact_name}.joblib')
    return apply_platt(calibrator, raw_test)


def make_hgb_factory(week06_ns: dict, feature_cols: list[str], params: dict) -> Callable:
    def factory():
        pre = week06_ns["build_preprocessor"](feature_cols, ["q1", "q2", "q3", "raceeth"])
        return week06_ns["build_estimator"](
            model_name="hgb",
            preprocessor=pre,
            seed=SEED,
            tuned_hgb_params=params,
        )

    return factory


def generate_core_predictions(
    df: pd.DataFrame,
    week06_ns: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_id: np.ndarray,
    params: dict,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    feature_sets = week06_ns["derive_feature_sets"](df.columns)
    y = df["y_qn26"].astype(int)
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    prediction_map: dict[str, np.ndarray] = {}
    rows = []
    for label, features in [
        ("baseline", feature_sets.baseline_features),
        ("expanded", feature_sets.full_features),
    ]:
        x = week06_ns["prepare_feature_frame"](df, features, ["q1", "q2", "q3", "raceeth"])
        x_train = x.loc[train_idx]
        x_test = x.loc[test_idx]
        factory = make_hgb_factory(week06_ns, features, params)
        for calibration in ["none", "platt"]:
            p = test_predictions(
                factory,
                x_train,
                y_train,
                x_test,
                fold_id,
                calibration,
                train_weight=None,
                artifact_name=f'{label}_platt_2023_frozen' if calibration == 'platt' else None,
            )
            key = f"{label}_{calibration}"
            prediction_map[key] = p
            row = {
                "feature_set": label,
                "calibration": calibration,
                "n_test": len(y_test),
                **metric_row(y_test.to_numpy(dtype=int), p),
            }
            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_DIR / "matched_hgb_2023_metrics_seed2026.csv", index=False)

    pred = pd.DataFrame(
        {
            "row_index": test_idx,
            "y_true": y_test.to_numpy(dtype=int),
            "weight": df.loc[test_idx, "weight"].to_numpy(dtype=float),
            "q1": df.loc[test_idx, "q1"].astype(object).to_numpy(),
            "q2": df.loc[test_idx, "q2"].astype(object).to_numpy(),
            "q3": df.loc[test_idx, "q3"].astype(object).to_numpy(),
            "raceeth": df.loc[test_idx, "raceeth"].astype(object).to_numpy(),
            **prediction_map,
        }
    )
    pred.to_csv(TABLES_DIR / "matched_hgb_2023_holdout_predictions_seed2026.csv", index=False)
    return pred, prediction_map


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    pos_draw = rng.choice(pos, size=len(pos), replace=True)
    neg_draw = rng.choice(neg, size=len(neg), replace=True)
    return np.concatenate([pos_draw, neg_draw])


def paired_bootstrap(pred: pd.DataFrame) -> pd.DataFrame:
    from joblib import Parallel, delayed

    y = pred["y_true"].to_numpy(dtype=int)
    arrays = {
        "uncalibrated": (
            pred["baseline_none"].to_numpy(dtype=float),
            pred["expanded_none"].to_numpy(dtype=float),
        ),
        "platt": (
            pred["baseline_platt"].to_numpy(dtype=float),
            pred["expanded_platt"].to_numpy(dtype=float),
        ),
    }

    point_rows = []
    for state, (p0, p1) in arrays.items():
        base = metric_row(y, p0)
        expanded = metric_row(y, p1)
        point_rows.extend(
            [
                {
                    "comparison": state,
                    "metric": "roc_auc_change",
                    "baseline": base["roc_auc"],
                    "expanded": expanded["roc_auc"],
                    "point_difference": expanded["roc_auc"] - base["roc_auc"],
                },
                {
                    "comparison": state,
                    "metric": "average_precision_change",
                    "baseline": base["average_precision"],
                    "expanded": expanded["average_precision"],
                    "point_difference": expanded["average_precision"] - base["average_precision"],
                },
                {
                    "comparison": state,
                    "metric": "brier_improvement",
                    "baseline": base["brier"],
                    "expanded": expanded["brier"],
                    "point_difference": base["brier"] - expanded["brier"],
                },
            ]
        )

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    def one_rep(rep: int) -> list[dict[str, float | int | str]]:
        rng = np.random.default_rng(PAIRED_BOOTSTRAP_SEED + rep)
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        yb = y[idx]
        rows = []
        for state, (p0, p1) in arrays.items():
            p0b = p0[idx]
            p1b = p1[idx]
            b_roc = float(roc_auc_score(yb, p0b))
            e_roc = float(roc_auc_score(yb, p1b))
            b_ap = float(average_precision_score(yb, p0b))
            e_ap = float(average_precision_score(yb, p1b))
            b_brier = float(brier_score_loss(yb, p0b))
            e_brier = float(brier_score_loss(yb, p1b))
            rows.append(
                {
                    "comparison": state,
                    "replicate": rep,
                    "roc_auc_change": e_roc - b_roc,
                    "average_precision_change": e_ap - b_ap,
                    "brier_improvement": b_brier - e_brier,
                }
            )
        return rows

    nested = Parallel(n_jobs=5, prefer="processes", batch_size=20)(
        delayed(one_rep)(rep) for rep in range(PAIRED_BOOTSTRAP_REPS)
    )
    draw_rows = [row for pair in nested for row in pair]
    draws = pd.DataFrame(draw_rows)
    draws.to_csv(TABLES_DIR / "paired_bootstrap_draws_2023_seed20260915.csv", index=False)

    points = pd.DataFrame(point_rows)
    summary_rows = []
    for row in points.to_dict(orient="records"):
        vals = draws.loc[draws["comparison"] == row["comparison"], row["metric"]].to_numpy(dtype=float)
        summary_rows.append(
            {
                **row,
                "ci95_low": float(np.quantile(vals, 0.025)),
                "ci95_high": float(np.quantile(vals, 0.975)),
                "bootstrap_replicates": PAIRED_BOOTSTRAP_REPS,
                "bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / "paired_improvement_ci_2023_seed20260915.csv", index=False)
    return summary


def survey_weight_sensitivity(
    df: pd.DataFrame,
    week06_ns: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_id: np.ndarray,
    params: dict,
) -> pd.DataFrame:
    feature_sets = week06_ns["derive_feature_sets"](df.columns)
    y = df["y_qn26"].astype(int)
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx].to_numpy(dtype=int)
    w_train = df.loc[train_idx, "weight"].to_numpy(dtype=float)
    w_test = df.loc[test_idx, "weight"].to_numpy(dtype=float)

    rows = []
    for label, features in [
        ("baseline", feature_sets.baseline_features),
        ("expanded", feature_sets.full_features),
    ]:
        x = week06_ns["prepare_feature_frame"](df, features, ["q1", "q2", "q3", "raceeth"])
        x_train = x.loc[train_idx]
        x_test = x.loc[test_idx]
        factory = make_hgb_factory(week06_ns, features, params)

        for fitting in ["not_survey_weighted", "cdc_weighted"]:
            train_weight = None if fitting == "not_survey_weighted" else w_train
            p = test_predictions(
                factory,
                x_train,
                y_train,
                x_test,
                fold_id,
                "platt",
                train_weight=train_weight,
            )
            for evaluation in ["empirical", "cdc_weighted"]:
                eval_weight = None if evaluation == "empirical" else w_test
                row = {
                    "feature_set": label,
                    "model_fitting": fitting,
                    "calibration": "platt",
                    "evaluation": evaluation,
                    "training_weight_normalization": "mean_1_within_fit" if train_weight is not None else "none",
                    **metric_row(y_test, p, eval_weight),
                }
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "survey_weight_sensitivity_2023_seed2026.csv", index=False)

    delta_rows = []
    for fitting in out["model_fitting"].unique():
        for evaluation in out["evaluation"].unique():
            sub = out[(out["model_fitting"] == fitting) & (out["evaluation"] == evaluation)].set_index("feature_set")
            b = sub.loc["baseline"]
            e = sub.loc["expanded"]
            delta_rows.append(
                {
                    "model_fitting": fitting,
                    "evaluation": evaluation,
                    "roc_auc_change": e["roc_auc"] - b["roc_auc"],
                    "average_precision_change": e["average_precision"] - b["average_precision"],
                    "brier_improvement": b["brier"] - e["brier"],
                    "calibration_slope_change": e["calibration_slope"] - b["calibration_slope"],
                    "calibration_intercept_change": e["calibration_intercept"] - b["calibration_intercept"],
                }
            )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(TABLES_DIR / "survey_weight_sensitivity_deltas_2023_seed2026.csv", index=False)
    return out


def reliability_diagram(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred["expanded_platt"].to_numpy(dtype=float)
    y = pred["y_true"].to_numpy(dtype=int)
    order = np.argsort(p, kind="mergesort")
    bin_id = np.empty(len(p), dtype=int)
    bin_id[order] = np.floor(np.arange(len(p)) * 10 / len(p)).astype(int)
    bin_id = np.clip(bin_id, 0, 9)

    rows = []
    for b in range(10):
        mask = bin_id == b
        rows.append(
            {
                "bin": b + 1,
                "n": int(mask.sum()),
                "mean_predicted_probability": float(p[mask].mean()),
                "observed_event_frequency": float(y[mask].mean()),
            }
        )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES_DIR / "reliability_bins_final_model_2023_seed2026.csv", index=False)

    fig = plt.figure(figsize=(6.4, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.plot(tbl["mean_predicted_probability"], tbl["observed_event_frequency"], marker="o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed event frequency")
    ax.set_title("2023 reliability diagram: expanded HGB with Platt scaling")
    ax.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.bar(tbl["mean_predicted_probability"], tbl["n"], width=0.045)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("n")
    ax2.grid(alpha=0.15, axis="y")
    fig.savefig(FIGURES_DIR / "reliability_diagram_final_model_2023_seed2026.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return tbl


def bootstrap_rank_error_ci(y: np.ndarray, p: np.ndarray, seed: int, reps: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ["roc_auc", "average_precision", "brier"]}
    for _ in range(reps):
        idx = stratified_bootstrap_indices(y, rng)
        yb = y[idx]
        pb = p[idx]
        draws["roc_auc"].append(float(roc_auc_score(yb, pb)))
        draws["average_precision"].append(float(average_precision_score(yb, pb)))
        draws["brier"].append(float(brier_score_loss(yb, pb)))
    return {
        key: (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
        for key, vals in draws.items()
    }


def calibration_wald_ci(y: np.ndarray, p: np.ndarray) -> dict[str, tuple[float, float]]:
    import statsmodels.api as sm

    prob = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    logit = np.log(prob / (1.0 - prob))
    design = sm.add_constant(logit, has_constant="add")
    fit = sm.GLM(np.asarray(y, dtype=int), design, family=sm.families.Binomial()).fit()
    ci = np.asarray(fit.conf_int(alpha=0.05), dtype=float)
    return {
        "calibration_intercept": (float(ci[0, 0]), float(ci[0, 1])),
        "calibration_slope": (float(ci[1, 0]), float(ci[1, 1])),
    }


def subgroup_diagnostics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_vars = ["q2", "q3", "raceeth", "q1"]
    for var_index, var in enumerate(group_vars):
        values = pred[var].astype(object).where(pred[var].notna(), "<missing>")
        for value_index, value in enumerate(sorted(values.unique().tolist(), key=str)):
            mask = values == value
            sub = pred.loc[mask]
            y = sub["y_true"].to_numpy(dtype=int)
            p = sub["expanded_platt"].to_numpy(dtype=float)
            n_pos = int(y.sum())
            n_neg = int(len(y) - n_pos)
            eligible = n_pos >= SUBGROUP_MIN_POS and n_neg >= SUBGROUP_MIN_NEG
            row = {
                "subgroup_variable": var,
                "subgroup_value": value,
                "n": len(y),
                "events": n_pos,
                "non_events": n_neg,
                "outcome_prevalence": float(y.mean()) if len(y) else np.nan,
                "calibration_adequate": eligible,
                "minimum_events_rule": SUBGROUP_MIN_POS,
                "minimum_non_events_rule": SUBGROUP_MIN_NEG,
            }
            if eligible:
                m = metric_row(y, p)
                rank_error_ci = bootstrap_rank_error_ci(
                    y,
                    p,
                    seed=SUBGROUP_BOOTSTRAP_SEED + var_index * 100 + value_index,
                    reps=SUBGROUP_BOOTSTRAP_REPS,
                )
                calibration_ci = calibration_wald_ci(y, p)
                row.update(m)
                for metric, (lo, hi) in {**rank_error_ci, **calibration_ci}.items():
                    row[f"{metric}_ci95_low"] = lo
                    row[f"{metric}_ci95_high"] = hi
                row["bootstrap_replicates"] = SUBGROUP_BOOTSTRAP_REPS
            else:
                for metric in ["roc_auc", "average_precision", "brier", "calibration_slope", "calibration_intercept"]:
                    row[metric] = np.nan
                    row[f"{metric}_ci95_low"] = np.nan
                    row[f"{metric}_ci95_high"] = np.nan
                row["bootstrap_replicates"] = 0
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "subgroup_calibration_diagnostics_2023_seed2026.csv", index=False)
    return out


def write_reproducibility_record() -> None:
    import matplotlib
    import scipy
    import sklearn

    record = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "analysis_seed": SEED,
        "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        "paired_bootstrap_replicates": PAIRED_BOOTSTRAP_REPS,
        "subgroup_bootstrap_seed": SUBGROUP_BOOTSTRAP_SEED,
        "subgroup_bootstrap_replicates": SUBGROUP_BOOTSTRAP_REPS,
        "probability_clip_epsilon": EPS,
        "average_precision_function": "sklearn.metrics.average_precision_score",
        "calibration_slope_intercept": "joint logistic recalibration on probability logit; C=1e6; lbfgs",
        "subgroup_adequacy_rule": f">={SUBGROUP_MIN_POS} events and >={SUBGROUP_MIN_NEG} non-events",
    }
    (OUT_ROOT / "reproducibility_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    build_ns, week06_ns = load_project_modules()
    df = load_modeling_table(build_ns)
    train_idx, test_idx, fold_id = load_frozen_indices(df, week06_ns)
    params = load_hgb_params()

    print("[1/6] training-side selection", flush=True)
    selection = select_from_training_cv()
    selected = selection.loc[selection["selected_training_cv"]].iloc[0]
    if not (
        selected["model"] == "hgb"
        and selected["featureset"] == "full"
        and selected["calibration_method"] == "platt"
    ):
        raise RuntimeError("Training-side selection did not reproduce HGB full with Platt scaling.")

    print("[2/6] matched 2023 predictions", flush=True)
    pred, _ = generate_core_predictions(df, week06_ns, train_idx, test_idx, fold_id, params)
    print("[3/6] paired bootstrap", flush=True)
    paired_bootstrap(pred)
    print("[4/6] survey-weight sensitivity", flush=True)
    survey_weight_sensitivity(df, week06_ns, train_idx, test_idx, fold_id, params)
    print("[5/6] reliability diagram", flush=True)
    reliability_diagram(pred)
    print("[6/6] subgroup diagnostics", flush=True)
    subgroup_diagnostics(pred)
    write_reproducibility_record()

    print("Revision R1 2023 analyses completed.")


if __name__ == "__main__":
    main()
