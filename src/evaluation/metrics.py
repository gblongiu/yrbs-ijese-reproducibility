from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression


def calibration_slope_intercept(y_true, y_prob):
    eps = 1e-6
    y_prob = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
    # Near-unregularized logistic regression on the logit scores. (penalty defaults to L2; avoid
    # passing penalty explicitly to keep compatibility with newer sklearn versions.)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit, y_true)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def compute_binary_metrics(y_true, y_prob) -> Dict[str, float]:
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }
