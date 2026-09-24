"""
Evaluation Metrics and Threshold Tuning for Software Supply Chain Compromise Detection.

Strictly follows evaluation principles:
1. AUPRC is the primary metric for extreme class imbalance.
2. Classification thresholds are selected strictly using validation data.
3. Test sets remain completely unweighted and unmodified.
4. Positive prevalence is reported for every evaluation split and subset.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full suite of classification metrics given true binary labels and predicted probabilities.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) == 0:
        return {
            "total_samples": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_prevalence": 0.0,
            "auprc": 0.0,
            "auroc": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "threshold": threshold,
        }

    total_samples = int(len(y_true_arr))
    pos_count = int(np.sum(y_true_arr == 1))
    neg_count = int(np.sum(y_true_arr == 0))
    prevalence = float(pos_count / total_samples) if total_samples > 0 else 0.0

    # 1. Continuous Ranking Metrics (AUPRC and AUROC)
    # AUPRC (Primary)
    if pos_count == 0 or neg_count == 0:
        auprc = prevalence
        auroc = 0.5
    else:
        try:
            auprc = float(average_precision_score(y_true_arr, y_prob_arr))
        except Exception:
            auprc = prevalence

        try:
            # Handle constant predicted probabilities (e.g. PrevalenceBaseline)
            if np.all(y_prob_arr == y_prob_arr[0]):
                auroc = 0.5
            else:
                auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except Exception:
            auroc = 0.5

    # 2. Discrete Decision Metrics at given threshold
    y_pred = (y_prob_arr >= threshold).astype(int)
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    cm_list = [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]]

    return {
        "total_samples": total_samples,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_prevalence": round(prevalence, 4),
        "auprc": round(auprc, 4),
        "auroc": round(auroc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm_list,
        "threshold": round(threshold, 4),
    }


def find_optimal_threshold(
    y_true_val: Union[np.ndarray, List[int]],
    y_prob_val: Union[np.ndarray, List[float]],
    num_steps: int = 100,
) -> Tuple[float, float]:
    """
    Find F1-maximizing threshold strictly using validation predictions.
    Does not inspect test data or labels.

    Returns:
        (optimal_threshold, best_val_f1)
    """
    y_true_arr = np.asarray(y_true_val, dtype=int)
    y_prob_arr = np.asarray(y_prob_val, dtype=float)

    if len(y_true_arr) == 0 or np.sum(y_true_arr == 1) == 0:
        return 0.5, 0.0

    # Grid search across candidate thresholds combining standard [0.01, 0.99] grid
    # with adaptive candidate points spanning the range of validation probabilities
    grid_thresholds = np.linspace(0.01, 0.99, num_steps)
    prob_min = float(np.min(y_prob_arr))
    prob_max = float(np.max(y_prob_arr))
    if prob_max > prob_min:
        adaptive_thresholds = np.linspace(prob_min, prob_max, num_steps)
        thresholds = np.unique(np.concatenate([grid_thresholds, adaptive_thresholds]))
    else:
        thresholds = grid_thresholds

    best_f1 = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (y_prob_arr >= thresh).astype(int)
        score = f1_score(y_true_arr, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(thresh)

    return round(best_thresh, 4), round(best_f1, 4)


def evaluate_with_subsets(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float,
    package_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate full test set and also evaluation subset excluding 'upgini'.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    full_metrics = compute_classification_metrics(y_true_arr, y_prob_arr, threshold=threshold)

    results = {
        "full": full_metrics,
    }

    if package_names is not None:
        pkg_arr = np.asarray(package_names)
        mask_non_upgini = np.array([p.lower() != "upgini" for p in pkg_arr])

        y_true_sub = y_true_arr[mask_non_upgini]
        y_prob_sub = y_prob_arr[mask_non_upgini]

        sub_metrics = compute_classification_metrics(y_true_sub, y_prob_sub, threshold=threshold)
        results["eval_excluding_upgini"] = sub_metrics

    return results
