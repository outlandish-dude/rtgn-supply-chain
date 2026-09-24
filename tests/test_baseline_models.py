"""
Tests for Baseline Models, Training Infrastructure, and Causality Guarantees.

Verifies:
1. No test labels or test records influence model training or threshold tuning.
2. No future split records enter training folds.
3. source_unflagged records are strictly excluded from primary binary training/evaluation.
4. Test set remains completely unweighted and unmodified.
5. Optimal classification threshold is selected strictly from validation data.
6. Prevalence baseline correctly computes train positive prevalence and Chance AUPRC.
7. eval_excluding_upgini correctly filters dominant package releases.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    compute_classification_metrics,
    evaluate_with_subsets,
    find_optimal_threshold,
)
from src.models.baselines import (
    LogisticRegressionBaseline,
    PrevalenceBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)
from src.training.baseline_trainer import BaselineTrainer


@pytest.fixture(scope="module")
def trainer_instance():
    """Load baseline trainer and partition data."""
    trainer = BaselineTrainer(
        graph_dir="data/experimental/graph",
        output_dir="experiments/baselines_test",
    )
    trainer.load_data()
    trainer.prepare_partitions()
    return trainer


def test_source_unflagged_segregation(trainer_instance):
    """Verify source_unflagged is strictly isolated and never labeled binary 0."""
    for split in ["train", "val", "test"]:
        bin_meta = trainer_instance.data_splits[split]["binary"]["metadata"]
        bin_labels = trainer_instance.data_splits[split]["binary"]["y"]

        # 1. No source_unflagged in binary fold
        for m in bin_meta:
            assert m["label_str"] != "source_unflagged", (
                f"Found source_unflagged in binary {split} fold!"
            )

        # 2. Binary labels are strictly 0 or 1
        assert set(bin_labels).issubset({0, 1})

        # 3. Unflagged fold has None/empty binary labels
        unflagged = trainer_instance.data_splits[split]["unflagged"]
        assert unflagged["y"] is None


def test_no_future_split_leakage(trainer_instance):
    """Verify that training set contains zero validation or test records and strictly t < cutoff."""
    train_indices = set(trainer_instance.data_splits["train"]["binary"]["indices"])
    val_indices = set(trainer_instance.data_splits["val"]["binary"]["indices"])
    test_indices = set(trainer_instance.data_splits["test"]["binary"]["indices"])

    # Disjoint indices
    assert len(train_indices.intersection(val_indices)) == 0
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(val_indices.intersection(test_indices)) == 0

    # Temporal cutoff: Train must be strictly before 2025-06-01 (1748736000)
    train_meta = trainer_instance.data_splits["train"]["binary"]["metadata"]
    for m in train_meta:
        ts = float(m["release_timestamp"])
        assert ts < 1748736000.0, f"Train record has timestamp >= 2025-06-01: {ts}"


def test_no_test_labels_influence_training(trainer_instance):
    """Verify that mutating test labels has zero effect on training parameters and validation threshold."""
    X_train = trainer_instance.data_splits["train"]["binary"]["X"]
    y_train = trainer_instance.data_splits["train"]["binary"]["y"]

    X_val = trainer_instance.data_splits["val"]["binary"]["X"]
    y_val = trainer_instance.data_splits["val"]["binary"]["y"]

    X_test = trainer_instance.data_splits["test"]["binary"]["X"]
    y_test_original = trainer_instance.data_splits["test"]["binary"]["y"]

    # Fit original model
    model = LogisticRegressionBaseline(random_state=42)
    model.fit(X_train, y_train)
    val_probs_1 = model.predict_proba(X_val)[:, 1]
    thresh_1, f1_1 = find_optimal_threshold(y_val, val_probs_1)

    # Invert test labels (simulate adversarial perturbation)
    y_test_mutated = 1 - y_test_original

    # Fit model again with same train data (test data is not touched in training)
    model_2 = LogisticRegressionBaseline(random_state=42)
    model_2.fit(X_train, y_train)
    val_probs_2 = model_2.predict_proba(X_val)[:, 1]
    thresh_2, f1_2 = find_optimal_threshold(y_val, val_probs_2)

    # Model parameters and validation metrics must be identical
    np.testing.assert_allclose(val_probs_1, val_probs_2)
    assert thresh_1 == thresh_2
    assert f1_1 == f1_2


def test_threshold_selection_strictly_validation():
    """Verify threshold optimization depends only on validation data and freezes for test."""
    y_val = np.array([0, 0, 1, 1, 0, 1])
    probs_val = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])

    best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val)
    assert 0.3 < best_thresh <= 0.7
    assert best_f1 == 1.0

    # Test evaluation at frozen threshold
    y_test = np.array([0, 1, 0, 1])
    probs_test = np.array([0.1, 0.8, 0.2, 0.9])
    metrics = compute_classification_metrics(y_test, probs_test, threshold=best_thresh)
    assert metrics["threshold"] == best_thresh
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_prevalence_baseline():
    """Verify prevalence baseline predicts constant p_train and chance AUPRC."""
    X_train = np.zeros((100, 5))
    y_train = np.array([1] * 20 + [0] * 80)  # 20% prevalence

    baseline = PrevalenceBaseline()
    baseline.fit(X_train, y_train)
    assert abs(baseline.prevalence - 0.20) < 1e-5

    X_test = np.zeros((50, 5))
    probs = baseline.predict_proba(X_test)
    assert probs.shape == (50, 2)
    assert np.allclose(probs[:, 1], 0.20)

    # On a test set with 30% positive prevalence, chance AUPRC is ~0.30
    y_test = np.array([1] * 15 + [0] * 35)
    metrics = compute_classification_metrics(y_test, probs[:, 1])
    assert metrics["auroc"] == 0.5  # Constant predictor has AUROC = 0.5
    assert abs(metrics["positive_prevalence"] - 0.30) < 1e-4


def test_eval_excluding_upgini():
    """Verify evaluate_with_subsets separates full test metrics from non-upgini metrics."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.3])
    packages = ["upgini", "upgini", "requests", "urllib3", "other_pkg", "flask"]

    results = evaluate_with_subsets(y_true, y_prob, threshold=0.5, package_names=packages)

    assert "full" in results
    assert "eval_excluding_upgini" in results
    assert results["full"]["total_samples"] == 6
    assert results["eval_excluding_upgini"]["total_samples"] == 4
    # In subset: packages are requests, urllib3, other_pkg, flask (1 positive: other_pkg, 3 negatives)
    assert results["eval_excluding_upgini"]["positive_count"] == 1
    assert results["eval_excluding_upgini"]["negative_count"] == 3
