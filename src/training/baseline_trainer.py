"""
Baseline Model Trainer and Evaluation Pipeline.

Orchestrates:
1. Loading features, masks, and metadata from data/experimental/graph/.
2. Strict binary supervised filtering:
   - positive: confirmed_malicious (1)
   - negative: assumed_benign_background (0)
   - source_unflagged: true_label_binary is null/NaN, excluded from training and primary metrics.
3. Training models using strictly their single primary imbalance mechanism.
4. Selecting optimal decision thresholds using validation data only.
5. Applying frozen thresholds to test data (both full test and excluding upgini).
6. Exporting predictions, models, metrics JSON/CSV, and threshold records.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
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

logger = logging.getLogger(__name__)


class BaselineTrainer:
    """Trainer and Evaluator for Classical Tabular Baselines."""

    def __init__(
        self,
        graph_dir: Union[str, Path] = "data/experimental/graph",
        output_dir: Union[str, Path] = "experiments/baselines",
        mode: str = "chrono",
    ) -> None:
        self.graph_dir = Path(graph_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode.lower()
        self.models_dir = self.output_dir / "models"
        self.preds_dir = self.output_dir / "predictions"
        self.metrics_dir = self.output_dir / "metrics"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.features: Optional[np.ndarray] = None
        self.split_masks: Optional[Dict[str, np.ndarray]] = None
        self.version_metadata: Optional[List[Dict[str, Any]]] = None

        # Partitions
        self.data_splits: Dict[str, Dict[str, Any]] = {}

    def load_data(self) -> None:
        """Load features, split masks, and node metadata."""
        feat_path = self.graph_dir / "package_version_features.pt"
        masks_path = self.graph_dir / "split_masks.pt"
        meta_path = self.graph_dir / "nodes_package_version.csv"

        logger.info(f"Loading features from {feat_path}...")
        self.features = torch.load(feat_path, weights_only=True).numpy()

        logger.info(f"Loading split masks from {masks_path}...")
        raw_masks = torch.load(masks_path, weights_only=True)
        self.split_masks = {k: v.numpy() for k, v in raw_masks.items()}

        logger.info(f"Loading version metadata from {meta_path}...")
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.version_metadata = list(reader)

        logger.info(f"Loaded {len(self.version_metadata)} total records with features {self.features.shape}.")

    def prepare_partitions(self) -> None:
        """
        Partition records into binary supervised train/val/test and separate unflagged subsets.
        """
        if self.features is None or self.split_masks is None or self.version_metadata is None:
            raise RuntimeError("Data must be loaded before preparing partitions.")

        splits = ["train", "val", "test"]
        if self.mode in ("tier_c", "graph_disjoint"):
            mask_keys = {
                "train": "train_mask_graph_disjoint",
                "val": "val_mask_graph_disjoint",
                "test": "test_mask_graph_disjoint",
            }
        else:
            mask_keys = {
                "train": "train_mask_chrono",
                "val": "val_mask_chrono",
                "test": "test_mask_chrono",
            }

        for split in splits:
            m = self.split_masks[mask_keys[split]]
            split_indices = np.where(m)[0]

            binary_indices: List[int] = []
            binary_labels: List[int] = []
            unflagged_indices: List[int] = []

            for idx in split_indices:
                lbl_str = self.version_metadata[idx]["label_str"]
                if lbl_str == "confirmed_malicious":
                    binary_indices.append(idx)
                    binary_labels.append(1)
                elif lbl_str == "assumed_benign_background":
                    binary_indices.append(idx)
                    binary_labels.append(0)
                elif lbl_str == "source_unflagged":
                    unflagged_indices.append(idx)

            bin_idx_arr = np.array(binary_indices, dtype=int)
            bin_labels_arr = np.array(binary_labels, dtype=int)
            unflagged_idx_arr = np.array(unflagged_indices, dtype=int)

            self.data_splits[split] = {
                "binary": {
                    "indices": bin_idx_arr,
                    "X": self.features[bin_idx_arr],
                    "y": bin_labels_arr,
                    "packages": [self.version_metadata[i]["package_name"] for i in bin_idx_arr],
                    "metadata": [self.version_metadata[i] for i in bin_idx_arr],
                },
                "unflagged": {
                    "indices": unflagged_idx_arr,
                    "X": self.features[unflagged_idx_arr],
                    "y": None,  # Strictly None / NaN
                    "packages": [self.version_metadata[i]["package_name"] for i in unflagged_idx_arr],
                    "metadata": [self.version_metadata[i] for i in unflagged_idx_arr],
                },
            }

            pos = int(np.sum(bin_labels_arr == 1))
            neg = int(np.sum(bin_labels_arr == 0))
            prev = pos / len(bin_labels_arr) if len(bin_labels_arr) > 0 else 0.0
            logger.info(
                f"Split '{split}': binary={len(bin_labels_arr)} (pos={pos}, neg={neg}, "
                f"prevalence={prev:.4f}), unflagged={len(unflagged_idx_arr)}"
            )

    def train_and_evaluate_all(self) -> Dict[str, Any]:
        """Train all baselines, select thresholds on val, and evaluate on test."""
        if not self.data_splits:
            self.prepare_partitions()

        X_train = self.data_splits["train"]["binary"]["X"]
        y_train = self.data_splits["train"]["binary"]["y"]

        X_val = self.data_splits["val"]["binary"]["X"]
        y_val = self.data_splits["val"]["binary"]["y"]

        X_test = self.data_splits["test"]["binary"]["X"]
        y_test = self.data_splits["test"]["binary"]["y"]
        test_pkgs = self.data_splits["test"]["binary"]["packages"]

        pos_train = int(np.sum(y_train == 1))
        neg_train = int(np.sum(y_train == 0))
        pos_val = int(np.sum(y_val == 1))
        neg_val = int(np.sum(y_val == 0))
        pos_test = int(np.sum(y_test == 1))
        neg_test = int(np.sum(y_test == 0))

        unflagged_train = len(self.data_splits["train"]["unflagged"]["indices"])
        unflagged_val = len(self.data_splits["val"]["unflagged"]["indices"])
        unflagged_test = len(self.data_splits["test"]["unflagged"]["indices"])

        models: Dict[str, Any] = {
            "prevalence": PrevalenceBaseline(),
            "logistic_regression": LogisticRegressionBaseline(),
            "random_forest": RandomForestBaseline(),
            "xgboost": XGBoostBaseline(),
        }

        all_metrics: Dict[str, Any] = {}
        all_thresholds: Dict[str, float] = {}
        summary_rows: List[Dict[str, Any]] = []

        for name, model in models.items():
            logger.info(f"--- Training Model: {name} ---")
            model.fit(X_train, y_train)

            # 1. Validation Predictions & Threshold Tuning
            val_probs = model.predict_proba(X_val)[:, 1]
            optimal_thresh, val_best_f1 = find_optimal_threshold(y_val, val_probs)
            all_thresholds[name] = optimal_thresh
            logger.info(f"[{name}] Optimal validation threshold selected: {optimal_thresh} (Val F1: {val_best_f1:.4f})")

            val_metrics = compute_classification_metrics(y_val, val_probs, threshold=optimal_thresh)

            # 2. Test Evaluation at Frozen Threshold
            test_probs = model.predict_proba(X_test)[:, 1]
            test_eval_results = evaluate_with_subsets(
                y_true=y_test,
                y_prob=test_probs,
                threshold=optimal_thresh,
                package_names=test_pkgs,
            )

            # 3. Save Model Artifact
            if name == "prevalence":
                model.save(self.models_dir / "prevalence_baseline.json")
            elif name == "logistic_regression":
                model.save(self.models_dir / "logistic_regression.joblib")
            elif name == "random_forest":
                model.save(self.models_dir / "random_forest.joblib")
            elif name == "xgboost":
                model.save(self.models_dir / "xgboost.json")

            # 4. Save Binary Predictions CSV
            self._save_binary_predictions(name, model, optimal_thresh)

            # 5. Save Qualitative Unflagged Predictions CSV
            self._save_unflagged_predictions(name, model)

            # Collect metrics payload
            all_metrics[name] = {
                "model": name,
                "mode": self.mode,
                "train_counts": {
                    "supervised_positives": pos_train,
                    "supervised_negatives": neg_train,
                    "ignored_source_unflagged": unflagged_train,
                },
                "val_counts": {
                    "supervised_positives": pos_val,
                    "supervised_negatives": neg_val,
                    "ignored_source_unflagged": unflagged_val,
                },
                "test_counts": {
                    "supervised_positives": pos_test,
                    "supervised_negatives": neg_test,
                    "ignored_source_unflagged": unflagged_test,
                },
                "validation_threshold": optimal_thresh,
                "validation_metrics": val_metrics,
                "test_metrics_full": test_eval_results["full"],
                "test_metrics_excluding_upgini": test_eval_results.get("eval_excluding_upgini", {}),
            }

            summary_rows.append({
                "model": name,
                "mode": self.mode,
                "train_pos": pos_train,
                "train_neg": neg_train,
                "val_pos": pos_val,
                "val_neg": neg_val,
                "test_pos": pos_test,
                "test_neg": neg_test,
                "val_threshold": optimal_thresh,
                "val_auprc": val_metrics["auprc"],
                "val_auroc": val_metrics["auroc"],
                "val_f1": val_metrics["f1"],
                "test_prevalence": test_eval_results["full"]["positive_prevalence"],
                "test_auprc": test_eval_results["full"]["auprc"],
                "test_auroc": test_eval_results["full"]["auroc"],
                "test_precision": test_eval_results["full"]["precision"],
                "test_recall": test_eval_results["full"]["recall"],
                "test_f1": test_eval_results["full"]["f1"],
                "test_no_upgini_auprc": test_eval_results.get("eval_excluding_upgini", {}).get("auprc", 0.0),
                "test_no_upgini_f1": test_eval_results.get("eval_excluding_upgini", {}).get("f1", 0.0),
            })

        # Save Metrics JSON
        with open(self.metrics_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        # Save Validation Thresholds JSON
        with open(self.metrics_dir / "validation_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(all_thresholds, f, indent=2)

        # Save Summary CSV
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.metrics_dir / "baseline_metrics_summary.csv", index=False)

        logger.info(f"Baseline training complete. Metrics saved to {self.metrics_dir}.")
        return all_metrics

    def _save_binary_predictions(
        self,
        model_name: str,
        model: Any,
        threshold: float,
    ) -> None:
        """Export binary predictions for train, val, and test splits."""
        records_to_export: List[Dict[str, Any]] = []

        for split in ["train", "val", "test"]:
            data = self.data_splits[split]["binary"]
            X = data["X"]
            y = data["y"]
            meta = data["metadata"]
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)

            for idx, r in enumerate(meta):
                records_to_export.append({
                    "version_id": r["version_id"],
                    "package_name": r["package_name"],
                    "version": r["version"],
                    "release_timestamp": r["release_timestamp"],
                    "split": split,
                    "true_label_str": r["label_str"],
                    "true_label_binary": int(y[idx]),
                    "predicted_probability": round(float(probs[idx]), 6),
                    "predicted_label_at_threshold": int(preds[idx]),
                })

        out_path = self.preds_dir / f"binary_predictions_{model_name}.csv"
        df = pd.DataFrame(records_to_export)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved binary predictions to {out_path}.")

    def _save_unflagged_predictions(
        self,
        model_name: str,
        model: Any,
    ) -> None:
        """Export qualitative/sensitivity prediction probabilities for source_unflagged releases."""
        records_to_export: List[Dict[str, Any]] = []

        for split in ["train", "val", "test"]:
            data = self.data_splits[split]["unflagged"]
            X = data["X"]
            meta = data["metadata"]
            if len(X) == 0:
                continue
            probs = model.predict_proba(X)[:, 1]

            for idx, r in enumerate(meta):
                records_to_export.append({
                    "version_id": r["version_id"],
                    "package_name": r["package_name"],
                    "version": r["version"],
                    "release_timestamp": r["release_timestamp"],
                    "split": split,
                    "true_label_str": r["label_str"],
                    "true_label_binary": "",  # Strictly empty / null
                    "predicted_probability": round(float(probs[idx]), 6),
                })

        out_path = self.preds_dir / f"unflagged_qualitative_predictions_{model_name}.csv"
        df = pd.DataFrame(records_to_export)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved qualitative unflagged predictions to {out_path}.")
