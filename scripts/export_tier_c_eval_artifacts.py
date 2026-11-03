#!/usr/bin/env python3
"""
scripts/export_tier_c_eval_artifacts.py

Executes causal chronological evaluation of the recovered Tier C RTGN checkpoint
on the test partition (test_mask_graph_disjoint) and saves:
1. experiments/rtgn_tier_c/metrics/rtgn_tier_c_metrics.json
2. experiments/rtgn_tier_c/predictions/binary_predictions_rtgn.csv

Non-destructive: Does not modify model, graph, features, or existing baselines.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.rtgn import RTGNModel
from src.training.loss import PackageBalancedLoss
from src.training.rtgn_trainer import RTGNTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_tier_c")


def main() -> int:
    ckpt_path = Path("experiments/rtgn_tier_c/checkpoints/best_rtgn_model.pt")
    if not ckpt_path.is_file():
        logger.error(f"Checkpoint not found at {ckpt_path.resolve()}")
        return 1

    out_metrics_dir = Path("experiments/rtgn_tier_c/metrics")
    out_preds_dir = Path("experiments/rtgn_tier_c/predictions")
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_preds_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    logger.info(f"Loading checkpoint from {ckpt_path} on {device}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    threshold = float(ckpt.get("threshold", 0.1882))

    data = torch.load("data/experimental/graph/temporal_hetero_graph.pt", weights_only=False)
    feats = torch.load("data/experimental/graph/package_version_features.pt")
    df_nodes = pd.read_csv("data/experimental/graph/nodes_package_version.csv")
    num_packages = data["package"].num_nodes

    model = RTGNModel(
        num_packages=num_packages,
        in_feature_dim=feats.shape[1],
        hidden_dim=64,
        time_dim=32,
        heads=4,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    trainer = RTGNTrainer(
        model=model,
        loss_fn=PackageBalancedLoss(),
        optimizer=torch.optim.AdamW(model.parameters()),
        device=device,
    )
    trainer.setup_data(train_split_key="train_mask_graph_disjoint")

    logger.info("Evaluating test_mask_graph_disjoint with cold-start initial state...")
    trainer.model.package_memory.reset_memory()

    # Chronological evaluation
    split_key = "test_mask_graph_disjoint"
    event_groups = trainer.get_event_groups(split_key)
    logger.info(f"Test split contains {len(event_groups)} causal event groups.")

    all_preds: List[float] = []
    all_targets: List[float] = []
    all_indices: List[int] = []
    is_cold_list: List[bool] = []

    with torch.no_grad():
        for i, g in enumerate(event_groups):
            t = g["time"]
            group_tensor = g["group_tensor"]
            group_pkgs = g["group_pkgs"]
            group_y = g["group_y_np"]

            cold_status = trainer.model.package_memory.is_cold_start(group_pkgs).cpu().numpy()

            edge_indices, edge_times = trainer.get_causal_edges(t)
            edge_attrs = trainer.model.encode_edge_times(edge_times, query_time=t)

            logits, probs, out_dict = trainer.model(
                trainer.version_features,
                trainer.all_package_ids,
                edge_indices,
                edge_attrs,
            )

            target_probs = probs[group_tensor].cpu().numpy()
            target_reps = out_dict["package_version"][group_tensor]

            all_preds.extend(target_probs.tolist())
            all_targets.extend(group_y.tolist())
            all_indices.extend(g["group_indices_list"])
            is_cold_list.extend(cold_status.tolist())

            # Dynamic memory update
            trainer.model.package_memory.update_memory(group_pkgs, target_reps.detach())

            if (i + 1) % 500 == 0 or (i + 1) == len(event_groups):
                logger.info(f"Progress: {i + 1}/{len(event_groups)} events evaluated.")

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    is_cold_arr = np.array(is_cold_list)

    # Supervised mask
    supervised_mask = (all_targets_arr == 0) | (all_targets_arr == 1)
    y_eval = all_targets_arr[supervised_mask]
    p_eval = all_preds_arr[supervised_mask]
    eval_indices = np.array(all_indices)[supervised_mask]

    from src.evaluation.metrics import compute_classification_metrics
    metrics = compute_classification_metrics(y_eval, p_eval, threshold=threshold)
    metrics["total_evaluated_events"] = len(all_targets_arr)
    metrics["ignored_unflagged_count"] = int((all_targets_arr == -1).sum())

    # Cold start and online adaptation breakdowns
    cold_sup_mask = supervised_mask & is_cold_arr
    adapt_sup_mask = supervised_mask & (~is_cold_arr)
    metrics["cold_start"] = compute_classification_metrics(
        all_targets_arr[cold_sup_mask],
        all_preds_arr[cold_sup_mask],
        threshold=threshold,
    ) if cold_sup_mask.any() else {}
    metrics["online_adaptation"] = compute_classification_metrics(
        all_targets_arr[adapt_sup_mask],
        all_preds_arr[adapt_sup_mask],
        threshold=threshold,
    ) if adapt_sup_mask.any() else {}

    # Save metrics JSON
    metrics_path = out_metrics_dir / "rtgn_tier_c_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": "tier_c",
            "checkpoint": str(ckpt_path),
            "best_epoch": ckpt.get("epoch", 4),
            "val_auprc": ckpt.get("val_auprc", 0.5905),
            "optimal_threshold": threshold,
            "test_metrics": metrics,
        }, f, indent=2)
    logger.info(f"Saved Tier C metrics JSON to {metrics_path}")

    # Save binary predictions CSV
    preds_csv_path = out_preds_dir / "binary_predictions_rtgn.csv"
    pred_labels = (p_eval >= threshold).astype(int)

    rows = []
    for idx, p_prob, p_bin, y_bin in zip(eval_indices, p_eval, pred_labels, y_eval):
        node_row = df_nodes.loc[idx]
        rows.append({
            "version_id": int(node_row["version_id"]),
            "package_name": str(node_row["package_name"]),
            "version": str(node_row["version"]),
            "release_timestamp": float(node_row["release_timestamp"]),
            "split": "test",
            "true_label_str": str(node_row["label_str"]),
            "true_label_binary": int(y_bin),
            "predicted_probability": round(float(p_prob), 6),
            "predicted_label_at_threshold": int(p_bin),
        })

    with open(preds_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "version_id", "package_name", "version", "release_timestamp",
            "split", "true_label_str", "true_label_binary",
            "predicted_probability", "predicted_label_at_threshold"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} binary test predictions to {preds_csv_path}")

    print("\n" + "=" * 60)
    print("RTGN TIER C TEST EVALUATION SUMMARY:")
    print(f"  Test AUPRC   : {metrics['auprc']:.4f}")
    print(f"  Test AUROC   : {metrics['auroc']:.4f}")
    print(f"  Test F1      : {metrics['f1']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  Threshold    : {threshold:.4f}")
    print(f"  Confusion Mat: {metrics['confusion_matrix']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
