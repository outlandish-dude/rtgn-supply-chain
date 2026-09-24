"""
Diagnostic Script to Inspect Raw Prediction Probabilities and Metric Behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rtgn import RTGNModel
from src.training.loss import PackageBalancedLoss
from src.training.rtgn_trainer import RTGNTrainer


def analyze_split(name: str, y_eval: np.ndarray, p_eval: np.ndarray) -> None:
    print(f"\n{'='*20} {name} PREDICTION DISTRIBUTION {'='*20}")
    print(f"Total supervised samples: {len(y_eval)}")
    pos_mask = (y_eval == 1)
    neg_mask = (y_eval == 0)

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    print(f"Positives (y=1): {n_pos}, Negatives (y=0): {n_neg}")

    p_pos = p_eval[pos_mask]
    p_neg = p_eval[neg_mask]

    print(f"Overall Probabilities:")
    print(f"  Min:    {p_eval.min():.6f}")
    print(f"  Max:    {p_eval.max():.6f}")
    print(f"  Mean:   {p_eval.mean():.6f}")
    print(f"  Median: {np.median(p_eval):.6f}")

    print(f"Positive Scores:")
    print(f"  Mean:   {p_pos.mean():.6f}" if n_pos > 0 else "  Mean: N/A")
    print(f"  Median: {np.median(p_pos):.6f}" if n_pos > 0 else "  Median: N/A")

    print(f"Negative Scores:")
    print(f"  Mean:   {p_neg.mean():.6f}" if n_neg > 0 else "  Mean: N/A")
    print(f"  Median: {np.median(p_neg):.6f}" if n_neg > 0 else "  Median: N/A")

    thresholds = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    print(f"\nFraction Above Thresholds:")
    print(f"{'Threshold':<12} | {'Positives Above':<18} | {'Negatives Above':<18}")
    print("-" * 54)
    for th in thresholds:
        frac_pos = float((p_pos >= th).mean()) if n_pos > 0 else 0.0
        frac_neg = float((p_neg >= th).mean()) if n_neg > 0 else 0.0
        print(f"{th:<12.2f} | {frac_pos:<18.4f} | {frac_neg:<18.4f}")

    # Ranking of positive examples (1 is highest score, N is lowest)
    # Higher rank (smaller rank index) means predicted more malicious
    sort_order = np.argsort(-p_eval)  # descending order
    ranks = np.empty_like(sort_order)
    ranks[sort_order] = np.arange(1, len(p_eval) + 1)
    pos_ranks = ranks[pos_mask]
    print(f"\nPositive Examples Rank Positions (1 = highest predicted score, {len(p_eval)} = lowest):")
    print(f"  Ranks of positives: {sorted(pos_ranks.tolist())}")
    print(f"  Mean rank of positives: {pos_ranks.mean():.1f} / {len(p_eval)}")
    print(f"  Mean rank of negatives: {ranks[neg_mask].mean():.1f} / {len(p_eval)}")

    # Sklearn direct computation
    try:
        auroc = roc_auc_score(y_eval, p_eval)
    except Exception as e:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_eval, p_eval)
    except Exception as e:
        auprc = float("nan")

    print(f"\nDirect Sklearn Metrics:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  Random Guess AUPRC (prevalence): {n_pos / len(y_eval):.4f}")


def run_diagnostics():
    data = torch.load("data/experimental/graph/temporal_hetero_graph.pt", weights_only=False)
    feats = torch.load("data/experimental/graph/package_version_features.pt")
    model = RTGNModel(num_packages=data["package"].num_nodes, in_feature_dim=13, hidden_dim=64, time_dim=32, heads=4)
    loss_fn = PackageBalancedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = RTGNTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    trainer.setup_data()
    trainer.load_checkpoint("experiments/rtgn_trial/checkpoints/best_rtgn_model.pt")

    # Collect raw validation predictions
    trainer.model.eval()
    val_targets = trainer.ordered_indices["val_mask_chrono"][:75]
    val_y = []
    val_p = []
    with torch.no_grad():
        for idx in val_targets:
            t = float(trainer.df_nodes.loc[idx, "release_timestamp"])
            e_idx, e_t = trainer.get_causal_edges(t)
            e_attr = trainer.model.encode_edge_times(e_t, t)
            l, probs, _ = trainer.model(trainer.version_features, trainer.all_package_ids, e_idx, e_attr)
            val_p.append(float(probs[idx].item()))
            val_y.append(int(trainer.df_nodes.loc[idx, "target_label"]))

    val_y_arr = np.array(val_y)
    val_p_arr = np.array(val_p)
    sup_val = (val_y_arr == 0) | (val_y_arr == 1)
    analyze_split("VALIDATION (75 prefix events)", val_y_arr[sup_val], val_p_arr[sup_val])

    # Collect raw test predictions
    test_targets = trainer.ordered_indices["test_mask_chrono"][:75]
    test_y = []
    test_p = []
    with torch.no_grad():
        for idx in test_targets:
            t = float(trainer.df_nodes.loc[idx, "release_timestamp"])
            e_idx, e_t = trainer.get_causal_edges(t)
            e_attr = trainer.model.encode_edge_times(e_t, t)
            l, probs, _ = trainer.model(trainer.version_features, trainer.all_package_ids, e_idx, e_attr)
            test_p.append(float(probs[idx].item()))
            test_y.append(int(trainer.df_nodes.loc[idx, "target_label"]))

    test_y_arr = np.array(test_y)
    test_p_arr = np.array(test_p)
    sup_test = (test_y_arr == 0) | (test_y_arr == 1)
    analyze_split("TEST (75 prefix events)", test_y_arr[sup_test], test_p_arr[sup_test])


if __name__ == "__main__":
    run_diagnostics()
