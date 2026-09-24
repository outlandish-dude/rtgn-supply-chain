"""
RTGN Training Pipeline Smoke Test.

Executes a small end-to-end forward -> loss -> backward -> optimizer.step() run
on real graph artifacts to verify that:
1. Supervised target mask isolates confirmed_malicious (y=1) and assumed_benign_background (y=0).
2. source_unflagged samples contribute zero supervised gradient.
3. All loss, logits, and parameter gradients are finite.
4. Optimizer step updates weights cleanly without tensor/device/shape issues.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rtgn import RTGNModel
from src.training.loss import PackageBalancedLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rtgn_smoke_test")


def run_smoke_test() -> None:
    logger.info("1. Loading temporal graph and metadata artifacts...")
    data = torch.load("data/experimental/graph/temporal_hetero_graph.pt", weights_only=False)
    feats = torch.load("data/experimental/graph/package_version_features.pt")
    splits = torch.load("data/experimental/graph/split_masks.pt")
    df_nodes = pd.read_csv("data/experimental/graph/nodes_package_version.csv")

    num_packages = data["package"].num_nodes
    num_versions = data["package_version"].num_nodes
    device = torch.device("cpu")

    logger.info(f"   Package nodes: {num_packages}, Version nodes: {num_versions}")
    logger.info(f"   Feature matrix shape: {feats.shape}, device: {feats.device}")

    # 2. Prepare edge indices and time encodings
    edge_index_dict = {
        "has_version": data["package", "has_version", "package_version"].edge_index,
        "version_of": data["package", "has_version", "package_version"].edge_index[[1, 0]],
        "depends_on": data["package_version", "depends_on", "package"].edge_index,
        "depended_by": data["package_version", "depends_on", "package"].edge_index[[1, 0]],
    }
    edge_time_dict = {
        "has_version": data["package", "has_version", "package_version"].edge_time,
        "version_of": data["package", "has_version", "package_version"].edge_time,
        "depends_on": data["package_version", "depends_on", "package"].edge_time,
        "depended_by": data["package_version", "depends_on", "package"].edge_time,
    }

    # 3. Model construction
    logger.info("2. Constructing RTGN model...")
    model = RTGNModel(
        num_packages=num_packages,
        in_feature_dim=feats.shape[1],
        hidden_dim=64,
        time_dim=32,
        heads=4,
        dropout=0.2,
    ).to(device)

    param_stats = model.count_parameters()
    logger.info(f"   Total parameters: {param_stats['total_parameters']:,}")
    logger.info(f"   Trainable parameters: {param_stats['trainable_parameters']:,}")

    query_time = float(data["package", "has_version", "package_version"].edge_time.max().item())
    edge_attrs = model.encode_edge_times(edge_time_dict, query_time)

    # 4. Chronological training split & loss weighting
    logger.info("3. Initializing loss weighting on chronological training fold...")
    train_mask = splits["train_mask_chrono"].numpy()
    train_indices = np.where(train_mask)[0]

    all_targets = torch.tensor(df_nodes["target_label"].values, dtype=torch.float32, device=device)
    all_weights = torch.tensor(df_nodes["sample_weight"].values, dtype=torch.float32, device=device)

    train_labels = all_targets[train_mask]
    train_sample_weights = all_weights[train_mask]

    loss_fn = PackageBalancedLoss().to(device)
    loss_stats = loss_fn.compute_weights(train_labels, train_sample_weights)

    logger.info(
        f"   Full train fold: {loss_stats['raw_pos_count']} pos, {loss_stats['raw_neg_count']} neg, "
        f"{loss_stats['raw_ignored_count']} ignored | Effective pos weight: {loss_stats['effective_pos_weight']:.4f}"
    )

    # 5. Select a small representative smoke batch from train split
    # Sample 16 positives, 32 negatives, and 16 ignored (source_unflagged)
    rng = np.random.RandomState(42)
    pos_idx = train_indices[df_nodes["target_label"].values[train_indices] == 1]
    neg_idx = train_indices[df_nodes["target_label"].values[train_indices] == 0]
    unf_idx = train_indices[df_nodes["target_label"].values[train_indices] == -1]

    sampled_pos = rng.choice(pos_idx, size=16, replace=False)
    sampled_neg = rng.choice(neg_idx, size=32, replace=False)
    sampled_unf = rng.choice(unf_idx, size=16, replace=False)

    batch_indices = np.concatenate([sampled_pos, sampled_neg, sampled_unf])
    rng.shuffle(batch_indices)
    batch_idx_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)

    batch_targets = all_targets[batch_idx_tensor]
    batch_weights = all_weights[batch_idx_tensor]

    n_pos = int((batch_targets == 1).sum().item())
    n_neg = int((batch_targets == 0).sum().item())
    n_unf = int((batch_targets == -1).sum().item())

    logger.info(f"4. Selected smoke batch of size {len(batch_indices)}:")
    logger.info(f"   Supervised positives: {n_pos}")
    logger.info(f"   Supervised negatives: {n_neg}")
    logger.info(f"   Ignored source_unflagged: {n_unf}")

    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 7. Forward Pass
    logger.info("5. Executing forward pass...")
    pkg_ids = torch.arange(num_packages, device=device)
    model.train()
    logits, probs, out_dict = model(feats, pkg_ids, edge_index_dict, edge_attrs)

    batch_logits = logits[batch_idx_tensor]

    assert torch.isfinite(batch_logits).all(), "Logits contain NaN or Inf values!"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]!"

    # 8. Compute Supervised Loss
    logger.info("6. Computing supervised loss...")
    loss = loss_fn(batch_logits, batch_targets, batch_weights)
    initial_loss_val = float(loss.item())

    assert torch.isfinite(loss), f"Loss is non-finite: {initial_loss_val}"
    logger.info(f"   Initial loss: {initial_loss_val:.4f}")

    # 9. Verify zero gradient for source_unflagged samples
    logger.info("7. Verifying gradient isolation for source_unflagged...")
    batch_logit_grads = torch.autograd.grad(loss, batch_logits, retain_graph=True)[0]
    unf_mask = (batch_targets == -1)

    max_unf_grad = float(batch_logit_grads[unf_mask].abs().max().item())
    min_sup_grad = float(batch_logit_grads[~unf_mask].abs().min().item())

    assert max_unf_grad == 0.0, f"source_unflagged received non-zero gradient: {max_unf_grad}!"
    assert min_sup_grad > 0.0, "Supervised samples received zero gradient!"
    logger.info(f"   source_unflagged max gradient: {max_unf_grad:.6e} (EXACT ZERO)")
    logger.info(f"   supervised min gradient: {min_sup_grad:.6e} (NON-ZERO)")

    # 10. Backward pass & check gradients
    logger.info("8. Executing backward pass...")
    optimizer.zero_grad()
    loss.backward()

    max_grad_mag = 0.0
    nan_grad_count = 0
    grad_param_count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_param_count += 1
            if not torch.isfinite(p.grad).all():
                nan_grad_count += 1
            mag = float(p.grad.abs().max().item())
            if mag > max_grad_mag:
                max_grad_mag = mag

    assert nan_grad_count == 0, f"{nan_grad_count} parameters had non-finite gradients!"
    assert grad_param_count > 0, "No parameter gradients computed!"
    assert max_grad_mag > 0.0, "All parameter gradients are zero!"
    logger.info(f"   Parameters with gradient: {grad_param_count}")
    logger.info(f"   Maximum parameter gradient magnitude: {max_grad_mag:.4e}")

    # 11. Optimizer step
    logger.info("9. Executing optimizer.step()...")
    optimizer.step()

    # 12. Post-step forward & loss
    model.eval()
    with torch.no_grad():
        post_logits, _, _ = model(feats, pkg_ids, edge_index_dict, edge_attrs)
        post_loss = loss_fn(post_logits[batch_idx_tensor], batch_targets, batch_weights)
        post_loss_val = float(post_loss.item())

    assert torch.isfinite(post_loss), f"Post-step loss is non-finite: {post_loss_val}"
    logger.info(f"   Post-step loss: {post_loss_val:.4f}")

    print("\n" + "=" * 60)
    print("RTGN SMOKE TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tensor/device shapes:")
    print(f"  - Version features: {list(feats.shape)} ({feats.device})")
    print(f"  - Package IDs:      {list(pkg_ids.shape)} ({pkg_ids.device})")
    print(f"  - Output Logits:     {list(logits.shape)} ({logits.device})")
    print(f"  - Batch Logits:      {list(batch_logits.shape)} ({batch_logits.device})")
    print(f"Smoke batch counts:")
    print(f"  - Supervised Positives (y=1): {n_pos}")
    print(f"  - Supervised Negatives (y=0): {n_neg}")
    print(f"  - Ignored Unflagged (y=-1):   {n_unf}")
    print(f"  - Total Smoke Batch Size:     {len(batch_indices)}")
    print(f"Loss & Gradients:")
    print(f"  - Initial Supervised Loss:    {initial_loss_val:.4f}")
    print(f"  - Post-step Supervised Loss:   {post_loss_val:.4f}")
    print(f"  - Max Gradient Magnitude:     {max_grad_mag:.4e}")
    print(f"  - Unflagged Supervised Grad:  {max_unf_grad:.6e} (strictly 0.0)")
    print(f"Model Parameters:")
    print(f"  - Total Parameters:           {param_stats['total_parameters']:,}")
    print(f"  - Trainable Parameters:       {param_stats['trainable_parameters']:,}")
    print("=" * 60)
    print("SMOKE TEST PASSED: Pipeline is ready for full training.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_smoke_test()
