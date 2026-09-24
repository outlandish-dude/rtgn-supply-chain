"""
Command-Line Runner for Real-Time Temporal Graph Neural Network (RTGN).

Executes:
1. Chronological event-by-event causal training with per-target edge filtering.
2. Read-before-write dynamic package memory updates.
3. Validation-based model selection and threshold freezing.
4. Tier A (chronological forecasting) and Tier C (unseen-package generalization).
5. Comprehensive metrics logging and checkpoint management.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rtgn import RTGNModel
from src.training.loss import PackageBalancedLoss
from src.training.rtgn_trainer import RTGNTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_rtgn")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_keep_awake(enable: bool = True) -> None:
    """Prevent Windows OS from sleeping during intensive model training."""
    if sys.platform == "win32":
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            if enable:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
                logger.info("Windows sleep prevention activated (SetThreadExecutionState).")
            else:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                logger.info("Windows sleep prevention cleared.")
        except Exception as e:
            logger.warning(f"Could not configure sleep prevention: {e}")


def train_pipeline() -> None:
    parser = argparse.ArgumentParser(description="Train and Evaluate RTGN Model")
    parser.add_argument("--mode", type=str, default="tier_a", choices=["tier_a", "tier_c"],
                        help="Benchmark evaluation tier: tier_a (chronological) or tier_c (package-disjoint)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--max-train-events", type=int, default=None,
                        help="Prefix limit of training events for short controlled trials (None for full training)")
    parser.add_argument("--max-val-events", type=int, default=None,
                        help="Prefix limit of validation events for short controlled trials (None for full validation)")
    parser.add_argument("--max-test-events", type=int, default=None,
                        help="Prefix limit of test events for short controlled trials (None for full test)")
    parser.add_argument("--output-dir", type=str, default="experiments/rtgn", help="Directory to save checkpoints and metrics")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RTGN TRAINING PIPELINE START")
    logger.info(f"Mode: {args.mode} | Epochs: {args.epochs} | LR: {args.lr} | Seed: {args.seed} | Device: {args.device}")
    if args.max_train_events is not None:
        logger.info(f"CONTROLLED TRIAL ACTIVE: max_train_events={args.max_train_events}, max_val_events={args.max_val_events}")
    logger.info("=" * 70)

    # 1. Load graph data to inspect package count
    graph_path = Path("data/experimental/graph/temporal_hetero_graph.pt")
    feats_path = Path("data/experimental/graph/package_version_features.pt")
    data = torch.load(graph_path, weights_only=False)
    feats = torch.load(feats_path)

    num_packages = data["package"].num_nodes
    in_feature_dim = feats.shape[1]

    # 2. Build RTGN Model (0 static package-ID embeddings, 2-layer HeteroGATv2, 64 hidden dim)
    model = RTGNModel(
        num_packages=num_packages,
        in_feature_dim=in_feature_dim,
        hidden_dim=64,
        time_dim=32,
        heads=4,
        dropout=0.2,
    )
    param_counts = model.count_parameters()
    logger.info(f"Initialized RTGNModel with {param_counts['total_parameters']:,} parameters (0 static ID tables).")

    # 3. Build Loss and Optimizer
    loss_fn = PackageBalancedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    config = vars(args)

    trainer = RTGNTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=args.device,
        checkpoint_dir=ckpt_dir,
    )

    trainer.setup_data()

    # Determine split keys based on evaluation tier
    if args.mode == "tier_a":
        train_split = "train_mask_chrono"
        val_split = "val_mask_chrono"
        test_split = "test_mask_chrono"
        is_tier_c = False
    else:
        train_split = "train_mask_graph_disjoint"
        val_split = "val_mask_graph_disjoint"
        test_split = "test_mask_graph_disjoint"
        is_tier_c = True

    history = []

    # 4. Chronological Training Loop
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")
        # In Tier A, reset memory at epoch start for clean replay of train fold
        trainer.model.package_memory.reset_memory()

        train_stats = trainer.train_epoch(split_key=train_split, max_events=args.max_train_events, epoch_num=epoch)
        logger.info(f"Epoch {epoch} Train Complete: Loss={train_stats['train_loss']:.4f} "
                    f"({train_stats['supervised_events']}/{train_stats['total_events']} events)")

        # Validation (processed chronologically starting from post-train memory state)
        val_metrics = trainer.evaluate_split(
            split_key=val_split,
            max_events=args.max_val_events,
            reset_memory=False,  # memory carries forward causally
            is_tier_c=is_tier_c,
        )
        logger.info(f"Epoch {epoch} Val: AUPRC={val_metrics['auprc']:.4f}, AUROC={val_metrics['auroc']:.4f}, "
                    f"F1={val_metrics['f1']:.4f} (Thresh={val_metrics['threshold']:.4f}, "
                    f"Pos={val_metrics['positive_count']}, Neg={val_metrics['negative_count']})")

        val_auprc = val_metrics["auprc"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_auprc": val_auprc,
            "val_auroc": val_metrics["auroc"],
            "val_f1": val_metrics["f1"],
            "optimal_threshold": val_metrics["threshold"],
        }
        history.append(epoch_record)

        # Checkpoint selection on validation AUPRC
        if val_auprc > trainer.best_val_auprc:
            trainer.best_val_auprc = val_auprc
            trainer.best_epoch = epoch
            trainer.optimal_threshold = val_metrics["threshold"]
            trainer.save_checkpoint(epoch, val_auprc, val_metrics["threshold"], filename="best_rtgn_model.pt")

        if scheduler is not None:
            scheduler.step()

    # 5. Final Evaluation on Test Set using Best Checkpoint
    logger.info("\n" + "=" * 70)
    logger.info(f"TRAINING COMPLETE. Best Epoch={trainer.best_epoch} with Val AUPRC={trainer.best_val_auprc:.4f}")
    logger.info(f"Loading best checkpoint from {ckpt_dir / 'best_rtgn_model.pt'}...")
    trainer.load_checkpoint(ckpt_dir / "best_rtgn_model.pt", reset_memory_to_clean=True)

    if not is_tier_c:
        # Tier A: Reconstruct causal memory from scratch under best restored model
        logger.info("Replaying train split memory under restored best checkpoint...")
        trainer.replay_split_memory(train_split, max_events=args.max_train_events)
        logger.info("Replaying val split memory under restored best checkpoint...")
        trainer.replay_split_memory(val_split, max_events=args.max_val_events)
    else:
        # Tier C: Unseen test packages strictly start in cold-start state
        trainer.model.package_memory.reset_memory()

    # Evaluate test split with frozen optimal threshold
    logger.info(f"Evaluating {test_split} using frozen threshold {trainer.optimal_threshold:.4f}...")
    test_metrics = trainer.evaluate_split(
        split_key=test_split,
        max_events=args.max_test_events,
        reset_memory=False,
        threshold=trainer.optimal_threshold,
        is_tier_c=is_tier_c,
    )

    logger.info(f"Test Evaluation: AUPRC={test_metrics['auprc']:.4f}, AUROC={test_metrics['auroc']:.4f}, "
                f"F1={test_metrics['f1']:.4f} (Threshold={test_metrics['threshold']:.4f}, "
                f"Pos={test_metrics['positive_count']}, Neg={test_metrics['negative_count']}, "
                f"Ignored={test_metrics.get('ignored_unflagged_count', 0)})")

    if is_tier_c and "cold_start" in test_metrics:
        logger.info(f"  Tier C Cold-Start: AUPRC={test_metrics['cold_start'].get('auprc', 'N/A')}, "
                    f"F1={test_metrics['cold_start'].get('f1', 'N/A')}")
        logger.info(f"  Tier C Online-Adapt: AUPRC={test_metrics['online_adaptation'].get('auprc', 'N/A')}, "
                    f"F1={test_metrics['online_adaptation'].get('f1', 'N/A')}")

    # Export metrics JSON
    metrics_path = metrics_dir / f"rtgn_{args.mode}_metrics.json"
    results_payload = {
        "mode": args.mode,
        "config": config,
        "history": history,
        "best_epoch": trainer.best_epoch,
        "best_val_auprc": trainer.best_val_auprc,
        "optimal_threshold": trainer.optimal_threshold,
        "test_metrics": test_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info(f"Exported metrics to {metrics_path}")
    logger.info("=" * 70)


def main() -> None:
    set_keep_awake(True)
    try:
        train_pipeline()
    finally:
        set_keep_awake(False)


if __name__ == "__main__":
    main()
