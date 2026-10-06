"""
Command-Line Runner for the RTGN Ablation Experiment Framework under Tier C.

Executes:
1. Full RTGN (Reference run results from validated Tier C evaluation)
2. No temporal encoding (HeteroGATv2 operates without edge harmonic time encodings)
3. No graph / message passing (HeteroGATv2 bypassed; version encoder maps directly to detection head)
4. No historical release features (Input restricted to 7 intrinsic/dependency release features)
5. No package balancing (Uniform sample weights w_i = 1.0; no package frequency downweighting)
6. Memory ablation (Cold-start vs. online-adaptation comparison incorporated from tracked evaluation)

Protocol:
- Same random seed = 42
- Same LR = 0.001
- Same epochs = 5
- Same Tier C masks (train_mask_graph_disjoint, val_mask_graph_disjoint, test_mask_graph_disjoint)
- Source_unflagged (-1) strictly excluded from loss, backward gradients, and primary binary metrics
- Validation-only checkpoint selection (highest validation AUPRC)
- Frozen validation threshold applied to test evaluation
- Separate metrics JSON and checkpoint per ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

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
logger = logging.getLogger("run_ablations")

# Reference Full RTGN Tier C benchmark results
REFERENCE_FULL_RTGN: Dict[str, Any] = {
    "ablation": "full_rtgn",
    "description": "Full RTGN (All components: 13 causal features, temporal encoding, HeteroGATv2, package memory, package balancing)",
    "best_epoch": 4,
    "val_auprc": 0.5905,
    "val_auroc": 0.9164,
    "val_f1": 0.7360,
    "frozen_threshold": 0.1882,
    "test_auprc": 0.3067,
    "test_auroc": 0.8144,
    "test_f1": 0.2428,
    "cold_start_auprc": 0.5272,
    "online_adaptation_auprc": 0.3029,
    "test_positives": 130,
    "test_assumed_benign": 1281,
    "test_source_unflagged": 1244,
}


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


def get_ablation_spec(name: str) -> Dict[str, Any]:
    """Retrieve architecture and training hyperparameter specification for each ablation."""
    specs = {
        "no_temporal_encoding": {
            "name": "no_temporal_encoding",
            "description": "No Temporal Encoding (Harmonic edge time encodings removed; GATv2 uses unconditioned topology)",
            "use_time_encoding": False,
            "use_graph": True,
            "in_feature_dim": 13,
            "use_historical_features": True,
            "use_package_balancing": True,
        },
        "no_graph": {
            "name": "no_graph",
            "description": "No Graph Message Passing (HeteroGATv2 bypassed; version encoder maps directly to detection head)",
            "use_time_encoding": True,
            "use_graph": False,
            "in_feature_dim": 13,
            "use_historical_features": True,
            "use_package_balancing": True,
        },
        "no_historical_features": {
            "name": "no_historical_features",
            "description": "No Historical Release Features (Input restricted to 7 intrinsic/dependency features; hist features removed)",
            "use_time_encoding": True,
            "use_graph": True,
            "in_feature_dim": 7,
            "use_historical_features": False,
            "use_package_balancing": True,
        },
        "no_package_balancing": {
            "name": "no_package_balancing",
            "description": "No Package Balancing (Uniform sample weights w_i = 1.0; no package concentration downweighting)",
            "use_time_encoding": True,
            "use_graph": True,
            "in_feature_dim": 13,
            "use_historical_features": True,
            "use_package_balancing": False,
        },
    }
    if name not in specs:
        raise ValueError(f"Unknown ablation name: {name}. Choices: {list(specs.keys())}")
    return specs[name]


def run_single_ablation(
    spec: Dict[str, Any],
    args: argparse.Namespace,
    base_output_dir: Path,
) -> Dict[str, Any]:
    """Execute training and evaluation pipeline for one ablation configuration."""
    set_seed(args.seed)
    ablation_name = spec["name"]
    logger.info("=" * 75)
    logger.info(f"STARTING ABLATION: {ablation_name.upper()}")
    logger.info(f"Description: {spec['description']}")
    logger.info(f"Config: in_dim={spec['in_feature_dim']}, use_time={spec['use_time_encoding']}, "
                f"use_graph={spec['use_graph']}, use_hist={spec['use_historical_features']}, "
                f"use_bal={spec['use_package_balancing']}")
    logger.info("=" * 75)

    ablation_dir = base_output_dir / ablation_name
    ckpt_dir = ablation_dir / "checkpoints"
    metrics_dir = ablation_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load graph metadata
    graph_path = Path("data/experimental/graph/temporal_hetero_graph.pt")
    data = torch.load(graph_path, weights_only=False)
    num_packages = data["package"].num_nodes

    # 2. Instantiate Model
    model = RTGNModel(
        num_packages=num_packages,
        in_feature_dim=spec["in_feature_dim"],
        hidden_dim=64,
        time_dim=32,
        heads=4,
        dropout=0.2,
        use_time_encoding=spec["use_time_encoding"],
        use_graph=spec["use_graph"],
    )
    param_counts = model.count_parameters()
    logger.info(f"Model initialized: {param_counts['total_parameters']:,} parameters.")

    # 3. Instantiate Loss and Optimizer
    loss_fn = PackageBalancedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    config = {
        "ablation_name": ablation_name,
        "ablation_spec": spec,
        "mode": args.mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
        "max_train_events": args.max_train_events,
        "max_val_events": args.max_val_events,
        "max_test_events": args.max_test_events,
    }

    trainer = RTGNTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=args.device,
        checkpoint_dir=ckpt_dir,
        use_historical_features=spec["use_historical_features"],
        use_package_balancing=spec["use_package_balancing"],
    )

    # Setup causal buffers and fold indices
    trainer.setup_data(train_split_key="train_mask_graph_disjoint")

    train_split = "train_mask_graph_disjoint"
    val_split = "val_mask_graph_disjoint"
    test_split = "test_mask_graph_disjoint"

    history = []

    # 4. Chronological Training Loop
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- [{ablation_name}] Epoch {epoch}/{args.epochs} ---")
        trainer.model.package_memory.reset_memory()

        train_stats = trainer.train_epoch(
            split_key=train_split,
            max_events=args.max_train_events,
            epoch_num=epoch,
        )
        logger.info(f"Epoch {epoch} Train: Loss={train_stats['train_loss']:.4f} "
                    f"({train_stats['supervised_events']}/{train_stats['total_events']} events)")

        val_metrics = trainer.evaluate_split(
            split_key=val_split,
            max_events=args.max_val_events,
            reset_memory=False,
            is_tier_c=True,
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

        if val_auprc > trainer.best_val_auprc:
            trainer.best_val_auprc = val_auprc
            trainer.best_epoch = epoch
            trainer.optimal_threshold = val_metrics["threshold"]
            trainer.save_checkpoint(epoch, val_auprc, val_metrics["threshold"], filename="best_ablation_model.pt")

        if scheduler is not None:
            scheduler.step()

    # 5. Restore best checkpoint and evaluate test split
    logger.info(f"\n[{ablation_name}] Best Epoch: {trainer.best_epoch} with Val AUPRC={trainer.best_val_auprc:.4f}")
    best_ckpt_path = ckpt_dir / "best_ablation_model.pt"
    trainer.load_checkpoint(best_ckpt_path, reset_memory_to_clean=True)

    # Tier C test unseen packages start strictly cold
    trainer.model.package_memory.reset_memory()

    test_metrics = trainer.evaluate_split(
        split_key=test_split,
        max_events=args.max_test_events,
        reset_memory=False,
        threshold=trainer.optimal_threshold,
        is_tier_c=True,
    )

    logger.info(f"[{ablation_name}] Test: AUPRC={test_metrics['auprc']:.4f}, AUROC={test_metrics['auroc']:.4f}, "
                f"F1={test_metrics['f1']:.4f} (Threshold={test_metrics['threshold']:.4f})")

    # Export individual ablation results
    results_payload = {
        "ablation_name": ablation_name,
        "ablation_spec": spec,
        "config": config,
        "history": history,
        "best_epoch": trainer.best_epoch,
        "best_val_auprc": trainer.best_val_auprc,
        "optimal_threshold": trainer.optimal_threshold,
        "test_metrics": test_metrics,
    }
    metrics_path = metrics_dir / "ablation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info(f"Saved ablation metrics to {metrics_path}")

    return {
        "ablation": ablation_name,
        "description": spec["description"],
        "best_epoch": trainer.best_epoch,
        "val_auprc": trainer.best_val_auprc,
        "optimal_threshold": trainer.optimal_threshold,
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_f1": test_metrics["f1"],
        "cold_start_auprc": test_metrics.get("cold_start", {}).get("auprc", None),
        "online_adaptation_auprc": test_metrics.get("online_adaptation", {}).get("auprc", None),
    }


def generate_ablation_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate Markdown and JSON summary tables of all ablations and comparisons."""
    summary_path = output_dir / "ablation_summary.json"
    md_path = output_dir / "ablation_summary.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown table formatting
    rows = []
    rows.append("# RTGN Tier C Ablation Study Results\n")
    rows.append("> **Benchmark:** Tier C (Unseen Package / Graph-Disjoint Partition)  ")
    rows.append("> **Evaluation Protocol:** Strictly chronological causal filtering, per-target edge snapshot, validation-frozen threshold $\\tau^*$, source_unflagged excluded.  ")
    rows.append("> **Random Seed:** 42 | **Epochs:** 5 | **Learning Rate:** 0.001\n")
    rows.append("## Ablation Comparison Matrix\n")
    rows.append("| Model Variant / Ablation | Val AUPRC | Frozen $\\tau^*$ | Test AUPRC | Test AUROC | Test F1 | $\\Delta$ AUPRC (vs Full) |")
    rows.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    ref_auprc = REFERENCE_FULL_RTGN["test_auprc"]

    for r in results:
        delta = r["test_auprc"] - ref_auprc if r.get("test_auprc") is not None else 0.0
        delta_str = f"{delta:+.4f}" if r["ablation"] != "full_rtgn" else "Reference"
        val_auprc_str = f"{r['val_auprc']:.4f}" if r.get("val_auprc") is not None else "N/A"
        thresh_str = f"{r['optimal_threshold']:.4f}" if r.get("optimal_threshold") is not None else "N/A"
        test_auprc_str = f"{r['test_auprc']:.4f}" if r.get("test_auprc") is not None else "N/A"
        test_auroc_str = f"{r['test_auroc']:.4f}" if r.get("test_auroc") is not None else "N/A"
        test_f1_str = f"{r['test_f1']:.4f}" if r.get("test_f1") is not None else "N/A"

        rows.append(f"| **{r['ablation']}** | {val_auprc_str} | {thresh_str} | {test_auprc_str} | {test_auroc_str} | {test_f1_str} | {delta_str} |")

    rows.append("\n## Memory Ablation (Cold-Start vs. Online-Adaptation)\n")
    rows.append("| Memory Protocol | Test AUPRC | Protocol Description |")
    rows.append("| :--- | :---: | :--- |")
    rows.append(f"| **Cold-Start** | {REFERENCE_FULL_RTGN['cold_start_auprc']:.4f} | Zero historical memory for unseen packages ($h_0$ at release 1) |")
    rows.append(f"| **Online-Adaptation** | {REFERENCE_FULL_RTGN['online_adaptation_auprc']:.4f} | Dynamic memory updated post-prediction across package releases |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    logger.info(f"Ablation summary exported to {summary_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTGN Ablation Study under Tier C")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all", "no_temporal_encoding", "no_graph", "no_historical_features", "no_package_balancing"],
                        help="Specific ablation to run, or 'all'")
    parser.add_argument("--mode", type=str, default="tier_c", choices=["tier_c"],
                        help="Evaluation mode (fixed to tier_c)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Require CUDA device; raises error if CUDA is not available")
    parser.add_argument("--output-dir", type=str, default="experiments/ablations_tier_c",
                        help="Output directory (e.g. Google Drive path or local folder)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Prefix limit of total events for short smoke testing across splits")
    parser.add_argument("--max-train-events", type=int, default=None,
                        help="Prefix limit of training events (overrides max-events if set)")
    parser.add_argument("--max-val-events", type=int, default=None,
                        help="Prefix limit of validation events (overrides max-events if set)")
    parser.add_argument("--max-test-events", type=int, default=None,
                        help="Prefix limit of test events (overrides max-events if set)")

    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by --require-cuda, but torch.cuda.is_available() is False.")

    if args.max_events is not None:
        if args.max_train_events is None:
            args.max_train_events = args.max_events
        if args.max_val_events is None:
            args.max_val_events = max(100, args.max_events // 2)
        if args.max_test_events is None:
            args.max_test_events = max(100, args.max_events // 2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_keep_awake(True)
    try:
        all_ablation_names = ["no_temporal_encoding", "no_graph", "no_historical_features", "no_package_balancing"]
        to_run = all_ablation_names if args.ablation == "all" else [args.ablation]

        # Start summary with Reference Full RTGN
        summary_records = [{
            "ablation": REFERENCE_FULL_RTGN["ablation"],
            "description": REFERENCE_FULL_RTGN["description"],
            "best_epoch": REFERENCE_FULL_RTGN["best_epoch"],
            "val_auprc": REFERENCE_FULL_RTGN["val_auprc"],
            "optimal_threshold": REFERENCE_FULL_RTGN["frozen_threshold"],
            "test_auprc": REFERENCE_FULL_RTGN["test_auprc"],
            "test_auroc": REFERENCE_FULL_RTGN["test_auroc"],
            "test_f1": REFERENCE_FULL_RTGN["test_f1"],
            "cold_start_auprc": REFERENCE_FULL_RTGN["cold_start_auprc"],
            "online_adaptation_auprc": REFERENCE_FULL_RTGN["online_adaptation_auprc"],
        }]

        for ab_name in to_run:
            spec = get_ablation_spec(ab_name)
            record = run_single_ablation(spec, args, output_dir)
            summary_records.append(record)

        generate_ablation_report(summary_records, output_dir)
        logger.info("\nALL REQUESTED ABLATIONS COMPLETED SUCCESSFULLY.")
    finally:
        set_keep_awake(False)


if __name__ == "__main__":
    main()
