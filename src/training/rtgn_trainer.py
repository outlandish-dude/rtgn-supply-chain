"""
Real-Time Temporal Graph Neural Network (RTGN) Training and Evaluation Runner.

Enforces strict temporal causality and methodological integrity:
1. Strict Chronological Ordering: Events processed in strictly ascending release_timestamp order.
2. Per-Target Causal Edge Filtering: For target at timestamp t, edge snapshot strictly enforces edge_time <= t.
3. Read-Before-Write Memory Protocol: Package memory is read before event prediction; memory update occurs
   only after prediction/loss step using detached representations.
4. Target Masking: confirmed_malicious (1) and assumed_benign_background (0) receive supervised loss.
   source_unflagged (-1) is strictly excluded from loss, backward gradients, and binary evaluation metrics.
5. Checkpointing & Reproducibility: Validated model selection using validation AUPRC, threshold freezing,
   and complete state dictionary persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.evaluation.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
)
from src.models.rtgn import RTGNModel
from src.training.loss import PackageBalancedLoss

logger = logging.getLogger(__name__)


class RTGNTrainer:
    """
    Causally Sound Chronological Trainer and Evaluator for RTGN.
    """

    def __init__(
        self,
        model: RTGNModel,
        loss_fn: PackageBalancedLoss,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Union[str, torch.device] = "cpu",
        checkpoint_dir: Union[str, Path] = "experiments/rtgn/checkpoints",
        max_grad_norm: float = 1.0,
        use_historical_features: bool = True,
        use_package_balancing: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.use_historical_features = use_historical_features
        self.use_package_balancing = use_package_balancing

        self.best_val_auprc: float = -1.0
        self.best_epoch: int = -1
        self.optimal_threshold: float = 0.5

        # Causal graph structures
        self.version_features: Optional[torch.Tensor] = None
        self.all_package_ids: Optional[torch.Tensor] = None
        self.df_nodes: Optional[pd.DataFrame] = None
        self.split_masks: Optional[Dict[str, np.ndarray]] = None

        # Pre-sorted causal edge buffers
        self.sorted_has_ver_index: Optional[torch.Tensor] = None
        self.sorted_has_ver_time: Optional[torch.Tensor] = None
        self.sorted_dep_on_index: Optional[torch.Tensor] = None
        self.sorted_dep_on_time: Optional[torch.Tensor] = None

        # Sorted event indices per split
        self.ordered_indices: Dict[str, np.ndarray] = {}
        self.is_initialized: bool = False

    def setup_data(
        self,
        hetero_graph_path: Union[str, Path] = "data/experimental/graph/temporal_hetero_graph.pt",
        features_path: Union[str, Path] = "data/experimental/graph/package_version_features.pt",
        split_masks_path: Union[str, Path] = "data/experimental/graph/split_masks.pt",
        nodes_csv_path: Union[str, Path] = "data/experimental/graph/nodes_package_version.csv",
        train_split_key: Optional[str] = None,
    ) -> None:
        """
        Load graph artifacts and pre-sort edge buffers and events chronologically.
        """
        logger.info("Loading graph artifacts for RTGNTrainer...")
        data = torch.load(hetero_graph_path, weights_only=False)
        self.version_features = torch.load(features_path).to(self.device)
        raw_masks = torch.load(split_masks_path)
        self.split_masks = {k: v.numpy() for k, v in raw_masks.items()}
        self.df_nodes = pd.read_csv(nodes_csv_path)

        # Handle historical features ablation
        if not self.use_historical_features:
            if hasattr(self.model, "in_feature_dim") and self.model.in_feature_dim == 7:
                self.version_features = self.version_features[:, :7]
                logger.info("Ablation: Historical release features sliced out (in_feature_dim=7).")
            else:
                self.version_features = self.version_features.clone()
                self.version_features[:, 7:] = 0.0
                logger.info("Ablation: Historical release features zero-masked (indices 7..12 = 0).")

        num_packages = data["package"].num_nodes
        self.all_package_ids = torch.arange(num_packages, device=self.device)

        # 1. Pre-sort edges chronologically for O(log E) causal slicing
        has_ver = data["package", "has_version", "package_version"]
        perm_has = torch.argsort(has_ver.edge_time)
        self.sorted_has_ver_index = has_ver.edge_index[:, perm_has].to(self.device)
        self.sorted_has_ver_time = has_ver.edge_time[perm_has].to(self.device)

        dep_on = data["package_version", "depends_on", "package"]
        perm_dep = torch.argsort(dep_on.edge_time)
        self.sorted_dep_on_index = dep_on.edge_index[:, perm_dep].to(self.device)
        self.sorted_dep_on_time = dep_on.edge_time[perm_dep].to(self.device)

        # 2. Sort target events chronologically per split
        timestamps = self.df_nodes["release_timestamp"].values
        chrono_order = np.argsort(timestamps)

        for split_key in ["train_mask_chrono", "val_mask_chrono", "test_mask_chrono",
                          "train_mask_graph_disjoint", "val_mask_graph_disjoint", "test_mask_graph_disjoint"]:
            if split_key in self.split_masks:
                mask = self.split_masks[split_key]
                # Filter sorted chronological order to those in mask
                split_indices = np.array([idx for idx in chrono_order if mask[idx]], dtype=np.int64)
                self.ordered_indices[split_key] = split_indices
                logger.info(f"Split {split_key}: {len(split_indices)} releases chronologically ordered.")

        # 3. Initialize loss weighting on training fold
        split_to_use = train_split_key or "train_mask_chrono"
        train_indices = self.ordered_indices[split_to_use]
        train_labels = torch.tensor(self.df_nodes.loc[train_indices, "target_label"].values, dtype=torch.float32)
        if self.use_package_balancing:
            train_weights = torch.tensor(self.df_nodes.loc[train_indices, "sample_weight"].values, dtype=torch.float32)
        else:
            train_weights = torch.ones_like(train_labels)
            logger.info("Ablation: Package balancing disabled (uniform sample weights = 1.0).")
        self.loss_fn.compute_weights(train_labels, train_weights)

        # 4. Precompute event groups per split for efficient execution
        self.event_groups: Dict[str, List[Dict[str, Any]]] = {}
        for split_key in self.ordered_indices:
            self.event_groups[split_key] = self._build_event_groups(split_key)
            logger.info(f"Split {split_key}: precomputed {len(self.event_groups[split_key])} causal event groups.")

        self.is_initialized = True
        logger.info("RTGNTrainer data and causal buffers initialized successfully.")

    def _build_event_groups(self, split_key: str) -> List[Dict[str, Any]]:
        """Group contiguous releases sharing identical timestamps for causal batching."""
        ordered_targets = self.ordered_indices[split_key]
        total_events = len(ordered_targets)
        groups = []
        i = 0
        while i < total_events:
            curr_idx = ordered_targets[i]
            t = float(self.df_nodes.loc[curr_idx, "release_timestamp"])

            j = i + 1
            while j < total_events and float(self.df_nodes.loc[ordered_targets[j], "release_timestamp"]) == t:
                j += 1

            group_indices = ordered_targets[i:j]
            group_tensor = torch.tensor(group_indices, dtype=torch.long, device=self.device)
            group_pkgs = torch.tensor(self.df_nodes.loc[group_indices, "package_id"].values, dtype=torch.long, device=self.device)
            group_targets = torch.tensor(self.df_nodes.loc[group_indices, "target_label"].values, dtype=torch.float32, device=self.device)
            if self.use_package_balancing:
                group_weights = torch.tensor(self.df_nodes.loc[group_indices, "sample_weight"].values, dtype=torch.float32, device=self.device)
            else:
                group_weights = torch.ones(len(group_indices), dtype=torch.float32, device=self.device)
            has_supervised = bool(((group_targets == 0) | (group_targets == 1)).any().item())
            group_y_np = self.df_nodes.loc[group_indices, "target_label"].values

            groups.append({
                "time": t,
                "group_tensor": group_tensor,
                "group_pkgs": group_pkgs,
                "group_targets": group_targets,
                "group_weights": group_weights,
                "has_supervised": has_supervised,
                "group_indices_list": group_indices.tolist(),
                "group_y_np": group_y_np,
            })
            i = j
        return groups

    def get_event_groups(self, split_key: str) -> List[Dict[str, Any]]:
        """Retrieve precomputed or dynamically constructed event groups."""
        if hasattr(self, "event_groups") and split_key in self.event_groups:
            return self.event_groups[split_key]
        return self._build_event_groups(split_key)

    def get_causal_edges(self, target_time: float) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Construct causal graph snapshot strictly containing edges where edge_time <= target_time.
        """
        t_tensor = torch.tensor(target_time, device=self.device)
        k_has = int(torch.searchsorted(self.sorted_has_ver_time, t_tensor, right=True).item())
        k_dep = int(torch.searchsorted(self.sorted_dep_on_time, t_tensor, right=True).item())

        causal_has_idx = self.sorted_has_ver_index[:, :k_has]
        causal_has_time = self.sorted_has_ver_time[:k_has]

        causal_dep_idx = self.sorted_dep_on_index[:, :k_dep]
        causal_dep_time = self.sorted_dep_on_time[:k_dep]

        edge_index_dict = {
            "has_version": causal_has_idx,
            "version_of": causal_has_idx[[1, 0]] if k_has > 0 else causal_has_idx,
            "depends_on": causal_dep_idx,
            "depended_by": causal_dep_idx[[1, 0]] if k_dep > 0 else causal_dep_idx,
        }

        edge_time_dict = {
            "has_version": causal_has_time,
            "version_of": causal_has_time,
            "depends_on": causal_dep_time,
            "depended_by": causal_dep_time,
        }

        return edge_index_dict, edge_time_dict

    def train_epoch(
        self,
        split_key: str = "train_mask_chrono",
        max_events: Optional[int] = None,
        epoch_num: int = 1,
    ) -> Dict[str, float]:
        """
        Execute one training epoch strictly in chronological event order.

        Protocol per target event:
        1. Query target timestamp t.
        2. Filter causal graph snapshot: edge_time <= t.
        3. Read package memory BEFORE current event update.
        4. Forward pass -> compute logits.
        5. Compute supervised loss (excluding source_unflagged).
        6. Backpropagate and step optimizer.
        7. Update package memory with observed release representation (detached).
        """
        if not self.is_initialized:
            raise RuntimeError("RTGNTrainer must be initialized via setup_data().")

        self.model.train()
        event_groups = self.get_event_groups(split_key)
        if max_events is not None:
            cum_count = 0
            sliced_groups = []
            for g in event_groups:
                sliced_groups.append(g)
                cum_count += len(g["group_indices_list"])
                if cum_count >= max_events:
                    break
            event_groups = sliced_groups

        total_loss = 0.0
        supervised_events = 0
        total_groups = len(event_groups)

        for event_idx, g in enumerate(event_groups):
            t = g["time"]
            group_tensor = g["group_tensor"]
            group_pkgs = g["group_pkgs"]
            group_targets = g["group_targets"]
            group_weights = g["group_weights"]
            has_supervised = g["has_supervised"]

            # 1 & 2. Construct causal snapshot strictly as of t
            edge_indices, edge_times = self.get_causal_edges(t)
            edge_attrs = self.model.encode_edge_times(edge_times, query_time=t)

            if has_supervised:
                # 3 & 4. Forward pass: memory read occurs inside forward before update
                logits, probs, out_dict = self.model(
                    self.version_features,
                    self.all_package_ids,
                    edge_indices,
                    edge_attrs,
                )

                target_logits = logits[group_tensor]
                target_reps = out_dict["package_version"][group_tensor]

                # 5 & 6. Supervised loss & optimization
                supervised_mask = (group_targets == 0) | (group_targets == 1)
                loss = self.loss_fn(target_logits, group_targets, group_weights, supervised_mask=supervised_mask)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                supervised_events += 1

                # 7. Post-prediction dynamic memory update (detached representation)
                with torch.no_grad():
                    self.model.package_memory.update_memory(group_pkgs, target_reps.detach())
            else:
                # Event contains only unflagged nodes: no gradient needed
                with torch.no_grad():
                    logits, probs, out_dict = self.model(
                        self.version_features,
                        self.all_package_ids,
                        edge_indices,
                        edge_attrs,
                    )
                    target_reps = out_dict["package_version"][group_tensor]
                    self.model.package_memory.update_memory(group_pkgs, target_reps.detach())

            if (event_idx + 1) % 1000 == 0 or (event_idx + 1) == total_groups:
                curr_avg = total_loss / supervised_events if supervised_events > 0 else 0.0
                logger.info(f"Epoch {epoch_num} Progress: {event_idx + 1}/{total_groups} events | Avg Loss={curr_avg:.4f}")

        avg_loss = total_loss / supervised_events if supervised_events > 0 else 0.0
        return {
            "train_loss": round(avg_loss, 4),
            "supervised_events": supervised_events,
            "total_events": total_groups,
        }

    def evaluate_split(
        self,
        split_key: str,
        max_events: Optional[int] = None,
        reset_memory: bool = False,
        threshold: Optional[float] = None,
        is_tier_c: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model chronologically on specified split.

        Args:
            split_key: name of split mask (e.g. val_mask_chrono, test_mask_chrono)
            max_events: optional prefix limit for testing
            reset_memory: if True, resets memory to initial state before evaluation.
            threshold: classification threshold (if None and evaluating val, tunes threshold).
            is_tier_c: if True, separates cold-start from online adaptation predictions.
        """
        if not self.is_initialized:
            raise RuntimeError("RTGNTrainer must be initialized via setup_data().")

        self.model.eval()
        if reset_memory:
            self.model.package_memory.reset_memory()

        event_groups = self.get_event_groups(split_key)
        if max_events is not None:
            cum_count = 0
            sliced_groups = []
            for g in event_groups:
                sliced_groups.append(g)
                cum_count += len(g["group_indices_list"])
                if cum_count >= max_events:
                    break
            event_groups = sliced_groups

        all_preds = []
        all_targets = []
        all_indices = []
        is_cold_list = []

        with torch.no_grad():
            for g in event_groups:
                t = g["time"]
                group_tensor = g["group_tensor"]
                group_pkgs = g["group_pkgs"]
                group_y = g["group_y_np"]

                # Track whether package was evaluated under cold start
                cold_status = self.model.package_memory.is_cold_start(group_pkgs).cpu().numpy()

                edge_indices, edge_times = self.get_causal_edges(t)
                edge_attrs = self.model.encode_edge_times(edge_times, query_time=t)

                logits, probs, out_dict = self.model(
                    self.version_features,
                    self.all_package_ids,
                    edge_indices,
                    edge_attrs,
                )

                target_probs = probs[group_tensor].cpu().numpy()
                target_reps = out_dict["package_version"][group_tensor]

                all_preds.extend(target_probs.tolist())
                all_targets.extend(group_y.tolist())
                all_indices.extend(g["group_indices_list"])
                is_cold_list.extend(cold_status.tolist())

                # Post-prediction memory update for online adaptation
                self.model.package_memory.update_memory(group_pkgs, target_reps.detach())

        all_preds_arr = np.array(all_preds)
        all_targets_arr = np.array(all_targets)
        is_cold_arr = np.array(is_cold_list)

        # Exclude source_unflagged from binary evaluation
        supervised_mask = (all_targets_arr == 0) | (all_targets_arr == 1)
        y_eval = all_targets_arr[supervised_mask]
        p_eval = all_preds_arr[supervised_mask]

        # Determine threshold
        if threshold is None:
            eval_threshold, _ = find_optimal_threshold(y_eval, p_eval)
        else:
            eval_threshold = threshold

        metrics = compute_classification_metrics(y_eval, p_eval, threshold=eval_threshold)
        metrics["total_evaluated_events"] = len(all_targets_arr)
        metrics["ignored_unflagged_count"] = int((all_targets_arr == -1).sum())

        if is_tier_c:
            # Report cold-start vs online adaptation metrics separately
            cold_sup_mask = supervised_mask & is_cold_arr
            adapt_sup_mask = supervised_mask & (~is_cold_arr)

            metrics["cold_start"] = compute_classification_metrics(
                all_targets_arr[cold_sup_mask],
                all_preds_arr[cold_sup_mask],
                threshold=eval_threshold,
            ) if cold_sup_mask.any() else {}

            metrics["online_adaptation"] = compute_classification_metrics(
                all_targets_arr[adapt_sup_mask],
                all_preds_arr[adapt_sup_mask],
                threshold=eval_threshold,
            ) if adapt_sup_mask.any() else {}

        return metrics

    def replay_split_memory(self, split_key: str, max_events: Optional[int] = None) -> None:
        """
        Causally replay a split's event history to evolve package memory under current model parameters.
        Runs with torch.no_grad(); does not calculate loss or update optimizer.
        """
        if not self.is_initialized:
            raise RuntimeError("RTGNTrainer must be initialized via setup_data().")

        self.model.eval()
        event_groups = self.get_event_groups(split_key)
        if max_events is not None:
            cum_count = 0
            sliced_groups = []
            for g in event_groups:
                sliced_groups.append(g)
                cum_count += len(g["group_indices_list"])
                if cum_count >= max_events:
                    break
            event_groups = sliced_groups

        with torch.no_grad():
            for g in event_groups:
                t = g["time"]
                group_tensor = g["group_tensor"]
                group_pkgs = g["group_pkgs"]

                edge_indices, edge_times = self.get_causal_edges(t)
                edge_attrs = self.model.encode_edge_times(edge_times, query_time=t)

                logits, probs, out_dict = self.model(
                    self.version_features,
                    self.all_package_ids,
                    edge_indices,
                    edge_attrs,
                )

                target_reps = out_dict["package_version"][group_tensor]
                self.model.package_memory.update_memory(group_pkgs, target_reps.detach())

    def save_checkpoint(self, epoch: int, val_auprc: float, threshold: float, filename: str = "best_model.pt") -> Path:
        """Save model weights, optimizer state, and training configuration."""
        save_path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": epoch,
            "val_auprc": val_auprc,
            "threshold": threshold,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path} (epoch={epoch}, val_auprc={val_auprc:.4f})")
        return save_path

    def load_checkpoint(self, filepath: Union[str, Path], reset_memory_to_clean: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint and restore model/optimizer states.
        Ensures package memory is clean (h0) and not contaminated by stale state.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if reset_memory_to_clean:
            self.model.package_memory.reset_memory()

        self.best_val_auprc = checkpoint.get("val_auprc", -1.0)
        self.optimal_threshold = checkpoint.get("threshold", 0.5)
        self.best_epoch = checkpoint.get("epoch", -1)
        logger.info(f"Loaded checkpoint from {filepath} (best_val_auprc={self.best_val_auprc:.4f}, memory reset to clean h0)")
        return checkpoint
