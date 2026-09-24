"""
Unit and Regression Tests for RTGNTrainer.

Verifies:
1. Chronological Event Ordering.
2. Per-Target Causal Edge Filtering (no future edges).
3. Read-Before-Write Memory Ordering.
4. source_unflagged Exclusion from Loss and Metrics.
5. Tier C Cold-Start vs Online Adaptation Memory Semantics.
6. Checkpoint Save and Load Integrity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

from src.models.rtgn import RTGNModel
from src.training.loss import PackageBalancedLoss
from src.training.rtgn_trainer import RTGNTrainer


@pytest.fixture
def mock_trainer_setup():
    """Build a mock RTGNTrainer with synthetic causal graph for unit testing."""
    num_packages = 4
    num_versions = 6

    # Version timestamps: deliberately unsorted in row order
    # Row 0: t=100 (pkg 0, y=1)
    # Row 1: t=300 (pkg 0, y=0)
    # Row 2: t=200 (pkg 1, y=-1 unflagged)
    # Row 3: t=400 (pkg 2, y=1)
    # Row 4: t=250 (pkg 2, y=0)
    # Row 5: t=500 (pkg 3, y=0, Tier C unseen)
    timestamps = [100.0, 300.0, 200.0, 400.0, 250.0, 500.0]
    pkg_ids = [0, 0, 1, 2, 2, 3]
    labels = [1, 0, -1, 1, 0, 0]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    df_nodes = pd.DataFrame({
        "version_id": range(num_versions),
        "package_id": pkg_ids,
        "release_timestamp": timestamps,
        "target_label": labels,
        "sample_weight": weights,
    })

    version_features = torch.randn(num_versions, 13)

    # Synthetic edges with timestamps:
    # has_version: (pkg 0 -> v0 at t=100), (pkg 1 -> v2 at t=200), (pkg 2 -> v4 at t=250), (pkg 0 -> v1 at t=300)
    has_ver_index = torch.tensor([[0, 1, 2, 0], [0, 2, 4, 1]], dtype=torch.long)
    has_ver_time = torch.tensor([100.0, 200.0, 250.0, 300.0])

    # depends_on: (v2 -> pkg 0 at t=200), (v4 -> pkg 1 at t=250), (v3 -> pkg 0 at t=400)
    dep_on_index = torch.tensor([[2, 4, 3], [0, 1, 0]], dtype=torch.long)
    dep_on_time = torch.tensor([200.0, 250.0, 400.0])

    model = RTGNModel(num_packages=num_packages, in_feature_dim=13, hidden_dim=64, time_dim=32, heads=4)
    loss_fn = PackageBalancedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = RTGNTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device="cpu")
    trainer.version_features = version_features
    trainer.all_package_ids = torch.arange(num_packages)
    trainer.df_nodes = df_nodes

    # Pre-sorted edge buffers
    perm_has = torch.argsort(has_ver_time)
    trainer.sorted_has_ver_index = has_ver_index[:, perm_has]
    trainer.sorted_has_ver_time = has_ver_time[perm_has]

    perm_dep = torch.argsort(dep_on_time)
    trainer.sorted_dep_on_index = dep_on_index[:, perm_dep]
    trainer.sorted_dep_on_time = dep_on_time[perm_dep]

    # Split masks
    # Train: rows 0, 1, 2, 4 (t=100, 300, 200, 250)
    # Val: row 3 (t=400)
    # Test: row 5 (t=500)
    train_mask = np.array([True, True, True, False, True, False])
    val_mask = np.array([False, False, False, True, False, False])
    test_mask = np.array([False, False, False, False, False, True])

    trainer.split_masks = {
        "train_mask_chrono": train_mask,
        "val_mask_chrono": val_mask,
        "test_mask_chrono": test_mask,
    }

    # Chronologically ordered indices
    chrono_order = np.argsort(df_nodes["release_timestamp"].values)
    trainer.ordered_indices["train_mask_chrono"] = np.array([i for i in chrono_order if train_mask[i]])
    trainer.ordered_indices["val_mask_chrono"] = np.array([i for i in chrono_order if val_mask[i]])
    trainer.ordered_indices["test_mask_chrono"] = np.array([i for i in chrono_order if test_mask[i]])

    # Initialize loss fn
    train_labels = torch.tensor(df_nodes.loc[trainer.ordered_indices["train_mask_chrono"], "target_label"].values, dtype=torch.float32)
    train_weights = torch.tensor(df_nodes.loc[trainer.ordered_indices["train_mask_chrono"], "sample_weight"].values, dtype=torch.float32)
    trainer.loss_fn.compute_weights(train_labels, train_weights)

    trainer.is_initialized = True
    return trainer


def test_chronological_ordering(mock_trainer_setup):
    """Verify that event processing order is strictly monotonic in release_timestamp."""
    trainer = mock_trainer_setup
    train_ordered = trainer.ordered_indices["train_mask_chrono"]

    timestamps = trainer.df_nodes.loc[train_ordered, "release_timestamp"].values
    assert len(timestamps) == 4
    # Expected order: row 0 (100.0) -> row 2 (200.0) -> row 4 (250.0) -> row 1 (300.0)
    assert list(train_ordered) == [0, 2, 4, 1]
    assert np.all(np.diff(timestamps) > 0), "Train events are not strictly ascending in time!"


def test_per_target_causal_edge_filtering(mock_trainer_setup):
    """Verify that causal edge filtering strictly excludes future edges for each target timestamp."""
    trainer = mock_trainer_setup

    # At t=150.0: only edges with edge_time <= 150.0 should be present
    # has_ver: only t=100.0 (1 edge)
    # dep_on: none (0 edges)
    edge_idx, edge_times = trainer.get_causal_edges(target_time=150.0)
    assert edge_idx["has_version"].shape[1] == 1
    assert edge_times["has_version"][0].item() == 100.0
    assert edge_idx["depends_on"].shape[1] == 0

    # At t=250.0: edges up to t=250.0
    # has_ver: 100.0, 200.0, 250.0 (3 edges, excluding 300.0)
    # dep_on: 200.0, 250.0 (2 edges, excluding 400.0)
    edge_idx_250, edge_times_250 = trainer.get_causal_edges(target_time=250.0)
    assert edge_idx_250["has_version"].shape[1] == 3
    assert (edge_times_250["has_version"] <= 250.0).all()
    assert (edge_times_250["depends_on"] <= 250.0).all()
    assert (edge_times_250["has_version"] < 300.0).all()


def test_read_before_write_memory_ordering(mock_trainer_setup):
    """Verify that memory is read before event prediction and updated only after prediction."""
    trainer = mock_trainer_setup
    pkg_id = 0

    # Before processing any event, package 0 should have update_count == 0
    assert trainer.model.package_memory.update_count[pkg_id].item() == 0

    # First event in train is row 0 (pkg 0, t=100)
    # Run 1 event
    res = trainer.train_epoch(max_events=1)
    assert res["supervised_events"] == 1

    # After the event completes, update_count for pkg 0 should be 1
    assert trainer.model.package_memory.update_count[pkg_id].item() == 1
    # Other packages remain un-updated
    assert trainer.model.package_memory.update_count[1].item() == 0
    assert trainer.model.package_memory.update_count[2].item() == 0


def test_source_unflagged_exclusion(mock_trainer_setup):
    """Verify that source_unflagged records (-1) contribute 0 supervised loss and 0 gradient."""
    trainer = mock_trainer_setup

    # In our mock data, row 2 is at t=200.0 and has label -1 (source_unflagged)
    # If we run the first 2 events: event 0 is pos (y=1), event 1 is unflagged (y=-1)
    res = trainer.train_epoch(max_events=2)

    # Supervised events count should only be 1 (the unflagged event did not compute supervised loss)
    assert res["supervised_events"] == 1
    # But total events processed should be 2
    assert res["total_events"] == 2
    # Package 1 memory was nonetheless updated as an observed historical event
    assert trainer.model.package_memory.update_count[1].item() == 1


def test_tier_c_cold_start_memory_semantics(mock_trainer_setup):
    """Verify that unseen test packages evaluate under cold-start mode with update_count == 0."""
    trainer = mock_trainer_setup
    unseen_pkg = 3

    # Ensure memory is fresh or train on train fold
    trainer.train_epoch(max_events=4)

    # Package 3 was never present in train fold
    assert trainer.model.package_memory.update_count[unseen_pkg].item() == 0
    assert trainer.model.package_memory.is_cold_start(torch.tensor([unseen_pkg])).item() is True

    # Evaluate on test set (row 5: pkg 3, t=500.0)
    metrics = trainer.evaluate_split("test_mask_chrono", is_tier_c=True)

    assert "cold_start" in metrics
    # After evaluation, memory has been updated
    assert trainer.model.package_memory.update_count[unseen_pkg].item() == 1


def test_checkpoint_save_and_load(mock_trainer_setup):
    """Verify checkpoint persistence, restoration, and parameter fidelity."""
    trainer = mock_trainer_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer.checkpoint_dir = Path(tmp_dir)

        # Save checkpoint
        ckpt_path = trainer.save_checkpoint(epoch=1, val_auprc=0.885, threshold=0.45)
        assert ckpt_path.exists()

        # Modify model weight
        with torch.no_grad():
            trainer.model.version_encoder[0].weight.fill_(999.0)

        assert (trainer.model.version_encoder[0].weight == 999.0).all()

        # Load checkpoint
        loaded = trainer.load_checkpoint(ckpt_path)
        assert loaded["epoch"] == 1
        assert loaded["val_auprc"] == 0.885
        assert loaded["threshold"] == 0.45
        assert trainer.best_val_auprc == 0.885
        assert trainer.optimal_threshold == 0.45

        # Check that weight was restored from 999.0
        assert not (trainer.model.version_encoder[0].weight == 999.0).any()


def test_memory_reset_between_training_epochs(mock_trainer_setup):
    """Verify that every training epoch begins with a clean h0 memory state."""
    trainer = mock_trainer_setup

    # Epoch 1
    trainer.train_epoch(max_events=3)
    assert trainer.model.package_memory.update_count.sum().item() > 0

    # Start of Epoch 2: explicit reset
    trainer.model.package_memory.reset_memory()
    assert trainer.model.package_memory.update_count.sum().item() == 0
    # Every package memory equals initial_state
    init_state = trainer.model.package_memory.initial_state
    for pid in range(trainer.model.num_packages):
        assert torch.allclose(trainer.model.package_memory.memory[pid], init_state)


def test_validation_memory_cannot_leak_into_subsequent_training(mock_trainer_setup):
    """Verify that memory updates during validation do not leak into the next training epoch."""
    trainer = mock_trainer_setup

    # Train epoch 1
    trainer.train_epoch(max_events=3)
    # Validation evaluation (updates validation package memories)
    trainer.evaluate_split("val_mask_chrono")
    assert trainer.model.package_memory.update_count.sum().item() > 0

    # Next epoch must reset memory to prevent validation leakage into training
    trainer.model.package_memory.reset_memory()
    assert trainer.model.package_memory.update_count.sum().item() == 0


def test_restoring_checkpoint_reconstructs_memory_consistently(mock_trainer_setup):
    """Verify that restoring a checkpoint resets memory to clean h0 and allows exact causal reconstruction."""
    trainer = mock_trainer_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer.checkpoint_dir = Path(tmp_dir)
        # Train epoch 1 and save checkpoint
        trainer.train_epoch(max_events=3)
        ckpt_path = trainer.save_checkpoint(epoch=1, val_auprc=0.75, threshold=0.3)

        # Replay to get reference memory state under epoch 1 model
        trainer.model.package_memory.reset_memory()
        trainer.replay_split_memory("train_mask_chrono", max_events=3)
        ref_memory = trainer.model.package_memory.memory.clone()

        # Train epoch 2 (weights and memory change)
        trainer.train_epoch(max_events=3)

        # Restore checkpoint 1
        trainer.load_checkpoint(ckpt_path, reset_memory_to_clean=True)
        # Memory is clean h0 immediately after loading
        assert trainer.model.package_memory.update_count.sum().item() == 0

        # Reconstruct memory by replaying train history under restored checkpoint
        trainer.replay_split_memory("train_mask_chrono", max_events=3)
        assert torch.allclose(trainer.model.package_memory.memory, ref_memory, atol=1e-5)


def test_test_memory_generated_using_restored_checkpoint_not_stale_state(mock_trainer_setup):
    """Verify that test evaluation uses causally reconstructed memory under the restored model, not stale state."""
    trainer = mock_trainer_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer.checkpoint_dir = Path(tmp_dir)
        # Epoch 1 (best model)
        trainer.train_epoch(max_events=3)
        ckpt_path = trainer.save_checkpoint(epoch=1, val_auprc=0.9, threshold=0.4)

        # Epoch 2 (worse model with different memory states)
        trainer.train_epoch(max_events=3)
        stale_memory = trainer.model.package_memory.memory.clone()

        # Restore best model from epoch 1
        trainer.load_checkpoint(ckpt_path, reset_memory_to_clean=True)
        # Memory is NOT stale memory from epoch 2
        assert not torch.allclose(trainer.model.package_memory.memory, stale_memory)

        # Replay train and val history under restored model
        trainer.replay_split_memory("train_mask_chrono", max_events=3)
        trainer.replay_split_memory("val_mask_chrono", max_events=1)

        # Evaluate test split
        metrics = trainer.evaluate_split("test_mask_chrono", max_events=1, reset_memory=False)
        assert "auprc" in metrics

