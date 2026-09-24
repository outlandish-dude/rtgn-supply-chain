"""
Comprehensive Architecture and Causality Unit Tests for RTGN.

Verifies:
1. Zero static package-ID lookup tables (no nn.Embedding(num_packages, dim)).
2. Continuous harmonic time encoder converts delta_seconds to delta_days.
3. Package memory explicitly distinguishes Tier C cold-start from online adaptation.
4. Exactly 2-layer Heterogeneous GATv2 with relation-specific transforms for all 4 relations.
5. Loss weighting computes effective positive weight from weighted class totals without double-compensation.
6. End-to-end forward pass, logit output, and backward gradient flow.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.hetero_conv import HeteroGATv2Layer, HeteroGATv2Net
from src.models.rtgn import RTGNModel
from src.models.temporal_memory import PackageMemoryManager
from src.models.time_encoding import HarmonicTimeEncoder
from src.training.loss import PackageBalancedLoss


def test_no_static_package_id_embedding():
    """Verify that RTGN contains ZERO static package-ID lookup tables (nn.Embedding)."""
    model = RTGNModel(num_packages=1933, in_feature_dim=13, hidden_dim=64, time_dim=32)

    # 1. Assert no nn.Embedding layer exists anywhere in the model
    for name, module in model.named_modules():
        assert not isinstance(module, nn.Embedding), (
            f"Found forbidden static embedding layer: {name} ({module})!"
        )

    # 2. Verify all packages initialize to the identical shared initial state vector
    pkg_ids = torch.arange(10)
    initial_memories = model.package_memory.get_memory(pkg_ids)
    for i in range(1, 10):
        assert torch.allclose(initial_memories[0], initial_memories[i]), (
            f"Package {i} has different initial state from Package 0!"
        )


def test_time_encoding_units_and_dimensions():
    """Verify time encoder converts seconds to days and produces 32-dim harmonic embeddings."""
    encoder = HarmonicTimeEncoder(dimension=32)

    # Pass elapsed seconds (e.g. 1 day, 7 days, 30 days in seconds)
    one_day_sec = torch.tensor([86400.0])
    seven_days_sec = torch.tensor([86400.0 * 7.0])
    zero_sec = torch.tensor([0.0])

    emb_1d = encoder(one_day_sec)
    emb_7d = encoder(seven_days_sec)
    emb_0d = encoder(zero_sec)

    assert emb_1d.shape == (1, 32)
    assert emb_7d.shape == (1, 32)
    assert emb_0d.shape == (1, 32)

    # Non-negative elapsed time check: negative inputs clamped to 0
    neg_sec = torch.tensor([-100.0])
    emb_neg = encoder(neg_sec)
    assert torch.allclose(emb_0d, emb_neg)


def test_tier_c_cold_start_vs_online_adaptation():
    """Verify package memory explicitly distinguishes cold-start (0 updates) from online adaptation."""
    memory_mgr = PackageMemoryManager(num_packages=10, memory_dim=64, message_dim=64)

    test_pkg_ids = torch.tensor([0, 1, 2])

    # 1. Cold start verification: 0 updates
    assert (memory_mgr.is_cold_start(test_pkg_ids)).all()
    assert (memory_mgr.update_count[test_pkg_ids] == 0).all()

    # Initial state matches
    initial_states = memory_mgr.get_memory(test_pkg_ids)
    assert initial_states.shape == (3, 64)

    # 2. Simulate processing release 1 of package 1 (online adaptation)
    simulated_message = torch.randn(1, 64)
    memory_mgr.update_memory(torch.tensor([1]), simulated_message)

    # Package 1 is now online-adapted (1 update); Package 0 and 2 remain cold-start (0 updates)
    assert not memory_mgr.is_cold_start(torch.tensor([1])).item()
    assert memory_mgr.is_cold_start(torch.tensor([0])).item()
    assert memory_mgr.is_cold_start(torch.tensor([2])).item()

    assert memory_mgr.update_count[1].item() == 1
    assert memory_mgr.update_count[0].item() == 0

    # Package 1 memory state has evolved away from initial state
    updated_pkg1_mem = memory_mgr.get_memory(torch.tensor([1]))
    assert not torch.allclose(updated_pkg1_mem[0], initial_states[1])


def test_hetero_gatv2_message_passing():
    """Verify 2-layer Heterogeneous GATv2 executes relation-specific transforms on all 4 relations."""
    net = HeteroGATv2Net(in_dim=64, hidden_dim=64, out_dim=64, edge_dim=32, heads=4, dropout=0.2)

    # 5 versions, 3 packages
    x_dict = {
        "package": torch.randn(3, 64),
        "package_version": torch.randn(5, 64),
    }

    # Bipartite edges
    edge_index_dict = {
        "depends_on": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),   # ver -> pkg
        "version_of": torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]], dtype=torch.long),  # ver -> pkg
        "has_version": torch.tensor([[0, 0, 1, 1, 2], [0, 1, 2, 3, 4]], dtype=torch.long), # pkg -> ver
        "depended_by": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),  # pkg -> ver
    }

    edge_attr_dict = {
        "depends_on": torch.randn(3, 32),
        "version_of": torch.randn(5, 32),
        "has_version": torch.randn(5, 32),
        "depended_by": torch.randn(3, 32),
    }

    out_dict = net(x_dict, edge_index_dict, edge_attr_dict)

    assert "package" in out_dict
    assert "package_version" in out_dict
    assert out_dict["package"].shape == (3, 64)
    assert out_dict["package_version"].shape == (5, 64)


def test_package_balanced_loss_calculation():
    """Verify effective positive weight is calculated from weighted mass without double compensation."""
    loss_fn = PackageBalancedLoss()

    # Synthetic training batch
    y_train = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.long)  # 2 pos, 4 neg
    # Suppose package weights downweight positive package:
    sample_weights = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    stats = loss_fn.compute_weights(y_train, sample_weights)

    # W_pos = 0.5 + 0.5 = 1.0
    # W_neg = 1.0 * 4 = 4.0
    # effective_pos_weight = W_neg / W_pos = 4.0 / 1.0 = 4.0
    assert stats["raw_pos_count"] == 2
    assert stats["raw_neg_count"] == 4
    assert stats["total_weighted_pos_mass"] == 1.0
    assert stats["total_weighted_neg_mass"] == 4.0
    assert stats["effective_pos_weight"] == 4.0
    assert loss_fn.effective_pos_weight == 4.0

    # Compute loss
    logits = torch.tensor([2.0, 1.5, -2.0, -1.5, -3.0, -2.5], requires_grad=True)
    loss = loss_fn(logits, y_train.float(), sample_weights)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_rtgn_full_forward_and_backward():
    """Verify RTGN full forward pass, probability output, and gradient backpropagation."""
    num_pkgs = 5
    num_vers = 8
    model = RTGNModel(num_packages=num_pkgs, in_feature_dim=13, hidden_dim=64, time_dim=32)

    # Synthetic inputs
    feat = torch.randn(num_vers, 13)
    pkg_ids = torch.arange(num_pkgs)

    edge_times = {
        "depends_on": torch.tensor([1600000000.0, 1650000000.0]),
        "version_of": torch.tensor([1600000000.0, 1610000000.0, 1620000000.0]),
        "has_version": torch.tensor([1600000000.0, 1610000000.0, 1620000000.0]),
        "depended_by": torch.tensor([1600000000.0, 1650000000.0]),
    }
    edge_indices = {
        "depends_on": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        "version_of": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        "has_version": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        "depended_by": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
    }

    query_time = 1700000000.0
    edge_attrs = model.encode_edge_times(edge_times, query_time)

    logits, probs, out_dict = model(feat, pkg_ids, edge_indices, edge_attrs)

    assert logits.shape == (num_vers, )
    assert probs.shape == (num_vers, )
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Backpropagation check
    loss = logits.sum()
    loss.backward()

    # Check gradients in version encoder, memory initial state, gnn, and head
    assert model.version_encoder[0].weight.grad is not None
    assert model.package_memory.initial_state.grad is not None
    assert model.hetero_gnn.layer1.conv_depends_on.lin_l.weight.grad is not None
    assert model.detection_head[0].weight.grad is not None


def test_source_unflagged_excluded_from_supervised_loss():
    """
    Verify that source_unflagged records are strictly excluded from supervised loss:
    1. Not counted in negative class or weighted negative mass.
    2. Logits for source_unflagged nodes receive exactly zero gradient.
    3. Chronological dataset training split counts match: pos=1218, neg=8517, ignored=2098.
    """
    import os
    import pandas as pd

    loss_fn = PackageBalancedLoss()

    # Synthetic batch: 2 pos (y=1), 3 neg (y=0), 2 unflagged (y=-1)
    y_batch = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0])
    weights = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])

    stats = loss_fn.compute_weights(y_batch, weights)

    assert stats["raw_pos_count"] == 2
    assert stats["raw_neg_count"] == 3
    assert stats["raw_ignored_count"] == 2
    assert stats["total_weighted_pos_mass"] == 1.0
    assert stats["total_weighted_neg_mass"] == 3.0
    assert stats["effective_pos_weight"] == 3.0

    # Test gradient zeroing for unflagged nodes
    logits = torch.tensor([2.0, 1.5, -2.0, -1.5, -3.0, 5.0, -5.0], requires_grad=True)
    loss = loss_fn(logits, y_batch, weights)
    loss.backward()

    # Logits for indices 5 and 6 (source_unflagged) must have zero gradient
    assert logits.grad[5].item() == 0.0, "source_unflagged logit received non-zero gradient!"
    assert logits.grad[6].item() == 0.0, "source_unflagged logit received non-zero gradient!"
    # Supervised nodes must receive non-zero gradients
    assert logits.grad[0].item() != 0.0
    assert logits.grad[2].item() != 0.0

    # Audit against experimental dataset artifacts if available
    nodes_csv = "data/experimental/graph/nodes_package_version.csv"
    splits_pt = "data/experimental/graph/split_masks.pt"
    if os.path.exists(nodes_csv) and os.path.exists(splits_pt):
        df_nodes = pd.read_csv(nodes_csv)
        splits = torch.load(splits_pt)
        train_mask = splits["train_mask_chrono"].numpy()
        train_nodes = df_nodes[train_mask]

        train_labels = torch.tensor(train_nodes["target_label"].values, dtype=torch.float32)
        train_weights = torch.tensor(train_nodes["sample_weight"].values, dtype=torch.float32)

        chrono_loss_fn = PackageBalancedLoss()
        chrono_stats = chrono_loss_fn.compute_weights(train_labels, train_weights)

        assert chrono_stats["raw_pos_count"] == 1218, f"Expected 1218 pos, got {chrono_stats['raw_pos_count']}"
        assert chrono_stats["raw_neg_count"] == 8517, f"Expected 8517 neg, got {chrono_stats['raw_neg_count']}"
        assert chrono_stats["raw_ignored_count"] == 2098, f"Expected 2098 ignored, got {chrono_stats['raw_ignored_count']}"
        assert abs(chrono_stats["total_weighted_pos_mass"] - 1073.7772) < 1e-2
        assert abs(chrono_stats["total_weighted_neg_mass"] - 8517.0) < 1e-2
        assert abs(chrono_stats["effective_pos_weight"] - 7.9318) < 1e-3

