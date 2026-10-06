"""
Unit Tests for RTGN Ablation Framework Configurations.

Verifies:
1. No temporal encoding: GATv2 operates without edge harmonic time encodings.
2. No graph / message passing: GNN is bypassed; version encoder maps directly to detection head.
3. No historical release features: input dimension restricted to 7 intrinsic/dependency features.
4. No package balancing: uniform sample weights w_i = 1.0 without package frequency downweighting.
5. End-to-end forward, loss, backward, and optimization for all ablation configurations.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.rtgn import RTGNModel
from src.training.loss import PackageBalancedLoss
from src.training.rtgn_trainer import RTGNTrainer


@pytest.fixture
def dummy_batch_data():
    """Create minimal synthetic batch for ablation architecture testing."""
    num_versions = 6
    num_packages = 3
    in_dim = 13

    version_feats = torch.randn(num_versions, in_dim)
    package_ids = torch.arange(num_packages)

    # Edge indices
    edge_index_dict = {
        "has_version": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        "version_of": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        "depends_on": torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
        "depended_by": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }

    edge_time_dict = {
        "has_version": torch.tensor([100.0, 200.0, 300.0]),
        "version_of": torch.tensor([100.0, 200.0, 300.0]),
        "depends_on": torch.tensor([150.0, 250.0]),
        "depended_by": torch.tensor([150.0, 250.0]),
    }

    return {
        "num_versions": num_versions,
        "num_packages": num_packages,
        "version_feats": version_feats,
        "package_ids": package_ids,
        "edge_index_dict": edge_index_dict,
        "edge_time_dict": edge_time_dict,
    }


def test_no_temporal_encoding_configuration(dummy_batch_data):
    """Verify that RTGN without temporal encoding bypasses time encoder and runs GATv2 without edge attrs."""
    d = dummy_batch_data
    model = RTGNModel(
        num_packages=d["num_packages"],
        in_feature_dim=13,
        hidden_dim=32,
        time_dim=16,
        heads=2,
        use_time_encoding=False,
        use_graph=True,
    )

    assert model.use_time_encoding is False
    assert model.time_encoder is None
    assert model.hetero_gnn.edge_dim is None

    # encode_edge_times must return None
    edge_attrs = model.encode_edge_times(d["edge_time_dict"], query_time=350.0)
    assert edge_attrs is None

    # Forward pass without edge_attr_dict
    logits, probs, out_dict = model(
        d["version_feats"],
        d["package_ids"],
        d["edge_index_dict"],
        edge_attr_dict=None,
    )

    assert logits.shape == (d["num_versions"],)
    assert probs.shape == (d["num_versions"],)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(probs).all()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_no_graph_configuration(dummy_batch_data):
    """Verify that RTGN without graph message passing bypasses hetero_gnn entirely."""
    d = dummy_batch_data
    model = RTGNModel(
        num_packages=d["num_packages"],
        in_feature_dim=13,
        hidden_dim=32,
        time_dim=16,
        heads=2,
        use_time_encoding=True,
        use_graph=False,
    )

    assert model.use_graph is False
    assert model.hetero_gnn is None

    # Forward pass bypasses GNN (eval mode to disable dropout for deterministic comparison)
    model.eval()
    with torch.no_grad():
        logits, probs, out_dict = model(
            d["version_feats"],
            d["package_ids"],
            edge_index_dict=None,
            edge_attr_dict=None,
        )

        assert logits.shape == (d["num_versions"],)
        assert probs.shape == (d["num_versions"],)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(probs).all()
        # Representation equals directly projected version features
        h_ver_expected = model.version_encoder(d["version_feats"])
        torch.testing.assert_close(out_dict["package_version"], h_ver_expected)


def test_no_historical_features_configuration(dummy_batch_data):
    """Verify that RTGN without historical release features takes 7-dimensional causal input."""
    d = dummy_batch_data
    feats_7d = d["version_feats"][:, :7]

    model = RTGNModel(
        num_packages=d["num_packages"],
        in_feature_dim=7,
        hidden_dim=32,
        time_dim=16,
        heads=2,
        use_time_encoding=True,
        use_graph=True,
    )

    assert model.in_feature_dim == 7
    assert model.version_encoder[0].in_features == 7

    edge_attrs = model.encode_edge_times(d["edge_time_dict"], query_time=350.0)
    logits, probs, out_dict = model(
        feats_7d,
        d["package_ids"],
        d["edge_index_dict"],
        edge_attrs,
    )

    assert logits.shape == (d["num_versions"],)
    assert probs.shape == (d["num_versions"],)
    assert torch.isfinite(logits).all()


def test_no_package_balancing_weights():
    """Verify that no package balancing produces uniform sample weights w_i = 1.0."""
    loss_fn = PackageBalancedLoss()
    y_train = torch.tensor([1.0, 0.0, 0.0, 1.0, -1.0])
    # Uniform weights
    uniform_weights = torch.ones_like(y_train)

    stats = loss_fn.compute_weights(y_train, uniform_weights)
    assert stats["raw_pos_count"] == 2
    assert stats["raw_neg_count"] == 2
    assert stats["raw_ignored_count"] == 1
    assert stats["total_weighted_pos_mass"] == 2.0
    assert stats["total_weighted_neg_mass"] == 2.0
    assert stats["effective_pos_weight"] == 1.0


@pytest.mark.parametrize(
    "ablation_name,in_dim,use_time,use_graph,use_bal",
    [
        ("no_temporal_encoding", 13, False, True, True),
        ("no_graph", 13, True, False, True),
        ("no_historical_features", 7, True, True, True),
        ("no_package_balancing", 13, True, True, False),
    ],
)
def test_ablation_training_step(dummy_batch_data, ablation_name, in_dim, use_time, use_graph, use_bal):
    """Verify that a full backward step and gradient update succeeds for every ablation configuration."""
    d = dummy_batch_data
    feats = d["version_feats"][:, :in_dim]

    model = RTGNModel(
        num_packages=d["num_packages"],
        in_feature_dim=in_dim,
        hidden_dim=32,
        time_dim=16,
        heads=2,
        use_time_encoding=use_time,
        use_graph=use_graph,
    )

    loss_fn = PackageBalancedLoss()
    train_targets = torch.tensor([1.0, 0.0, 1.0, 0.0, -1.0, -1.0])
    sample_weights = torch.tensor([2.0, 1.0, 3.0, 0.5, 1.0, 1.0]) if use_bal else torch.ones_like(train_targets)
    loss_fn.compute_weights(train_targets, sample_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    edge_indices = d["edge_index_dict"] if use_graph else None
    edge_attrs = model.encode_edge_times(d["edge_time_dict"], query_time=350.0) if use_time else None

    # Forward
    logits, probs, out_dict = model(feats, d["package_ids"], edge_indices, edge_attrs)
    assert logits.shape == (d["num_versions"],)

    # Supervised loss (source_unflagged indices 4 and 5 must be excluded)
    sup_mask = (train_targets == 0) | (train_targets == 1)
    loss = loss_fn(logits, train_targets, sample_weights, supervised_mask=sup_mask)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    optimizer.zero_grad()
    loss.backward()

    # Verify all gradients are finite
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()

    optimizer.step()
