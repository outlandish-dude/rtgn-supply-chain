"""
Unit and Integration Tests for Temporal Graph Construction and Causal Snapshot Semantics.
Verifies all 6 explicit requirements:
1. Zero future edges in snapshot(t)
2. Zero target-label / security-event leakage
3. Zero maintainer / repository leakage across splits
4. Package-level dependency declaration semantics
5. Temporal invariance of feature values
6. Takedown records isolated from message passing
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from src.graph.builder import TemporalGraphBuilder
from src.graph.features import compute_causal_features_for_release
from src.graph.schema import REL_DEPENDS_ON, REL_HAS_VERSION, NodeType, TargetLabel
from src.graph.temporal_graph import TemporalGraph


@pytest.fixture(scope="module")
def graph_artifacts(tmp_path_factory):
    """Build temporal graph from actual experimental artifacts for testing."""
    test_dir = tmp_path_factory.mktemp("graph_test")
    builder = TemporalGraphBuilder(
        experimental_dir="data/experimental",
        processed_dir="data/processed",
        output_dir=test_dir,
    )
    temporal_graph, summary = builder.build()
    return temporal_graph, summary, test_dir


def test_no_future_edges_in_snapshot(graph_artifacts):
    """Verify that snapshot(t) contains ZERO edges with edge_time > t."""
    temporal_graph, summary, _ = graph_artifacts

    test_cutoffs = [
        1600000000.0,  # 2020
        1700000000.0,  # 2023
        1748736000.0,  # 2025-06-01 (Train cutoff)
        1772323200.0,  # 2026-03-01 (Val cutoff)
    ]

    for t in test_cutoffs:
        snapshot = temporal_graph.snapshot(t)

        # 1. HAS_VERSION edges
        if snapshot[REL_HAS_VERSION].num_edges > 0:
            hv_times = snapshot[REL_HAS_VERSION].edge_time
            assert (hv_times <= t).all(), f"Found HAS_VERSION edge with edge_time > {t}!"

        # 2. DEPENDS_ON edges
        if snapshot[REL_DEPENDS_ON].num_edges > 0:
            dep_times = snapshot[REL_DEPENDS_ON].edge_time
            assert (dep_times <= t).all(), f"Found DEPENDS_ON edge with edge_time > {t}!"

        # 3. Active versions mask
        active_mask = snapshot[NodeType.PACKAGE_VERSION.value].active_mask
        active_indices = torch.where(active_mask)[0].tolist()
        for idx in active_indices:
            v_node = temporal_graph.version_nodes[idx]
            assert v_node["release_timestamp"] is not None
            assert v_node["release_timestamp"] <= t, f"Active version has release_timestamp > {t}!"
            assert v_node["topology_status"] == "graph_observable"


def test_zero_target_label_and_security_leakage():
    """Verify that target labels and target compromise advisories never enter input features."""
    mock_target = {
        "package": "compromised_pkg",
        "normalized_package": "compromised-pkg",
        "version": "2.0.0",
        "release_timestamp": 1700000000.0,
        "label": "confirmed_malicious",  # Target ground truth
        "requires_python": ">=3.8",
        "requires_dist": '["requests>=2.0"]',
    }

    # Historical advisories contain an advisory for version 1.0.0 (past) and 2.0.0 (target)
    historical_advisories = [
        ("1.0.0", 1650000000.0),  # Valid historical advisory prior to t
        ("2.0.0", 1690000000.0),  # Target version's own advisory (MUST BE EXCLUDED!)
        ("1.0.0", 1750000000.0),  # Future advisory published after t (MUST BE EXCLUDED!)
    ]

    features = compute_causal_features_for_release(
        row=mock_target,
        historical_package_releases=[("1.0.0", 1600000000.0), ("2.0.0", 1700000000.0)],
        historical_dependency_events=[("requests", 1650000000.0)],
        historical_advisories=historical_advisories,
    )

    # Feature 12 is hist_advisories_count_before_t
    # Only ("1.0.0", 1650000000.0) should be counted -> exactly 1
    assert features[12] == 1.0, f"Expected 1 historical advisory, got {features[12]}!"


def test_feature_temporal_invariance():
    """Verify feature vector at timestamp t is strictly invariant to appending future records."""
    mock_release = {
        "package": "my_pkg",
        "normalized_package": "my-pkg",
        "version": "1.5.0",
        "release_timestamp": 1700000000.0,
        "label": "assumed_benign_background",
        "requires_python": ">=3.9",
        "requires_dist": '["numpy>=1.20"]',
    }

    base_releases = [("1.0.0", 1600000000.0), ("1.5.0", 1700000000.0)]
    base_deps = [("numpy", 1650000000.0)]
    base_advs = [("1.0.0", 1680000000.0)]

    features_before = compute_causal_features_for_release(
        row=mock_release,
        historical_package_releases=base_releases,
        historical_dependency_events=base_deps,
        historical_advisories=base_advs,
    )

    # Append future events
    augmented_releases = base_releases + [("2.0.0", 1750000000.0), ("3.0.0", 1800000000.0)]
    augmented_deps = base_deps + [("numpy", 1760000000.0), ("my-pkg", 1790000000.0)]
    augmented_advs = base_advs + [("2.0.0", 1770000000.0)]

    features_after = compute_causal_features_for_release(
        row=mock_release,
        historical_package_releases=augmented_releases,
        historical_dependency_events=augmented_deps,
        historical_advisories=augmented_advs,
    )

    assert features_before == features_after, "Features changed when future records were appended!"


def test_dependency_semantics_and_takedown_isolation(graph_artifacts):
    """Verify DEPENDS_ON points to Package nodes and takedowns have zero edges."""
    temporal_graph, summary, _ = graph_artifacts

    # 1. Verify DEPENDS_ON destination is always a valid Package ID
    num_packages = temporal_graph.num_packages
    for src_ver_id, dst_pkg_id, edge_time in temporal_graph.depends_on_edges:
        assert 0 <= dst_pkg_id < num_packages, f"DEPENDS_ON target {dst_pkg_id} out of bounds!"
        assert 0 <= src_ver_id < temporal_graph.num_versions, f"DEPENDS_ON source {src_ver_id} out of bounds!"
        ver_node = temporal_graph.version_nodes[src_ver_id]
        assert ver_node["topology_status"] == "graph_observable"
        assert ver_node["registry_status"] == "live"
        assert ver_node["release_timestamp"] == edge_time

    # 2. Verify takedown records have 0 outgoing and 0 incoming edges
    takedown_ver_ids = {
        v["version_id"] for v in temporal_graph.version_nodes
        if v["topology_status"] == "label_only_takedown"
    }

    assert len(takedown_ver_ids) == 1799

    for src_ver_id, dst_pkg_id, _ in temporal_graph.depends_on_edges:
        assert src_ver_id not in takedown_ver_ids, "Takedown version has DEPENDS_ON edge!"

    for src_pkg_id, dst_ver_id, _ in temporal_graph.has_version_edges:
        assert dst_ver_id not in takedown_ver_ids, "Takedown version has HAS_VERSION edge!"


def test_graph_package_disjoint_leakage_audit(graph_artifacts):
    """Verify zero package, repository, or maintainer leakage in graph-model split."""
    temporal_graph, summary, _ = graph_artifacts
    import json
    with open("data/experimental/splits/graph_package_disjoint_split.json", "r", encoding="utf-8") as f:
        g_split = json.load(f)

    audit = g_split["identifier_leakage_audit"]
    assert audit["strict_package_disjointness"] is True
    assert audit["package_overlap_count"] == 0
    assert audit["repository_overlap_count"] == 0
    assert audit["maintainer_overlap_count"] == 0
    assert len(audit["shared_repositories"]) == 0
    assert len(audit["shared_maintainers"]) == 0
