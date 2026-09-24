"""Unit tests for the experimental dataset augmentation and split layer."""

from __future__ import annotations

import math
from pathlib import Path
import pytest
from src.data.experimental.builder import ExperimentalDatasetBuilder
from src.data.experimental.sampler import BackgroundSampler


def test_sampler_historical_cutoff_metrics(tmp_path: Path):
    """Verify that releases uploaded AFTER cutoff timestamp are strictly excluded from metrics."""
    cutoff = 1700000000.0  # arbitrary test cutoff
    sampler = BackgroundSampler(
        raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "exp",
        malicious_labels_path=tmp_path / "empty.csv",
        cutoff_timestamp=cutoff,
    )

    mock_payload = {
        "releases": {
            "1.0.0": [{"upload_time_iso_8601": "2020-01-01T00:00:00Z"}],  # ts < cutoff
            "2.0.0": [{"upload_time_iso_8601": "2023-01-01T00:00:00Z"}],  # ts < cutoff
            "3.0.0": [{"upload_time_iso_8601": "2026-01-01T00:00:00Z"}],  # ts > cutoff (FUTURE!)
        }
    }

    metrics = sampler.compute_historical_metrics("test_pkg", mock_payload)
    # Only 1.0.0 and 2.0.0 should be counted
    assert metrics["historical_release_count"] == 2
    assert metrics["latest_historical_release_timestamp"] < cutoff


def test_three_distinct_labels(tmp_path: Path):
    """Verify 3 distinct labels: confirmed_malicious, source_unflagged, assumed_benign_background."""
    builder = ExperimentalDatasetBuilder(
        processed_dir=tmp_path / "proc",
        experimental_dir=tmp_path / "exp",
    )

    processed_dir = tmp_path / "proc"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy package_versions.csv
    import csv
    with open(processed_dir / "package_versions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "package", "normalized_package", "version", "release_timestamp",
            "label", "label_scope", "requires_dist", "requires_python",
            "repository_url", "homepage_url", "source"
        ])
        writer.writeheader()
        writer.writerow({
            "package": "mal_pkg", "normalized_package": "mal-pkg", "version": "1.0",
            "release_timestamp": 1600000000.0, "label": "confirmed_malicious",
            "label_scope": "all_versions", "requires_dist": '["req1"]', "requires_python": "",
            "repository_url": "", "homepage_url": "", "source": "pypi"
        })
        writer.writerow({
            "package": "compromised_pkg", "normalized_package": "compromised-pkg", "version": "0.9",
            "release_timestamp": 1500000000.0, "label": "unlabeled / assumed_benign",
            "label_scope": "unlabeled", "requires_dist": "", "requires_python": "",
            "repository_url": "", "homepage_url": "", "source": "pypi"
        })

    with open(processed_dir / "malicious_labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "package", "normalized_package", "version", "label",
            "label_scope", "source", "in_pypi_registry"
        ])
        writer.writeheader()
        writer.writerow({
            "package": "mal_pkg", "normalized_package": "mal-pkg", "version": "__ALL__",
            "label": "confirmed_malicious", "label_scope": "all_versions", "source": "datadog",
            "in_pypi_registry": "True"
        })

    bg_versions = [{
        "package": "clean_bg", "normalized_package": "clean-bg", "version": "1.0",
        "release_timestamp": 1650000000.0, "requires_dist": '["dep"]',
        "requires_python": "", "repository_url": "", "homepage_url": "", "source": "pypi_background"
    }]

    summary = builder.build(bg_versions)
    dist = summary["label_distribution"]

    assert dist["confirmed_malicious"] == 1
    assert dist["source_unflagged"] == 1
    assert dist["assumed_benign_background"] == 1


def test_square_root_package_weighting(tmp_path: Path):
    """Verify package-balanced square root weighting reduces dominant package influence."""
    builder = ExperimentalDatasetBuilder(
        processed_dir=tmp_path / "proc",
        experimental_dir=tmp_path / "exp",
        t_train_val=1700000000.0,
    )

    records = [
        # Heavy package with 100 releases in train
        *[{
            "normalized_package": "heavy_pkg", "label": "confirmed_malicious",
            "release_timestamp": 1600000000.0, "registry_status": "live"
        } for _ in range(100)],
        # Light package with 1 release in train
        {
            "normalized_package": "light_pkg", "label": "confirmed_malicious",
            "release_timestamp": 1600000000.0, "registry_status": "live"
        }
    ]

    train_indices = list(range(len(records)))
    report = builder._compute_and_apply_training_weights(records, train_indices)

    heavy_rep = next(p for p in report["per_package_breakdown"] if p["package"] == "heavy_pkg")
    light_rep = next(p for p in report["per_package_breakdown"] if p["package"] == "light_pkg")

    # Before weighting: heavy_pkg is 100/101 = 99.01% of mass
    assert heavy_rep["positive_mass_before_pct"] > 95.0
    # After square-root weighting: heavy_pkg weight is damped by 1/sqrt(100) = 0.1
    # while light_pkg weight is 1/sqrt(1) = 1.0. Relative influence is dramatically rebalanced!
    assert heavy_rep["raw_damped_weight"] == 0.1
    assert light_rep["raw_damped_weight"] == 1.0
    assert heavy_rep["positive_mass_after_pct"] < heavy_rep["positive_mass_before_pct"]


def test_connected_component_leak_free_split(tmp_path: Path):
    """Verify that packages sharing a repo or maintainer are assigned to the same split."""
    builder = ExperimentalDatasetBuilder(
        processed_dir=tmp_path / "proc",
        experimental_dir=tmp_path / "exp",
        raw_dir=tmp_path / "raw",
    )

    records = [
        # Package A and B share repository github:org/shared_repo
        {"normalized_package": "pkg_a", "label": "confirmed_malicious", "release_timestamp": 1600000000.0, "registry_status": "live", "repository_url": "github:org/shared_repo"},
        {"normalized_package": "pkg_b", "label": "source_unflagged", "release_timestamp": 1600000000.0, "registry_status": "live", "repository_url": "github:org/shared_repo"},
        # Package C is independent
        {"normalized_package": "pkg_c", "label": "assumed_benign_background", "release_timestamp": 1600000000.0, "registry_status": "live", "repository_url": "github:org/repo_c"},
    ]

    pkg_identifiers = builder._extract_package_identifiers(records)
    assert "repo:github:org/shared_repo" in pkg_identifiers["pkg_a"]
    assert "repo:github:org/shared_repo" in pkg_identifiers["pkg_b"]

    components = builder._build_connected_components(records, pkg_identifiers)
    # pkg_a and pkg_b must be in the same component
    shared_comp = next(c for c in components if "pkg_a" in c)
    assert "pkg_b" in shared_comp

    disjoint_split, audit = builder._build_package_disjoint_split(records, pkg_identifiers, components)
    assert audit["strict_package_disjointness"] is True
    assert audit["repository_overlap_count"] == 0
    assert audit["maintainer_overlap_count"] == 0

    # pkg_a and pkg_b must be in the exact same split
    split_a = "train" if "pkg_a" in disjoint_split["train_packages"] else ("val" if "pkg_a" in disjoint_split["val_packages"] else "test")
    split_b = "train" if "pkg_b" in disjoint_split["train_packages"] else ("val" if "pkg_b" in disjoint_split["val_packages"] else "test")
    assert split_a == split_b


def test_graph_package_disjoint_split(tmp_path: Path):
    """Verify graph-model split includes only packages with observable topology and excludes takedowns."""
    builder = ExperimentalDatasetBuilder(
        processed_dir=tmp_path / "proc",
        experimental_dir=tmp_path / "exp",
        raw_dir=tmp_path / "raw",
    )

    records = [
        # Live with observed deps
        {"normalized_package": "pkg_graph", "label": "confirmed_malicious", "registry_status": "live", "dependency_metadata_status": "observed", "topology_status": "graph_observable", "repository_url": ""},
        # Same package has a version with missing deps
        {"normalized_package": "pkg_graph", "label": "confirmed_malicious", "registry_status": "live", "dependency_metadata_status": "missing", "topology_status": "live_unobservable_topology", "repository_url": ""},
        # Takedown package (404)
        {"normalized_package": "pkg_takedown", "label": "confirmed_malicious", "registry_status": "takedown", "dependency_metadata_status": "missing", "topology_status": "label_only_takedown", "repository_url": ""},
    ]

    pkg_identifiers = builder._extract_package_identifiers(records)
    components = builder._build_connected_components(records, pkg_identifiers)

    graph_split, audit = builder._build_graph_package_disjoint_split(records, pkg_identifiers, components)

    assert audit["strict_package_disjointness"] is True
    assert graph_split["counts"]["total_packages"] == 1  # Only pkg_graph
    assert "pkg_takedown" not in graph_split["packages"]["train_packages"]
    assert "pkg_takedown" not in graph_split["packages"]["val_packages"]
    assert "pkg_takedown" not in graph_split["packages"]["test_packages"]

    # In whatever split pkg_graph is, graph_observable and unobservable indices are separated
    target_split = "train" if "pkg_graph" in graph_split["packages"]["train_packages"] else ("validation" if "pkg_graph" in graph_split["packages"]["val_packages"] else "test")
    assert len(graph_split[target_split]["graph_observable_indices"]) == 1
    assert len(graph_split[target_split]["unobservable_topology_indices"]) == 1
    assert graph_split[target_split]["graph_observable_indices"][0] == 0
    assert graph_split[target_split]["unobservable_topology_indices"][0] == 1

