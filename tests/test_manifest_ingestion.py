"""Unit tests for Datadog manifest ingestion adapter."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from src.data.datadog.adapter import DatadogManifestAdapter


def test_manifest_adapter_parsing_rules(tmp_path: Path):
    manifest_data = {
        "pure_malware_pkg": None,
        "compromised_pkg": ["1.0.0", "1.0.1"],
    }
    manifest_file = tmp_path / "test_manifest.json"
    output_file = tmp_path / "test_labels.csv"

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    adapter = DatadogManifestAdapter(manifest_path=manifest_file, output_path=output_file)
    records = adapter.parse_records()

    assert len(records) == 3

    # Check null rule
    r_null = next(r for r in records if r["package"] == "pure_malware_pkg")
    assert r_null["version"] == "__ALL__"
    assert r_null["label"] == "confirmed_malicious"
    assert r_null["label_scope"] == "all_versions"
    assert r_null["source"] == "datadog"

    # Check explicit version rule
    r_ver1 = next(r for r in records if r["package"] == "compromised_pkg" and r["version"] == "1.0.0")
    assert r_ver1["label"] == "confirmed_malicious"
    assert r_ver1["label_scope"] == "specific_version"
    assert r_ver1["source"] == "datadog"

    r_ver2 = next(r for r in records if r["package"] == "compromised_pkg" and r["version"] == "1.0.1")
    assert r_ver2["label"] == "confirmed_malicious"
    assert r_ver2["label_scope"] == "specific_version"
    assert r_ver2["source"] == "datadog"


def test_manifest_validation_metrics(tmp_path: Path):
    manifest_data = {
        "pkg_a": None,
        "pkg_b": None,
        "pkg_c": ["0.1.0", "0.2.0"],
    }
    manifest_file = tmp_path / "test_manifest.json"
    output_file = tmp_path / "test_labels.csv"

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    adapter = DatadogManifestAdapter(manifest_path=manifest_file, output_path=output_file)
    val = adapter.validate()

    assert val["total_manifest_packages"] == 3
    assert val["all_version_malicious_packages"] == 2
    assert val["specific_version_packages"] == 1
    assert val["total_explicit_malicious_versions"] == 2
    assert val["total_normalized_label_rows"] == 4
    assert val["duplicate_package_version_rows"] == 0
    assert val["invalid_empty_package_names"] == 0
    assert val["invalid_version_strings"] == 0
