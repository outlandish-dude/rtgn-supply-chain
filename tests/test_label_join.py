"""Unit tests for label joining rules and canonical normalization."""

from __future__ import annotations

from pathlib import Path
import pytest
from src.data.pipeline import SupplyChainDataPipeline


def test_label_join_all_versions_rule(tmp_path: Path):
    pipeline = SupplyChainDataPipeline(
        raw_dir=tmp_path / "raw",
        intermediate_dir=tmp_path / "inter",
        processed_dir=tmp_path / "proc",
    )

    pypi_rows = [
        {"package": "Malware_Pkg", "version": "1.0.0", "release_timestamp": 1600000000.0},
        {"package": "Malware_Pkg", "version": "1.1.0", "release_timestamp": 1610000000.0},
    ]

    raw_labels = [
        {
            "package": "malware-pkg",
            "version": "__ALL__",
            "label": "confirmed_malicious",
            "label_scope": "all_versions",
            "source": "datadog",
        }
    ]

    processed, mal_labels, stats = pipeline._join_labels(
        pypi_rows, raw_labels, ["Malware_Pkg"]
    )

    # Every PyPI version of Malware_Pkg should be marked confirmed_malicious with all_versions scope
    assert len(processed) == 2
    for r in processed:
        assert r["label"] == "confirmed_malicious"
        assert r["label_scope"] == "all_versions"
        assert r["label_source"] == "datadog"


def test_label_join_specific_version_rule(tmp_path: Path):
    pipeline = SupplyChainDataPipeline(
        raw_dir=tmp_path / "raw",
        intermediate_dir=tmp_path / "inter",
        processed_dir=tmp_path / "proc",
    )

    pypi_rows = [
        {"package": "Legit-Pkg", "version": "1.0.0", "release_timestamp": 1600000000.0},
        {"package": "Legit-Pkg", "version": "1.0.1", "release_timestamp": 1610000000.0},  # compromised
        {"package": "Legit-Pkg", "version": "1.0.2", "release_timestamp": 1620000000.0},
    ]

    raw_labels = [
        {
            "package": "legit.pkg",  # PEP 503 normalization test
            "version": "1.0.1",
            "label": "confirmed_malicious",
            "label_scope": "specific_version",
            "source": "datadog",
        }
    ]

    processed, mal_labels, stats = pipeline._join_labels(
        pypi_rows, raw_labels, ["Legit-Pkg"]
    )

    assert len(processed) == 3

    v100 = next(r for r in processed if r["version"] == "1.0.0")
    v101 = next(r for r in processed if r["version"] == "1.0.1")
    v102 = next(r for r in processed if r["version"] == "1.0.2")

    # ONLY 1.0.1 is confirmed_malicious
    assert v101["label"] == "confirmed_malicious"
    assert v101["label_scope"] == "specific_version"

    # Other versions MUST NOT be labeled malicious
    assert v100["label"] == "unlabeled / assumed_benign"
    assert v100["label_scope"] == "unlabeled"

    assert v102["label"] == "unlabeled / assumed_benign"
    assert v102["label_scope"] == "unlabeled"
