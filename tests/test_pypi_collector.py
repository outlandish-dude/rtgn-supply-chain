"""Unit tests for PyPI metadata collector and temporal non-leakage."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from src.data.pypi.collector import PyPICollector


def test_pypi_collector_caching(tmp_path: Path):
    collector = PyPICollector(raw_dir=tmp_path / "raw_pypi", output_path=tmp_path / "out.csv")
    pkg_name = "test-pkg"
    cache_path = collector._get_pkg_cache_path(pkg_name)

    # Pre-populate cache
    cache_data = {
        "package": pkg_name,
        "status_code": 200,
        "payload": {
            "info": {"name": pkg_name, "version": "1.0.0"},
            "releases": {"1.0.0": []},
        },
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)

    payload, status = collector.fetch_package_data(pkg_name)
    assert status == 200
    assert payload["info"]["name"] == pkg_name


def test_pypi_negative_caching_404(tmp_path: Path):
    collector = PyPICollector(raw_dir=tmp_path / "raw_pypi", output_path=tmp_path / "out.csv")
    pkg_name = "malware-removed-pkg"
    cache_path = collector._get_pkg_cache_path(pkg_name)

    # Pre-populate negative 404 cache
    cache_data = {
        "package": pkg_name,
        "status_code": 404,
        "message": "Not Found",
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)

    payload, status = collector.fetch_package_data(pkg_name)
    assert status == 404
    assert payload is None


def test_pypi_no_backward_dependency_leakage(tmp_path: Path):
    """
    CRITICAL TEMPORAL INTEGRITY TEST:
    Verify that an older release (0.1.0) does NOT inherit requires_dist
    from the latest release (2.0.0) when older release metadata is unavailable.
    """
    collector = PyPICollector(raw_dir=tmp_path / "raw_pypi", output_path=tmp_path / "out.csv")
    
    mock_payload = {
        "info": {
            "name": "example-pkg",
            "version": "2.0.0",
            "requires_dist": ["cryptography>=40.0.0", "requests>=2.28.0"],
            "project_urls": {"Homepage": "https://example.com"},
        },
        "releases": {
            "0.1.0": [
                {
                    "upload_time_iso_8601": "2020-01-01T00:00:00Z",
                    "filename": "example-pkg-0.1.0.tar.gz",
                }
            ],
            "2.0.0": [
                {
                    "upload_time_iso_8601": "2023-01-01T00:00:00Z",
                    "filename": "example-pkg-2.0.0.tar.gz",
                }
            ],
        },
    }

    # Simulate release-specific cache for 0.1.0 as having 404 (no release json)
    rel_cache = collector._get_release_cache_path("example-pkg", "0.1.0")
    with open(rel_cache, "w", encoding="utf-8") as f:
        json.dump({"status_code": 404}, f)

    rows = collector.parse_package_releases("example-pkg", mock_payload)

    v010 = next(r for r in rows if r["version"] == "0.1.0")
    v200 = next(r for r in rows if r["version"] == "2.0.0")

    # 2.0.0 should have requires_dist
    assert "cryptography" in v200["requires_dist"]

    # 0.1.0 MUST NOT have 2.0.0's dependencies copied backward!
    assert v010["requires_dist"] == ""  # explicitly missing / empty
    assert v010["release_timestamp"] == 1577836800.0  # Jan 1 2020
