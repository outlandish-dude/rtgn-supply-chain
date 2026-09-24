"""Unit tests for OSV collector and GitHub advisory adapter."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from src.data.advisories.github_adapter import GitHubAdvisoryAdapter
from src.data.osv.collector import OSVCollector


def test_github_adapter_unauthenticated_deferred():
    adapter = GitHubAdvisoryAdapter(token=None)
    assert not adapter.is_authenticated
    status = adapter.status()
    assert status["status"] == "deferred_missing_credentials"
    advisories, meta = adapter.query_package_advisories("requests")
    assert advisories == []
    assert meta["status"] == "deferred_missing_credentials"


def test_osv_collector_caching(tmp_path: Path):
    collector = OSVCollector(raw_dir=tmp_path / "raw_osv", output_path=tmp_path / "out.csv")
    vuln_id = "GHSA-test-1234"
    cache_path = collector._get_vuln_cache_path(vuln_id)

    mock_vuln = {
        "id": vuln_id,
        "summary": "Mock vulnerability",
        "published": "2023-01-01T00:00:00Z",
        "modified": "2023-01-02T00:00:00Z",
        "affected": [],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(mock_vuln, f)

    res = collector.fetch_vuln_detail(vuln_id)
    assert res is not None
    assert res["id"] == vuln_id
    assert res["summary"] == "Mock vulnerability"
