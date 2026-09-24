"""
End-to-End Software Supply-Chain Ingestion Pipeline.
Orchestrates:
1. Datadog Manifest Ingestion & Label Parsing
2. Safe PyPI Metadata Collection with release-specific dependency isolation
3. OSV Advisory Querying at exact package-version granularity
4. Deterministic Label Joining obeying __ALL__ vs explicit version rules
5. Persistence of intermediate and final processed CSV/JSON artifacts
Strictly non-executable and temporally leak-free.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from packaging.requirements import InvalidRequirement, Requirement

from src.data.advisories.github_adapter import GitHubAdvisoryAdapter
from src.data.datadog.adapter import DatadogManifestAdapter
from src.data.normalization import normalize_package_name, normalize_version
from src.data.osv.collector import OSVCollector
from src.data.pypi.collector import PyPICollector

logger = logging.getLogger(__name__)


class SupplyChainDataPipeline:
    """
    Complete reproducible ingestion pipeline for software supply-chain data.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        intermediate_dir: str | Path = "data/intermediate",
        processed_dir: str | Path = "data/processed",
        cache_enabled: bool = True,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.intermediate_dir = Path(intermediate_dir)
        self.processed_dir = Path(processed_dir)
        self.cache_enabled = cache_enabled

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.datadog_adapter = DatadogManifestAdapter(
            manifest_path=self.raw_dir / "datadog" / "pypi_manifest.json",
            output_path=self.intermediate_dir / "datadog_pypi_labels.csv",
        )
        self.pypi_collector = PyPICollector(
            raw_dir=self.raw_dir / "pypi",
            output_path=self.intermediate_dir / "package_versions.csv",
            cache_enabled=cache_enabled,
        )
        self.osv_collector = OSVCollector(
            raw_dir=self.raw_dir / "osv",
            output_path=self.intermediate_dir / "osv_events.csv",
            cache_enabled=cache_enabled,
        )
        self.github_adapter = GitHubAdvisoryAdapter(
            raw_dir=self.raw_dir / "advisories",
            cache_enabled=cache_enabled,
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute full reproducible pipeline.
        """
        logger.info("Step 1: Parsing and exporting Datadog manifest labels...")
        manifest_val = self.datadog_adapter.validate()
        self.datadog_adapter.export_csv()
        raw_labels = self.datadog_adapter.parse_records()

        # Unique package names from manifest
        manifest_data = self.datadog_adapter.load_manifest()
        all_manifest_packages = list(manifest_data.keys())

        logger.info(f"Step 2: Collecting PyPI metadata for {len(all_manifest_packages)} packages...")
        pypi_rows, pypi_stats = self.pypi_collector.collect(all_manifest_packages)
        self.pypi_collector.export_csv(pypi_rows)

        # Build list of package-version tuples for OSV query
        # 1. From collected PyPI releases
        osv_targets: List[Tuple[str, str]] = [
            (row["package"], row["version"]) for row in pypi_rows
        ]
        # 2. For packages not found on PyPI (e.g. 404 malware), query package at __ALL__
        found_pkg_names = {normalize_package_name(r["package"]) for r in pypi_rows}
        for pkg in all_manifest_packages:
            if normalize_package_name(pkg) not in found_pkg_names:
                osv_targets.append((pkg, "__ALL__"))

        logger.info(f"Step 3: Querying OSV at package-version granularity ({len(osv_targets)} items)...")
        osv_events, osv_stats = self.osv_collector.collect_for_package_versions(osv_targets)
        self.osv_collector.export_csv(osv_events)

        logger.info("Step 4: Checking GitHub Advisory Adapter status...")
        gh_status = self.github_adapter.status()

        logger.info("Step 5: Joining Datadog labels onto PyPI version table...")
        processed_versions, malicious_labels_rows, label_stats = self._join_labels(
            pypi_rows, raw_labels, all_manifest_packages
        )

        logger.info("Step 6: Constructing security events table...")
        security_events = self._build_security_events(osv_events)

        logger.info("Step 7: Persisting processed datasets...")
        self._export_processed_versions(processed_versions)
        self._export_malicious_labels(malicious_labels_rows)
        self._export_security_events(security_events)

        logger.info("Step 8: Generating machine-readable ingestion summary...")
        summary = self._generate_ingestion_summary(
            all_manifest_packages=all_manifest_packages,
            processed_versions=processed_versions,
            malicious_labels_rows=malicious_labels_rows,
            security_events=security_events,
            osv_events=osv_events,
            pypi_stats=pypi_stats,
            osv_stats=osv_stats,
            manifest_val=manifest_val,
            gh_status=gh_status,
        )

        summary_path = self.processed_dir / "ingestion_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary written to {summary_path}")

        return summary

    def _join_labels(
        self,
        pypi_rows: List[Dict[str, Any]],
        raw_labels: List[Dict[str, str]],
        all_manifest_packages: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Join Datadog labels onto PyPI version rows:
        - If package is marked __ALL__: all collected PyPI versions get confirmed_malicious (all_versions)
        - If package has explicit versions: only those exact versions get confirmed_malicious (specific_version)
        - All other versions get unlabeled / assumed_benign
        """
        all_version_pkgs: Set[str] = set()
        specific_version_map: Dict[Tuple[str, str], Dict[str, str]] = {}

        for l in raw_labels:
            norm_pkg = normalize_package_name(l["package"])
            if l["version"] == "__ALL__":
                all_version_pkgs.add(norm_pkg)
            else:
                norm_ver, _ = normalize_version(l["version"])
                specific_version_map[(norm_pkg, norm_ver)] = l

        processed_versions: List[Dict[str, Any]] = []
        confirmed_malicious_versions_count = 0
        unlabeled_versions_count = 0

        for r in pypi_rows:
            norm_pkg = normalize_package_name(r["package"])
            norm_ver, _ = normalize_version(r["version"])

            if norm_pkg in all_version_pkgs:
                label = "confirmed_malicious"
                label_scope = "all_versions"
                label_source = "datadog"
                confirmed_malicious_versions_count += 1
            elif (norm_pkg, norm_ver) in specific_version_map:
                label = "confirmed_malicious"
                label_scope = "specific_version"
                label_source = "datadog"
                confirmed_malicious_versions_count += 1
            else:
                label = "unlabeled / assumed_benign"
                label_scope = "unlabeled"
                label_source = "none"
                unlabeled_versions_count += 1

            processed_versions.append({
                "package": r["package"],
                "normalized_package": norm_pkg,
                "version": r["version"],
                "release_timestamp": r["release_timestamp"],
                "label": label,
                "label_scope": label_scope,
                "label_source": label_source,
                "requires_python": r.get("requires_python", ""),
                "requires_dist": r.get("requires_dist", ""),
                "repository_url": r.get("repository_url", ""),
                "homepage_url": r.get("homepage_url", ""),
                "source": "pypi",
            })

        # Build malicious_labels.csv covering all Datadog confirmed malicious rows
        found_norm_pkgs = {normalize_package_name(r["package"]) for r in pypi_rows}
        malicious_labels_rows: List[Dict[str, Any]] = []

        for l in raw_labels:
            norm_pkg = normalize_package_name(l["package"])
            norm_ver, _ = normalize_version(l["version"])
            in_pypi = norm_pkg in found_norm_pkgs
            malicious_labels_rows.append({
                "package": l["package"],
                "normalized_package": norm_pkg,
                "version": l["version"],
                "label": l["label"],
                "label_scope": l["label_scope"],
                "source": l["source"],
                "in_pypi_registry": in_pypi,
            })

        stats = {
            "confirmed_malicious_versions": confirmed_malicious_versions_count,
            "unlabeled_versions": unlabeled_versions_count,
            "total_malicious_label_rows": len(malicious_labels_rows),
        }
        return processed_versions, malicious_labels_rows, stats

    def _build_security_events(
        self, osv_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize OSV events into canonical security events.
        """
        security_events: List[Dict[str, Any]] = []
        for idx, ev in enumerate(osv_events):
            v_id = ev["osv_id"]
            pkg = ev["package"]
            norm_pkg = normalize_package_name(pkg)
            ver = ev["version"]
            norm_ver, _ = normalize_version(ver)
            event_id = f"osv:{v_id}@{norm_pkg}@{norm_ver}"

            is_malware = v_id.startswith("MAL-") or "malicious" in ev.get("summary", "").lower()
            event_type = "malware_report" if is_malware else "vulnerability_advisory"

            security_events.append({
                "event_id": event_id,
                "event_type": event_type,
                "package": pkg,
                "normalized_package": norm_pkg,
                "version": ver,
                "advisory_id": v_id,
                "published_timestamp": ev["published_timestamp"],
                "modified_timestamp": ev["modified_timestamp"],
                "summary": ev.get("summary", ""),
                "affected_ranges": ev.get("affected_ranges", ""),
                "references": ev.get("references", ""),
                "source": "osv",
            })
        return security_events

    def _export_processed_versions(self, rows: List[Dict[str, Any]]) -> Path:
        out_path = self.processed_dir / "package_versions.csv"
        fieldnames = [
            "package",
            "normalized_package",
            "version",
            "release_timestamp",
            "label",
            "label_scope",
            "label_source",
            "requires_python",
            "requires_dist",
            "repository_url",
            "homepage_url",
            "source",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return out_path

    def _export_malicious_labels(self, rows: List[Dict[str, Any]]) -> Path:
        out_path = self.processed_dir / "malicious_labels.csv"
        fieldnames = [
            "package",
            "normalized_package",
            "version",
            "label",
            "label_scope",
            "source",
            "in_pypi_registry",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return out_path

    def _export_security_events(self, rows: List[Dict[str, Any]]) -> Path:
        out_path = self.processed_dir / "security_events.csv"
        fieldnames = [
            "event_id",
            "event_type",
            "package",
            "normalized_package",
            "version",
            "advisory_id",
            "published_timestamp",
            "modified_timestamp",
            "summary",
            "affected_ranges",
            "references",
            "source",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return out_path

    def _generate_ingestion_summary(
        self,
        all_manifest_packages: List[str],
        processed_versions: List[Dict[str, Any]],
        malicious_labels_rows: List[Dict[str, Any]],
        security_events: List[Dict[str, Any]],
        osv_events: List[Dict[str, Any]],
        pypi_stats: Dict[str, Any],
        osv_stats: Dict[str, Any],
        manifest_val: Dict[str, Any],
        gh_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate final JSON summary."""
        # Calculate dependency edges from requires_dist
        dependency_edges_count = 0
        for r in processed_versions:
            req_dist_str = r.get("requires_dist")
            if req_dist_str:
                try:
                    req_list = json.loads(req_dist_str)
                    if isinstance(req_list, list):
                        dependency_edges_count += len(req_list)
                except Exception:
                    pass

        # Calculate date range
        timestamps = [
            r["release_timestamp"]
            for r in processed_versions
            if r.get("release_timestamp") not in (None, "")
        ]
        earliest_ts = min(timestamps) if timestamps else None
        latest_ts = max(timestamps) if timestamps else None

        earliest_iso = (
            datetime.fromtimestamp(earliest_ts, tz=timezone.utc).isoformat()
            if earliest_ts
            else None
        )
        latest_iso = (
            datetime.fromtimestamp(latest_ts, tz=timezone.utc).isoformat()
            if latest_ts
            else None
        )

        # Missing metadata counts
        missing_release_ts = sum(
            1 for r in processed_versions if r.get("release_timestamp") in (None, "")
        )
        missing_requires_dist = sum(
            1 for r in processed_versions if r.get("requires_dist") in (None, "", "[]")
        )
        missing_repo_url = sum(
            1 for r in processed_versions if not r.get("repository_url")
        )

        malicious_pkgs_set = {
            r["normalized_package"]
            for r in malicious_labels_rows
        }
        malicious_versions_count = sum(
            1 for r in processed_versions if r["label"] == "confirmed_malicious"
        )
        unlabeled_versions_count = sum(
            1 for r in processed_versions if r["label"] == "unlabeled / assumed_benign"
        )
        unique_vuln_ids = len({r["osv_id"] for r in osv_events if r.get("osv_id")})

        return {
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "packages": len(all_manifest_packages),
            "versions": len(processed_versions),
            "malicious_packages": len(malicious_pkgs_set),
            "malicious_versions": malicious_versions_count,
            "unlabeled_versions": unlabeled_versions_count,
            "vulnerabilities": unique_vuln_ids,
            "security_events": len(security_events),
            "dependency_edges": dependency_edges_count,
            "api_failures": {
                "pypi_failures": pypi_stats.get("api_failures", 0),
                "osv_failures": osv_stats.get("api_failures", 0),
            },
            "missing_metadata": {
                "versions_missing_release_timestamp": missing_release_ts,
                "versions_missing_requires_dist": missing_requires_dist,
                "versions_missing_repository_url": missing_repo_url,
            },
            "date_range": {
                "earliest_release_timestamp": earliest_ts,
                "earliest_release_iso": earliest_iso,
                "latest_release_timestamp": latest_ts,
                "latest_release_iso": latest_iso,
            },
            "sources_used": [
                "datadog_pypi_manifest (local file: data/raw/datadog/pypi_manifest.json)",
                "pypi_json_api (https://pypi.org/pypi/<package>/json)",
                "pypi_release_json_api (https://pypi.org/pypi/<package>/<version>/json)",
                "osv_querybatch_api (https://api.osv.dev/v1/querybatch)",
                "osv_vulns_api (https://api.osv.dev/v1/vulns/<id>)",
                f"github_advisory_database ({gh_status['status']})",
            ],
            "github_advisory_status": gh_status,
            "pypi_registry_status": {
                "found_on_pypi": pypi_stats.get("found_on_pypi", 0),
                "not_found_404_takedown": pypi_stats.get("not_found_404", 0),
            },
            "datadog_manifest_validation": manifest_val,
        }
