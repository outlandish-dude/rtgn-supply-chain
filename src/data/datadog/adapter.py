"""
Datadog PyPI Manifest Ingestion Adapter.
Parses data/raw/datadog/pypi_manifest.json and generates
data/intermediate/datadog_pypi_labels.csv adhering to strict label scoping:
- null -> version=__ALL__, label=confirmed_malicious, label_scope=all_versions, source=datadog
- list -> one row per version, label=confirmed_malicious, label_scope=specific_version, source=datadog
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.data.normalization import normalize_package_name, normalize_version

logger = logging.getLogger(__name__)


class DatadogManifestAdapter:
    """
    Adapter to parse and normalize the Datadog PyPI malware manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path = "data/raw/datadog/pypi_manifest.json",
        output_path: str | Path = "data/intermediate/datadog_pypi_labels.csv",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.output_path = Path(output_path)

    def load_manifest(self) -> Dict[str, Optional[List[str]]]:
        """Load raw JSON manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path.resolve()}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def parse_records(self) -> List[Dict[str, str]]:
        """
        Parse manifest into list of label records.
        """
        data = self.load_manifest()
        records: List[Dict[str, str]] = []

        for pkg_name, versions in data.items():
            if versions is None:
                records.append({
                    "package": pkg_name,
                    "version": "__ALL__",
                    "label": "confirmed_malicious",
                    "label_scope": "all_versions",
                    "source": "datadog",
                })
            elif isinstance(versions, list):
                for ver in versions:
                    records.append({
                        "package": pkg_name,
                        "version": str(ver),
                        "label": "confirmed_malicious",
                        "label_scope": "specific_version",
                        "source": "datadog",
                    })
            else:
                logger.warning(f"Unexpected entry format for {pkg_name}: {versions}")

        return records

    def export_csv(self) -> Path:
        """
        Parse and export to CSV.
        """
        records = self.parse_records()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["package", "version", "label", "label_scope", "source"]

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Exported {len(records)} label records to {self.output_path}")
        return self.output_path

    def validate(self) -> Dict[str, Any]:
        """
        Execute comprehensive validation checks on the manifest and parsed rows.
        """
        data = self.load_manifest()
        total_packages = len(data)
        all_version_pkgs = sum(1 for v in data.values() if v is None)
        specific_version_pkgs = sum(1 for v in data.values() if isinstance(v, list))
        
        explicit_versions: List[Tuple[str, str]] = []
        for pkg, v_list in data.items():
            if isinstance(v_list, list):
                for ver in v_list:
                    explicit_versions.append((pkg, str(ver)))

        records = self.parse_records()
        total_normalized_rows = len(records)

        # Check duplicates
        seen_pairs = set()
        duplicate_rows: List[Tuple[str, str]] = []
        for r in records:
            norm_pkg = normalize_package_name(r["package"])
            norm_ver, _ = normalize_version(r["version"])
            pair = (norm_pkg, norm_ver)
            if pair in seen_pairs:
                duplicate_rows.append((r["package"], r["version"]))
            seen_pairs.add(pair)

        # Check invalid/empty package names
        invalid_package_names = [
            pkg for pkg in data.keys() if not pkg or not isinstance(pkg, str) or not pkg.strip()
        ]

        # Check invalid PEP 440 version strings among explicit versions
        invalid_versions: List[Tuple[str, str]] = []
        for pkg, ver in explicit_versions:
            _, is_valid = normalize_version(ver)
            if not is_valid:
                invalid_versions.append((pkg, ver))

        return {
            "total_manifest_packages": total_packages,
            "all_version_malicious_packages": all_version_pkgs,
            "specific_version_packages": specific_version_pkgs,
            "total_explicit_malicious_versions": len(explicit_versions),
            "total_normalized_label_rows": total_normalized_rows,
            "duplicate_package_version_rows": len(duplicate_rows),
            "duplicate_samples": duplicate_rows[:5],
            "invalid_empty_package_names": len(invalid_package_names),
            "invalid_package_samples": invalid_package_names[:5],
            "invalid_version_strings": len(invalid_versions),
            "invalid_version_samples": invalid_versions[:5],
        }
