"""
Validation script for Datadog PyPI Malware Manifest.
Prints the 8 required validation metrics and exports data/intermediate/datadog_pypi_labels.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datadog.adapter import DatadogManifestAdapter


def main() -> None:
    manifest_path = PROJECT_ROOT / "data" / "raw" / "datadog" / "pypi_manifest.json"
    output_path = PROJECT_ROOT / "data" / "intermediate" / "datadog_pypi_labels.csv"

    adapter = DatadogManifestAdapter(manifest_path=manifest_path, output_path=output_path)
    val = adapter.validate()
    exported = adapter.export_csv()

    print("=" * 65)
    print("DATADOG PYPI MANIFEST VALIDATION REPORT")
    print("=" * 65)
    print(f"Total manifest packages:             {val['total_manifest_packages']}")
    print(f"All-version malicious package count:  {val['all_version_malicious_packages']}")
    print(f"Specific-version package count:      {val['specific_version_packages']}")
    print(f"Total explicit malicious-version count: {val['total_explicit_malicious_versions']}")
    print(f"Total normalized label rows:         {val['total_normalized_label_rows']}")
    print(f"Duplicate package/version rows:      {val['duplicate_package_version_rows']}")
    print(f"Invalid/empty package names:         {val['invalid_empty_package_names']}")
    print(f"Invalid version strings:             {val['invalid_version_strings']}")
    print("=" * 65)
    print(f"Successfully exported normalized labels to: {exported}")


if __name__ == "__main__":
    main()
