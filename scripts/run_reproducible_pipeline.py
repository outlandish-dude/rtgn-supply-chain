"""
CLI script to run the full reproducible software supply-chain data ingestion pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import SupplyChainDataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    start_time = time.time()
    logger.info("Initializing reproducible supply-chain data pipeline...")

    pipeline = SupplyChainDataPipeline(
        raw_dir=PROJECT_ROOT / "data" / "raw",
        intermediate_dir=PROJECT_ROOT / "data" / "intermediate",
        processed_dir=PROJECT_ROOT / "data" / "processed",
        cache_enabled=True,
    )

    summary = pipeline.run()
    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("DATA PIPELINE INGESTION SUMMARY")
    print("=" * 65)
    print(f"Total manifest packages:        {summary['packages']}")
    print(f"Packages found on PyPI:         {summary['pypi_registry_status']['found_on_pypi']}")
    print(f"Packages 404/takedown on PyPI:  {summary['pypi_registry_status']['not_found_404_takedown']}")
    print(f"Total collected versions:       {summary['versions']}")
    print(f"Confirmed malicious packages:   {summary['malicious_packages']}")
    print(f"Confirmed malicious versions:   {summary['malicious_versions']}")
    print(f"Unlabeled / assumed benign ver: {summary['unlabeled_versions']}")
    print(f"Unique OSV vulnerabilities:     {summary['vulnerabilities']}")
    print(f"Security events recorded:       {summary['security_events']}")
    print(f"Dependency relationship edges:  {summary['dependency_edges']}")
    print(f"API failures:                   {summary['api_failures']}")
    print(f"Date range:                     {summary['date_range']['earliest_release_iso']} -> {summary['date_range']['latest_release_iso']}")
    print(f"Pipeline execution time:        {elapsed:.2f} seconds")
    print("=" * 65)
    print(f"Summary JSON saved to: {PROJECT_ROOT / 'data' / 'processed' / 'ingestion_summary.json'}")


if __name__ == "__main__":
    main()
