"""
Safe PyPI Metadata Collector.
Queries the official PyPI JSON API for package metadata, release history, timestamps,
release-specific dependencies (without backwards temporal leakage), and URLs.
Implements robust disk caching (including negative 404 caching) in data/raw/pypi/.
Strictly non-executable: never downloads source distributions or wheels.
Supports multi-threaded batch fetching with ThreadPoolExecutor for fast execution.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.data.normalization import normalize_package_name, normalize_repo_url, normalize_version

logger = logging.getLogger(__name__)


class PyPICollector:
    """
    Safe, resilient and fast metadata collector for the official PyPI registry.
    """

    PACKAGE_URL = "https://pypi.org/pypi/{package}/json"
    RELEASE_URL = "https://pypi.org/pypi/{package}/{version}/json"

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/pypi",
        output_path: str | Path = "data/intermediate/package_versions.csv",
        cache_enabled: bool = True,
        max_workers: int = 10,
        request_delay: float = 0.02,
        timeout: float = 12.0,
        max_retries: int = 3,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = Path(output_path)
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries

        # Thread-local storage for sessions to avoid thread contention
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        if not hasattr(self._thread_local, "session"):
            session = requests.Session()
            retries = Retry(
                total=self.max_retries,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update({"User-Agent": "SupplyChainGNN-Research/0.2.0"})
            self._thread_local.session = session
        return self._thread_local.session

    def _get_pkg_cache_path(self, package_name: str) -> Path:
        norm = normalize_package_name(package_name)
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", norm)[:64]
        return self.raw_dir / f"pkg_{safe_name}.json"

    def _get_release_cache_path(self, package_name: str, version: str) -> Path:
        norm_pkg = normalize_package_name(package_name)
        norm_ver, _ = normalize_version(version)
        safe_pkg = re.sub(r"[^a-zA-Z0-9_\-]", "_", norm_pkg)[:48]
        safe_ver = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", norm_ver)[:32]
        return self.raw_dir / f"rel_{safe_pkg}_{safe_ver}.json"

    def fetch_package_data(self, package_name: str) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Fetch package level metadata from PyPI JSON API with disk caching.
        Returns (payload, status_code). If 404, payload is None.
        """
        cache_path = self._get_pkg_cache_path(package_name)

        if self.cache_enabled and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                status = cached.get("status_code", 200)
                if status == 404:
                    return None, 404
                return cached.get("payload"), 200
            except Exception as e:
                logger.warning(f"Error reading cache {cache_path}: {e}")

        url = self.PACKAGE_URL.format(package=package_name)
        session = self._get_session()
        try:
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            resp = session.get(url, timeout=self.timeout)

            if resp.status_code == 404:
                if self.cache_enabled:
                    cache_obj = {
                        "package": package_name,
                        "status_code": 404,
                        "url": url,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_obj, f, indent=2)
                return None, 404

            resp.raise_for_status()
            payload = resp.json()

            if self.cache_enabled:
                cache_obj = {
                    "package": package_name,
                    "status_code": resp.status_code,
                    "url": url,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "payload": payload,
                }
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_obj, f, indent=2)

            return payload, 200

        except Exception as e:
            logger.error(f"Failed to fetch PyPI package data for {package_name}: {e}")
            return None, 500

    def fetch_release_specific_data(
        self, package_name: str, version: str
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Fetch version-specific metadata from PyPI JSON API with disk caching.
        Used to strictly obtain release-specific requires_dist and prevent backwards leakage.
        """
        cache_path = self._get_release_cache_path(package_name, version)

        if self.cache_enabled and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                status = cached.get("status_code", 200)
                if status == 404:
                    return None, 404
                return cached.get("payload"), 200
            except Exception as e:
                logger.warning(f"Error reading release cache {cache_path}: {e}")

        url = self.RELEASE_URL.format(package=package_name, version=version)
        session = self._get_session()
        try:
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            resp = session.get(url, timeout=self.timeout)

            if resp.status_code == 404:
                if self.cache_enabled:
                    cache_obj = {
                        "package": package_name,
                        "version": version,
                        "status_code": 404,
                        "url": url,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_obj, f, indent=2)
                return None, 404

            resp.raise_for_status()
            payload = resp.json()

            if self.cache_enabled:
                cache_obj = {
                    "package": package_name,
                    "version": version,
                    "status_code": resp.status_code,
                    "url": url,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "payload": payload,
                }
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_obj, f, indent=2)

            return payload, 200

        except Exception as e:
            logger.debug(f"Failed to fetch release metadata for {package_name}@{version}: {e}")
            return None, 500

    def parse_package_releases(
        self, package_name: str, payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract releases, timestamps, and metadata for a package payload.
        Strictly release-specific: does not copy latest info.requires_dist to older releases.
        """
        info = payload.get("info", {})
        raw_name = info.get("name") or package_name
        norm_name = normalize_package_name(raw_name)

        latest_version = info.get("version")
        latest_requires_dist = info.get("requires_dist")

        # Project URLs
        project_urls = info.get("project_urls") or {}
        repo_candidate = (
            project_urls.get("Source")
            or project_urls.get("Repository")
            or project_urls.get("Source Code")
            or project_urls.get("Code")
            or info.get("home_page")
        )
        repo_url = normalize_repo_url(repo_candidate) if repo_candidate else None
        homepage_url = project_urls.get("Homepage") or info.get("home_page")

        releases = payload.get("releases", {})
        version_rows: List[Dict[str, Any]] = []

        for ver_str, file_list in releases.items():
            norm_ver, _ = normalize_version(ver_str)

            # Earliest upload timestamp among distribution files
            earliest_upload_ts: Optional[float] = None
            requires_python: Optional[str] = None

            if file_list:
                timestamps: List[float] = []
                for file_info in file_list:
                    upload_iso = file_info.get("upload_time_iso_8601") or file_info.get("upload_time")
                    if upload_iso:
                        try:
                            dt = datetime.fromisoformat(upload_iso.replace("Z", "+00:00"))
                            timestamps.append(dt.timestamp())
                        except Exception:
                            pass
                    if not requires_python and file_info.get("requires_python"):
                        requires_python = file_info.get("requires_python")

                if timestamps:
                    earliest_upload_ts = min(timestamps)

            # Determine requires_dist strictly per release:
            requires_dist: Optional[List[str]] = None
            if ver_str == latest_version and latest_requires_dist is not None:
                requires_dist = latest_requires_dist
            else:
                # Query version-specific endpoint for historical releases
                rel_payload, rel_code = self.fetch_release_specific_data(raw_name, ver_str)
                if rel_payload:
                    rel_info = rel_payload.get("info", {})
                    requires_dist = rel_info.get("requires_dist")
                    if not requires_python:
                        requires_python = rel_info.get("requires_python")
                else:
                    # Explicitly missing: do not backfill from latest release
                    requires_dist = None

            version_rows.append({
                "package": raw_name,
                "normalized_package": norm_name,
                "version": ver_str,
                "release_timestamp": earliest_upload_ts,
                "requires_python": requires_python or "",
                "requires_dist": json.dumps(requires_dist) if requires_dist is not None else "",
                "repository_url": repo_url or "",
                "homepage_url": homepage_url or "",
                "source": "pypi",
            })

        return version_rows

    def collect(
        self, package_names: List[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Collect metadata for a list of package names using ThreadPoolExecutor for speed.
        Returns (rows, stats).
        """
        unique_pkgs = sorted(list(set(package_names)))
        logger.info(
            f"Collecting PyPI metadata for {len(unique_pkgs)} unique packages (max_workers={self.max_workers})..."
        )

        pkg_results: Dict[str, Tuple[Optional[Dict[str, Any]], int]] = {}
        processed_count = 0

        # Concurrent package fetching
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pkg = {
                executor.submit(self.fetch_package_data, pkg): pkg
                for pkg in unique_pkgs
            }
            for future in as_completed(future_to_pkg):
                pkg = future_to_pkg[future]
                processed_count += 1
                try:
                    payload, status = future.result()
                    pkg_results[pkg] = (payload, status)
                except Exception as e:
                    logger.error(f"Worker exception for package {pkg}: {e}")
                    pkg_results[pkg] = (None, 500)

                if processed_count % 200 == 0 or processed_count == len(unique_pkgs):
                    logger.info(f"Progress: [{processed_count}/{len(unique_pkgs)}] PyPI packages queried.")

        rows: List[Dict[str, Any]] = []
        stats = {
            "total_queried": len(unique_pkgs),
            "found_on_pypi": 0,
            "not_found_404": 0,
            "api_failures": 0,
            "total_versions_collected": 0,
        }

        # Process found packages deterministically in sorted order
        for pkg in unique_pkgs:
            payload, status = pkg_results.get(pkg, (None, 500))
            if status == 200 and payload:
                stats["found_on_pypi"] += 1
                pkg_rows = self.parse_package_releases(pkg, payload)
                rows.extend(pkg_rows)
                stats["total_versions_collected"] += len(pkg_rows)
            elif status == 404:
                stats["not_found_404"] += 1
            else:
                stats["api_failures"] += 1

        return rows, stats

    def export_csv(self, rows: List[Dict[str, Any]]) -> Path:
        """Export collected version rows to CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "package",
            "normalized_package",
            "version",
            "release_timestamp",
            "requires_python",
            "requires_dist",
            "repository_url",
            "homepage_url",
            "source",
        ]
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} package version records to {self.output_path}")
        return self.output_path
