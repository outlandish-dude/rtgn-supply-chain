"""
Background Population Sampler.
Implements leak-free, objective background sampling for the software supply-chain dataset.
Eligibility, stratification metrics, and hard-negative selection are computed STRICTLY
using information available on or before T_train_val = 2025-06-01 (Epoch: 1748736000.0).
Strictly non-executable: fetches JSON metadata only.
"""

from __future__ import annotations

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

# Strict cutoff for background eligibility and stratification
T_TRAIN_VAL_CUTOFF = 1748736000.0  # 2025-06-01 00:00:00 UTC

# Component B: Top 40 Clean Historical Hard-Negative Dependency Targets
# Extracted empirically from dependencies declared by packages prior to 2025-06-01
HARD_NEGATIVE_TARGETS = [
    "requests", "pandas", "numpy", "pytest", "pydantic",
    "torchvision", "psutil", "ipython", "pytest-cov", "scikit-learn",
    "coverage", "torch", "torchmetrics", "bitsandbytes", "lightning-utilities",
    "matplotlib", "click", "python-dateutil", "pyyaml", "fastparquet",
    "uvicorn", "fastapi", "python-json-logger", "tensorboardx", "deepspeed",
    "rich", "catboost", "tiktoken", "tqdm", "openai",
    "pytest-timeout", "tensorboard", "pytest-rerunfailures", "pyjwt", "pillow",
    "onnx", "omegaconf", "hydra-core", "scipy", "jsonargparse"
]

# Component A: 60 General Stratified Background Candidates across 4 strata
GENERAL_BACKGROUND_CANDIDATES = [
    # Stratum 1 candidates: High-centrality infrastructure
    "setuptools", "wheel", "pip", "certifi", "urllib3",
    "six", "packaging", "idna", "charset-normalizer", "colorama",
    "jinja2", "markupsafe", "werkzeug", "attrs", "typing-extensions",
    # Stratum 2 candidates: Active domain utilities
    "httpx", "sqlalchemy", "aiohttp", "anyio", "starlette",
    "sniffio", "yarl", "multidict", "websockets", "cryptography",
    "bcrypt", "redis", "celery", "flake8", "mypy",
    # Stratum 3 candidates: Stable low-velocity utilities
    "decorator", "wrapt", "flit-core", "tomli", "iniconfig",
    "pluggy", "pyparsing", "docutils", "termcolor", "tabulate",
    "bleach", "inflection", "sortedcontainers", "more-itertools", "pyasn1",
    # Stratum 4 candidates: Nascent / sparse leaf packages
    "wcwidth", "curtsies", "cwcwidth", "dbgpu", "fpdf2",
    "fqdn", "mido", "openpyxl", "pyfluidsynth", "simplejson",
    "smart-open", "threadpoolctl", "tinycss2", "webcolors", "pytz"
]


class BackgroundSampler:
    """
    Selects, fetches metadata for, and stratifies background PyPI packages.
    """

    PACKAGE_URL = "https://pypi.org/pypi/{package}/json"
    RELEASE_URL = "https://pypi.org/pypi/{package}/{version}/json"

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/pypi_background",
        output_dir: str | Path = "data/experimental",
        malicious_labels_path: str | Path = "data/processed/malicious_labels.csv",
        max_workers: int = 8,
        cache_enabled: bool = True,
        cutoff_timestamp: float = T_TRAIN_VAL_CUTOFF,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.malicious_labels_path = Path(malicious_labels_path)
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self.cutoff_timestamp = cutoff_timestamp

        self._thread_local = threading.local()
        self.excluded_packages = self._load_excluded_packages()

    def _get_session(self) -> requests.Session:
        if not hasattr(self._thread_local, "session"):
            s = requests.Session()
            retries = Retry(
                total=3,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retries, pool_connections=15, pool_maxsize=15)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            s.headers.update({"User-Agent": "SupplyChainGNN-Research/0.2.0"})
            self._thread_local.session = s
        return self._thread_local.session

    def _load_excluded_packages(self) -> Set[str]:
        """Load all packages that must be strictly excluded from background sampling."""
        excluded = set()
        if self.malicious_labels_path.exists():
            import csv
            with open(self.malicious_labels_path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    excluded.add(normalize_package_name(r["normalized_package"]))
        return excluded

    def _get_pkg_cache_path(self, package_name: str) -> Path:
        norm = normalize_package_name(package_name)
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", norm)[:64]
        return self.raw_dir / f"pkg_{safe}.json"

    def _get_release_cache_path(self, package_name: str, version: str) -> Path:
        norm_p = normalize_package_name(package_name)
        norm_v, _ = normalize_version(version)
        safe_p = re.sub(r"[^a-zA-Z0-9_\-]", "_", norm_p)[:48]
        safe_v = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", norm_v)[:32]
        return self.raw_dir / f"rel_{safe_p}_{safe_v}.json"

    def fetch_package_data(self, package_name: str) -> Tuple[Optional[Dict[str, Any]], int]:
        cache_path = self._get_pkg_cache_path(package_name)
        if self.cache_enabled and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                return cached.get("payload"), cached.get("status_code", 200)
            except Exception:
                pass

        url = self.PACKAGE_URL.format(package=package_name)
        session = self._get_session()
        try:
            resp = session.get(url, timeout=12.0)
            if resp.status_code == 404:
                return None, 404
            resp.raise_for_status()
            payload = resp.json()
            if self.cache_enabled:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "package": package_name,
                        "status_code": 200,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "payload": payload,
                    }, f, indent=2)
            return payload, 200
        except Exception as e:
            logger.error(f"Error fetching background pkg {package_name}: {e}")
            return None, 500

    def fetch_release_data(self, package_name: str, version: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_release_cache_path(package_name, version)
        if self.cache_enabled and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                return cached.get("payload")
            except Exception:
                pass

        url = self.RELEASE_URL.format(package=package_name, version=version)
        session = self._get_session()
        try:
            resp = session.get(url, timeout=12.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            payload = resp.json()
            if self.cache_enabled:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "package": package_name,
                        "version": version,
                        "status_code": 200,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "payload": payload,
                    }, f, indent=2)
            return payload
        except Exception:
            return None

    def compute_historical_metrics(
        self, package_name: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute stratification metrics STRICTLY using releases uploaded at or before cutoff.
        """
        releases = payload.get("releases", {})
        hist_releases = {}
        first_ts = None
        latest_hist_ts = None

        for ver_str, file_list in releases.items():
            if not file_list:
                continue
            upload_timestamps = []
            for f in file_list:
                iso = f.get("upload_time_iso_8601") or f.get("upload_time")
                if iso:
                    try:
                        ts = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
                        upload_timestamps.append(ts)
                    except Exception:
                        pass
            if upload_timestamps:
                earliest_upload = min(upload_timestamps)
                if earliest_upload <= self.cutoff_timestamp:
                    hist_releases[ver_str] = earliest_upload
                    if first_ts is None or earliest_upload < first_ts:
                        first_ts = earliest_upload
                    if latest_hist_ts is None or earliest_upload > latest_hist_ts:
                        latest_hist_ts = earliest_upload

        n_rel = len(hist_releases)
        if first_ts and latest_hist_ts:
            age_days = (latest_hist_ts - first_ts) / 86400.0
            freq = n_rel / (age_days / 365.25 + 0.1)
        else:
            age_days = 0.0
            freq = 0.0

        return {
            "historical_release_count": n_rel,
            "historical_age_days": round(age_days, 1),
            "historical_release_frequency": round(freq, 2),
            "first_release_timestamp": first_ts,
            "latest_historical_release_timestamp": latest_hist_ts,
        }

    def assign_stratum(self, metrics: Dict[str, Any], in_degree: int) -> str:
        """
        Assign stratum based on operational thresholds on measurable metrics.
        """
        n_rel = metrics["historical_release_count"]
        age_days = metrics["historical_age_days"]
        freq = metrics["historical_release_frequency"]

        # Stratum 1: Core Infrastructure
        if n_rel >= 40 and age_days >= 1800 and in_degree >= 50:
            return "stratum_1_core_infrastructure"
        # Stratum 2: Active Domain Utilities
        elif n_rel >= 15 and freq >= 1.5:
            return "stratum_2_active_domain_utilities"
        # Stratum 3: Stable Low-Velocity Utilities
        elif n_rel >= 5 and age_days >= 730:
            return "stratum_3_stable_low_velocity"
        # Stratum 4: Nascent / Sparse Leaf Packages
        else:
            return "stratum_4_nascent_sparse_leaf"

    def sample_cohort(
        self, historical_in_degrees: Dict[str, int]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Select 100 packages (40 hard-negatives + 60 general stratified) and extract all release rows.
        """
        all_candidates = []
        for p in HARD_NEGATIVE_TARGETS:
            norm = normalize_package_name(p)
            if norm not in self.excluded_packages:
                all_candidates.append((p, "hard_negative_target"))

        for p in GENERAL_BACKGROUND_CANDIDATES:
            norm = normalize_package_name(p)
            if norm not in self.excluded_packages and norm not in [c[0] for c in all_candidates]:
                all_candidates.append((p, "general_stratified"))

        logger.info(f"Querying metadata for {len(all_candidates)} background candidate packages...")
        pkg_metadata_map = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cand = {
                executor.submit(self.fetch_package_data, cand[0]): cand
                for cand in all_candidates
            }
            for future in as_completed(future_to_cand):
                cand = future_to_cand[future]
                pkg_name, origin = cand
                try:
                    payload, status = future.result()
                    if status == 200 and payload:
                        pkg_metadata_map[pkg_name] = (payload, origin)
                except Exception as e:
                    logger.error(f"Error fetching {pkg_name}: {e}")

        # Compute historical metrics strictly as of cutoff
        stratified_packages: List[Dict[str, Any]] = []
        all_background_versions: List[Dict[str, Any]] = []

        for pkg_name, (payload, origin) in pkg_metadata_map.items():
            norm_pkg = normalize_package_name(pkg_name)
            in_deg = historical_in_degrees.get(norm_pkg, 0)
            hist_metrics = self.compute_historical_metrics(pkg_name, payload)
            stratum = self.assign_stratum(hist_metrics, in_deg)

            stratified_packages.append({
                "package": pkg_name,
                "normalized_package": norm_pkg,
                "origin_type": origin,
                "stratum": stratum,
                "historical_metrics_as_of_cutoff": hist_metrics,
                "historical_downstream_in_degree": in_deg,
                "selection_cutoff_timestamp": self.cutoff_timestamp,
                "selection_cutoff_iso": datetime.fromtimestamp(
                    self.cutoff_timestamp, tz=timezone.utc
                ).isoformat(),
            })

            # Extract releases (both historical and future releases are extracted,
            # with future releases properly landing in Val/Test splits)
            info = payload.get("info", {})
            latest_v = info.get("version")
            latest_reqs = info.get("requires_dist")
            repo_candidate = (
                (info.get("project_urls") or {}).get("Source")
                or (info.get("project_urls") or {}).get("Repository")
                or info.get("home_page")
            )
            repo_url = normalize_repo_url(repo_candidate) if repo_candidate else None
            homepage_url = (info.get("project_urls") or {}).get("Homepage") or info.get("home_page")

            releases = payload.get("releases", {})
            for ver_str, file_list in releases.items():
                if not file_list:
                    continue
                norm_ver, _ = normalize_version(ver_str)
                timestamps = []
                req_python = None
                for f in file_list:
                    iso = f.get("upload_time_iso_8601") or f.get("upload_time")
                    if iso:
                        try:
                            timestamps.append(
                                datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
                            )
                        except Exception:
                            pass
                    if not req_python and f.get("requires_python"):
                        req_python = f.get("requires_python")

                upload_ts = min(timestamps) if timestamps else None

                # Release-specific requires_dist without backwards leakage
                reqs_dist = None
                if ver_str == latest_v and latest_reqs is not None:
                    reqs_dist = latest_reqs
                else:
                    rel_p = self.fetch_release_data(pkg_name, ver_str)
                    if rel_p:
                        rel_info = rel_p.get("info", {})
                        reqs_dist = rel_info.get("requires_dist")
                        if not req_python:
                            req_python = rel_info.get("requires_python")

                has_deps = bool(reqs_dist)
                all_background_versions.append({
                    "package": pkg_name,
                    "normalized_package": norm_pkg,
                    "version": ver_str,
                    "release_timestamp": upload_ts,
                    "label": "assumed_benign_background",
                    "label_scope": "background",
                    "registry_status": "live",
                    "dependency_metadata_status": "observed" if has_deps else "missing",
                    "requires_python": req_python or "",
                    "requires_dist": json.dumps(reqs_dist) if reqs_dist is not None else "",
                    "repository_url": repo_url or "",
                    "homepage_url": homepage_url or "",
                    "source": "pypi_background",
                })

        # Save metadata summary
        bg_meta_path = self.output_dir / "background_packages.json"
        with open(bg_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "candidate_selection_cutoff_timestamp": self.cutoff_timestamp,
                "candidate_selection_cutoff_iso": datetime.fromtimestamp(
                    self.cutoff_timestamp, tz=timezone.utc
                ).isoformat(),
                "total_background_packages": len(stratified_packages),
                "strata_counts": {
                    s: sum(1 for p in stratified_packages if p["stratum"] == s)
                    for s in [
                        "stratum_1_core_infrastructure",
                        "stratum_2_active_domain_utilities",
                        "stratum_3_stable_low_velocity",
                        "stratum_4_nascent_sparse_leaf",
                    ]
                },
                "origin_composition": {
                    "hard_negative_target": sum(
                        1 for p in stratified_packages if p["origin_type"] == "hard_negative_target"
                    ),
                    "general_stratified": sum(
                        1 for p in stratified_packages if p["origin_type"] == "general_stratified"
                    ),
                },
                "packages": stratified_packages,
            }, f, indent=2)

        logger.info(
            f"Background sampling complete: {len(stratified_packages)} packages, "
            f"{len(all_background_versions)} releases."
        )
        return stratified_packages, all_background_versions
