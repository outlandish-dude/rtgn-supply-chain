"""
Causally-Sound Feature Extraction for PackageVersion Nodes.
Strictly adheres to zero future lookahead:
Every feature is computed strictly "as of" t = release_timestamp of the target version.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from packaging.version import InvalidVersion, Version

from src.data.normalization import normalize_package_name

logger = logging.getLogger(__name__)

# Canonical documentation of every feature dimension
FEATURE_SPECIFICATION: List[Dict[str, str]] = [
    {
        "name": "version_major",
        "type": "integer",
        "description": "Major version number parsed via PEP 440 from release metadata.",
        "temporal_provenance": "Available at upload timestamp t (intrinsic release metadata).",
    },
    {
        "name": "version_minor",
        "type": "integer",
        "description": "Minor version number parsed via PEP 440 from release metadata.",
        "temporal_provenance": "Available at upload timestamp t (intrinsic release metadata).",
    },
    {
        "name": "version_patch",
        "type": "integer",
        "description": "Patch/micro version number parsed via PEP 440 from release metadata.",
        "temporal_provenance": "Available at upload timestamp t (intrinsic release metadata).",
    },
    {
        "name": "is_prerelease",
        "type": "binary (0/1)",
        "description": "1 if release version is marked as pre-release (alpha/beta/rc/dev), 0 otherwise.",
        "temporal_provenance": "Available at upload timestamp t (intrinsic release metadata).",
    },
    {
        "name": "has_requires_python",
        "type": "binary (0/1)",
        "description": "1 if release specifies requires_python environment constraint, 0 otherwise.",
        "temporal_provenance": "Available at upload timestamp t (intrinsic release metadata).",
    },
    {
        "name": "requires_python_len",
        "type": "integer",
        "description": "Character length of requires_python constraint string.",
        "temporal_provenance": "Available at upload timestamp t (intrinsic release metadata).",
    },
    {
        "name": "declared_dependency_count",
        "type": "integer",
        "description": "Number of declared dependencies in requires_dist for this specific release.",
        "temporal_provenance": "Available at upload timestamp t (release-specific dependency declaration).",
    },
    {
        "name": "hist_release_count_before_t",
        "type": "integer",
        "description": "Number of releases of this package uploaded strictly prior to timestamp t.",
        "temporal_provenance": "Computed strictly from historical releases where release_timestamp < t.",
    },
    {
        "name": "hist_age_days_at_t",
        "type": "float",
        "description": "Days elapsed since the package's earliest historical release prior to t.",
        "temporal_provenance": "Computed strictly from historical releases where release_timestamp < t.",
    },
    {
        "name": "time_since_prev_release_days",
        "type": "float",
        "description": "Days elapsed since the package's immediate preceding release uploaded prior to t.",
        "temporal_provenance": "Computed strictly from historical releases where release_timestamp < t.",
    },
    {
        "name": "hist_release_frequency",
        "type": "float",
        "description": "Historical release cadence (releases/year) as of timestamp t.",
        "temporal_provenance": "Computed strictly from historical releases where release_timestamp < t.",
    },
    {
        "name": "dynamic_downstream_in_degree",
        "type": "integer",
        "description": "Number of distinct external releases uploaded prior to t declaring a dependency on this package.",
        "temporal_provenance": "Computed strictly from dependency edges where edge_time < t.",
    },
    {
        "name": "hist_advisories_count_before_t",
        "type": "integer",
        "description": "Count of distinct public OSV advisories published strictly prior to t affecting prior releases.",
        "temporal_provenance": "Computed strictly from OSV records where published_timestamp < t and affected_version != target_version.",
    },
]

FEATURE_NAMES = [f["name"] for f in FEATURE_SPECIFICATION]
NUM_FEATURES = len(FEATURE_NAMES)


def parse_version_tuple(version_str: str) -> Tuple[int, int, int, int]:
    """Parse major, minor, patch, is_prerelease safely using packaging."""
    try:
        v = Version(version_str)
        major = v.major if v.major is not None else 0
        minor = v.minor if v.minor is not None else 0
        patch = v.micro if v.micro is not None else 0
        is_pre = 1 if (v.is_prerelease or v.is_devrelease) else 0
        return min(major, 999), min(minor, 999), min(patch, 999), is_pre
    except (InvalidVersion, TypeError):
        # Fallback numeric regex extraction
        import re
        parts = re.findall(r"\d+", str(version_str))
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        is_pre = 1 if any(tok in str(version_str).lower() for tok in ["a", "b", "rc", "dev"]) else 0
        return min(major, 999), min(minor, 999), min(patch, 999), is_pre


import bisect


def compute_causal_features_for_release(
    row: Dict[str, Any],
    historical_package_releases: List[Tuple[str, float]],  # List of (version_str, upload_ts) for same package
    historical_dependency_events: Any,                      # Dict[str, List[float]] OR List[Tuple[str, float]]
    historical_advisories: List[Tuple[str, float]],         # List of (affected_version, published_ts) for same package
) -> List[float]:
    """
    Compute causal feature vector for a single PackageVersion strictly as of t = release_timestamp.
    """
    raw_ts = row.get("release_timestamp")
    t = float(raw_ts) if raw_ts is not None and str(raw_ts).strip() != "" else None

    # 1. Version intrinsic features (upload time t)
    v_str = row.get("version", "")
    major, minor, patch, is_pre = parse_version_tuple(v_str)

    req_py = row.get("requires_python", "") or ""
    has_req_py = 1.0 if req_py.strip() else 0.0
    req_py_len = float(min(len(req_py.strip()), 100))

    req_dist_str = row.get("requires_dist", "") or ""
    dep_count = 0.0
    if req_dist_str and req_dist_str != "[]" and req_dist_str != '""':
        try:
            parsed_deps = json.loads(req_dist_str)
            if isinstance(parsed_deps, list):
                dep_count = float(len(parsed_deps))
        except Exception:
            pass

    # If release has no timestamp (e.g. takedown record), historical dynamic features are 0
    if t is None:
        return [
            float(major), float(minor), float(patch), float(is_pre),
            has_req_py, req_py_len, dep_count,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

    # 2. Dynamic package release velocity strictly prior to t
    # Find all releases with ts < t
    prior_releases = [
        (v, ts) for v, ts in historical_package_releases
        if ts < t
    ]
    prior_releases.sort(key=lambda x: x[1])

    hist_rel_count = float(len(prior_releases))
    if prior_releases:
        t_first = prior_releases[0][1]
        t_prev = prior_releases[-1][1]
        age_days = max(0.0, (t - t_first) / 86400.0)
        time_since_prev = max(0.0, (t - t_prev) / 86400.0)
        age_years = age_days / 365.25
        cadence = hist_rel_count / (age_years + 0.1)
    else:
        age_days = 0.0
        time_since_prev = 0.0
        cadence = 0.0

    # 3. Dynamic downstream in-degree strictly prior to t
    target_norm_pkg = row.get("normalized_package", "")
    if isinstance(historical_dependency_events, dict):
        dep_ts_list = historical_dependency_events.get(target_norm_pkg, [])
        downstream_count = bisect.bisect_left(dep_ts_list, t)
    elif historical_dependency_events and isinstance(historical_dependency_events[0], (int, float)):
        downstream_count = bisect.bisect_left(historical_dependency_events, t)
    else:
        downstream_count = sum(
            1 for tgt, edge_ts in historical_dependency_events
            if tgt == target_norm_pkg and edge_ts < t
        )

    # 4. Historical OSV advisories strictly prior to t affecting prior releases
    # Target release's own compromise advisory or Datadog label is strictly excluded
    hist_advisories_count = 0
    for aff_ver, pub_ts in historical_advisories:
        if pub_ts < t and aff_ver != v_str:
            hist_advisories_count += 1

    return [
        float(major),
        float(minor),
        float(patch),
        float(is_pre),
        has_req_py,
        req_py_len,
        dep_count,
        hist_rel_count,
        round(age_days, 2),
        round(time_since_prev, 2),
        round(cadence, 2),
        float(downstream_count),
        float(hist_advisories_count),
    ]
