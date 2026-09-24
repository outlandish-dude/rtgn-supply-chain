"""
Deterministic normalization utilities for package names, versions, and repository URLs.
Follows PEP 503 for package name canonicalization and PEP 440 for version parsing.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from packaging.version import InvalidVersion, Version


def normalize_package_name(name: str) -> str:
    """Normalize package names following PEP 503."""
    if not name or not isinstance(name, str):
        return ""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def normalize_version(version_str: str) -> Tuple[str, bool]:
    """
    Normalize version string using PEP 440.
    Returns (normalized_version_string, is_valid_pep440).
    For special markers such as '__ALL__', returns ('__ALL__', True).
    """
    if not version_str or not isinstance(version_str, str):
        return "", False
    clean_v = version_str.strip()
    if clean_v == "__ALL__":
        return "__ALL__", True
    try:
        parsed = Version(clean_v)
        return str(parsed), True
    except InvalidVersion:
        # Preserve raw version string but flag invalid PEP 440
        return clean_v, False


def normalize_repo_url(url: Optional[str]) -> Optional[str]:
    """Normalize git/GitHub repository URLs."""
    if not url or not isinstance(url, str):
        return None
    clean = url.strip().rstrip("/")
    clean = re.sub(r"^git\+", "", clean)
    clean = re.sub(r"\.git$", "", clean)
    clean = re.sub(r"^http://", "https://", clean)
    match = re.search(r"github\.com/([^/]+)/([^/]+)", clean, re.IGNORECASE)
    if match:
        owner, repo = match.group(1).lower(), match.group(2).lower()
        return f"github:{owner}/{repo}"
    return clean
