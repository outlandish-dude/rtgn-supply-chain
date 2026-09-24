"""
GitHub Advisory Database Ingestion Adapter.
Provides a formal interface for querying GitHub Security Advisories (GHSA).
Strictly adheres to authentication boundaries:
If GITHUB_TOKEN is not configured in the environment, the adapter explicitly defers
direct collection, records this limitation in audit metadata, and does NOT fabricate credentials
or silently conflate OSV data with direct GitHub API responses.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GitHubAdvisoryAdapter:
    """
    Adapter for GitHub Advisory Database.
    Requires GITHUB_TOKEN authentication for high-volume or GraphQL queries.
    """

    GRAPHQL_URL = "https://api.github.com/graphql"

    def __init__(
        self,
        token: Optional[str] = None,
        raw_dir: str | Path = "data/raw/advisories",
        cache_enabled: bool = True,
    ) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.is_authenticated = bool(self.token and self.token.strip())

        if not self.is_authenticated:
            logger.info(
                "GitHub Advisory direct integration deferred: GITHUB_TOKEN not configured. "
                "Advisory data will rely exclusively on public OSV PyPI feeds without credential fabrication."
            )

    def status(self) -> Dict[str, Any]:
        """Return adapter operational status."""
        return {
            "source": "github_advisory_database",
            "authenticated": self.is_authenticated,
            "status": "active" if self.is_authenticated else "deferred_missing_credentials",
            "reason": (
                "Operational"
                if self.is_authenticated
                else "GITHUB_TOKEN unavailable. Direct GitHub Advisory API queries deferred per Rule 10."
            ),
        }

    def query_package_advisories(
        self, package_name: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Query advisories for a package. If unauthenticated, returns empty list with status metadata.
        """
        if not self.is_authenticated:
            return [], {
                "package": package_name,
                "status": "deferred_missing_credentials",
                "record_count": 0,
            }

        # If authenticated, query GraphQL API (implementation hook)
        # Note: In current environment, GITHUB_TOKEN is not configured
        return [], {
            "package": package_name,
            "status": "unimplemented_authenticated_hook",
            "record_count": 0,
        }
