"""
OSV (Open Source Vulnerabilities) Collector.
Queries OSV API at exact package-version granularity for the PyPI ecosystem.
Uses batch queries (POST https://api.osv.dev/v1/querybatch) with local caching in data/raw/osv/.
Preserves published/modified timestamps and affected ranges without temporal leakage.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.data.normalization import normalize_package_name

logger = logging.getLogger(__name__)


class OSVCollector:
    """
    Ingestion collector for Open Source Vulnerabilities (OSV).
    """

    BATCH_URL = "https://api.osv.dev/v1/querybatch"
    VULN_URL = "https://api.osv.dev/v1/vulns/{id}"

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/osv",
        output_path: str | Path = "data/intermediate/osv_events.csv",
        cache_enabled: bool = True,
        batch_size: int = 500,
        request_delay: float = 0.1,
        timeout: float = 15.0,
        max_retries: int = 3,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = Path(output_path)
        self.cache_enabled = cache_enabled
        self.batch_size = batch_size
        self.request_delay = request_delay
        self.timeout = timeout

        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": "SupplyChainGNN-Research/0.2.0"})

    def _get_vuln_cache_path(self, vuln_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", vuln_id)
        return self.raw_dir / f"vuln_{safe_id}.json"

    def fetch_vuln_detail(self, vuln_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch full vulnerability record by ID with local disk caching.
        """
        cache_path = self._get_vuln_cache_path(vuln_id)

        if self.cache_enabled and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading OSV vuln cache {cache_path}: {e}")

        url = self.VULN_URL.format(id=vuln_id)
        try:
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            payload = resp.json()

            if self.cache_enabled:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

            return payload
        except Exception as e:
            logger.error(f"Failed to fetch OSV vuln {vuln_id}: {e}")
            return None

    def query_batch(
        self, query_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch query to OSV API.
        """
        if not query_items:
            return []

        try:
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            resp = self.session.post(
                self.BATCH_URL,
                json={"queries": query_items},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"OSV batch query failed for chunk of size {len(query_items)}: {e}")
            return [{} for _ in query_items]

    def collect_for_package_versions(
        self, package_versions: List[Tuple[str, str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Query OSV at exact package-version granularity.
        package_versions is a list of (package_name, version_str) tuples.
        """
        logger.info(f"Querying OSV for {len(package_versions)} package/version items...")
        events: List[Dict[str, Any]] = []
        stats = {
            "total_queries": len(package_versions),
            "vuln_references_found": 0,
            "unique_vuln_ids": 0,
            "events_recorded": 0,
            "api_failures": 0,
        }

        # Deduplicate queries
        unique_pairs = list(dict.fromkeys(package_versions))
        vuln_cache: Dict[str, Optional[Dict[str, Any]]] = {}

        # Process in batches
        for i in range(0, len(unique_pairs), self.batch_size):
            chunk = unique_pairs[i : i + self.batch_size]
            query_items = []
            for pkg, ver in chunk:
                item: Dict[str, Any] = {"package": {"name": pkg, "ecosystem": "PyPI"}}
                if ver != "__ALL__":
                    item["version"] = ver
                query_items.append(item)

            results = self.query_batch(query_items)
            if not results:
                stats["api_failures"] += 1
                continue

            for (pkg, ver), res in zip(chunk, results):
                vulns = res.get("vulns", [])
                if not vulns:
                    continue

                stats["vuln_references_found"] += len(vulns)
                for v_stub in vulns:
                    v_id = v_stub.get("id")
                    if not v_id:
                        continue

                    if v_id not in vuln_cache:
                        v_detail = self.fetch_vuln_detail(v_id)
                        vuln_cache[v_id] = v_detail
                    else:
                        v_detail = vuln_cache[v_id]

                    # Parse timestamps and affected ranges
                    pub_ts: Optional[float] = None
                    mod_ts: Optional[float] = None
                    affected_ranges: List[str] = []
                    references: List[str] = []
                    summary: str = ""

                    if v_detail:
                        summary = v_detail.get("summary") or ""
                        if v_detail.get("published"):
                            try:
                                dt = datetime.fromisoformat(
                                    v_detail["published"].replace("Z", "+00:00")
                                )
                                pub_ts = dt.timestamp()
                            except Exception:
                                pass
                        if v_detail.get("modified"):
                            try:
                                dt = datetime.fromisoformat(
                                    v_detail["modified"].replace("Z", "+00:00")
                                )
                                mod_ts = dt.timestamp()
                            except Exception:
                                pass

                        for aff in v_detail.get("affected", []):
                            for r in aff.get("ranges", []):
                                r_type = r.get("type", "")
                                events_list = [
                                    f"{k}:{v}"
                                    for ev in r.get("events", [])
                                    for k, v in ev.items()
                                ]
                                affected_ranges.append(f"{r_type}({', '.join(events_list)})")

                        references = [
                            ref.get("url", "")
                            for ref in v_detail.get("references", [])
                            if ref.get("url")
                        ]
                    else:
                        # Fallback from stub if detail fetch failed
                        if v_stub.get("modified"):
                            try:
                                dt = datetime.fromisoformat(
                                    v_stub["modified"].replace("Z", "+00:00")
                                )
                                mod_ts = dt.timestamp()
                            except Exception:
                                pass

                    events.append({
                        "osv_id": v_id,
                        "package": pkg,
                        "version": ver,
                        "published_timestamp": pub_ts if pub_ts is not None else "",
                        "modified_timestamp": mod_ts if mod_ts is not None else "",
                        "affected_ranges": json.dumps(affected_ranges),
                        "summary": summary,
                        "references": json.dumps(references[:5]),
                        "source": "osv",
                    })

        stats["unique_vuln_ids"] = len(vuln_cache)
        stats["events_recorded"] = len(events)
        logger.info(f"OSV collection complete: {stats}")
        return events, stats

    def export_csv(self, events: List[Dict[str, Any]]) -> Path:
        """Export OSV events to CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "osv_id",
            "package",
            "version",
            "published_timestamp",
            "modified_timestamp",
            "affected_ranges",
            "summary",
            "references",
            "source",
        ]
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)

        logger.info(f"Exported {len(events)} OSV event records to {self.output_path}")
        return self.output_path
