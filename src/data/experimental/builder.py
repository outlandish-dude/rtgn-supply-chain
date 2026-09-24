"""
Experimental Dataset Builder.
Orchestrates:
1. Integration of processed releases, background releases, and takedown records.
2. Label classification into: confirmed_malicious, source_unflagged, assumed_benign_background.
3. Separation of topology observability: registry_status (live|takedown) and dependency_metadata_status (observed|missing).
4. Chronological splitting (Train < 2025-06-01 <= Val < 2026-03-01 <= Test).
5. Empirical square-root package-balanced sample weights computed on the actual training split.
6. Package-disjoint splitting (P_train ∩ P_val ∩ P_test = ∅) with maintainer/repository leakage audit.
Strictly non-modifying to data/processed/.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.normalization import normalize_package_name, normalize_version

logger = logging.getLogger(__name__)

# Chronological split boundaries
T_TRAIN_VAL = 1748736000.0  # 2025-06-01 00:00:00 UTC
T_VAL_TEST = 1772323200.0   # 2026-03-01 00:00:00 UTC


class ExperimentalDatasetBuilder:
    """
    Constructs the augmented experimental dataset and reproducible experimental splits.
    """

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        experimental_dir: str | Path = "data/experimental",
        raw_dir: str | Path = "data/raw",
        t_train_val: float = T_TRAIN_VAL,
        t_val_test: float = T_VAL_TEST,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.experimental_dir = Path(experimental_dir)
        self.raw_dir = Path(raw_dir)
        self.experimental_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir = self.experimental_dir / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        self.t_train_val = t_train_val
        self.t_val_test = t_val_test

    def build(
        self, background_versions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build augmented dataset, compute weights, and generate splits.
        """
        logger.info("Loading existing processed datasets...")
        processed_versions = []
        with open(self.processed_dir / "package_versions.csv", "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                processed_versions.append(r)

        malicious_labels = []
        with open(self.processed_dir / "malicious_labels.csv", "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                malicious_labels.append(r)

        # 1. Transform Processed Versions with 3 distinct labels and observability statuses
        combined_records: List[Dict[str, Any]] = []

        for r in processed_versions:
            orig_label = r["label"]
            if orig_label == "confirmed_malicious":
                label = "confirmed_malicious"
            else:
                label = "source_unflagged"

            req_dist = r.get("requires_dist")
            has_deps = bool(req_dist and req_dist != "[]" and req_dist != '""')

            if has_deps:
                top_status = "graph_observable"
            else:
                top_status = "live_unobservable_topology"

            combined_records.append({
                "package": r["package"],
                "normalized_package": r["normalized_package"],
                "version": r["version"],
                "release_timestamp": float(r["release_timestamp"]) if r["release_timestamp"] else None,
                "label": label,
                "label_scope": r.get("label_scope", "unknown"),
                "registry_status": "live",
                "dependency_metadata_status": "observed" if has_deps else "missing",
                "topology_status": top_status,
                "sample_weight": 1.0,  # Will be updated on train split
                "requires_python": r.get("requires_python", ""),
                "requires_dist": r.get("requires_dist", ""),
                "repository_url": r.get("repository_url", ""),
                "homepage_url": r.get("homepage_url", ""),
                "source": r.get("source", "pypi"),
            })

        # 2. Add Background Versions
        for r in background_versions:
            req_dist = r.get("requires_dist")
            has_deps = bool(req_dist and req_dist != "[]" and req_dist != '""')

            if has_deps:
                top_status = "graph_observable"
            else:
                top_status = "live_unobservable_topology"

            combined_records.append({
                "package": r["package"],
                "normalized_package": r["normalized_package"],
                "version": r["version"],
                "release_timestamp": float(r["release_timestamp"]) if r["release_timestamp"] else None,
                "label": "assumed_benign_background",
                "label_scope": "background",
                "registry_status": "live",
                "dependency_metadata_status": "observed" if has_deps else "missing",
                "topology_status": top_status,
                "sample_weight": 1.0,
                "requires_python": r.get("requires_python", ""),
                "requires_dist": r.get("requires_dist", ""),
                "repository_url": r.get("repository_url", ""),
                "homepage_url": r.get("homepage_url", ""),
                "source": "pypi_background",
            })

        # 3. Add Takedown Records (404 on live PyPI) for completeness
        live_norm_pkgs = {r["normalized_package"] for r in combined_records}
        takedown_records_count = 0

        for m in malicious_labels:
            norm_p = m["normalized_package"]
            if m.get("in_pypi_registry") == "False" or norm_p not in live_norm_pkgs:
                combined_records.append({
                    "package": m["package"],
                    "normalized_package": norm_p,
                    "version": m["version"],
                    "release_timestamp": None,
                    "label": "confirmed_malicious",
                    "label_scope": m.get("label_scope", "all_versions"),
                    "registry_status": "takedown",
                    "dependency_metadata_status": "missing",
                    "topology_status": "label_only_takedown",
                    "sample_weight": 1.0,
                    "requires_python": "",
                    "requires_dist": "",
                    "repository_url": "",
                    "homepage_url": "",
                    "source": "datadog_takedown",
                })
                takedown_records_count += 1

        # 4. Generate Chronological Split (Unchanged)
        chrono_split = self._build_chronological_split(combined_records)

        # 5. Compute Empirical Package-Balanced Sample Weights on Actual Train Split
        weight_report = self._compute_and_apply_training_weights(combined_records, chrono_split["train_indices"])

        # 6. Extract Persistent Identifiers & Connected Components
        pkg_identifiers = self._extract_package_identifiers(combined_records)
        components = self._build_connected_components(combined_records, pkg_identifiers)

        # 7. Generate Leak-Free Package-Disjoint Split (Package + Maintainer + Repository Grouped)
        disjoint_split, leakage_audit = self._build_package_disjoint_split(
            combined_records, pkg_identifiers, components
        )

        # 8. Generate Graph-Model Package-Disjoint Split (Restricted to Graph-Observable Universe)
        graph_split, graph_leakage_audit = self._build_graph_package_disjoint_split(
            combined_records, pkg_identifiers, components
        )

        # 9. Persist Artifacts
        self._export_experimental_csv(combined_records)

        chrono_path = self.splits_dir / "chronological_split.json"
        with open(chrono_path, "w", encoding="utf-8") as f:
            json.dump({
                "split_type": "chronological_temporal_forecasting",
                "cutoff_train_val_timestamp": self.t_train_val,
                "cutoff_train_val_iso": datetime.fromtimestamp(self.t_train_val, tz=timezone.utc).isoformat(),
                "cutoff_val_test_timestamp": self.t_val_test,
                "cutoff_val_test_iso": datetime.fromtimestamp(self.t_val_test, tz=timezone.utc).isoformat(),
                "counts": {
                    "train": len(chrono_split["train_indices"]),
                    "validation": len(chrono_split["val_indices"]),
                    "test": len(chrono_split["test_indices"]),
                    "takedown_catalog": len(chrono_split["takedown_indices"]),
                },
                "package_weighting_report": weight_report,
                "train_indices": chrono_split["train_indices"],
                "validation_indices": chrono_split["val_indices"],
                "test_indices": chrono_split["test_indices"],
                "takedown_indices": chrono_split["takedown_indices"],
            }, f, indent=2)

        disjoint_path = self.splits_dir / "package_disjoint_split.json"
        with open(disjoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "split_type": "package_disjoint_unseen_attack",
                "counts": {
                    "train": len(disjoint_split["train_indices"]),
                    "validation": len(disjoint_split["val_indices"]),
                    "test": len(disjoint_split["test_indices"]),
                },
                "packages": {
                    "train_packages": sorted(list(disjoint_split["train_packages"])),
                    "validation_packages": sorted(list(disjoint_split["val_packages"])),
                    "test_packages": sorted(list(disjoint_split["test_packages"])),
                },
                "identifier_leakage_audit": leakage_audit,
                "train_indices": disjoint_split["train_indices"],
                "validation_indices": disjoint_split["val_indices"],
                "test_indices": disjoint_split["test_indices"],
            }, f, indent=2)

        graph_split_path = self.splits_dir / "graph_package_disjoint_split.json"
        with open(graph_split_path, "w", encoding="utf-8") as f:
            json.dump(graph_split, f, indent=2)

        summary = {
            "total_records": len(combined_records),
            "registry_live_records": len(combined_records) - takedown_records_count,
            "registry_takedown_records": takedown_records_count,
            "label_distribution": {
                "live_confirmed_malicious": sum(
                    1 for r in combined_records
                    if r["label"] == "confirmed_malicious" and r["registry_status"] == "live"
                ),
                "confirmed_malicious": sum(
                    1 for r in combined_records
                    if r["label"] == "confirmed_malicious" and r["registry_status"] == "live"
                ),
                "source_unflagged": sum(1 for r in combined_records if r["label"] == "source_unflagged"),
                "assumed_benign_background": sum(1 for r in combined_records if r["label"] == "assumed_benign_background"),
                "takedown_catalog_records": takedown_records_count,
            },
            "observability_distribution": {
                "live_observed_topology": sum(
                    1 for r in combined_records
                    if r["registry_status"] == "live" and r["dependency_metadata_status"] == "observed"
                ),
                "live_missing_topology": sum(
                    1 for r in combined_records
                    if r["registry_status"] == "live" and r["dependency_metadata_status"] == "missing"
                ),
                "takedown_missing_topology": takedown_records_count,
            },
            "chronological_split_counts": {
                "train": len(chrono_split["train_indices"]),
                "validation": len(chrono_split["val_indices"]),
                "test": len(chrono_split["test_indices"]),
                "takedown_catalog": len(chrono_split["takedown_indices"]),
            },
            "package_disjoint_split_counts": {
                "train": len(disjoint_split["train_indices"]),
                "validation": len(disjoint_split["val_indices"]),
                "test": len(disjoint_split["test_indices"]),
            },
            "graph_package_disjoint_split_counts": {
                "train": graph_split["counts"]["train_records"],
                "validation": graph_split["counts"]["val_records"],
                "test": graph_split["counts"]["test_records"],
                "train_graph_observable": graph_split["counts"]["train_graph_observable"],
                "train_live_unobservable": graph_split["counts"]["train_live_unobservable"],
                "val_graph_observable": graph_split["counts"]["val_graph_observable"],
                "val_live_unobservable": graph_split["counts"]["val_live_unobservable"],
                "test_graph_observable": graph_split["counts"]["test_graph_observable"],
                "test_live_unobservable": graph_split["counts"]["test_live_unobservable"],
            },
            "weight_report": weight_report,
            "leakage_audit": leakage_audit,
            "graph_leakage_audit": graph_leakage_audit,
        }

        logger.info(f"Experimental dataset build complete: {summary['total_records']} records.")
        return summary

    def _build_chronological_split(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, List[int]]:
        """Partition indices into Train, Val, Test based strictly on release_timestamp."""
        train_idx = []
        val_idx = []
        test_idx = []
        takedown_idx = []

        for idx, r in enumerate(records):
            ts = r["release_timestamp"]
            if ts is None or r["registry_status"] == "takedown":
                takedown_idx.append(idx)
            elif ts < self.t_train_val:
                train_idx.append(idx)
            elif ts < self.t_val_test:
                val_idx.append(idx)
            else:
                test_idx.append(idx)

        return {
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
            "takedown_indices": takedown_idx,
        }

    def _compute_and_apply_training_weights(
        self, records: List[Dict[str, Any]], train_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Compute empirical square-root package-balanced weights on the actual training split.
        Per-package report includes: raw_count, training_count, raw_weight, normalized_weight,
        and percentage of positive training mass.
        """
        train_pos_counts: Dict[str, int] = {}
        for idx in train_indices:
            r = records[idx]
            if r["label"] == "confirmed_malicious":
                pkg = r["normalized_package"]
                train_pos_counts[pkg] = train_pos_counts.get(pkg, 0) + 1

        total_pos_train = sum(train_pos_counts.values())
        if total_pos_train == 0:
            return {}

        # 1. Raw square-root damped weight: w_raw(v) = 1 / sqrt(N_{p, train})
        raw_weights: Dict[str, float] = {
            pkg: 1.0 / math.sqrt(cnt) for pkg, cnt in train_pos_counts.items()
        }

        # 2. Compute normalizer such that sum of weights equals total_pos_train (mean = 1.0)
        sum_weighted_pos = sum(
            train_pos_counts[pkg] * raw_weights[pkg] for pkg in train_pos_counts
        )
        scale_factor = total_pos_train / sum_weighted_pos if sum_weighted_pos > 0 else 1.0

        norm_weights: Dict[str, float] = {
            pkg: max(0.05, min(5.0, raw_weights[pkg] * scale_factor))
            for pkg in raw_weights
        }

        # Apply weights to training records in place
        for idx in train_indices:
            r = records[idx]
            if r["label"] == "confirmed_malicious":
                r["sample_weight"] = round(norm_weights.get(r["normalized_package"], 1.0), 4)
            else:
                r["sample_weight"] = 1.0

        # Detailed per-package reporting
        package_reports = []
        for pkg, cnt in sorted(train_pos_counts.items(), key=lambda x: x[1], reverse=True):
            raw_w = raw_weights[pkg]
            norm_w = norm_weights[pkg]
            mass_before = (cnt / total_pos_train) * 100.0
            mass_after = (cnt * norm_w / total_pos_train) * 100.0

            package_reports.append({
                "package": pkg,
                "training_positive_count": cnt,
                "raw_damped_weight": round(raw_w, 4),
                "normalized_weight": round(norm_w, 4),
                "positive_mass_before_pct": round(mass_before, 2),
                "positive_mass_after_pct": round(mass_after, 2),
            })

        upgini_rep = next((p for p in package_reports if p["package"] == "upgini"), None)
        return {
            "total_positive_training_samples": total_pos_train,
            "unique_positive_training_packages": len(train_pos_counts),
            "upgini_summary": upgini_rep,
            "per_package_breakdown": package_reports,
        }

    def _extract_package_identifiers(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """Extract repository URLs and maintainer/author names/emails for each package."""
        packages = {r["normalized_package"] for r in records}
        pkg_identifiers: Dict[str, Set[str]] = {p: set() for p in packages}

        # 1. Repositories from records
        for r in records:
            p = r["normalized_package"]
            repo = r.get("repository_url")
            if repo and repo.strip():
                pkg_identifiers[p].add("repo:" + repo.strip().lower())

        # 2. Maintainers and authors from cached raw metadata
        raw_files = glob.glob(str(self.raw_dir / "pypi*" / "pkg_*.json"))
        for pf in raw_files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                p = normalize_package_name(d.get("package", ""))
                if p in pkg_identifiers:
                    info = (d.get("payload") or {}).get("info") or {}
                    for k in ["author", "maintainer", "author_email", "maintainer_email"]:
                        val = info.get(k)
                        if not val or not isinstance(val, str):
                            continue
                        val = val.strip()
                        if val.lower() in ["none", "null", "unknown", "n/a", "na", ""]:
                            continue
                        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", val)
                        for em in emails:
                            pkg_identifiers[p].add("email:" + em.lower())
                        clean_name = re.sub(r"<[^>]+>", "", val)
                        clean_name = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", clean_name)
                        clean_name = clean_name.strip(',; -\'\"').lower()
                        if clean_name and clean_name not in ["none", "null", "unknown", "n/a", "na"]:
                            pkg_identifiers[p].add("name:" + clean_name)
            except Exception:
                pass

        return pkg_identifiers

    def _build_connected_components(
        self, records: List[Dict[str, Any]], pkg_identifiers: Dict[str, Set[str]]
    ) -> List[List[str]]:
        """Cluster packages into connected components based on shared identifiers."""
        packages = sorted(list({r["normalized_package"] for r in records}))
        id_to_pkgs: Dict[str, Set[str]] = {}
        for p, ids in pkg_identifiers.items():
            for ident in ids:
                id_to_pkgs.setdefault(ident, set()).add(p)

        adj: Dict[str, Set[str]] = {p: set() for p in packages}
        for ident, pkgs in id_to_pkgs.items():
            if len(pkgs) > 1:
                for p1 in pkgs:
                    for p2 in pkgs:
                        if p1 != p2:
                            adj[p1].add(p2)

        visited: Set[str] = set()
        components: List[List[str]] = []
        for p in packages:
            if p not in visited:
                comp = []
                q = [p]
                visited.add(p)
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                components.append(sorted(comp))

        return components

    def _build_package_disjoint_split(
        self,
        records: List[Dict[str, Any]],
        pkg_identifiers: Dict[str, Set[str]],
        components: List[List[str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Construct package-disjoint split ensuring P_train ∩ P_val ∩ P_test = ∅,
        grouping packages by shared maintainers and repositories, and preserving
        approximate 70/15/15 proportions.
        """
        mal_pkgs = {r["normalized_package"] for r in records if r["label"] == "confirmed_malicious"}

        mal_comps = [c for c in components if any(p in mal_pkgs for p in c)]
        benign_comps = [c for c in components if not any(p in mal_pkgs for p in c)]

        target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

        def partition_comps(comps: List[List[str]]) -> Dict[str, Set[str]]:
            tot = sum(len(c) for c in comps)
            sorted_c = sorted(comps, key=lambda c: (len(c), c[0]), reverse=True)
            splits: Dict[str, Set[str]] = {s: set() for s in target_ratios}
            counts: Dict[str, int] = {s: 0 for s in target_ratios}
            for c in sorted_c:
                c_len = len(c)
                best_s = min(
                    target_ratios.keys(),
                    key=lambda s: (counts[s] + c_len) / (target_ratios[s] * tot if tot > 0 else 1.0)
                )
                splits[best_s].update(c)
                counts[best_s] += c_len
            return splits

        mal_splits = partition_comps(mal_comps)
        benign_splits = partition_comps(benign_comps)

        train_p = mal_splits["train"] | benign_splits["train"]
        val_p = mal_splits["val"] | benign_splits["val"]
        test_p = mal_splits["test"] | benign_splits["test"]

        train_idx = []
        val_idx = []
        test_idx = []

        for idx, r in enumerate(records):
            p = r["normalized_package"]
            if p in train_p:
                train_idx.append(idx)
            elif p in val_p:
                val_idx.append(idx)
            else:
                test_idx.append(idx)

        pkg_overlap = len(train_p & val_p) + len(train_p & test_p) + len(val_p & test_p)

        shared_repos = []
        shared_maints = []
        id_to_splits: Dict[str, Set[str]] = {}
        for s_name, p_set in [("train", train_p), ("val", val_p), ("test", test_p)]:
            for p in p_set:
                for ident in pkg_identifiers.get(p, []):
                    id_to_splits.setdefault(ident, set()).add(s_name)

        for ident, s_set in id_to_splits.items():
            if len(s_set) > 1:
                if ident.startswith("repo:"):
                    shared_repos.append(ident)
                else:
                    shared_maints.append(ident)

        leakage_audit = {
            "strict_package_disjointness": pkg_overlap == 0,
            "package_overlap_count": pkg_overlap,
            "repository_overlap_count": len(shared_repos),
            "maintainer_overlap_count": len(shared_maints),
            "shared_repositories": shared_repos,
            "shared_maintainers": shared_maints,
            "unique_packages_per_split": {
                "train_packages": len(train_p),
                "val_packages": len(val_p),
                "test_packages": len(test_p),
            },
            "connected_components_total": len(components),
            "multi_package_connected_components": sum(1 for c in components if len(c) > 1),
            "neighbor_dependency_policy": (
                "Observable dependency neighbors from training packages are permitted during message passing "
                "provided edge_time <= target_release_time. Target package nodes themselves are strictly unseen."
            ),
        }

        disjoint_split = {
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
            "train_packages": train_p,
            "val_packages": val_p,
            "test_packages": test_p,
        }

        return disjoint_split, leakage_audit

    def _build_graph_package_disjoint_split(
        self,
        records: List[Dict[str, Any]],
        pkg_identifiers: Dict[str, Set[str]],
        components: List[List[str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Construct package-disjoint split restricted to the package universe that has
        at least one graph_observable release. Excludes label-only takedown packages.
        Separates graph_observable and live_unobservable_topology records.
        """
        graph_pkgs = {
            r["normalized_package"] for r in records
            if r["topology_status"] == "graph_observable"
        }
        mal_pkgs = {r["normalized_package"] for r in records if r["label"] == "confirmed_malicious"}

        graph_components: List[List[str]] = []
        for c in components:
            gc = [p for p in c if p in graph_pkgs]
            if gc:
                graph_components.append(sorted(gc))

        graph_mal_comps = [c for c in graph_components if any(p in mal_pkgs for p in c)]
        graph_benign_comps = [c for c in graph_components if not any(p in mal_pkgs for p in c)]

        target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

        def partition_comps(comps: List[List[str]]) -> Dict[str, Set[str]]:
            tot = sum(len(c) for c in comps)
            sorted_c = sorted(comps, key=lambda c: (len(c), c[0]), reverse=True)
            splits: Dict[str, Set[str]] = {s: set() for s in target_ratios}
            counts: Dict[str, int] = {s: 0 for s in target_ratios}
            for c in sorted_c:
                c_len = len(c)
                best_s = min(
                    target_ratios.keys(),
                    key=lambda s: (counts[s] + c_len) / (target_ratios[s] * tot if tot > 0 else 1.0)
                )
                splits[best_s].update(c)
                counts[best_s] += c_len
            return splits

        g_mal_splits = partition_comps(graph_mal_comps)
        g_benign_splits = partition_comps(graph_benign_comps)

        train_p = g_mal_splits["train"] | g_benign_splits["train"]
        val_p = g_mal_splits["val"] | g_benign_splits["val"]
        test_p = g_mal_splits["test"] | g_benign_splits["test"]

        split_data: Dict[str, Dict[str, Any]] = {
            "train": {"graph_observable": [], "live_unobservable": [], "packages": train_p},
            "validation": {"graph_observable": [], "live_unobservable": [], "packages": val_p},
            "test": {"graph_observable": [], "live_unobservable": [], "packages": test_p},
        }

        for idx, r in enumerate(records):
            p = r["normalized_package"]
            top = r["topology_status"]
            if r["registry_status"] == "takedown":
                continue

            target_split = None
            if p in train_p:
                target_split = "train"
            elif p in val_p:
                target_split = "validation"
            elif p in test_p:
                target_split = "test"

            if target_split:
                if top == "graph_observable":
                    split_data[target_split]["graph_observable"].append(idx)
                elif top == "live_unobservable_topology":
                    split_data[target_split]["live_unobservable"].append(idx)

        pkg_overlap = len(train_p & val_p) + len(train_p & test_p) + len(val_p & test_p)
        shared_repos = []
        shared_maints = []
        id_to_splits: Dict[str, Set[str]] = {}
        for s_name, p_set in [("train", train_p), ("val", val_p), ("test", test_p)]:
            for p in p_set:
                for ident in pkg_identifiers.get(p, []):
                    id_to_splits.setdefault(ident, set()).add(s_name)

        for ident, s_set in id_to_splits.items():
            if len(s_set) > 1:
                if ident.startswith("repo:"):
                    shared_repos.append(ident)
                else:
                    shared_maints.append(ident)

        graph_leakage_audit = {
            "strict_package_disjointness": pkg_overlap == 0,
            "package_overlap_count": pkg_overlap,
            "repository_overlap_count": len(shared_repos),
            "maintainer_overlap_count": len(shared_maints),
            "shared_repositories": shared_repos,
            "shared_maintainers": shared_maints,
        }

        graph_split_payload = {
            "split_type": "graph_model_package_disjoint_unseen_attack",
            "description": (
                "Restricted to the package universe with at least one graph_observable release. "
                "Excludes label-only takedown packages. Keeps live_unobservable_topology records separate."
            ),
            "counts": {
                "total_packages": len(graph_pkgs),
                "train_packages": len(train_p),
                "val_packages": len(val_p),
                "test_packages": len(test_p),
                "train_records": len(split_data["train"]["graph_observable"]) + len(split_data["train"]["live_unobservable"]),
                "val_records": len(split_data["validation"]["graph_observable"]) + len(split_data["validation"]["live_unobservable"]),
                "test_records": len(split_data["test"]["graph_observable"]) + len(split_data["test"]["live_unobservable"]),
                "train_graph_observable": len(split_data["train"]["graph_observable"]),
                "train_live_unobservable": len(split_data["train"]["live_unobservable"]),
                "train_confirmed_malicious": sum(1 for idx in (split_data["train"]["graph_observable"] + split_data["train"]["live_unobservable"]) if records[idx]["label"] == "confirmed_malicious"),
                "val_graph_observable": len(split_data["validation"]["graph_observable"]),
                "val_live_unobservable": len(split_data["validation"]["live_unobservable"]),
                "val_confirmed_malicious": sum(1 for idx in (split_data["validation"]["graph_observable"] + split_data["validation"]["live_unobservable"]) if records[idx]["label"] == "confirmed_malicious"),
                "test_graph_observable": len(split_data["test"]["graph_observable"]),
                "test_live_unobservable": len(split_data["test"]["live_unobservable"]),
                "test_confirmed_malicious": sum(1 for idx in (split_data["test"]["graph_observable"] + split_data["test"]["live_unobservable"]) if records[idx]["label"] == "confirmed_malicious"),
            },
            "packages": {
                "train_packages": sorted(list(train_p)),
                "val_packages": sorted(list(val_p)),
                "test_packages": sorted(list(test_p)),
            },
            "identifier_leakage_audit": graph_leakage_audit,
            "train": {
                "package_count": len(train_p),
                "total_records": len(split_data["train"]["graph_observable"]) + len(split_data["train"]["live_unobservable"]),
                "graph_observable_indices": split_data["train"]["graph_observable"],
                "unobservable_topology_indices": split_data["train"]["live_unobservable"],
            },
            "validation": {
                "package_count": len(val_p),
                "total_records": len(split_data["validation"]["graph_observable"]) + len(split_data["validation"]["live_unobservable"]),
                "graph_observable_indices": split_data["validation"]["graph_observable"],
                "unobservable_topology_indices": split_data["validation"]["live_unobservable"],
            },
            "test": {
                "package_count": len(test_p),
                "total_records": len(split_data["test"]["graph_observable"]) + len(split_data["test"]["live_unobservable"]),
                "graph_observable_indices": split_data["test"]["graph_observable"],
                "unobservable_topology_indices": split_data["test"]["live_unobservable"],
            },
        }

        return graph_split_payload, graph_leakage_audit

    def _export_experimental_csv(self, records: List[Dict[str, Any]]) -> Path:
        out_path = self.experimental_dir / "experimental_versions.csv"
        fieldnames = [
            "package",
            "normalized_package",
            "version",
            "release_timestamp",
            "label",
            "label_scope",
            "registry_status",
            "dependency_metadata_status",
            "topology_status",
            "sample_weight",
            "requires_python",
            "requires_dist",
            "repository_url",
            "homepage_url",
            "source",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Exported {len(records)} experimental version records to {out_path}")
        return out_path
