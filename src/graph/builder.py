"""
Temporal Heterogeneous Graph Builder.
Constructs inspectable node/edge tables, causal feature matrices, split masks,
and serialized PyG HeteroData objects with explicit edge_time.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
from packaging.requirements import InvalidRequirement, Requirement

from src.data.normalization import normalize_package_name
from src.graph.features import (
    FEATURE_NAMES,
    FEATURE_SPECIFICATION,
    NUM_FEATURES,
    compute_causal_features_for_release,
)
from src.graph.schema import (
    REL_DEPENDS_ON,
    REL_HAS_VERSION,
    NodeType,
    TargetLabel,
)
from src.graph.temporal_graph import TemporalGraph

logger = logging.getLogger(__name__)


class TemporalGraphBuilder:
    """
    Builds the temporal software supply-chain network from experimental dataset artifacts.
    """

    def __init__(
        self,
        experimental_dir: str | Path = "data/experimental",
        processed_dir: str | Path = "data/processed",
        output_dir: str | Path = "data/experimental/graph",
    ) -> None:
        self.experimental_dir = Path(experimental_dir)
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> Tuple[TemporalGraph, Dict[str, Any]]:
        """
        Execute graph construction pipeline and export artifacts.
        """
        logger.info("Starting temporal graph construction...")

        # 1. Load Experimental Versions
        versions_csv = self.experimental_dir / "experimental_versions.csv"
        records: List[Dict[str, Any]] = []
        with open(versions_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                records.append(r)

        logger.info(f"Loaded {len(records)} experimental version records.")

        # 2. Build Package Nodes
        # Unique canonical packages
        pkg_names = sorted(list({r["normalized_package"] for r in records}))
        pkg_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(pkg_names)}

        # Package metadata
        pkg_records: Dict[str, Dict[str, Any]] = {}
        for r in records:
            p = r["normalized_package"]
            if p not in pkg_records:
                pkg_records[p] = {
                    "package_id": pkg_to_id[p],
                    "normalized_name": p,
                    "display_name": r["package"],
                    "has_live_releases": False,
                    "is_takedown": False,
                }
            if r["registry_status"] == "live":
                pkg_records[p]["has_live_releases"] = True
            elif r["registry_status"] == "takedown":
                pkg_records[p]["is_takedown"] = True

        package_nodes_list = [pkg_records[p] for p in pkg_names]

        # 3. Build PackageVersion Nodes
        version_nodes_list: List[Dict[str, Any]] = []
        ver_key_to_id: Dict[Tuple[str, str], int] = {}

        for idx, r in enumerate(records):
            p = r["normalized_package"]
            v = r["version"]
            ver_key_to_id[(p, v)] = idx

            label_str = r["label"]
            if label_str == "confirmed_malicious":
                target_label = TargetLabel.CONFIRMED_MALICIOUS.value
            elif label_str == "assumed_benign_background":
                target_label = TargetLabel.ASSUMED_BENIGN.value
            else:
                target_label = TargetLabel.SOURCE_UNFLAGGED.value

            ts = float(r["release_timestamp"]) if r["release_timestamp"] else None
            weight = float(r.get("sample_weight", 1.0))

            version_nodes_list.append({
                "version_id": idx,
                "package_id": pkg_to_id[p],
                "normalized_package": p,
                "package_name": r["package"],
                "version": v,
                "release_timestamp": ts,
                "label_str": label_str,
                "target_label": target_label,
                "sample_weight": weight,
                "registry_status": r["registry_status"],
                "dependency_metadata_status": r["dependency_metadata_status"],
                "topology_status": r["topology_status"],
            })

        # 4. Load Public Security Events (for historical public context only)
        security_csv = self.processed_dir / "security_events.csv"
        pkg_to_advisories: Dict[str, List[Tuple[str, float]]] = {p: [] for p in pkg_names}
        if security_csv.exists():
            with open(security_csv, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    p = r["normalized_package"]
                    aff_v = r["version"]
                    pub_ts_raw = r.get("published_timestamp")
                    if pub_ts_raw and p in pkg_to_advisories:
                        try:
                            pub_ts = float(pub_ts_raw)
                            pkg_to_advisories[p].append((aff_v, pub_ts))
                        except Exception:
                            pass

        # 5. Pre-index Historical Package Releases & Ecosystem Dependency Events
        pkg_to_releases: Dict[str, List[Tuple[str, float]]] = {p: [] for p in pkg_names}
        ecosystem_dep_events: List[Tuple[str, float]] = []
        pkg_to_downstream_ts: Dict[str, List[float]] = {p: [] for p in pkg_names}

        for r in records:
            p = r["normalized_package"]
            ts = float(r["release_timestamp"]) if r["release_timestamp"] else None
            if ts is not None:
                pkg_to_releases[p].append((r["version"], ts))

                # Extract declared dependencies for dynamic in-degree calculation
                req_dist_str = r.get("requires_dist")
                if req_dist_str and req_dist_str != "[]" and req_dist_str != '""':
                    try:
                        req_list = json.loads(req_dist_str)
                        if isinstance(req_list, list):
                            for req_s in req_list:
                                try:
                                    parsed_req = Requirement(req_s)
                                    target_p = normalize_package_name(parsed_req.name)
                                    ecosystem_dep_events.append((target_p, ts))
                                    if target_p in pkg_to_downstream_ts:
                                        pkg_to_downstream_ts[target_p].append(ts)
                                except InvalidRequirement:
                                    pass
                    except Exception:
                        pass

        for p in pkg_to_downstream_ts:
            pkg_to_downstream_ts[p].sort()
        for p in pkg_to_releases:
            pkg_to_releases[p].sort(key=lambda x: x[1])

        # 6. Build Causal Edges with Explicit edge_time
        has_version_edges: List[Tuple[int, int, float]] = []
        depends_on_edges: List[Tuple[int, int, float]] = []

        for idx, r in enumerate(records):
            p = r["normalized_package"]
            pkg_id = pkg_to_id[p]
            ts = float(r["release_timestamp"]) if r["release_timestamp"] else None

            # HAS_VERSION edge: Package -> PackageVersion at release upload time
            if ts is not None:
                has_version_edges.append((pkg_id, idx, ts))

            # DEPENDS_ON edge: PackageVersion -> Package at release upload time
            # Strictly for observable topology (takedown and missing metadata excluded)
            if ts is not None and r["topology_status"] == "graph_observable":
                req_dist_str = r.get("requires_dist")
                if req_dist_str and req_dist_str != "[]" and req_dist_str != '""':
                    try:
                        req_list = json.loads(req_dist_str)
                        if isinstance(req_list, list):
                            seen_targets = set()
                            for req_s in req_list:
                                try:
                                    parsed_req = Requirement(req_s)
                                    target_p = normalize_package_name(parsed_req.name)
                                    # Link only to known packages in universe, avoiding duplicates per release
                                    if target_p in pkg_to_id and target_p not in seen_targets:
                                        seen_targets.add(target_p)
                                        target_pkg_id = pkg_to_id[target_p]
                                        depends_on_edges.append((idx, target_pkg_id, ts))
                                except InvalidRequirement:
                                    pass
                    except Exception:
                        pass

        logger.info(
            f"Constructed {len(has_version_edges)} HAS_VERSION edges and "
            f"{len(depends_on_edges)} DEPENDS_ON edges."
        )

        # 7. Compute Causal Feature Matrix for PackageVersion Nodes
        logger.info("Computing causal zero-lookahead feature matrix for versions...")
        features_list = []
        for r in records:
            p = r["normalized_package"]
            feat_vec = compute_causal_features_for_release(
                row=r,
                historical_package_releases=pkg_to_releases.get(p, []),
                historical_dependency_events=pkg_to_downstream_ts,
                historical_advisories=pkg_to_advisories.get(p, []),
            )
            features_list.append(feat_vec)

        version_features = torch.tensor(features_list, dtype=torch.float32)
        version_labels = torch.tensor([v["target_label"] for v in version_nodes_list], dtype=torch.long)
        version_weights = torch.tensor([v["sample_weight"] for v in version_nodes_list], dtype=torch.float32)

        # 8. Construct Split Masks
        split_masks = self._load_and_build_split_masks(records)

        # 9. Instantiate TemporalGraph Container
        temporal_graph = TemporalGraph(
            package_nodes=package_nodes_list,
            version_nodes=version_nodes_list,
            has_version_edges=has_version_edges,
            depends_on_edges=depends_on_edges,
            version_features=version_features,
            version_labels=version_labels,
            version_weights=version_weights,
            split_masks=split_masks,
        )

        # 10. Persist Inspectable Graph Artifacts
        summary = self._export_graph_artifacts(
            package_nodes_list=package_nodes_list,
            version_nodes_list=version_nodes_list,
            has_version_edges=has_version_edges,
            depends_on_edges=depends_on_edges,
            temporal_graph=temporal_graph,
            split_masks=split_masks,
        )

        logger.info("Temporal graph construction successfully completed.")
        return temporal_graph, summary

    def _load_and_build_split_masks(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Construct boolean index masks for chronological and graph package-disjoint splits."""
        num_versions = len(records)
        masks = {
            "train_mask_chrono": torch.zeros(num_versions, dtype=torch.bool),
            "val_mask_chrono": torch.zeros(num_versions, dtype=torch.bool),
            "test_mask_chrono": torch.zeros(num_versions, dtype=torch.bool),
            "takedown_mask_chrono": torch.zeros(num_versions, dtype=torch.bool),
            "train_mask_graph_disjoint": torch.zeros(num_versions, dtype=torch.bool),
            "val_mask_graph_disjoint": torch.zeros(num_versions, dtype=torch.bool),
            "test_mask_graph_disjoint": torch.zeros(num_versions, dtype=torch.bool),
        }

        # Chronological Split
        chrono_path = self.experimental_dir / "splits" / "chronological_split.json"
        if chrono_path.exists():
            with open(chrono_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            masks["train_mask_chrono"][c_data["train_indices"]] = True
            masks["val_mask_chrono"][c_data["validation_indices"]] = True
            masks["test_mask_chrono"][c_data["test_indices"]] = True
            masks["takedown_mask_chrono"][c_data["takedown_indices"]] = True

        # Graph Package-Disjoint Split
        graph_split_path = self.experimental_dir / "splits" / "graph_package_disjoint_split.json"
        if graph_split_path.exists():
            with open(graph_split_path, "r", encoding="utf-8") as f:
                g_data = json.load(f)
            tr_idx = g_data["train"]["graph_observable_indices"] + g_data["train"]["unobservable_topology_indices"]
            val_idx = g_data["validation"]["graph_observable_indices"] + g_data["validation"]["unobservable_topology_indices"]
            te_idx = g_data["test"]["graph_observable_indices"] + g_data["test"]["unobservable_topology_indices"]
            masks["train_mask_graph_disjoint"][tr_idx] = True
            masks["val_mask_graph_disjoint"][val_idx] = True
            masks["test_mask_graph_disjoint"][te_idx] = True

        return masks

    def _export_graph_artifacts(
        self,
        package_nodes_list: List[Dict[str, Any]],
        version_nodes_list: List[Dict[str, Any]],
        has_version_edges: List[Tuple[int, int, float]],
        depends_on_edges: List[Tuple[int, int, float]],
        temporal_graph: TemporalGraph,
        split_masks: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Export tabular CSVs, PyG binary objects, and schema documentation."""
        # 1. Export Node Tables
        nodes_pkg_path = self.output_dir / "nodes_package.csv"
        with open(nodes_pkg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["package_id", "normalized_name", "display_name", "has_live_releases", "is_takedown"])
            writer.writeheader()
            writer.writerows(package_nodes_list)

        nodes_ver_path = self.output_dir / "nodes_package_version.csv"
        with open(nodes_ver_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "version_id", "package_id", "normalized_package", "package_name", "version",
                "release_timestamp", "label_str", "target_label", "sample_weight",
                "registry_status", "dependency_metadata_status", "topology_status"
            ])
            writer.writeheader()
            writer.writerows(version_nodes_list)

        # 2. Export Edge Tables with explicit edge_time
        edges_hv_path = self.output_dir / "edges_has_version.csv"
        with open(edges_hv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["package_id", "version_id", "edge_time"])
            for e in has_version_edges:
                writer.writerow([e[0], e[1], e[2]])

        edges_dep_path = self.output_dir / "edges_depends_on.csv"
        with open(edges_dep_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["version_id", "target_package_id", "edge_time"])
            for e in depends_on_edges:
                writer.writerow([e[0], e[1], e[2]])

        # 3. Export Features (PyTorch tensor & CSV)
        feat_pt_path = self.output_dir / "package_version_features.pt"
        torch.save(temporal_graph.version_features, feat_pt_path)

        feat_csv_path = self.output_dir / "package_version_features.csv"
        with open(feat_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["version_id"] + FEATURE_NAMES)
            for idx, vec in enumerate(temporal_graph.version_features.tolist()):
                writer.writerow([idx] + vec)

        # 4. Export PyG HeteroData container
        pyg_data = temporal_graph.to_full_pyg()
        pyg_pt_path = self.output_dir / "temporal_hetero_graph.pt"
        torch.save(pyg_data, pyg_pt_path)

        # 5. Export Split Masks
        masks_pt_path = self.output_dir / "split_masks.pt"
        torch.save(split_masks, masks_pt_path)

        # 6. Feature Metadata JSON
        feat_meta_path = self.output_dir / "feature_metadata.json"
        with open(feat_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "num_features": NUM_FEATURES,
                "feature_names": FEATURE_NAMES,
                "features": FEATURE_SPECIFICATION,
            }, f, indent=2)

        # 7. Graph Summary Metadata
        summary = {
            "node_counts": {
                "package": len(package_nodes_list),
                "package_version": len(version_nodes_list),
            },
            "edge_counts": {
                "has_version": len(has_version_edges),
                "depends_on": len(depends_on_edges),
            },
            "feature_matrix_shape": list(temporal_graph.version_features.shape),
            "chronological_split_counts": {
                "train": int(split_masks["train_mask_chrono"].sum()),
                "validation": int(split_masks["val_mask_chrono"].sum()),
                "test": int(split_masks["test_mask_chrono"].sum()),
                "takedown_catalog": int(split_masks["takedown_mask_chrono"].sum()),
            },
            "graph_package_disjoint_split_counts": {
                "train": int(split_masks["train_mask_graph_disjoint"].sum()),
                "validation": int(split_masks["val_mask_graph_disjoint"].sum()),
                "test": int(split_masks["test_mask_graph_disjoint"].sum()),
            },
            "topology_observability": {
                "graph_observable": sum(1 for v in version_nodes_list if v["topology_status"] == "graph_observable"),
                "live_unobservable_topology": sum(1 for v in version_nodes_list if v["topology_status"] == "live_unobservable_topology"),
                "label_only_takedown": sum(1 for v in version_nodes_list if v["topology_status"] == "label_only_takedown"),
            },
        }

        summary_path = self.output_dir / "graph_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
