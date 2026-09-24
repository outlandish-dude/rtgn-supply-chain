"""
Temporal Graph Container with Causal Snapshot Filtering Interface.
Guarantees that snapshot(t) exposes ONLY edges where edge_time <= t.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch_geometric.data import HeteroData

from src.graph.schema import REL_DEPENDS_ON, REL_HAS_VERSION, NodeType

logger = logging.getLogger(__name__)


class TemporalGraph:
    """
    In-memory representation of the temporal heterogeneous software supply-chain network.
    Exposes causal snapshot interface get_graph_state(t) / snapshot(t).
    """

    def __init__(
        self,
        package_nodes: List[Dict[str, Any]],
        version_nodes: List[Dict[str, Any]],
        has_version_edges: List[Tuple[int, int, float]],  # (src_pkg_id, dst_ver_id, edge_time)
        depends_on_edges: List[Tuple[int, int, float]],   # (src_ver_id, dst_pkg_id, edge_time)
        version_features: torch.Tensor,
        version_labels: torch.Tensor,
        version_weights: torch.Tensor,
        split_masks: Dict[str, torch.Tensor],
    ) -> None:
        self.package_nodes = package_nodes
        self.version_nodes = version_nodes
        self.has_version_edges = has_version_edges
        self.depends_on_edges = depends_on_edges
        self.version_features = version_features
        self.version_labels = version_labels
        self.version_weights = version_weights
        self.split_masks = split_masks

        # Internal tensor conversions for vectorized filtering
        self.num_packages = len(package_nodes)
        self.num_versions = len(version_nodes)

        # HAS_VERSION tensors
        if has_version_edges:
            self._hv_src = torch.tensor([e[0] for e in has_version_edges], dtype=torch.long)
            self._hv_dst = torch.tensor([e[1] for e in has_version_edges], dtype=torch.long)
            self._hv_time = torch.tensor([e[2] for e in has_version_edges], dtype=torch.float32)
        else:
            self._hv_src = torch.empty(0, dtype=torch.long)
            self._hv_dst = torch.empty(0, dtype=torch.long)
            self._hv_time = torch.empty(0, dtype=torch.float32)

        # DEPENDS_ON tensors
        if depends_on_edges:
            self._dep_src = torch.tensor([e[0] for e in depends_on_edges], dtype=torch.long)
            self._dep_dst = torch.tensor([e[1] for e in depends_on_edges], dtype=torch.long)
            self._dep_time = torch.tensor([e[2] for e in depends_on_edges], dtype=torch.float32)
        else:
            self._dep_src = torch.empty(0, dtype=torch.long)
            self._dep_dst = torch.empty(0, dtype=torch.long)
            self._dep_time = torch.empty(0, dtype=torch.float32)

        # Version timestamps
        self._version_times = torch.tensor([
            float(v["release_timestamp"]) if v["release_timestamp"] is not None else float("inf")
            for v in version_nodes
        ], dtype=torch.float32)

        self._topology_status = [v["topology_status"] for v in version_nodes]

    def get_graph_state(self, t: float) -> HeteroData:
        """
        Causal temporal filter:
        Constructs a HeteroData snapshot exposing strictly edges where edge_time <= t.
        """
        data = HeteroData()

        # Node counts and features
        data[NodeType.PACKAGE.value].num_nodes = self.num_packages
        data[NodeType.PACKAGE_VERSION.value].x = self.version_features
        data[NodeType.PACKAGE_VERSION.value].y = self.version_labels
        data[NodeType.PACKAGE_VERSION.value].sample_weight = self.version_weights
        data[NodeType.PACKAGE_VERSION.value].num_nodes = self.num_versions

        # Causal version active mask (uploaded at or before t AND observable)
        active_versions = (self._version_times <= t) & torch.tensor(
            [s == "graph_observable" for s in self._topology_status], dtype=torch.bool
        )
        data[NodeType.PACKAGE_VERSION.value].active_mask = active_versions

        # 1. Filter HAS_VERSION edges (edge_time <= t)
        hv_mask = self._hv_time <= t
        if hv_mask.any():
            data[REL_HAS_VERSION].edge_index = torch.stack([self._hv_src[hv_mask], self._hv_dst[hv_mask]], dim=0)
            data[REL_HAS_VERSION].edge_time = self._hv_time[hv_mask]
        else:
            data[REL_HAS_VERSION].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[REL_HAS_VERSION].edge_time = torch.empty(0, dtype=torch.float32)

        # 2. Filter DEPENDS_ON edges (edge_time <= t)
        dep_mask = self._dep_time <= t
        if dep_mask.any():
            data[REL_DEPENDS_ON].edge_index = torch.stack([self._dep_src[dep_mask], self._dep_dst[dep_mask]], dim=0)
            data[REL_DEPENDS_ON].edge_time = self._dep_time[dep_mask]
        else:
            data[REL_DEPENDS_ON].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[REL_DEPENDS_ON].edge_time = torch.empty(0, dtype=torch.float32)

        # Add split masks
        for k, mask in self.split_masks.items():
            setattr(data, k, mask)

        data.snapshot_timestamp = float(t)
        return data

    def snapshot(self, t: float) -> HeteroData:
        """Alias for get_graph_state(t)."""
        return self.get_graph_state(t)

    def to_full_pyg(self) -> HeteroData:
        """
        Export complete graph with explicit edge_time tensors on every relation.
        """
        data = HeteroData()
        data[NodeType.PACKAGE.value].num_nodes = self.num_packages
        data[NodeType.PACKAGE_VERSION.value].x = self.version_features
        data[NodeType.PACKAGE_VERSION.value].y = self.version_labels
        data[NodeType.PACKAGE_VERSION.value].sample_weight = self.version_weights
        data[NodeType.PACKAGE_VERSION.value].num_nodes = self.num_versions

        data[REL_HAS_VERSION].edge_index = torch.stack([self._hv_src, self._hv_dst], dim=0)
        data[REL_HAS_VERSION].edge_time = self._hv_time

        data[REL_DEPENDS_ON].edge_index = torch.stack([self._dep_src, self._dep_dst], dim=0)
        data[REL_DEPENDS_ON].edge_time = self._dep_time

        for k, mask in self.split_masks.items():
            setattr(data, k, mask)

        return data
