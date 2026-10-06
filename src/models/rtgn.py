"""
Real-Time Temporal Graph Neural Network (RTGN) Architecture.

Comprehensive Model combining:
1. 13-dimensional Causal Feature Projection.
2. Continuous Harmonic Time Encoding (days).
3. Dynamic Stateful Package Memory (GRU, NO static package-ID embeddings).
4. 2-Layer Relation-Aware Heterogeneous GATv2 Message Passing.
5. 2-Layer MLP Detection Head for release compromise probability.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.hetero_conv import HeteroGATv2Net
from src.models.temporal_memory import PackageMemoryManager
from src.models.time_encoding import HarmonicTimeEncoder

logger = logging.getLogger(__name__)


class RTGNModel(nn.Module):
    """
    Real-Time Temporal Graph Neural Network for Release-Level Compromise Detection.
    """

    def __init__(
        self,
        num_packages: int,
        in_feature_dim: int = 13,
        hidden_dim: int = 64,
        time_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.2,
        use_time_encoding: bool = True,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        self.num_packages = num_packages
        self.in_feature_dim = in_feature_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.use_time_encoding = use_time_encoding
        self.use_graph = use_graph

        # 1. Feature encoder for PackageVersion causal features (in_feature_dim -> hidden_dim)
        self.version_encoder = nn.Sequential(
            nn.Linear(in_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Continuous harmonic time encoder for edge elapsed time (days -> 32)
        if self.use_time_encoding:
            self.time_encoder = HarmonicTimeEncoder(dimension=time_dim)
        else:
            self.time_encoder = None

        # 3. Dynamic package temporal memory (NO static package-ID embeddings)
        self.package_memory = PackageMemoryManager(
            num_packages=num_packages,
            memory_dim=hidden_dim,
            message_dim=hidden_dim,
        )

        # 4. Relation-aware Heterogeneous GATv2 (2 layers)
        if self.use_graph:
            self.hetero_gnn = HeteroGATv2Net(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                edge_dim=time_dim if self.use_time_encoding else None,
                heads=heads,
                dropout=dropout,
            )
        else:
            self.hetero_gnn = None

        # 5. Detection Head: 2-layer MLP (64 -> 32 -> 1)
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def encode_edge_times(
        self,
        edge_time_dict: Dict[str, torch.Tensor],
        query_time: float,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode elapsed time from each edge timestamp to query timestamp:
            delta_seconds = max(0, query_time - edge_time)
            delta_days = delta_seconds / 86400.0
        Returns None if use_time_encoding is False.
        """
        if not self.use_time_encoding or self.time_encoder is None:
            return None

        attr_dict = {}
        for rel_name, edge_times in edge_time_dict.items():
            if edge_times.numel() == 0:
                attr_dict[rel_name] = torch.empty((0, self.time_dim), device=edge_times.device)
            else:
                delta_sec = torch.clamp(query_time - edge_times, min=0.0)
                attr_dict[rel_name] = self.time_encoder(delta_sec)
        return attr_dict

    def forward(
        self,
        version_features: torch.Tensor,
        package_ids: torch.Tensor,
        edge_index_dict: Optional[Dict[str, torch.Tensor]] = None,
        edge_attr_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through RTGN.

        Args:
            version_features: Tensor of shape (N_v, in_feature_dim)
            package_ids: Tensor of shape (M_p, ) indicating package indices
            edge_index_dict: dict of edge indices per relation (or None if no graph)
            edge_attr_dict: dict of harmonic time encodings per relation (or None)
        Returns:
            logits: (N_v, ) unnormalized classification scores
            probabilities: (N_v, ) sigmoid probabilities
            node_representations: dict with updated 'package' and 'package_version' tensors
        """
        # 1. Project version features
        h_ver = self.version_encoder(version_features)

        # 2. Query package memory states (dynamic state, no static package ID embedding)
        h_pkg = self.package_memory.get_memory(package_ids)

        # 3. Heterogeneous GATv2 message passing (or identity bypass if use_graph=False)
        if self.use_graph and self.hetero_gnn is not None and edge_index_dict is not None:
            x_dict = {
                "package": h_pkg,
                "package_version": h_ver,
            }
            out_dict = self.hetero_gnn(x_dict, edge_index_dict, edge_attr_dict)
        else:
            out_dict = {
                "package": h_pkg,
                "package_version": h_ver,
            }

        # 4. Predict compromise logits on version representations
        h_ver_out = out_dict["package_version"]
        logits = self.detection_head(h_ver_out).squeeze(-1)
        probs = torch.sigmoid(logits)

        return logits, probs, out_dict
