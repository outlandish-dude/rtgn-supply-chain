"""
Relation-Aware Heterogeneous GATv2 Message Passing for RTGN.

Implements:
- Exactly 2 heterogeneous GATv2 layers.
- Relation-specific projections and attention heads:
    1. DEPENDS_ON:  PackageVersion -> Package (release depends on package)
    2. DEPENDED_BY: Package -> PackageVersion (reverse: packages/releases that depend on this package)
    3. HAS_VERSION: Package -> PackageVersion (package owns release)
    4. VERSION_OF:  PackageVersion -> Package (release belongs to package)
- Edge attributes: Continuous harmonic time encoding of elapsed days.
- Hidden dimension: 64, Attention heads: 4 (16 per head -> 64 concatenated), Dropout: 0.2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class HeteroGATv2Layer(nn.Module):
    """
    Single relation-aware Heterogeneous GATv2 convolution layer with continuous edge time attributes.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 64,
        edge_dim: Optional[int] = 32,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by heads ({heads})")
        head_dim = out_dim // heads

        self.heads = heads
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        # 1. Relation-specific GATv2 convolutions
        # DEPENDS_ON: PackageVersion -> Package
        self.conv_depends_on = GATv2Conv(
            in_channels=(in_dim, in_dim),
            out_channels=head_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )

        # DEPENDED_BY: Package -> PackageVersion
        self.conv_depended_by = GATv2Conv(
            in_channels=(in_dim, in_dim),
            out_channels=head_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )

        # HAS_VERSION: Package -> PackageVersion
        self.conv_has_version = GATv2Conv(
            in_channels=(in_dim, in_dim),
            out_channels=head_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )

        # VERSION_OF: PackageVersion -> Package
        self.conv_version_of = GATv2Conv(
            in_channels=(in_dim, in_dim),
            out_channels=head_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )

        # 2. Residual self-projections
        self.self_proj_ver = nn.Linear(in_dim, out_dim)
        self.self_proj_pkg = nn.Linear(in_dim, out_dim)

        # 3. Layer normalization and dropout
        self.norm_ver = nn.LayerNorm(out_dim)
        self.norm_pkg = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        edge_attr_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: {'package': (M, in_dim), 'package_version': (N, in_dim)}
            edge_index_dict: dict mapping relation names to (2, E_r) tensors
            edge_attr_dict: optional dict mapping relation names to (E_r, edge_dim) time encodings
        """
        x_pkg = x_dict["package"]
        x_ver = x_dict["package_version"]

        # Accumulators for incoming messages
        pkg_msgs = []
        ver_msgs = []

        # Relation 1: DEPENDS_ON (PackageVersion -> Package)
        rel_dep = "depends_on"
        if rel_dep in edge_index_dict and edge_index_dict[rel_dep].numel() > 0:
            edge_attr = (
                edge_attr_dict[rel_dep]
                if (edge_attr_dict is not None and self.edge_dim is not None and rel_dep in edge_attr_dict)
                else None
            )
            out_pkg = self.conv_depends_on(
                (x_ver, x_pkg),
                edge_index_dict[rel_dep],
                edge_attr=edge_attr,
            )
            pkg_msgs.append(out_pkg)

        # Relation 2: VERSION_OF (PackageVersion -> Package)
        rel_vof = "version_of"
        if rel_vof in edge_index_dict and edge_index_dict[rel_vof].numel() > 0:
            edge_attr = (
                edge_attr_dict[rel_vof]
                if (edge_attr_dict is not None and self.edge_dim is not None and rel_vof in edge_attr_dict)
                else None
            )
            out_pkg2 = self.conv_version_of(
                (x_ver, x_pkg),
                edge_index_dict[rel_vof],
                edge_attr=edge_attr,
            )
            pkg_msgs.append(out_pkg2)

        # Relation 3: HAS_VERSION (Package -> PackageVersion)
        rel_hver = "has_version"
        if rel_hver in edge_index_dict and edge_index_dict[rel_hver].numel() > 0:
            edge_attr = (
                edge_attr_dict[rel_hver]
                if (edge_attr_dict is not None and self.edge_dim is not None and rel_hver in edge_attr_dict)
                else None
            )
            out_ver = self.conv_has_version(
                (x_pkg, x_ver),
                edge_index_dict[rel_hver],
                edge_attr=edge_attr,
            )
            ver_msgs.append(out_ver)

        # Relation 4: DEPENDED_BY (Package -> PackageVersion)
        rel_dpby = "depended_by"
        if rel_dpby in edge_index_dict and edge_index_dict[rel_dpby].numel() > 0:
            edge_attr = (
                edge_attr_dict[rel_dpby]
                if (edge_attr_dict is not None and self.edge_dim is not None and rel_dpby in edge_attr_dict)
                else None
            )
            out_ver2 = self.conv_depended_by(
                (x_pkg, x_ver),
                edge_index_dict[rel_dpby],
                edge_attr=edge_attr,
            )
            ver_msgs.append(out_ver2)

        # Aggregate with residual connection
        # Package node update
        h_pkg = self.self_proj_pkg(x_pkg)
        if pkg_msgs:
            h_pkg = h_pkg + torch.stack(pkg_msgs, dim=0).mean(dim=0)
        h_pkg = self.norm_pkg(self.act(h_pkg))
        h_pkg = self.dropout(h_pkg)

        # Version node update
        h_ver = self.self_proj_ver(x_ver)
        if ver_msgs:
            h_ver = h_ver + torch.stack(ver_msgs, dim=0).mean(dim=0)
        h_ver = self.norm_ver(self.act(h_ver))
        h_ver = self.dropout(h_ver)

        return {
            "package": h_pkg,
            "package_version": h_ver,
        }


class HeteroGATv2Net(nn.Module):
    """
    2-Layer Relation-Aware Heterogeneous GATv2 Network.
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 64,
        out_dim: int = 64,
        edge_dim: Optional[int] = 32,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        self.layer1 = HeteroGATv2Layer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
        )
        self.layer2 = HeteroGATv2Layer(
            in_dim=hidden_dim,
            out_dim=out_dim,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        edge_attr_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        h1 = self.layer1(x_dict, edge_index_dict, edge_attr_dict)
        h2 = self.layer2(h1, edge_index_dict, edge_attr_dict)
        return h2
