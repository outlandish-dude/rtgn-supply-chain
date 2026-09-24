"""
Schema and data containers for Temporal Heterogeneous Graph Construction.
Minimal, defensible, source-attested schema:
- Nodes: Package, PackageVersion
- Edges: HAS_VERSION (Package -> PackageVersion), DEPENDS_ON (PackageVersion -> Package)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NodeType(str, Enum):
    PACKAGE = "package"
    PACKAGE_VERSION = "package_version"


class EdgeType(str, Enum):
    HAS_VERSION = "has_version"
    DEPENDS_ON = "depends_on"


# Formal PyG edge type triples
REL_HAS_VERSION: Tuple[str, str, str] = (NodeType.PACKAGE.value, EdgeType.HAS_VERSION.value, NodeType.PACKAGE_VERSION.value)
REL_DEPENDS_ON: Tuple[str, str, str] = (NodeType.PACKAGE_VERSION.value, EdgeType.DEPENDS_ON.value, NodeType.PACKAGE.value)


class TargetLabel(int, Enum):
    """
    Standard binary target labels for RTGN compromise detection.
    - POSITIVE = 1 (confirmed_malicious)
    - NEGATIVE = 0 (assumed_benign_background)
    - UNLABELED = -1 (source_unflagged releases, preserved for sensitivity analysis)
    """
    ASSUMED_BENIGN = 0
    CONFIRMED_MALICIOUS = 1
    SOURCE_UNFLAGGED = -1


@dataclass
class PackageNode:
    node_id: int
    name: str
    normalized_name: str
    in_registry: bool
    is_takedown: bool


@dataclass
class PackageVersionNode:
    node_id: int
    package_name: str
    normalized_package: str
    version: str
    release_timestamp: Optional[float]
    label_str: str
    target_label: int  # 1 for confirmed_malicious, 0 for assumed_benign, -1 for source_unflagged
    sample_weight: float
    registry_status: str
    dependency_metadata_status: str
    topology_status: str


@dataclass
class TemporalEdge:
    edge_id: int
    src_type: str
    src_id: int
    rel_type: str
    dst_type: str
    dst_id: int
    edge_time: float  # Release timestamp of the version creating the relationship
