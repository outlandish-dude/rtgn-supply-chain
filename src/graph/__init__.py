"""
Graph module for Software Supply-Chain Compromise Detection.
"""

from src.graph.builder import TemporalGraphBuilder
from src.graph.features import FEATURE_NAMES, FEATURE_SPECIFICATION, NUM_FEATURES
from src.graph.schema import (
    REL_DEPENDS_ON,
    REL_HAS_VERSION,
    EdgeType,
    NodeType,
    TargetLabel,
)
from src.graph.temporal_graph import TemporalGraph

__all__ = [
    "TemporalGraph",
    "TemporalGraphBuilder",
    "NodeType",
    "EdgeType",
    "TargetLabel",
    "REL_HAS_VERSION",
    "REL_DEPENDS_ON",
    "FEATURE_NAMES",
    "FEATURE_SPECIFICATION",
    "NUM_FEATURES",
]
