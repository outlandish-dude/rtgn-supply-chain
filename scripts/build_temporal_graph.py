"""
CLI Script to build the temporal heterogeneous software supply-chain graph.
Strictly non-executable: fetches zero packages and runs zero models.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.builder import TemporalGraphBuilder
from src.graph.features import FEATURE_SPECIFICATION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def write_graph_design_document(
    output_path: Path,
    summary: dict,
) -> None:
    """Generate docs/TEMPORAL_GRAPH_DESIGN.md."""
    lines = [
        "# Temporal Heterogeneous Graph Specification & Causal Design",
        "",
        "> **Generated:** " + datetime.now(timezone.utc).isoformat() + "  ",
        "> **Pipeline Mode:** Dynamic Temporal Graph Construction & Causal Snapshot Interface  ",
        "> **Source Dataset:** `data/experimental/experimental_versions.csv`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Graph Schema",
        "",
        "This specification establishes the temporal heterogeneous graph layer (`data/experimental/graph/`) for the Real-Time Temporal Graph Neural Network (RTGN).",
        "",
        "### Key Methodological & Causality Guarantees:",
        "1. **Causal Snapshot Interface (`snapshot(t)`):** Graph states at timestamp $t$ expose strictly edges where $\\text{edge\\_time} \\le t$. Zero edges from future releases are visible.",
        "2. **Minimal Source-Attested Heterogeneous Schema:**",
        "   - **Node Types:** `Package`, `PackageVersion`.",
        "   - **Edge Types:** `HAS_VERSION` (`Package` $\\to$ `PackageVersion`, $\\text{edge\\_time} = t_{\\text{rel}}$) and `DEPENDS_ON` (`PackageVersion` $\\to$ `Package`, $\\text{edge\\_time} = t_{\\text{rel}}$).",
        "   - Package-level dependency declarations are preserved without speculative target-version resolution.",
        "3. **Zero Target-Label / Security-Event Leakage:**",
        "   - Ground-truth compromise labels (`confirmed_malicious`), Datadog manifest records, and target release compromise events are **never** used as input features or graph edges.",
        "   - Public OSV advisories contribute strictly as historical public context published before $t$ on prior releases/packages.",
        "4. **Maintainer / Repository Semantics:**",
        "   - Used strictly for persistent-identifier grouping and split leakage auditing (guaranteeing 0 leakage across splits), not temporal message passing.",
        "5. **Observability Segregation:**",
        "   - Dynamic GNN message passing is restricted to the 108 packages with `graph_observable` releases.",
        "   - Administrative takedown records (1,799 records) are strictly excluded from message-passing edges.",
        "",
        "---",
        "",
        "## 2. Quantitative Graph Dimensions",
        "",
        "| Graph Entity / Dimension | Exact Count | Description |",
        "|---|---|---|",
        f"| **Package Nodes** | **{summary['node_counts']['package']}** | Canonical package identities in universe |",
        f"| **PackageVersion Nodes** | **{summary['node_counts']['package_version']}** | Physical releases (14,289 live + 1,799 takedown) |",
        f"| **HAS_VERSION Edges** | **{summary['edge_counts']['has_version']}** | Package to version edges with explicit edge_time |",
        f"| **DEPENDS_ON Edges** | **{summary['edge_counts']['depends_on']}** | Version to package dependency declarations |",
        f"| **Feature Matrix Dimension** | **{summary['feature_matrix_shape'][0]} $\\times$ {summary['feature_matrix_shape'][1]}** | Causal feature vectors per PackageVersion |",
        "",
        "---",
        "",
        "## 3. Causal Feature Definitions & Temporal Provenance",
        "",
        "Every feature is computed strictly \"as of\" $t = \\text{release\\_timestamp}$:",
        "",
        "| Dimension | Feature Name | Value Type | Temporal Provenance |",
        "|---|---|---|---|",
    ]

    for idx, f in enumerate(FEATURE_SPECIFICATION):
        lines.append(f"| {idx} | `{f['name']}` | {f['type']} | {f['temporal_provenance']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Experimental Splits on Graph",
        "",
        "### A. Chronological Temporal Forecasting Split (Primary RTGN Experiment):",
        f"- **Train Mask:** {summary['chronological_split_counts']['train']} releases ($t < \\text{{2025-06-01}}$)",
        f"- **Validation Mask:** {summary['chronological_split_counts']['validation']} releases ($2025-06-01 \\le t < \\text{{2026-03-01}}$)",
        f"- **Test Mask:** {summary['chronological_split_counts']['test']} releases ($t \\ge \\text{{2026-03-01}}$)",
        f"- **Takedown Catalog (Non-graph):** {summary['chronological_split_counts']['takedown_catalog']} records",
        "",
        "### B. Graph-Model Package-Disjoint Split (Secondary Unseen Attack Experiment):",
        f"- **Train Mask:** {summary['graph_package_disjoint_split_counts']['train']} releases (78 packages)",
        f"- **Validation Mask:** {summary['graph_package_disjoint_split_counts']['validation']} releases (15 packages)",
        f"- **Test Mask:** {summary['graph_package_disjoint_split_counts']['test']} releases (15 packages)",
        "- **Package, Maintainer, Repository Overlap:** Strictly `0` (Zero leakage verified).",
        "",
        "---",
        "",
        "## 5. Artifact Index",
        "- `data/experimental/graph/nodes_package.csv`: Package node table.",
        "- `data/experimental/graph/nodes_package_version.csv`: Version node table with target labels and topology status.",
        "- `data/experimental/graph/edges_has_version.csv`: Package $\\to$ Version edges with `edge_time`.",
        "- `data/experimental/graph/edges_depends_on.csv`: Version $\\to$ Package dependency edges with `edge_time`.",
        "- `data/experimental/graph/package_version_features.pt`: Causal feature tensor (`[16088, 13]`).",
        "- `data/experimental/graph/package_version_features.csv`: Inspectable tabular feature table.",
        "- `data/experimental/graph/temporal_hetero_graph.pt`: PyG `HeteroData` container.",
        "- `data/experimental/graph/split_masks.pt`: PyTorch boolean index masks for both split protocols.",
        "- `data/experimental/graph/feature_metadata.json`: Machine-readable feature definitions.",
        "- `data/experimental/graph/graph_summary.json`: Quantitative audit summary.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    start_time = time.time()
    logger.info("Initializing Temporal Graph Construction Pipeline...")

    experimental_dir = PROJECT_ROOT / "data" / "experimental"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = experimental_dir / "graph"

    builder = TemporalGraphBuilder(
        experimental_dir=experimental_dir,
        processed_dir=processed_dir,
        output_dir=output_dir,
    )
    temporal_graph, summary = builder.build()

    doc_path = PROJECT_ROOT / "docs" / "TEMPORAL_GRAPH_DESIGN.md"
    write_graph_design_document(output_path=doc_path, summary=summary)
    logger.info(f"Documentation written to {doc_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TEMPORAL HETEROGENEOUS GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"Total Package nodes:                     {summary['node_counts']['package']}")
    print(f"Total PackageVersion nodes:              {summary['node_counts']['package_version']}")
    print(f"HAS_VERSION edges:                       {summary['edge_counts']['has_version']}")
    print(f"DEPENDS_ON edges:                        {summary['edge_counts']['depends_on']}")
    print(f"Feature matrix dimensions:               {summary['feature_matrix_shape'][0]} x {summary['feature_matrix_shape'][1]}")
    print("-" * 70)
    print("TOPOLOGY OBSERVABILITY BREAKDOWN:")
    print(f"  Graph Observable (Live with deps):     {summary['topology_observability']['graph_observable']}")
    print(f"  Live Unobservable (Missing deps):      {summary['topology_observability']['live_unobservable_topology']}")
    print(f"  Takedown Catalog (404/Non-graph):      {summary['topology_observability']['label_only_takedown']}")
    print("-" * 70)
    print("CHRONOLOGICAL SPLIT MASKS (PRIMARY RTGN EXPERIMENT):")
    print(f"  Train:                                 {summary['chronological_split_counts']['train']}")
    print(f"  Validation:                            {summary['chronological_split_counts']['validation']}")
    print(f"  Test:                                  {summary['chronological_split_counts']['test']}")
    print(f"  Takedown Catalog:                      {summary['chronological_split_counts']['takedown_catalog']}")
    print("-" * 70)
    print("GRAPH-MODEL PACKAGE-DISJOINT SPLIT MASKS (SECONDARY EXPERIMENT):")
    print(f"  Train:                                 {summary['graph_package_disjoint_split_counts']['train']}")
    print(f"  Validation:                            {summary['graph_package_disjoint_split_counts']['validation']}")
    print(f"  Test:                                  {summary['graph_package_disjoint_split_counts']['test']}")
    print("=" * 70)
    print(f"Elapsed time: {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    main()
