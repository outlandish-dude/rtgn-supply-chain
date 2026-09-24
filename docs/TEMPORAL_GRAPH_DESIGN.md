# Temporal Heterogeneous Graph Specification & Causal Design

> **Generated:** 2026-09-22T20:06:52.817323+00:00  
> **Pipeline Mode:** Dynamic Temporal Graph Construction & Causal Snapshot Interface  
> **Source Dataset:** `data/experimental/experimental_versions.csv`  

---

## 1. Executive Summary & Graph Schema

This specification establishes the temporal heterogeneous graph layer (`data/experimental/graph/`) for the Real-Time Temporal Graph Neural Network (RTGN).

### Key Methodological & Causality Guarantees:
1. **Causal Snapshot Interface (`snapshot(t)`):** Graph states at timestamp $t$ expose strictly edges where $\text{edge\_time} \le t$. Zero edges from future releases are visible.
2. **Minimal Source-Attested Heterogeneous Schema:**
   - **Node Types:** `Package`, `PackageVersion`.
   - **Edge Types:** `HAS_VERSION` (`Package` $\to$ `PackageVersion`, $\text{edge\_time} = t_{\text{rel}}$) and `DEPENDS_ON` (`PackageVersion` $\to$ `Package`, $\text{edge\_time} = t_{\text{rel}}$).
   - Package-level dependency declarations are preserved without speculative target-version resolution.
3. **Zero Target-Label / Security-Event Leakage:**
   - Ground-truth compromise labels (`confirmed_malicious`), Datadog manifest records, and target release compromise events are **never** used as input features or graph edges.
   - Public OSV advisories contribute strictly as historical public context published before $t$ on prior releases/packages.
4. **Maintainer / Repository Semantics:**
   - Used strictly for persistent-identifier grouping and split leakage auditing (guaranteeing 0 leakage across splits), not temporal message passing.
5. **Observability Segregation:**
   - Dynamic GNN message passing is restricted to the 108 packages with `graph_observable` releases.
   - Administrative takedown records (1,799 records) are strictly excluded from message-passing edges.

---

## 2. Quantitative Graph Dimensions

| Graph Entity / Dimension | Exact Count | Description |
|---|---|---|
| **Package Nodes** | **1933** | Canonical package identities in universe |
| **PackageVersion Nodes** | **16088** | Physical releases (14,289 live + 1,799 takedown) |
| **HAS_VERSION Edges** | **14289** | Package to version edges with explicit edge_time |
| **DEPENDS_ON Edges** | **71677** | Version to package dependency declarations |
| **Feature Matrix Dimension** | **16088 $\times$ 13** | Causal feature vectors per PackageVersion |

---

## 3. Causal Feature Definitions & Temporal Provenance

Every feature is computed strictly "as of" $t = \text{release\_timestamp}$:

| Dimension | Feature Name | Value Type | Temporal Provenance |
|---|---|---|---|
| 0 | `version_major` | integer | Available at upload timestamp t (intrinsic release metadata). |
| 1 | `version_minor` | integer | Available at upload timestamp t (intrinsic release metadata). |
| 2 | `version_patch` | integer | Available at upload timestamp t (intrinsic release metadata). |
| 3 | `is_prerelease` | binary (0/1) | Available at upload timestamp t (intrinsic release metadata). |
| 4 | `has_requires_python` | binary (0/1) | Available at upload timestamp t (intrinsic release metadata). |
| 5 | `requires_python_len` | integer | Available at upload timestamp t (intrinsic release metadata). |
| 6 | `declared_dependency_count` | integer | Available at upload timestamp t (release-specific dependency declaration). |
| 7 | `hist_release_count_before_t` | integer | Computed strictly from historical releases where release_timestamp < t. |
| 8 | `hist_age_days_at_t` | float | Computed strictly from historical releases where release_timestamp < t. |
| 9 | `time_since_prev_release_days` | float | Computed strictly from historical releases where release_timestamp < t. |
| 10 | `hist_release_frequency` | float | Computed strictly from historical releases where release_timestamp < t. |
| 11 | `dynamic_downstream_in_degree` | integer | Computed strictly from dependency edges where edge_time < t. |
| 12 | `hist_advisories_count_before_t` | integer | Computed strictly from OSV records where published_timestamp < t and affected_version != target_version. |

---

## 4. Benchmark Tiers & Experimental Splits on Graph

### Tier A: Chronological Temporal Forecasting with Package-Overlapping Cohorts
- **Train Mask:** 11,833 releases ($t < \text{2025-06-01}$); 1,218 confirmed malicious (1,209 graph-observable)
- **Validation Mask:** 1,276 releases ($2025-06-01 \le t < \text{2026-03-01}$); 151 confirmed malicious (8 graph-observable)
- **Test Mask:** 1,180 releases ($t \ge \text{2026-03-01}$); **126 confirmed malicious** (**98 graph-observable**)
- **Takedown Catalog (Non-graph catalog records):** 1,799 records

### Tier B: Temporal Forecasting Excluding Upgini (`eval_excluding_upgini`)
- Evaluates the Tier A test fold with the dominant package `upgini` removed.
- **Test Size:** 509 binary releases (481 assumed benign, **28 confirmed malicious**; 0 graph-observable).

### Tier C: Package/Maintainer/Repository-Disjoint Unseen-Package Generalization
*(Deliberately redesigned so the test partition contains multiple unseen malicious packages/releases. Do not call this a temporal forecasting benchmark unless an explicit chronological constraint is also applied).*
- **Train Partition (87 packages):** 9,739 releases; 1,297 confirmed malicious (**1,278 graph-observable** across 7 malicious packages: `upgini`, `definitely-not-requests`, `a-b27`, `botocote`, `grandslam`, `secretInspector`, `security-util`), 6,672 assumed benign, 1,770 source unflagged.
- **Validation Partition (24 packages):** 1,895 releases; 68 confirmed malicious (**7 graph-observable** across 6 malicious packages: `fastapi-toolkit`, `liblxml3`, `mirrorbot`, `ReverseShell`, `MastoWrapper`, `pymatcha`), 1,609 assumed benign, 218 source unflagged.
- **Test Partition (24 packages):** 2,655 releases; **130 confirmed malicious** across 6 unseen malicious packages:
  - **30 of those 130 releases are `graph_observable`** and eligible for dynamic RTGN message passing (`convocations`: 22, `poppo213`: 7, `SpyWare`: 1).
  - The remaining 100 malicious records are non-graph-observable/missing dependency metadata (`convocations`: 78, `pip-doom`: 11, `SpyWare`: 8, `taya`: 2, `wordsdict`: 1) and must **NOT** be counted as graph-message-passing positives.
  - 1,281 assumed benign releases (764 graph-observable), 1,244 source unflagged releases.
- **Leakage Audit:** Strictly `0` package, `0` maintainer, and `0` repository overlap across Train, Val, and Test.

### Tier D: Unflagged Sibling Diagnostic with No Binary Ground Truth
- Evaluates model predictions on the **3,232 surviving live releases** from the 17 packages with specific-version compromises (`label_scope == specific_version`, e.g. `litellm`, `ultralytics`, `lightning`, `durabletask`).
- 29 malicious specific_version records exist in the Datadog provenance, but 0 are currently live graph-observable releases (yanked prior to ingestion). Therefore, `specific_version` is **not** a primary supervised RTGN benchmark.
- Surviving sibling releases remain a separate `source_unflagged` diagnostic cohort with null binary ground truth.

---

## 5. Artifact Index
- `data/experimental/graph/nodes_package.csv`: Package node table.
- `data/experimental/graph/nodes_package_version.csv`: Version node table with target labels and topology status.
- `data/experimental/graph/edges_has_version.csv`: Package $\to$ Version edges with `edge_time`.
- `data/experimental/graph/edges_depends_on.csv`: Version $\to$ Package dependency edges with `edge_time`.
- `data/experimental/graph/package_version_features.pt`: Causal feature tensor (`[16088, 13]`).
- `data/experimental/graph/package_version_features.csv`: Inspectable tabular feature table.
- `data/experimental/graph/temporal_hetero_graph.pt`: PyG `HeteroData` container.
- `data/experimental/graph/split_masks.pt`: PyTorch boolean index masks for both split protocols.
- `data/experimental/graph/feature_metadata.json`: Machine-readable feature definitions.
- `data/experimental/graph/graph_summary.json`: Quantitative audit summary.
