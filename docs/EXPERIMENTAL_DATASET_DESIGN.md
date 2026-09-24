# Software Supply-Chain Experimental Dataset Design & Sampling Audit

> **Generated:** 2026-09-22T19:37:23.546042+00:00  
> **Pipeline Mode:** Dataset Augmentation & Leakage-Safe Splitting  
> **Processed Source Dataset:** `data/processed/` (Preserved 100% Unchanged)  

---

## 1. Executive Summary & Experimental Dimensions

This specification establishes the experimental dataset layer (`data/experimental/`) for research on **temporal release-level compromise detection** in software supply chains.

### Key Methodological Guarantees:
1. **Strict Temporal Task Semantics:** The learning objective is *temporal release-level compromise detection*. `release_timestamp` represents physical upload time, NOT malicious attack onset. `label_scope` is recorded as source provenance, NOT an attack-start timestamp. No attack detection delay is computed for the general dataset.
2. **Candidate-Selection Temporal Semantics:** Historical dependency-target discovery is derived strictly from releases uploaded prior to $T_{\text{train\_val}} = \text{2025-06-01 00:00:00 UTC}$ (Epoch: `1748736000.0`). Temporal candidate exclusion utilizes temporally-attested OSV malware records published before cutoff. The complete Datadog manifest exclusion is maintained as a conservative open-world label guardrail without inventing synthetic disclosure/onset timestamps.
3. **Three Distinct Label Classes:**
   - `confirmed_malicious`: Verified compromised releases from Datadog manifest (1,495 live releases in registry).
   - `source_unflagged`: Uncompromised releases of Datadog packages (NOT treated as confirmed benign; 3,232 releases).
   - `assumed_benign_background`: Sampled background cohort (NOT claimed to be malware-free; 9,562 releases).
   - `label_only_takedown`: Pruned administrative takedowns (1,799 catalog records; isolated from graph message passing).
4. **Strict Graph Observability Definition:**
   - `graph_observable` is applied STRICTLY when `registry_status == "live"` AND `dependency_metadata_status == "observed"` (9,876 releases).
   - Live releases with missing `requires_dist` are classified as `live_unobservable_topology` (4,413 releases).
   - Pruned malware packages (404) are classified as `label_only_takedown` (1,799 records) and excluded from dynamic GNN message passing.
5. **Empirical Package-Balanced Sample Weighting:** Damped square-root weights ($w = 1/\sqrt{N_{p,\text{train}}}$) computed on the actual training split to prevent `upgini` from dominating gradients without discarding empirical samples.
6. **Leak-Free Package-Disjoint Splits:** Packages sharing a repository or maintainer/author are grouped into connected components, guaranteeing 0 package, 0 repository, and 0 maintainer leakage across splits.
7. **Dedicated Graph-Model Split:** A separate package-disjoint split restricted to the 108 packages with observable graph topology, explicitly separating graph-observable and unobservable records.

---

## 2. Quantitative Dataset Summary

| Dimension / Metric | Count / Specification | Methodological Role |
|---|---|---|
| **Total Experimental Records** | **16088** | Combined experimental universe |
| **Live PyPI Records** | **14289** | Versions with live PyPI registry metadata |
| **Takedown Records (404)** | **1799** | Pure malware removed by PyPI admins (non-graph) |
| **Live Confirmed Malicious Versions** | **1495** | Collected live malicious releases |
| **Takedown Malicious Catalog Records** | **1799** | Package-level Datadog takedown entries |
| **Source Unflagged Versions** | **3232** | Pre/post-compromise releases of attacked packages |
| **Assumed Benign Background Versions** | **9562** | Objective background cohort releases |
| **Live with Observed Dependency Topology** | **9876** | `graph_observable` eligible for GNN edges |
| **Live with Missing Dependencies** | **4413** | `live_unobservable_topology` isolated nodes |
| **Sampled Background Packages** | **100 packages** | Stratified across 4 operational strata |

> [!IMPORTANT]
> **Quantitative Distinction:** Malicious evidence is reported as 1,495 live confirmed malicious versions and 1,799 takedown catalog records. They are NOT derived or summed as 3,294 malicious versions because 1,799 represents package-level administrative takedown entries, not collected releases.

---

## 3. Background Cohort Stratification & Audit

### Candidate Selection & Stratification Methodology:
- **Cutoff Timestamp:** `2025-06-01 00:00:00 UTC` (`1748736000.0`).
- **Component B (Supply-Chain Hard-Negatives):** 40 packages drawn from the clean downstream dependency targets ($D_{\text{in}} \ge 1$) of releases uploaded before the cutoff.
- **Component A (General Stratified Background):** 60 packages sampled across the ecosystem.
- **Total Candidates Evaluated:** 100 packages.

### Stratum Distribution & Quota Audit:
| Stratum | Definition & Operational Thresholds | Requested Quota | Pre-Selection Candidates | Realized Selected |
|---|---|---|---|---|
| **Stratum 1: Core Infrastructure** | $N_{\text{rel}} \ge 40$, $\text{Age} \ge 1800\text{d}$, $D_{\text{in}} \ge 50$ | 25 | 42 | **42** |
| **Stratum 2: Active Domain Utilities** | $15 \le N_{\text{rel}} < 100$, $F_{\text{rel}} \ge 1.5\text{/yr}$ | 30 | 47 | **47** |
| **Stratum 3: Stable Low-Velocity Utilities** | $5 \le N_{\text{rel}} < 20$, $\text{Age} \ge 730\text{d}$ | 25 | 10 | **10** |
| **Stratum 4: Nascent / Sparse Leaf Packages** | $1 \le N_{\text{rel}} < 5$ | 20 | 1 | **1** |
| **Total** | | 100 | 100 | **100** |

### Audit Explanation for S1/S2 vs S3/S4 Imbalance:
1. **Empirical Nature of Hard-Negative Dependency Targets:** All 40 hard-negative candidates were selected based on high downstream in-degree ($D_{\text{in}} \ge 1$) prior to June 1, 2025. In the open PyPI ecosystem, foundational libraries that are heavily depended upon (e.g., `requests`, `numpy`, `pandas`, `click`, `torch`, `pydantic`) are universally mature, long-lived, and frequently updated. Consequently, 27 qualified as Stratum 1 and 13 qualified as Stratum 2. None qualified as Stratum 3 or 4.
2. **Empirical Longevity of Leaf Candidates:** Among the 60 general background candidates, packages nominated intuitively for S3/S4 (e.g., `openpyxl`, `simplejson`, `pytz`, `fpdf2`, `smart-open`, `mido`) have existed on PyPI for 5 to 15 years. Over their lifespans, regular maintenance caused their release counts to exceed 15 ($N_{\text{rel}} \ge 15$) and release velocity to exceed 1.5/yr ($F_{\text{rel}} \ge 1.5$/yr). Under objective metrics evaluated strictly as of the cutoff, they classify as Stratum 2 rather than Stratum 4.
3. **Objective Metric Adherence:** The pipeline avoided subjective reclassification, strictly honoring the mathematical thresholds computed as of $T < \text{2025-06-01}$.

---

## 4. Chronological Split & Package Weighting Report

- **Train Boundary:** $t < 2025-06-01$ (`1748736000.0`)  
- **Validation Boundary:** $2025-06-01 \le t < 2026-03-01$ (`1772323200.0`)  
- **Test Boundary:** $t \ge 2026-03-01$ (Untouched Future Holdout)  

### Chronological Split Counts:
- **Train Set:** 11833 releases
- **Validation Set:** 1276 releases
- **Test Set:** 1180 releases
- **Takedown Catalog (Non-graph):** 1799 records

### Upgini Concentration Mitigation Report:
- **Positive Training Samples:** 1218
- **Unique Positive Packages in Train:** 18
- **Upgini Training Count:** 1059
- **Upgini Raw Damped Weight:** 0.0307
- **Upgini Normalized Weight:** 0.5131
- **Upgini Positive Mass Before Weighting:** 86.95%
- **Upgini Positive Mass After Weighting:** 44.61%

#### Per-Package Weight Breakdown (Top 10):
| Package | Train Count | Raw Weight | Norm Weight | Mass Before (%) | Mass After (%) |
|---|---|---|---|---|---|
| `upgini` | 1059 | 0.0307 | 0.5131 | 86.95% | 44.61% |
| `convocations` | 71 | 0.1187 | 1.9816 | 5.83% | 11.55% |
| `fastapi-toolkit` | 23 | 0.2085 | 3.4817 | 1.89% | 6.57% |
| `liblxml3` | 16 | 0.25 | 4.1744 | 1.31% | 5.48% |
| `mirrorbot` | 12 | 0.2887 | 4.8201 | 0.99% | 4.75% |
| `spyware` | 9 | 0.3333 | 5.0 | 0.74% | 3.69% |
| `reverseshell` | 8 | 0.3536 | 5.0 | 0.66% | 3.28% |
| `poppo213` | 7 | 0.378 | 5.0 | 0.57% | 2.87% |
| `definitely-not-requests` | 2 | 0.7071 | 5.0 | 0.16% | 0.82% |
| `mastowrapper` | 2 | 0.7071 | 5.0 | 0.16% | 0.82% |

---

## 5. Strict Package-Disjoint Split & Identifier Leakage Audit

To evaluate **zero-overlap unseen attack compromise detection**, packages are partitioned into train, validation, and test splits with persistent-identifier grouping.

### Leakage Prevention Methodology:
- **Connected Component Grouping:** Packages sharing a common repository (e.g., `github:psf/requests` shared by `requests` and `definitely-not-requests`), common maintainer/author name, or maintainer/author email are grouped into the same split.
- **Partition Proportions:** Preserves approximate 70% Train / 15% Validation / 15% Test proportions.

### Audit Verification Results:
- **Strict Package Disjointness ($P_{\text{train}} \cap P_{\text{val}} \cap P_{\text{test}} = \emptyset$):** `PASSED` (Overlap: 0)
- **Repository Overlap Across Splits:** `PASSED` (0 shared repositories)
- **Maintainer/Author Overlap Across Splits:** `PASSED` (0 shared maintainers)
- **Train Packages:** 1353 (69.99%)
- **Validation Packages:** 290 (15.00%)
- **Test Packages:** 290 (15.00%)
- **Total Connected Components:** 1913 (Multi-package groups: 12)

---

## 6. Graph-Model Package-Disjoint Split (RTGN Universe)

For graph neural network models (e.g., RTGN, GCN, GraphSAGE), message passing requires observable dependency topology. This dedicated split is **strictly restricted to the 108 packages with at least one `graph_observable` release**, completely excluding label-only takedowns.

### Graph-Model Split Record Counts:
| Split | Packages | Total Releases | Graph-Observable | Live Unobservable | Confirmed Malicious |
|---|---|---|---|---|---|
| **Train** | 78 | 10282 | 7750 | 2532 | 1435 |
| **Validation** | 15 | 1865 | 1177 | 688 | 9 |
| **Test** | 15 | 1239 | 949 | 290 | 7 |

- **Graph Split Disjointness Audit:** `PASSED` (0 shared packages, 0 shared repos, 0 shared maintainers).
- **Takedown Packages in Graph Split:** `0` (strictly excluded).

---

## 7. Artifact Index
- `data/experimental/background_packages.json`: 100 background packages with objective stratification metrics.
- `data/experimental/experimental_versions.csv`: Augmented version table with 3 labels, observability flags, and empirical sample weights.
- `data/experimental/splits/chronological_split.json`: Chronological train/val/test splits.
- `data/experimental/splits/package_disjoint_split.json`: Strict leak-free unseen attack train/val/test splits (Full dataset).
- `data/experimental/splits/graph_package_disjoint_split.json`: Graph-model package-disjoint split restricted to observable topology.
