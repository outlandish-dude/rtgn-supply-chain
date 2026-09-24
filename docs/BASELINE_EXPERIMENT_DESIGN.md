# Baseline Modeling Specification & Experimental Design

> **Pipeline Stage:** Classical & Feature Tabular Baselines  
> **Task:** Temporal Release-Level Compromise Detection  
> **Input Matrix:** `data/experimental/graph/package_version_features.pt` ($16,088 \times 13$)  

---

## 1. Primary Experimental Setup

### A. Binary Label Semantics
- **Positive Class ($y=1$):** `confirmed_malicious` releases (Datadog verified malicious uploads).
- **Negative Class ($y=0$):** `assumed_benign_background` releases (rigorously stratified background packages).
- **`source_unflagged` (Non-Benign Sibling Releases):**
  - These releases belong to packages with identified compromised versions, but were not flagged.
  - They are **strictly excluded** from binary training and formal evaluation ($y \in \{0, 1\}$).
  - `true_label_binary` remains null/empty.
  - Model predictions are generated separately for qualitative/sensitivity analysis and never conflated with scored test metrics.

### B. Evaluation Benchmark Tiers

1. **Tier A: Chronological Temporal Forecasting with Package-Overlapping Cohorts**
   - Evaluates full chronological test set ($t \ge \text{2026-03-01}$).
   - **Test Size:** 607 binary releases (126 confirmed malicious, 481 assumed benign).
   - **Graph-Observable Malicious Positives:** 98 releases.

2. **Tier B: Temporal Forecasting Excluding Upgini (`eval_excluding_upgini`)**
   - Evaluates the Tier A test fold with dominant package `upgini` removed.
   - **Test Size:** 509 binary releases (28 confirmed malicious, 481 assumed benign).
   - **Graph-Observable Malicious Positives:** 0 releases.

3. **Tier C: Package/Maintainer/Repository-Disjoint Unseen-Package Generalization**
   - Evaluates on 24 completely unseen packages (6 malicious, 15 benign, 3 unflagged).
   - *(Do not describe as temporal forecasting unless an explicit chronological constraint is also applied).*
   - **Test Malicious Positives:** **130 confirmed-malicious releases** across 6 unseen malicious packages (`convocations`, `pip-doom`, `poppo213`, `SpyWare`, `taya`, `wordsdict`).
   - **Graph-Observable Positives:** **30 of those 130 releases** are `graph_observable` and eligible for dynamic RTGN message passing (`convocations`: 22, `poppo213`: 7, `SpyWare`: 1).
   - The remaining 100 malicious records are non-graph-observable / missing dependency metadata and must **NOT** be counted as graph-message-passing positives.
   - **Test Assumed Benign Negatives:** 1,281 releases (764 graph-observable).

4. **Tier D: Unflagged Sibling Diagnostic with No Binary Ground Truth**
   - Evaluates model predictions on the **3,232 surviving live releases** from the 17 packages with specific-version compromises (`label_scope == specific_version`, e.g. `litellm`, `ultralytics`, `lightning`, `durabletask`).
   - 29 malicious specific_version records exist in the Datadog provenance, but 0 are currently live graph-observable releases (yanked prior to ingestion). Therefore, `specific_version` is **not** a primary supervised RTGN benchmark.
   - Surviving sibling releases remain a separate `source_unflagged` diagnostic cohort with null binary ground truth.
- **Test Protocol:**
  - Evaluated on **Full Test Set** (607 binary releases).
  - Evaluated on **Test Set Excluding Upgini** (`eval_excluding_upgini`) to audit generalizability across non-dominant packages.
  - Test set remains completely unweighted and unmodified.

---

## 4. Frozen Validation Thresholds

- **prevalence:** $\tau^* = 0.01$
- **logistic_regression:** $\tau^* = 0.01$
- **random_forest:** $\tau^* = 0.0397$
- **xgboost:** $\tau^* = 0.01$

---

## 5. Persistent Artifacts Index
- `experiments/baselines/models/`: Serialized trained models (`.joblib`, `.json`).
- `experiments/baselines/predictions/binary_predictions_{model}.csv`: Scored binary predictions across splits.
- `experiments/baselines/predictions/unflagged_qualitative_predictions_{model}.csv`: Qualitative predictions on unflagged sibling releases.
- `experiments/baselines/metrics/baseline_metrics.json`: Detailed metric payload.
- `experiments/baselines/metrics/baseline_metrics_summary.csv`: Summary metrics comparison table.
- `experiments/baselines/metrics/validation_thresholds.json`: Validation-selected thresholds.
