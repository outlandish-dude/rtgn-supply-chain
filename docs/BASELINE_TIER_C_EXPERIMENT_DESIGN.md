# Tier C Baseline Modeling Specification & Experimental Design

> **Pipeline Stage:** Classical & Feature Tabular Baselines (Tier C Unseen Packages)  
> **Task:** Unseen Package / Disjoint Partition Compromise Detection  
> **Input Matrix:** `data/experimental/graph/package_version_features.pt` ($16,088 \times 13$)  

---

## 1. Tier C Experimental Setup

### A. Binary Label Semantics
- **Positive Class ($y=1$):** `confirmed_malicious` releases (Datadog verified malicious uploads).
- **Negative Class ($y=0$):** `assumed_benign_background` releases (rigorously stratified background packages).
- **`source_unflagged` (Non-Benign Sibling Releases):**
  - These releases belong to packages with identified compromised versions, but were not flagged.
  - They are **strictly excluded** from binary training and formal evaluation ($y \in \{0, 1\}$).
  - `true_label_binary` remains null/empty.

### B. Exact Binary Split Sample Counts

| Split | Partition Type | Confirmed Malicious ($y=1$) | Assumed Benign ($y=0$) | Total Supervised Binary Samples | Ignored Sibling ($y=-1$) |
|---|---|---|---|---|---|
| **Train** | `train_mask_graph_disjoint` | **1,297** | **6,672** | **7,969** | **1,770** |
| **Validation** | `val_mask_graph_disjoint` | **68** | **1,609** | **1,677** | **218** |
| **Test** | `test_mask_graph_disjoint` | **130** | **1,281** | **1,411** | **1,244** |

---

## 2. Model Architectures & Imbalance Protocols

1. **Prevalence Baseline:** Constant positive prevalence of Tier C training data.
2. **Logistic Regression:** Train-fitted StandardScaler + LogisticRegression(class_weight='balanced').
3. **Random Forest:** RandomForestClassifier(class_weight='balanced', n_estimators=300, max_depth=12).
4. **XGBoost:** XGBClassifier(scale_pos_weight = N_neg / N_pos = 6672 / 1297 = 5.14).

---

## 3. Frozen Validation Thresholds

- **prevalence:** $\tau^* = 0.01$
- **logistic_regression:** $\tau^* = 0.0188$
- **random_forest:** $\tau^* = 0.1469$
- **xgboost:** $\tau^* = 0.0097$

---

## 5. Persistent Artifacts Index
- `BASELINE_TIER_C_EXPERIMENT_DESIGN/models/`: Serialized trained models (`.joblib`, `.json`).
- `BASELINE_TIER_C_EXPERIMENT_DESIGN/predictions/binary_predictions_{model}.csv`: Scored binary predictions across splits.
- `BASELINE_TIER_C_EXPERIMENT_DESIGN/predictions/unflagged_qualitative_predictions_{model}.csv`: Qualitative predictions on unflagged sibling releases.
- `BASELINE_TIER_C_EXPERIMENT_DESIGN/metrics/baseline_metrics.json`: Detailed metric payload.
- `BASELINE_TIER_C_EXPERIMENT_DESIGN/metrics/baseline_metrics_summary.csv`: Summary metrics comparison table.
- `BASELINE_TIER_C_EXPERIMENT_DESIGN/metrics/validation_thresholds.json`: Validation-selected thresholds.
