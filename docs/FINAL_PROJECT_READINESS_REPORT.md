# Final Project Readiness & Verification Report

**Project Title:** Temporal Heterogeneous Graph Learning for Early Detection and Propagation-Aware Analysis of Software Supply-Chain Compromise  
**Course:** Deep Learning Project (5th Sem)  
**Evaluation Target:** Faculty Demonstration & Formal Defense  
**Date:** 2026-10-30  
**Status:** Complete, Verified, and Demonstration-Ready  

---

## Executive Summary

The research implementation, model verification, baseline benchmarking, ablation framework, and presentation tooling for the **Relational Temporal Graph Network (RTGN)** software supply-chain compromise detection system are complete.

The Tier C (unseen package inductive split) model checkpoint has been fully recovered, verified at the tensor level against the architecture definition, and evaluated. All strict protection rules were maintained: no model weights were retrained or modified, no graph splits or feature schemas were altered, and inference on benchmark packages was confirmed to be bit-identical.

---

## 1. Improvements Completed

### A. Faculty Demonstration Script (`scripts/demo_rtgn.py`)
- **Dynamic Feature Introspection:** Replaced outdated hardcoded feature names with the authentic 13-feature schema loaded directly from `data/experimental/graph/feature_metadata.json` (PEP 440 semantic version parsing, intrinsic constraints, release cadence deltas).
- **Graph Topology & Dataset Statistics:** Upgraded display to output exact heterogeneous node counts ($16,088$ PackageVersions, $1,805$ Packages) and relation counts ($13,858$ `version_of`, $16,088$ `has_version`, $3,409$ `depends_on`, $3,409$ `depended_by`).
- **Supervision & Label Segregation:** Explicitly formatted releases across three strict semantic categories:
  - Confirmed Malicious ($y=1$): Verified compromise events from Datadog security research.
  - Assumed-Benign ($y=0$): Rigorously stratified background packages.
  - Supervision-Excluded ($y=-1$): Unflagged sibling releases from compromised packages, prevented from leaking label noise into binary supervision.
- **Checkpoint Metadata & Calibration:** Script now dynamically retrieves and displays the saved checkpoint epoch (Epoch 4), validation metrics (AUPRC $0.5905$, AUROC $0.9164$, F1 $0.7360$), and frozen validation threshold ($\tau^* = 0.1882$).
- **Prediction Explanation & Topological Evidence (Section 7):** Added an explanation section for every release inspected:
  - Package name and version.
  - Raw logit and calibrated maliciousness probability.
  - Frozen decision threshold and binary alert classification.
  - SemVer jump classification (Major, Minor, Patch, Initial/Pre-release).
  - Historical release cadence and inter-release time delta.
  - Dependency graph context: direct upstream dependencies and downstream dependents.
  - Strict empirical disclaimer distinguishing observed graph/metadata evidence from model probabilistic output.

### B. Evaluation Visualizations (`scripts/generate_evaluation_figures.py`)
Generated 5 publication-ready, high-resolution figures in `experiments/final_figures/`:
1. `confusion_matrix.png`: Normalized and raw confusion matrix for RTGN on the Tier C test split ($N=1,411$).
2. `baseline_comparison.png`: Grouped comparative bar chart of RTGN vs. Logistic Regression, Random Forest, XGBoost, and Prevalence baseline across AUPRC, AUROC, and F1.
3. `ablation_comparison.png`: Architectural ablation and memory comparison with explicit status distinction (Full RTGN converged benchmark, Cold-Start vs. Online-Adaptation dynamic memory, and unconverged 1-epoch smoke tests).
4. `precision_recall_curve.png`: Comparative PR curves across models, marking the random background prevalence ($9.21\%$).
5. `roc_curve.png`: Comparative ROC curves with random-guess diagonal.

### C. Error Analysis Report (`docs/ERROR_ANALYSIS.md`)
Authored an evidence-based report detailing:
- Breakdown of all 22 false positives and 109 false negatives on the Tier C test set at frozen threshold $\tau^* = 0.1882$ ($TP=21, FP=22, FN=109, TN=1259 \implies \text{Precision} = 0.4884 \text{ [48.84\%]}, \text{Recall} = 0.1615 \text{ [16.15\%]}, F_1 = 0.2428$).
- Explicit distinction of the alternative high-sensitivity exploratory operating point: $TP=59, FP=298, FN=71 \implies \text{Precision} = 59/357 \approx 0.1653 \text{ [16.53\%]}, \text{Recall} = 59/130 \approx 0.4538 \text{ [45.38\%]}, F_1 = 118/487 \approx 0.2423$.
- Package-level detection patterns: Perfect compromise isolation on multi-release attacks (`pip-doom`, `poppo213`, `taya`).
- Forensic case study of the `convocations 0.1` false negative (an isolated, zero-dependency initial upload containing an obfuscated remote payload).
- Explicit limitations of metadata-only and graph-topological detection without dynamic sandboxing or AST payload analysis.

### D. Documentation Overhaul (`README.md` & `docs/`)
- Unified repository architecture, dataset statistics, feature metadata, benchmark tables, and running instructions in `README.md`.
- Synchronized `docs/TIER_C_VERIFICATION_REPORT.md` and `docs/TIER_C_FORENSIC_AUDIT.md`.
- Explicitly documented Tier A local test incompleteness, the absence of multi-seed variance runs, and bounded scope disclaimers.

---

## 2. Files Created and Modified

| File | Status | Description |
| :--- | :---: | :--- |
| `scripts/demo_rtgn.py` | **Modified** | Polished faculty demo with dynamic metadata, topological context, and prediction explanations. |
| `scripts/generate_evaluation_figures.py` | **Created** | Script generating the 5 evaluation figures from authentic benchmark data. |
| `scripts/export_tier_c_eval_artifacts.py` | **Created** | Standalone script to compute and persist release-level test set predictions. |
| `docs/ERROR_ANALYSIS.md` | **Created** | Comprehensive false positive, false negative, and architectural boundary analysis. |
| `docs/FINAL_PROJECT_READINESS_REPORT.md` | **Created** | Final project presentation and faculty readiness audit. |
| `docs/TIER_C_VERIFICATION_REPORT.md` | **Modified** | Updated audit status confirming local checkpoint recovery and state-dict verification. |
| `README.md` | **Created** | Main repository documentation, architecture overview, and evaluation commands. |
| `experiments/final_figures/*.png` | **Generated** | 5 publication figures for presentation slides and faculty evaluation. |

---

## 3. Tests Executed and Results

The full test suite was executed via pytest:
```bash
pytest -v tests/test_rtgn_architecture.py tests/test_rtgn_ablations.py tests/test_baseline_models.py
```

### Results Summary
- **Total Tests:** 21
- **Passed:** 21 (100%)
- **Failed:** 0
- **Execution Time:** 43.02 seconds

### Breakdown by Module:
1. `tests/test_rtgn_architecture.py` (7 tests):
   - `test_no_static_package_id_embedding`: **PASSED** (Confirms inductive capability; no package ID lookup tables).
   - `test_time_encoding_units_and_dimensions`: **PASSED** (Validates Bochner Fourier temporal projections).
   - `test_tier_c_cold_start_vs_online_adaptation`: **PASSED** (Validates package GRU memory behavior).
   - `test_hetero_gatv2_message_passing`: **PASSED** (Validates edge-typed heterogeneous attention).
   - `test_package_balanced_loss_calculation`: **PASSED** (Validates inverse frequency package weighting).
   - `test_rtgn_full_forward_and_backward`: **PASSED** (Validates forward and backward graph pass).
   - `test_source_unflagged_excluded_from_supervised_loss`: **PASSED** (Zero loss contribution from $y=-1$).

2. `tests/test_rtgn_ablations.py` (8 tests):
   - `test_no_temporal_encoding_configuration`: **PASSED**
   - `test_no_graph_configuration`: **PASSED**
   - `test_no_historical_features_configuration`: **PASSED**
   - `test_no_package_balancing_weights`: **PASSED**
   - Ablation single-step training across all 4 variants: **PASSED**

3. `tests/test_baseline_models.py` (6 tests):
   - `test_source_unflagged_segregation`: **PASSED**
   - `test_no_future_split_leakage`: **PASSED**
   - `test_no_test_labels_influence_training`: **PASSED**
   - `test_threshold_selection_strictly_validation`: **PASSED**
   - `test_prevalence_baseline`: **PASSED**
   - `test_eval_excluding_upgini`: **PASSED**

---

## 4. Checkpoint Integrity Confirmation

The recovered checkpoint at `experiments/rtgn_tier_c/checkpoints/best_rtgn_model.pt` was verified using `scripts/verify_rtgn_checkpoint.py`:

- **File Path:** `experiments/rtgn_tier_c/checkpoints/best_rtgn_model.pt`
- **File Size on Disk:** **1,752,093 bytes** ($1.67\text{ MB}$, containing AdamW optimizer moments and metadata; raw parameter weights = $517,380\text{ bytes} \approx 505\text{ KB}$ for $129,345$ float32 weights)
- **Checkpoint Epoch:** Epoch 4
- **Parameters Verified:** $129,345$ learnable parameters across $87$ parameter tensors
- **Buffers Verified:** $3$ persistent tensors:
  - `time_encoder.frequencies`: $[16]$
  - `package_memory.memory`: $[1805, 64]$
  - `package_memory.update_count`: $[1805]$
- **Total Tensors Loaded:** $90 / 90$ ($100\%$ matched)
- **Missing Keys:** $0$
- **Unexpected Keys:** $0$
- **Shape Mismatches:** $0$
- **Validation Metrics Stored in Checkpoint:**
  - AUPRC: $0.5905$
  - AUROC: $0.9164$
  - F1 Score: $0.7360$
  - Optimal Threshold $\tau^*$: $0.1882$

---

## 5. Before / After Prediction Comparison

Inference predictions were evaluated on the three benchmark packages specified in the protection rules:

| Package Release | Label | Pre-Polishing Prediction | Post-Polishing Prediction | Logit | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`pip-doom 0.1.0`** | Confirmed Malicious ($y=1$) | $P = 0.9921$ ($99.21\%$) | $P = 0.9921$ ($99.21\%$) | $+4.8286$ | **ALERT [MALICIOUS]** |
| **`torchvision 0.1.6`** | Assumed Benign ($y=0$) | $P = 0.0050$ ($0.50\%$) | $P = 0.0050$ ($0.50\%$) | $-5.2915$ | **CLEAN [BENIGN]** |
| **`convocations 0.1`** | Confirmed Malicious ($y=1$) | $P = 0.000008$ ($0.00\%$) | $P = 0.000008$ ($0.00\%$) | $-11.7376$ | **CLEAN [FALSE NEGATIVE]** |

**Conclusion:** Predictions are identical to 6 decimal places. All model logits and probabilities were preserved with zero drift.

---

## 6. Remaining Limitations

1. **Metadata-Only Detection Frontier:**
   - RTGN operates strictly on PyPI release metadata, SemVer jump characteristics, release cadence deltas, and dependency topology.
   - It cannot inspect package tarball contents, abstract syntax trees (ASTs), obfuscated bytecodes, or network calls.
   - Initial solo releases (`version=0.1`) that introduce malicious payloads without unusual SemVer anomalies or dependency graph connections (e.g. `convocations 0.1`) are undetectable via metadata alone.

2. **Absence of 5-Seed Variance Experiments:**
   - The reported benchmark results represent a single verified run (Seed 42).
   - Due to Colab GPU compute constraints and local environment limitations, 5-seed statistical variance bounds (mean $\pm$ standard deviation) were not completed.

3. **Tier A Local Evaluation Status:**
   - The Tier A (chronological split) was originally trained on Google Colab.
   - The primary model recovered and locally audited is the Tier C (unseen package inductive split) model. Local re-execution of Tier A has not been performed.

4. **Non-Predictive Attack Graph Traversal:**
   - Heterogeneous graph attention aggregates information across historical releases and dependencies, but does not simulate proactive multistep lateral movement or dynamic attack path simulation.

---

## 7. Exact Commands for Faculty Demonstration

### Command 1: Fast Comprehensive Checkpoint Verification
Verifies model architecture, tensor keys, parameter count, and causal test event inference:
```powershell
python scripts/verify_rtgn_checkpoint.py --checkpoint experiments/rtgn_tier_c/checkpoints/best_rtgn_model.pt
```

### Command 2: Malicious Package Demonstration (`pip-doom`)
Demonstrates detection of a zero-day malicious upload with high SemVer anomalies:
```powershell
python scripts/demo_rtgn.py --tier tier_c --package pip-doom
```
*Expected output: Probability $99.21\%$, Classification: ALERT [MALICIOUS].*

### Command 3: Benign Package Demonstration (`torchvision`)
Demonstrates clean classification of an established legitimate library release:
```powershell
python scripts/demo_rtgn.py --tier tier_c --package torchvision
```
*Expected output: Probability $0.50\%$, Classification: CLEAN [BENIGN].*

### Command 4: False Negative Boundary Case Demonstration (`convocations`)
Demonstrates rigorous academic transparency on an isolated initial-release compromise:
```powershell
python scripts/demo_rtgn.py --tier tier_c --package convocations
```
*Expected output: Probability $0.00\%$, Classification: CLEAN [BENIGN] (False Negative).*

### Command 5: Run Full Pytest Verification Suite
Runs the 21 architectural, ablation, and baseline unit tests:
```powershell
pytest -v tests/test_rtgn_architecture.py tests/test_rtgn_ablations.py tests/test_baseline_models.py
```

### Command 6: Generate Evaluation Figures
Generates the 5 publication charts in `experiments/final_figures/`:
```powershell
python scripts/generate_evaluation_figures.py
```

---

## 8. Explicit Declaration of Incomplete Work

To maintain strict academic integrity:
1. **Tier A Local Test Incompleteness:** The local repository contains the recovered Tier C checkpoint (`experiments/rtgn_tier_c/checkpoints/best_rtgn_model.pt`). A separate Tier A local checkpoint was not re-evaluated locally.
2. **5-Seed Experiments:** A 5-seed Monte Carlo replication study was planned in early proposals but was not executed due to resource constraints. All figures and tables accurately cite single-seed (Seed 42) verified metrics.
3. **Real-time PyPI Streaming / Production Deployment:** The pipeline operates on historical Datadog/PyPI experimental snapshots and is not deployed as a live daemon service.
