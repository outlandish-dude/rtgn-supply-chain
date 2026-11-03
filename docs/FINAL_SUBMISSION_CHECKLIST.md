# Final Project Submission & Consistency Checklist

**Project Title:** Temporal Heterogeneous Graph Learning for Early Detection and Propagation-Aware Analysis of Software Supply-Chain Compromise  
**Course:** Deep Learning (5th Semester)  
**Submission Date:** 2026-10-30  
**Verification Status:** Fully Verified, Corrected, and Locked for Defense  

---

## 1. Summary of Completed Corrections

### A. Authoritative Tier C Operating Point (Frozen Validation Threshold $\tau^* = 0.1882$)
- **Validation-Frozen Threshold:** $\tau^* = \mathbf{0.1882}$
- **Test Confusion Matrix ($N = 1,411$):**
  - $\text{True Positives (TP)} = 21$
  - $\text{False Positives (FP)} = 22$ (Specificity = $98.28\%$, False Alarm Rate = $1.72\%$)
  - $\text{False Negatives (FN)} = 109$
  - $\text{True Negatives (TN)} = 1,259$
- **Test Precision:** $\frac{21}{21 + 22} = \frac{21}{43} \approx \mathbf{0.4884}$ ($48.84\%$)
- **Test Recall:** $\frac{21}{21 + 109} = \frac{21}{130} \approx \mathbf{0.1615}$ ($16.15\%$)
- **Test $F_1$ Score:** $\frac{2 \times 21}{2 \times 21 + 22 + 109} = \frac{42}{173} \approx \mathbf{0.2428}$
- **Continuous Ranking Metrics:** Test AUPRC = **$0.3067$**, Test AUROC = **$0.8144$**

### B. High-Sensitivity Exploratory Operating Point (Clearly Segregated)
- **Regime:** Low-threshold exploratory evaluation ($\tau \approx 0.000006$) from earlier Colab runs
- **Confusion Counts:** $TP = 59, FP = 298, FN = 71, TN = 983$
- **Precision:** $\frac{59}{59 + 298} = \frac{59}{357} \approx \mathbf{0.1653}$ ($16.53\%$)
- **Recall:** $\frac{59}{59 + 71} = \frac{59}{130} \approx \mathbf{0.4538}$ ($45.38\%$)
- **$F_1$ Score:** $\frac{2 \times 59}{2 \times 59 + 298 + 71} = \frac{118}{487} \approx \mathbf{0.2423}$
- **Documentation Status:** Formally documented in [docs/ERROR_ANALYSIS.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/docs/ERROR_ANALYSIS.md) and [docs/TIER_C_VERIFICATION_REPORT.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/docs/TIER_C_VERIFICATION_REPORT.md) as a separate SOC trade-off point, explicitly distinguishing it from the frozen threshold benchmark.

### C. Ablation & Evidence Status Audit
- **Full RTGN Benchmark:** Fully trained, converged Colab GPU model (5 epochs, best epoch 4).
- **Dynamic Memory Lifecycle:** Empirically verified on the recovered checkpoint:
  - Cold-Start AUPRC: **$0.5272$** ($N = 21$ initial releases; $6$ pos, $15$ neg)
  - Online-Adaptation AUPRC: **$0.3029$** ($N = 1,390$ sequential releases; $124$ pos, $1,266$ neg)
- **Structural Component Variants:** `no_temporal_encoding`, `no_graph`, `no_historical_features`, `no_package_balancing`:
  - **Explicitly labeled as 1-epoch pipeline smoke tests** (`scripts/run_ablations.py --smoke-test`).
  - Documented as unconverged gradient checks; **not** presented as converged benchmark comparisons.
  - Updated in [experiments/ablation_smoke/ablation_summary.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/experiments/ablation_smoke/ablation_summary.md) and [`experiments/final_figures/ablation_comparison.png`](file:///e:/college/5th%20sem/theory/dl/dl%20project/experiments/final_figures/ablation_comparison.png).

### D. Checkpoint Integrity & Physical File Footprint
- **File Path:** `experiments/rtgn_tier_c/checkpoints/best_rtgn_model.pt`
- **File Size on Disk:** **1,752,093 bytes** ($1.67\text{ MB}$, including AdamW optimizer state moments, scheduler, and configuration dictionary).
- **Raw Parameter Weights:** $129,345 \text{ params} \times 4\text{ bytes} = 517,380\text{ bytes} \approx 505.25\text{ KB}$ (plus metadata headers $\approx 552\text{ KB}$).
- **SHA256 Hash:** `6a4544c6cc133e482c462b0389a36ed5823a4b5f252cc445b250900db6708155`
- **Total Tensors:** **90** tensors ($87$ trainable parameters + $3$ buffers; 0 missing, 0 unexpected).

---

## 2. Documentation Synchronization Index

| Document File | Key Corrected Content | Verification Status |
| :--- | :--- | :---: |
| [README.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/README.md) | Authoritative Tier C table ($P=0.4884, R=0.1615, F_1=0.2428$), note on alternative point ($P=59/357, R=59/130, F_1=118/487$), ablation smoke-test disclaimer. | **Synchronized** |
| [docs/ERROR_ANALYSIS.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/docs/ERROR_ANALYSIS.md) | Corrected $F_1 = 118/487 \approx 0.2423$ calculation for alternative point; preserved 22 FP / 109 FN analysis at $\tau^* = 0.1882$. | **Synchronized** |
| [docs/TIER_C_VERIFICATION_REPORT.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/docs/TIER_C_VERIFICATION_REPORT.md) | Corrected reference table to $P=0.4884, R=0.1615$; segregated alternative point ($P=0.1653, R=0.4538, F_1=0.2423$); updated file size to 1,752,093 bytes. | **Synchronized** |
| [docs/FINAL_PROJECT_READINESS_REPORT.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/docs/FINAL_PROJECT_READINESS_REPORT.md) | Synchronized operating points, clarified ablation smoke-test status, and updated physical file size. | **Synchronized** |
| [docs/FINAL_CONSISTENCY_AUDIT.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/docs/FINAL_CONSISTENCY_AUDIT.md) | Exhaustive forensic audit with mathematical derivations, SHA256 hashes, and data-flow integrity assessments. | **Synchronized** |
| [experiments/ablation_smoke/ablation_summary.md](file:///e:/college/5th%20sem/theory/dl/dl%20project/experiments/ablation_smoke/ablation_summary.md) | Formally declared 4 component variants as 1-epoch smoke tests; verified Cold-Start ($0.5272$) and Online-Adaptation ($0.3029$). | **Synchronized** |

---

## 3. Evaluation Figures Status (`experiments/final_figures/`)

All 5 figures regenerated and verified:
1. [`confusion_matrix.png`](file:///e:/college/5th%20sem/theory/dl/dl%20project/experiments/final_figures/confusion_matrix.png): Raw counts ($TN=1259, FP=22, FN=109, TP=21$) and normalized proportions dynamically verified from `rtgn_tier_c_metrics.json`.
2. [`baseline_comparison.png`](file:///e:/college/5th%20sem/theory/dl/dl%20project/experiments/final_figures/baseline_comparison.png): Grouped comparative bar chart dynamically verified from `baseline_metrics_summary.csv` and `rtgn_tier_c_metrics.json`.
3. [`ablation_comparison.png`](file:///e:/college/5th%20sem/theory/dl/dl%20project/experiments/final_figures/ablation_comparison.png): Explicitly marks the 4 component variants as "Smoke Test Only (Unconverged, 1 Epoch)", alongside the converged Full RTGN ($0.3067$) and verified Dynamic Memory Lifecycle ($0.5272 / 0.3029$).
4. [`precision_recall_curve.png`](file:///e:/college/5th%20sem/theory/dl/dl%20project/experiments/final_figures/precision_recall_curve.png): Dynamically computed from CSV prediction files across all 5 models.
5. [`roc_curve.png`](file:///e:/college/5th%20sem/theory/dl/dl%20project/experiments/final_figures/roc_curve.png): Dynamically computed ROC curves from CSV prediction files.

---

## 4. Verification Tests & Invariance Check

- **Unit Test Suite:** `pytest -v tests/test_rtgn_architecture.py tests/test_rtgn_ablations.py tests/test_baseline_models.py`
  - **21 passed, 0 failed (100% pass rate).**
- **Benchmark Prediction Invariance:**
  - `pip-doom 0.1.0`: Probability **$0.9921$**, Logit $+4.8286 \implies$ `ALERT [MALICIOUS]`
  - `torchvision 0.1.6`: Probability **$0.0050$**, Logit $-5.2915 \implies$ `CLEAN [BENIGN]`
  - `convocations 0.1`: Probability **$0.000008$**, Logit $-11.7376 \implies$ `CLEAN [FALSE NEGATIVE]`
- **Preservation Rules Maintained:** No model retraining, no checkpoint alteration, no graph tensor modifications, no split changes.
