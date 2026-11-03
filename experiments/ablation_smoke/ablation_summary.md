# RTGN Tier C Ablation Study & Execution Status

> **Benchmark:** Tier C (Unseen Package / Graph-Disjoint Partition)  
> **Evaluation Protocol:** Strictly chronological causal filtering, per-target edge snapshot, validation-frozen threshold $\tau^*$, source_unflagged excluded.  
> **Evidence Status:** Full RTGN is fully trained & converged; Dynamic Memory Lifecycle is evaluated on the checkpoint; Structural variants are 1-epoch smoke-test checks.

---

## 1. Architectural Variants Execution & Evidence Status

| Model Variant | Experiment Type | Execution Status | Test AUPRC | Test AUROC | Test F1 | Notes & Evidence Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full RTGN** | Full Training (5 Ep.) | **Converged Benchmark** | **0.3067** | **0.8144** | **0.2428** | Colab GPU Run (Epoch 4 recovered & verified locally) |
| **No Temporal Encoding** | Component Variant | **Smoke-Test Only** | 0.0000* | 0.5000* | 0.0000* | 1-epoch smoke run (pipeline gradient check; unconverged) |
| **No Graph (GNN Bypass)** | Component Variant | **Smoke-Test Only** | 0.0000* | 0.5000* | 0.0000* | 1-epoch smoke run (pipeline gradient check; unconverged) |
| **No Historical Features** | Component Variant | **Smoke-Test Only** | 0.0000* | 0.5000* | 0.0000* | 1-epoch smoke run (pipeline gradient check; unconverged) |
| **No Package Balancing** | Component Variant | **Smoke-Test Only** | 0.0000* | 0.5000* | 0.0000* | 1-epoch smoke run (pipeline gradient check; unconverged) |

> [!IMPORTANT]
> **Ablation Transparency Declaration:**
> The 4 structural component variants were executed exclusively as 1-epoch smoke tests (`scripts/run_ablations.py --smoke-test`) to confirm code execution and backward pass. They did not undergo 5-epoch convergence training. Their 0.0000 metrics reflect unconverged smoke-test states and must not be presented as converged comparative results.

---

## 2. Dynamic Memory Lifecycle (Empirically Evaluated on Checkpoint)

| Memory Protocol | Test AUPRC | Sample Count | Protocol Description & Execution Evidence |
| :--- | :---: | :---: | :--- |
| **Cold-Start** | **0.5272** | 21 releases | Evaluated strictly on initial unseen releases with $h_0$ (6 pos, 15 neg) |
| **Online-Adaptation** | **0.3029** | 1,390 releases | Dynamic memory updated post-prediction sequentially (124 pos, 1,266 neg) |
| **Background Prevalence** | **0.0921** | 1,411 releases | Random chance baseline on test split |

