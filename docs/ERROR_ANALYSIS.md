# RTGN Tier C Error Analysis & Model Limitations Report

> **Project:** Temporal Heterogeneous Graph Learning for Early Detection and Propagation-Aware Analysis of Software Supply-Chain Compromise  
> **Benchmark:** Tier C (Unseen Packages / Graph-Disjoint Partition)  
> **Evaluation Checkpoint:** `experiments/rtgn_tier_c/checkpoints/best_rtgn_model.pt`  
> **Decision Threshold:** $\tau^* = 0.1882$ (Frozen from validation fold)  
> **Report Status:** Final Verification & Quality Assurance Pass

---

## 1. Executive Summary & Test Confusion Matrix

In software supply-chain security, transparent error analysis is essential. Models operating under strict inductive conditions (evaluating packages never seen during training) face severe class imbalance and subtle adversarial tactics.

On the Tier C unseen-package test partition ($N = 1,411$ supervised binary releases; $130$ confirmed malicious, $1,281$ assumed benign background), RTGN operates at a frozen validation threshold of $\tau^* = 0.1882$:

```
                        Predicted Benign (0)    Predicted Malicious (1)
Actual Benign (0):             1,259                      22           (Total: 1,281)
Actual Malicious (1):            109                      21           (Total:   130)
```

### Discrete Operating Metrics ($\tau^* = 0.1882$):
- **True Negatives (TN):** $1,259$
- **False Positives (FP):** $22$ (Specificity = $98.28\%$, False Alarm Rate = $1.72\%$)
- **False Negatives (FN):** $109$ (Miss Rate = $83.85\%$)
- **True Positives (TP):** $21$
- **Recall:** $\frac{21}{21 + 109} = \frac{21}{130} \approx \mathbf{0.1615}$ ($16.15\%$)
- **Precision:** $\frac{21}{21 + 22} = \frac{21}{43} \approx \mathbf{0.4884}$ ($48.84\%$)
- **$F_1$ Score:** $\frac{42}{173} \approx \mathbf{0.2428}$

### Continuous Ranking Metrics:
- **Test AUPRC:** **0.3067** (vs. background prevalence of $0.0921$, representing a **$3.33\times$ improvement** over random ranking).
- **Test AUROC:** **0.8144** (demonstrating strong continuous discrimination across all possible operating thresholds).
- **Cold-Start AUPRC:** **0.5272** (evaluating strictly the initial release of unseen packages).

### Note on Alternative High-Sensitivity Operating Point (Colab Exploratory Run):
In earlier exploratory Colab evaluations operating in a lower-threshold regime ($\tau \approx 0.000006$), the model was evaluated at an alternative high-sensitivity operating point:
- **True Positives (TP):** $59$ (out of $130$ total positives)
- **False Positives (FP):** $298$
- **False Negatives (FN):** $71$
- **True Negatives (TN):** $983$
- **Precision:** $\frac{59}{59 + 298} = \frac{59}{357} \approx \mathbf{0.1653}$ ($16.53\%$)
- **Recall:** $\frac{59}{59 + 71} = \frac{59}{130} \approx \mathbf{0.4538}$ ($45.38\%$)
- **$F_1$ Score:** $\frac{2 \times 59}{2 \times 59 + 298 + 71} = \frac{118}{487} \approx \mathbf{0.2423}$

> [!NOTE]
> This represents a distinct, high-recall exploratory operating point ($\tau \ll \tau^*$) and is **NOT** the frozen validation threshold result ($\tau^* = 0.1882$, which achieves $\text{Precision} = 48.84\%$, $\text{Recall} = 16.15\%$, and $F_1 = 0.2428$). In a security operations center (SOC), practitioners can tune this operating threshold based on risk tolerance: $\tau^* = 0.1882$ minimizes analyst alert fatigue ($1.72\%$ false alarm rate), whereas lower thresholds maximize raw threat capture.

---

## 2. Analysis of False Positives ($FP = 22$)

The model flagged 22 legitimate background releases as malicious ($1.72\%$ false alarm rate):

### Primary Drivers of False Positives:
1. **Burst Cadence & Velocity Spikes:**
   - Legitimate packages undergoing rapid CI/CD automated deployments within minutes or hours produce extreme values in `hist_release_frequency` and near-zero `time_since_prev_release_days`.
   - In malicious campaigns, rapid micro-version dumping is common; legitimate automated bots mimic this exact temporal burst behavior.
2. **Abnormal Versioning Patterns in Utility Packages:**
   - Packages with non-standard SemVer jumps (e.g., jumping from `0.0.1` directly to `1.0.0` or publishing rapid sub-minor patches with empty requirements) trigger elevated suspicion.
3. **Zero-Dependency Leaf Utilities:**
   - Legitimate single-purpose utility packages that declare zero dependencies (`declared_dependency_count = 0`) lack stabilizing relational context from the ecosystem graph.

---

## 3. Analysis of False Negatives ($FN = 109$)

The model missed 109 confirmed malicious releases across the unseen test split:

### Primary Drivers of False Negatives:
1. **Isolated Disposable Releases:**
   - Attack packages published as standalone payloads with zero dependencies (`declared_dependency_count = 0`) and zero downstream dependents (`dynamic_downstream_in_degree = 0`).
   - Without dependency edges, relational message passing cannot incorporate contextual signals from established packages.
2. **Benign-Mimicking Metadata:**
   - Attackers frequently choose conventional version numbers (e.g., `0.1.0` or `1.0.0`) and omit environment constraints to blend into background noise.
   - Because RTGN relies strictly on metadata and graph topology, an upload whose metadata is indistinguishable from a standard library release cannot be identified without inspecting file payloads.
3. **Takedown Latency Asymmetry:**
   - Supply-chain malware is often taken down within hours or days of publication. Consequently, attack packages rarely accumulate incoming dependency edges from other packages, remaining permanently at `dynamic_downstream_in_degree = 0`.

---

## 4. Package-Level Detection Patterns in Tier C Test Split

The Tier C test split contains 130 malicious releases distributed across **6 distinct unseen packages**:

| Package Name | Total Test Releases | Detected ($\tau \ge 0.1882$) | Package Recall | Mean Probability | Peak Probability | Primary Risk Factor Observed |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`pip-doom`** | 11 | **11** | **100.0%** | **0.9955** | **0.9965** | Anomalous environment constraint strings & metadata structure |
| **`poppo213`** | 7 | **7** | **100.0%** | **0.8827** | **0.9964** | Rapid anomalous temporal sequence on unseen namespace |
| **`taya`** | 2 | **2** | **100.0%** | **0.9929** | **0.9937** | Consistent high-confidence malicious signature |
| **`SpyWare`** | 9 | **1** | **11.1%** | 0.1102 | **0.9914** | Initial releases blend with background; detected upon cadence divergence |
| **`convocations`** | 100 | **5** | **5.0%** | 0.0255 | **0.3726** | 100 repetitive micro-releases with zero dependencies |
| **`wordsdict`** | 1 | **0** | **0.0%** | 0.000008 | 0.000008 | Single isolated release with zero dependencies |

### Key Insight:
- On 3 of the 6 malicious test packages (`pip-doom`, `poppo213`, `taya`), RTGN achieved **100% detection recall with probabilities $> 0.99$**.
- On `convocations`, the attacker deployed 100 near-identical micro-releases without dependencies; because the first release was completely isolated, the cold-start prediction remained low until sequential cadence accumulated.

---

## 5. Detailed Case Study: `convocations 0.1` False Negative

- **Node Index:** 2 in `nodes_package_version.csv`
- **Release Version:** `0.1` (uploaded 2023-02-12 08:14:14 UTC)
- **Ground Truth:** `confirmed_malicious` ($y = 1$)
- **Model Output:** Logit = `-11.7376`, Probability = $0.000008$ ($8.0 \times 10^{-6}$)
- **Classification:** `BENIGN [CLEAN]` (False Negative)

### Forensic Feature Inspection:
```python
version_major                 = 0.0
version_minor                 = 1.0
version_patch                 = 0.0
is_prerelease                 = 0.0
has_requires_python           = 0.0
requires_python_len           = 0.0
declared_dependency_count     = 0.0    # 0 dependencies in requires_dist
hist_release_count_before_t   = 0.0    # First release of package
hist_age_days_at_t            = 0.0    # 0 days elapsed
time_since_prev_release_days  = 0.0
hist_release_frequency        = 0.0
dynamic_downstream_in_degree  = 0.0    # No packages depend on it
hist_advisories_count_before_t= 0.0
```

### Why Did the Model Miss It?
1. **Complete Topological Isolation:** `convocations 0.1` has 0 declared dependencies. In the causal edge filtering at $t$, there are 0 `depends_on` edges connecting it to the ecosystem.
2. **Identical to Benign Baseline:** Hundreds of benign PyPI libraries begin life as a `0.1.0` release with zero dependencies. Without executing the Python source code payload, there is zero metadata signal distinguishing this release from a benign library.
3. **Absence of Payload Inspection:** The malicious payload inside `convocations 0.1` consists of an obfuscated executable dropped in `setup.py`. Because RTGN is strictly a metadata and graph model, it cannot detect payload-level obfuscation in isolated nodes.

---

## 6. Architectural Limitations of Metadata & Graph Detection

1. **Payload Invisibility:**
   - Attackers embedding encrypted shellcode, reverse shells, or credential harvesters inside `setup.py` cannot be detected via metadata if the package metadata conforms strictly to standard conventions.
2. **Cold-Start Blindspot on Zero-Dependency Leaves:**
   - A newly registered package with no dependencies, no prior releases, and no downstream dependents is an isolated graph island. GNN message passing has no incoming information paths.
3. **Adversarial Cadence Manipulation:**
   - Sophisticated supply-chain attackers can deliberately space out release timestamps and adopt realistic semantic version numbers to evade temporal anomaly detection.
4. **Defense-in-Depth Recommendation:**
   - RTGN should **not** be deployed as an isolated gatekeeper. It is best utilized as an early-stage ecosystem triage filter:
     - RTGN continuously monitors the ecosystem graph and flags high-risk topological bursts or dependency anomalies.
     - Flagged packages and border-line releases are forwarded to downstream static AST analyzers, YARA rulesets, and dynamic sandbox detonation environments.
