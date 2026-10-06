"""
CLI Script to train and evaluate classical and feature baselines.

Executes:
1. Prevalence Baseline
2. Logistic Regression (class_weight='balanced')
3. Random Forest (class_weight='balanced')
4. XGBoost (scale_pos_weight=N_neg/N_pos)

Generates:
- experiments/baselines/models/
- experiments/baselines/predictions/
- experiments/baselines/metrics/
- docs/BASELINE_EXPERIMENT_DESIGN.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.baseline_trainer import BaselineTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def write_baseline_documentation(
    doc_path: Path,
    summary_metrics_path: Path,
    thresholds_path: Path,
    mode: str = "chrono",
) -> None:
    """Generate docs/BASELINE_EXPERIMENT_DESIGN.md or BASELINE_TIER_C_EXPERIMENT_DESIGN.md."""
    thresholds = {}
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    if mode == "tier_c":
        lines = [
            "# Tier C Baseline Modeling Specification & Experimental Design",
            "",
            "> **Pipeline Stage:** Classical & Feature Tabular Baselines (Tier C Unseen Packages)  ",
            "> **Task:** Unseen Package / Disjoint Partition Compromise Detection  ",
            "> **Input Matrix:** `data/experimental/graph/package_version_features.pt` ($16,088 \\times 13$)  ",
            "",
            "---",
            "",
            "## 1. Tier C Experimental Setup",
            "",
            "### A. Binary Label Semantics",
            "- **Positive Class ($y=1$):** `confirmed_malicious` releases (Datadog verified malicious uploads).",
            "- **Negative Class ($y=0$):** `assumed_benign_background` releases (rigorously stratified background packages).",
            "- **`source_unflagged` (Non-Benign Sibling Releases):**",
            "  - These releases belong to packages with identified compromised versions, but were not flagged.",
            "  - They are **strictly excluded** from binary training and formal evaluation ($y \\in \\{0, 1\\}$).",
            "  - `true_label_binary` remains null/empty.",
            "",
            "### B. Exact Binary Split Sample Counts",
            "",
            "| Split | Partition Type | Confirmed Malicious ($y=1$) | Assumed Benign ($y=0$) | Total Supervised Binary Samples | Ignored Sibling ($y=-1$) |",
            "|---|---|---|---|---|---|",
            "| **Train** | `train_mask_graph_disjoint` | **1,297** | **6,672** | **7,969** | **1,770** |",
            "| **Validation** | `val_mask_graph_disjoint` | **68** | **1,609** | **1,677** | **218** |",
            "| **Test** | `test_mask_graph_disjoint` | **130** | **1,281** | **1,411** | **1,244** |",
            "",
            "---",
            "",
            "## 2. Model Architectures & Imbalance Protocols",
            "",
            "1. **Prevalence Baseline:** Constant positive prevalence of Tier C training data.",
            "2. **Logistic Regression:** Train-fitted StandardScaler + LogisticRegression(class_weight='balanced').",
            "3. **Random Forest:** RandomForestClassifier(class_weight='balanced', n_estimators=300, max_depth=12).",
            "4. **XGBoost:** XGBClassifier(scale_pos_weight = N_neg / N_pos = 6672 / 1297 = 5.14).",
            "",
            "---",
            "",
            "## 3. Frozen Validation Thresholds",
            "",
        ]
    else:
        lines = [
            "# Baseline Modeling Specification & Experimental Design",
            "",
            "> **Pipeline Stage:** Classical & Feature Tabular Baselines  ",
            "> **Task:** Temporal Release-Level Compromise Detection  ",
            "> **Input Matrix:** `data/experimental/graph/package_version_features.pt` ($16,088 \\times 13$)  ",
            "",
            "---",
            "",
            "## 1. Primary Experimental Setup",
            "",
            "### A. Binary Label Semantics",
            "- **Positive Class ($y=1$):** `confirmed_malicious` releases (Datadog verified malicious uploads).",
            "- **Negative Class ($y=0$):** `assumed_benign_background` releases (rigorously stratified background packages).",
            "- **`source_unflagged` (Non-Benign Sibling Releases):**",
            "  - These releases belong to packages with identified compromised versions, but were not flagged.",
            "  - They are **strictly excluded** from binary training and formal evaluation ($y \\in \\{0, 1\\}$).",
            "  - `true_label_binary` remains null/empty.",
            "  - Model predictions are generated separately for qualitative/sensitivity analysis and never conflated with scored test metrics.",
            "",
            "### B. Exact Binary Split Sample Counts",
            "",
            "| Split | Cutoff Interval | Confirmed Malicious ($y=1$) | Assumed Benign ($y=0$) | Total Supervised Binary Samples | Positive Prevalence |",
            "|---|---|---|---|---|---|",
            "| **Train** | $t < \\text{2025-06-01}$ | **1,218** | **8,517** | **9,735** | **12.51%** |",
            "| **Validation** | $2025-06-01 \\le t < \\text{2026-03-01}$ | **151** | **564** | **715** | **21.12%** |",
            "| **Test** | $t \\ge \\text{2026-03-01}$ | **126** | **481** | **607** | **20.76%** |",
            "",
            "---",
            "",
            "## 2. Model Architectures & Imbalance Protocols",
            "",
            "Each classical model uses **exactly one** primary class-imbalance mechanism on the training loss:",
            "",
            "1. **Prevalence Baseline:**",
            "   - Predicts constant positive prevalence of training data ($p_{\\text{train}} = 1,218 / 9,735 = 0.1251$).",
            "   - Serves as empirical chance-level lower bound for AUPRC ($AUPRC_{\\text{test}} = \\text{prevalence}_{\\text{test}} = 0.2076$) and AUROC ($0.5$).",
            "",
            "2. **Logistic Regression:**",
            "   - Pipeline: `StandardScaler` fitted **strictly on training fold** $\\to$ `LogisticRegression(class_weight='balanced', max_iter=1000)`.  ",
            "   - Single imbalance mechanism: `class_weight='balanced'`.  ",
            "",
            "3. **Random Forest:**",
            "   - `RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, class_weight='balanced', random_state=42)`.  ",
            "   - Single imbalance mechanism: `class_weight='balanced'`.  ",
            "",
            "4. **XGBoost:**",
            "   - `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=6.99, random_state=42)`.  ",
            "   - Single imbalance mechanism: `scale_pos_weight = N_neg / N_pos = 8,517 / 1,218 \\approx 6.99`.  ",
            "",
            "> **Note on Sample Weights:** The dataset's package-balancing `sample_weight` is NOT mixed into baseline training loss, leaving package-balancing as an isolated future ablation.",
            "",
            "---",
            "",
            "## 3. Threshold Optimization & Test Evaluation",
            "",
            "- **Primary Continuous Metric:** AUPRC (Average Precision).",
            "- **Threshold Selection Protocol:**",
            "  - The classification threshold $\\tau^*$ is determined by maximizing the F1 score strictly on **validation predictions**:",
            "    $$\\tau^* = \\arg\\max_{\\tau \\in [0.01, 0.99]} F_1(\\text{val}, \\tau)$$  ",
            "  - The chosen threshold $\\tau^*$ is frozen and applied to test data.",
            "- **Test Protocol:**",
            "  - Evaluated on **Full Test Set** (607 binary releases).",
            "  - Evaluated on **Test Set Excluding Upgini** (`eval_excluding_upgini`) to audit generalizability across non-dominant packages.",
            "  - Test set remains completely unweighted and unmodified.",
            "",
            "---",
            "",
            "## 4. Frozen Validation Thresholds",
            "",
        ]

    for model_name, tau in thresholds.items():
        lines.append(f"- **{model_name}:** $\\tau^* = {tau}$")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Persistent Artifacts Index",
        f"- `{doc_path.stem}/models/`: Serialized trained models (`.joblib`, `.json`).",
        f"- `{doc_path.stem}/predictions/binary_predictions_{{model}}.csv`: Scored binary predictions across splits.",
        f"- `{doc_path.stem}/predictions/unflagged_qualitative_predictions_{{model}}.csv`: Qualitative predictions on unflagged sibling releases.",
        f"- `{doc_path.stem}/metrics/baseline_metrics.json`: Detailed metric payload.",
        f"- `{doc_path.stem}/metrics/baseline_metrics_summary.csv`: Summary metrics comparison table.",
        f"- `{doc_path.stem}/metrics/validation_thresholds.json`: Validation-selected thresholds.",
    ])

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Documentation written to {doc_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical baseline models.")
    parser.add_argument("--mode", type=str, default="chrono", choices=["chrono", "tier_a", "tier_c", "graph_disjoint"],
                        help="Benchmark evaluation tier: chrono/tier_a or tier_c/graph_disjoint")
    parser.add_argument("--graph-dir", type=str, default="data/experimental/graph")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    mode = "tier_c" if args.mode in ("tier_c", "graph_disjoint") else "chrono"
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "experiments/baselines_tier_c" if mode == "tier_c" else "experiments/baselines"

    start_time = time.time()
    logger.info(f"Initializing Baseline Modeling Pipeline (Mode: {mode})...")

    trainer = BaselineTrainer(
        graph_dir=PROJECT_ROOT / args.graph_dir,
        output_dir=PROJECT_ROOT / output_dir,
        mode=mode,
    )
    trainer.load_data()
    metrics = trainer.train_and_evaluate_all()

    doc_filename = "BASELINE_TIER_C_EXPERIMENT_DESIGN.md" if mode == "tier_c" else "BASELINE_EXPERIMENT_DESIGN.md"
    doc_path = PROJECT_ROOT / "docs" / doc_filename
    write_baseline_documentation(
        doc_path=doc_path,
        summary_metrics_path=trainer.metrics_dir / "baseline_metrics_summary.csv",
        thresholds_path=trainer.metrics_dir / "validation_thresholds.json",
        mode=mode,
    )

    elapsed = time.time() - start_time
    logger.info(f"Baseline pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
