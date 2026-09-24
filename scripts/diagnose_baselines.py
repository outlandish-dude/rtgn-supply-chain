"""
Comprehensive Baseline Shortcut, Leakage, and Cohort Diagnostic Script.

Performs:
1. Single-Feature Ablation (LR per feature, 13 features)
2. Feature-Group Ablation (Intrinsic, History, Graph, Security)
3. Security-Context Ablation (retrain LR, RF, XGB without feature 12)
4. Source-Unflagged Diagnostic (probabilities, distribution, thresholds)
5. Package-Level Concentration Analysis
6. Hard-Negative Diagnostic on source_unflagged cohort
7. Package-Disjoint Baseline Evaluation (with positive test count check)
8. Label/Package Structure Table for top malicious packages
9. Structured Conclusion & Recommendation synthesis
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_classification_metrics,
    evaluate_with_subsets,
    find_optimal_threshold,
)
from src.graph.features import FEATURE_NAMES
from src.models.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)
from src.training.baseline_trainer import BaselineTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_diagnostics():
    logger.info("Initializing Diagnostic Pipeline...")
    output_dir = PROJECT_ROOT / "experiments" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trainer data
    trainer = BaselineTrainer(
        graph_dir=PROJECT_ROOT / "data" / "experimental" / "graph",
        output_dir=PROJECT_ROOT / "experiments" / "baselines",
    )
    trainer.load_data()
    trainer.prepare_partitions()

    X_train = trainer.data_splits["train"]["binary"]["X"]
    y_train = trainer.data_splits["train"]["binary"]["y"]

    X_val = trainer.data_splits["val"]["binary"]["X"]
    y_val = trainer.data_splits["val"]["binary"]["y"]

    X_test = trainer.data_splits["test"]["binary"]["X"]
    y_test = trainer.data_splits["test"]["binary"]["y"]
    test_pkgs = trainer.data_splits["test"]["binary"]["packages"]

    # =========================================================================
    # 1. SINGLE-FEATURE ABLATION
    # =========================================================================
    logger.info("--- 1. SINGLE-FEATURE ABLATION ---")
    single_feat_results = []
    flagged_features = []

    for i, feat_name in enumerate(FEATURE_NAMES):
        X_tr_i = X_train[:, [i]]
        X_val_i = X_val[:, [i]]
        X_te_i = X_test[:, [i]]

        lr = LogisticRegressionBaseline(random_state=42)
        lr.fit(X_tr_i, y_train)

        val_probs = lr.predict_proba(X_val_i)[:, 1]
        opt_thresh, val_f1 = find_optimal_threshold(y_val, val_probs)

        test_probs = lr.predict_proba(X_te_i)[:, 1]
        metrics = compute_classification_metrics(y_test, test_probs, threshold=opt_thresh)

        flag = False
        # Flag unusually high discrimination
        if metrics["auprc"] >= 0.70 or metrics["auroc"] >= 0.85:
            flag = True
            flagged_features.append(feat_name)

        single_feat_results.append({
            "feature_index": i,
            "feature_name": feat_name,
            "val_threshold": opt_thresh,
            "val_f1": val_f1,
            "test_auprc": metrics["auprc"],
            "test_auroc": metrics["auroc"],
            "test_f1": metrics["f1"],
            "flagged": flag,
        })
        logger.info(
            f"Feature {i:2d} ({feat_name:30s}): Test AUPRC={metrics['auprc']:.4f}, "
            f"AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f} {'[FLAGGED]' if flag else ''}"
        )

    # =========================================================================
    # 2. FEATURE-GROUP ABLATION
    # =========================================================================
    logger.info("--- 2. FEATURE-GROUP ABLATION ---")
    groups = {
        "A_intrinsic_release": [0, 1, 2, 3, 4, 5, 6],
        "B_package_history": [7, 8, 9, 10],
        "C_graph_context": [11],
        "D_security_context": [12],
    }

    group_results = {}
    for gname, indices in groups.items():
        X_tr_g = X_train[:, indices]
        X_val_g = X_val[:, indices]
        X_te_g = X_test[:, indices]

        # LR
        lr = LogisticRegressionBaseline(random_state=42)
        lr.fit(X_tr_g, y_train)
        val_probs_lr = lr.predict_proba(X_val_g)[:, 1]
        opt_thresh_lr, _ = find_optimal_threshold(y_val, val_probs_lr)
        test_probs_lr = lr.predict_proba(X_te_g)[:, 1]
        m_lr = compute_classification_metrics(y_test, test_probs_lr, threshold=opt_thresh_lr)

        # RF
        rf = RandomForestBaseline(random_state=42)
        rf.fit(X_tr_g, y_train)
        val_probs_rf = rf.predict_proba(X_val_g)[:, 1]
        opt_thresh_rf, _ = find_optimal_threshold(y_val, val_probs_rf)
        test_probs_rf = rf.predict_proba(X_te_g)[:, 1]
        m_rf = compute_classification_metrics(y_test, test_probs_rf, threshold=opt_thresh_rf)

        group_results[gname] = {
            "feature_indices": indices,
            "feature_names": [FEATURE_NAMES[idx] for idx in indices],
            "logistic_regression": {
                "test_auprc": m_lr["auprc"],
                "test_auroc": m_lr["auroc"],
                "test_f1": m_lr["f1"],
                "threshold": opt_thresh_lr,
            },
            "random_forest": {
                "test_auprc": m_rf["auprc"],
                "test_auroc": m_rf["auroc"],
                "test_f1": m_rf["f1"],
                "threshold": opt_thresh_rf,
            },
        }
        logger.info(
            f"Group {gname:20s}: LR AUPRC={m_lr['auprc']:.4f}, AUROC={m_lr['auroc']:.4f} | "
            f"RF AUPRC={m_rf['auprc']:.4f}, AUROC={m_rf['auroc']:.4f}"
        )

    # =========================================================================
    # 3. SECURITY-CONTEXT ABLATION (Remove feature 12)
    # =========================================================================
    logger.info("--- 3. SECURITY-CONTEXT ABLATION (Remove feature 12) ---")
    no_sec_indices = list(range(12))  # Features 0..11
    X_tr_nosec = X_train[:, no_sec_indices]
    X_val_nosec = X_val[:, no_sec_indices]
    X_te_nosec = X_test[:, no_sec_indices]

    sec_ablation_results = {}
    models_to_test = {
        "logistic_regression": LogisticRegressionBaseline(random_state=42),
        "random_forest": RandomForestBaseline(random_state=42),
        "xgboost": XGBoostBaseline(random_state=42),
    }

    for mname, m in models_to_test.items():
        m.fit(X_tr_nosec, y_train)
        val_probs = m.predict_proba(X_val_nosec)[:, 1]
        opt_thresh, val_f1 = find_optimal_threshold(y_val, val_probs)
        test_probs = m.predict_proba(X_te_nosec)[:, 1]

        sub_res = evaluate_with_subsets(y_test, test_probs, threshold=opt_thresh, package_names=test_pkgs)
        sec_ablation_results[mname] = {
            "val_threshold": opt_thresh,
            "full_test": sub_res["full"],
            "eval_excluding_upgini": sub_res.get("eval_excluding_upgini", {}),
        }
        logger.info(
            f"Without Sec Feat [{mname}]: Test AUPRC={sub_res['full']['auprc']:.4f}, "
            f"AUROC={sub_res['full']['auroc']:.4f}, F1={sub_res['full']['f1']:.4f} | "
            f"No-Upgini AUPRC={sub_res.get('eval_excluding_upgini', {}).get('auprc', 0.0):.4f}"
        )

    # =========================================================================
    # 4. SOURCE-UNFLAGGED DIAGNOSTIC
    # =========================================================================
    logger.info("--- 4. SOURCE-UNFLAGGED DIAGNOSTIC ---")
    preds_dir = PROJECT_ROOT / "experiments" / "baselines" / "predictions"
    unflagged_diagnostic = {}

    for mname in ["logistic_regression", "random_forest", "xgboost"]:
        unflagged_csv = preds_dir / f"unflagged_qualitative_predictions_{mname}.csv"
        if not unflagged_csv.exists():
            continue
        df_unf = pd.read_csv(unflagged_csv)
        probs = df_unf["predicted_probability"].values

        mean_p = float(np.mean(probs))
        median_p = float(np.median(probs))
        pct_ge_05 = float(np.mean(probs >= 0.5) * 100)
        pct_ge_09 = float(np.mean(probs >= 0.9) * 100)

        # Package-level mean probability
        pkg_dist = df_unf.groupby("package_name")["predicted_probability"].agg(["count", "mean", "median"]).reset_index()
        pkg_dist_records = pkg_dist.to_dict(orient="records")

        unflagged_diagnostic[mname] = {
            "total_unflagged": len(probs),
            "mean_probability": round(mean_p, 4),
            "median_probability": round(median_p, 4),
            "pct_ge_0_5": round(pct_ge_05, 2),
            "pct_ge_0_9": round(pct_ge_09, 2),
            "package_breakdown": pkg_dist_records,
        }
        logger.info(
            f"Unflagged [{mname}]: Mean={mean_p:.4f}, Med={median_p:.4f}, "
            f"% >= 0.5: {pct_ge_05:.1f}%, % >= 0.9: {pct_ge_09:.1f}%"
        )

    # =========================================================================
    # 5. PACKAGE-LEVEL CONCENTRATION
    # =========================================================================
    logger.info("--- 5. PACKAGE-LEVEL CONCENTRATION ---")
    package_concentration = {}
    for mname in ["logistic_regression", "random_forest", "xgboost"]:
        bin_csv = preds_dir / f"binary_predictions_{mname}.csv"
        unf_csv = preds_dir / f"unflagged_qualitative_predictions_{mname}.csv"
        df_bin = pd.read_csv(bin_csv)
        df_unf = pd.read_csv(unf_csv)

        # Category distributions
        mal_probs = df_bin[df_bin["true_label_binary"] == 1]["predicted_probability"].values
        ben_probs = df_bin[df_bin["true_label_binary"] == 0]["predicted_probability"].values
        unf_probs = df_unf["predicted_probability"].values

        package_concentration[mname] = {
            "confirmed_malicious": {
                "count": len(mal_probs),
                "mean": round(float(np.mean(mal_probs)), 4),
                "median": round(float(np.median(mal_probs)), 4),
                "std": round(float(np.std(mal_probs)), 4),
            },
            "assumed_benign_background": {
                "count": len(ben_probs),
                "mean": round(float(np.mean(ben_probs)), 4),
                "median": round(float(np.median(ben_probs)), 4),
                "std": round(float(np.std(ben_probs)), 4),
            },
            "source_unflagged": {
                "count": len(unf_probs),
                "mean": round(float(np.mean(unf_probs)), 4),
                "median": round(float(np.median(unf_probs)), 4),
                "std": round(float(np.std(unf_probs)), 4),
            },
        }

    # =========================================================================
    # 6. HARD-NEGATIVE DIAGNOSTIC (source_unflagged cohort alert rate)
    # =========================================================================
    logger.info("--- 6. HARD-NEGATIVE DIAGNOSTIC ---")
    thresholds_path = PROJECT_ROOT / "experiments" / "baselines" / "metrics" / "validation_thresholds.json"
    thresholds = {}
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    hard_negative_diag = {}
    for mname in ["logistic_regression", "random_forest", "xgboost"]:
        df_unf = pd.read_csv(preds_dir / f"unflagged_qualitative_predictions_{mname}.csv")
        tau = thresholds.get(mname, 0.5)

        # Per split alert rate
        split_alert_rates = {}
        for sp in ["train", "val", "test"]:
            sub = df_unf[df_unf["split"] == sp]["predicted_probability"].values
            if len(sub) > 0:
                alert_rate = float(np.mean(sub >= tau) * 100)
                split_alert_rates[sp] = {
                    "count": len(sub),
                    "alert_rate_pct": round(alert_rate, 2),
                    "mean_prob": round(float(np.mean(sub)), 4),
                }

        hard_negative_diag[mname] = {
            "validation_threshold": tau,
            "overall_alert_rate_pct": round(float(np.mean(df_unf["predicted_probability"].values >= tau) * 100), 2),
            "by_split": split_alert_rates,
        }
        logger.info(
            f"Hard-Negative Alert Rate [{mname} at tau={tau}]: "
            f"Overall={hard_negative_diag[mname]['overall_alert_rate_pct']:.1f}% "
            f"(Test Alert Rate={split_alert_rates.get('test', {}).get('alert_rate_pct', 0.0):.1f}%)"
        )

    # =========================================================================
    # 7. PACKAGE-DISJOINT BASELINE EVALUATION
    # =========================================================================
    logger.info("--- 7. PACKAGE-DISJOINT BASELINE EVALUATION ---")
    masks = trainer.split_masks
    meta = trainer.version_metadata
    feats = trainer.features

    disjoint_splits = {}
    for sp in ["train", "val", "test"]:
        m_disjoint = masks[f"{sp}_mask_graph_disjoint"]
        sp_indices = np.where(m_disjoint)[0]

        bin_idx = []
        bin_y = []
        for idx in sp_indices:
            lbl = meta[idx]["label_str"]
            if lbl == "confirmed_malicious":
                bin_idx.append(idx)
                bin_y.append(1)
            elif lbl == "assumed_benign_background":
                bin_idx.append(idx)
                bin_y.append(0)

        disjoint_splits[sp] = {
            "indices": np.array(bin_idx, dtype=int),
            "X": feats[np.array(bin_idx, dtype=int)],
            "y": np.array(bin_y, dtype=int),
            "pos": int(np.sum(np.array(bin_y) == 1)),
            "neg": int(np.sum(np.array(bin_y) == 0)),
            "packages": [meta[i]["package_name"] for i in bin_idx],
        }
        logger.info(
            f"Disjoint Split [{sp}]: Total={len(bin_y)}, Pos={disjoint_splits[sp]['pos']}, "
            f"Neg={disjoint_splits[sp]['neg']}"
        )

    disjoint_test_pos = disjoint_splits["test"]["pos"]
    disjoint_results = {
        "train_pos": disjoint_splits["train"]["pos"],
        "train_neg": disjoint_splits["train"]["neg"],
        "val_pos": disjoint_splits["val"]["pos"],
        "val_neg": disjoint_splits["val"]["neg"],
        "test_pos": disjoint_test_pos,
        "test_neg": disjoint_splits["test"]["neg"],
        "models": {},
    }

    if disjoint_test_pos > 0 and disjoint_splits["val"]["pos"] > 0:
        X_tr_dj = disjoint_splits["train"]["X"]
        y_tr_dj = disjoint_splits["train"]["y"]
        X_val_dj = disjoint_splits["val"]["X"]
        y_val_dj = disjoint_splits["val"]["y"]
        X_te_dj = disjoint_splits["test"]["X"]
        y_te_dj = disjoint_splits["test"]["y"]

        dj_models = {
            "logistic_regression": LogisticRegressionBaseline(random_state=42),
            "random_forest": RandomForestBaseline(random_state=42),
            "xgboost": XGBoostBaseline(random_state=42),
        }
        for mname, m in dj_models.items():
            m.fit(X_tr_dj, y_tr_dj)
            val_p = m.predict_proba(X_val_dj)[:, 1]
            tau_dj, _ = find_optimal_threshold(y_val_dj, val_p)
            te_p = m.predict_proba(X_te_dj)[:, 1]
            m_dj = compute_classification_metrics(y_te_dj, te_p, threshold=tau_dj)
            disjoint_results["models"][mname] = {
                "threshold": tau_dj,
                "test_auprc": m_dj["auprc"],
                "test_auroc": m_dj["auroc"],
                "test_precision": m_dj["precision"],
                "test_recall": m_dj["recall"],
                "test_f1": m_dj["f1"],
                "test_prevalence": m_dj["positive_prevalence"],
            }
            logger.info(
                f"Disjoint [{mname}]: Test AUPRC={m_dj['auprc']:.4f}, AUROC={m_dj['auroc']:.4f}, "
                f"F1={m_dj['f1']:.4f} (test pos count = {disjoint_test_pos})"
            )
    else:
        logger.warning(
            f"Package-disjoint test set has {disjoint_test_pos} positive examples. "
            "Evaluation may be uninformative or undefined."
        )

    # =========================================================================
    # 8. LABEL / PACKAGE STRUCTURE TABLE FOR TOP MALICIOUS PACKAGES
    # =========================================================================
    logger.info("--- 8. LABEL/PACKAGE STRUCTURE ---")
    # Tally positive counts by package across train, val, test
    pkg_stats = {}
    for idx, r in enumerate(meta):
        p = r["package_name"]
        lbl = r["label_str"]
        split = None
        if masks["train_mask_chrono"][idx]:
            split = "train"
        elif masks["val_mask_chrono"][idx]:
            split = "val"
        elif masks["test_mask_chrono"][idx]:
            split = "test"

        if p not in pkg_stats:
            pkg_stats[p] = {
                "package": p,
                "train_pos": 0,
                "val_pos": 0,
                "test_pos": 0,
                "train_unf": 0,
                "val_unf": 0,
                "test_unf": 0,
                "total_versions": 0,
            }
        pkg_stats[p]["total_versions"] += 1
        if lbl == "confirmed_malicious" and split:
            pkg_stats[p][f"{split}_pos"] += 1
        elif lbl == "source_unflagged" and split:
            pkg_stats[p][f"{split}_unf"] += 1

    # Load predicted probabilities per package from test predictions
    df_test_lr = pd.read_csv(preds_dir / "binary_predictions_logistic_regression.csv")
    df_test_rf = pd.read_csv(preds_dir / "binary_predictions_random_forest.csv")
    df_test_xgb = pd.read_csv(preds_dir / "binary_predictions_xgboost.csv")

    test_lr_pkg_means = df_test_lr.groupby("package_name")["predicted_probability"].mean().to_dict()
    test_rf_pkg_means = df_test_rf.groupby("package_name")["predicted_probability"].mean().to_dict()
    test_xgb_pkg_means = df_test_xgb.groupby("package_name")["predicted_probability"].mean().to_dict()

    # Filter to packages with any malicious releases
    mal_pkgs = [p for p, s in pkg_stats.items() if (s["train_pos"] + s["val_pos"] + s["test_pos"]) > 0]
    mal_pkgs.sort(key=lambda p: (pkg_stats[p]["train_pos"] + pkg_stats[p]["val_pos"] + pkg_stats[p]["test_pos"]), reverse=True)

    structure_table = []
    for p in mal_pkgs:
        s = pkg_stats[p]
        structure_table.append({
            "package": p,
            "train_positive_count": s["train_pos"],
            "validation_positive_count": s["val_pos"],
            "test_positive_count": s["test_pos"],
            "total_versions": s["total_versions"],
            "lr_mean_test_prob": round(test_lr_pkg_means.get(p, float("nan")), 4),
            "rf_mean_test_prob": round(test_rf_pkg_means.get(p, float("nan")), 4),
            "xgb_mean_test_prob": round(test_xgb_pkg_means.get(p, float("nan")), 4),
        })

    # Save all diagnostic artifacts
    diagnostics_payload = {
        "single_feature_ablation": single_feat_results,
        "feature_group_ablation": group_results,
        "security_context_ablation": sec_ablation_results,
        "source_unflagged_diagnostic": unflagged_diagnostic,
        "package_level_concentration": package_concentration,
        "hard_negative_diagnostic": hard_negative_diag,
        "package_disjoint_baseline": disjoint_results,
        "top_malicious_packages_structure": structure_table,
    }

    with open(output_dir / "diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics_payload, f, indent=2)

    logger.info(f"Diagnostics complete. Full results written to {output_dir / 'diagnostic_results.json'}.")
    return diagnostics_payload


if __name__ == "__main__":
    run_diagnostics()
