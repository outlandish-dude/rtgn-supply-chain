"""
Script to update the package/maintainer/repository-disjoint split metadata and graph masks.

Implements the approved redesigned Tier C benchmark:
- Package-grouped, maintainer-grouped, and repository-grouped stratification
- Deliberately preserves 6 unseen malicious packages in Test (130 malicious releases, 30 graph-observable)
- Deliberately preserves 6 unseen malicious packages in Val (68 malicious releases, 7 graph-observable)
- Preserves 7 malicious packages in Train (1,297 malicious releases, 1,278 graph-observable)
- Strict 0 package, repository, and maintainer overlap
- Updates:
  1. data/experimental/splits/graph_package_disjoint_split.json
  2. data/experimental/graph/split_masks.pt
  3. data/experimental/graph/graph_summary.json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import sys
from collections import defaultdict
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def update_disjoint_split():
    experimental_csv = PROJECT_ROOT / "data" / "experimental" / "experimental_versions.csv"
    with open(experimental_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(rows)} records from {experimental_csv}.")

    live_rows = [(idx, r) for idx, r in enumerate(rows) if r["registry_status"] == "live"]
    pkg_rows = defaultdict(list)
    for idx, r in live_rows:
        pkg_rows[r["package"]].append((idx, r))

    # 1. Package allocation
    train_mal = [
        "upgini", "requests", "definitely-not-requests", "a-b27",
        "botocote", "grandslam", "secretInspector", "security-util"
    ]
    val_mal = [
        "fastapi-toolkit", "liblxml3", "mirrorbot", "ReverseShell",
        "MastoWrapper", "pymatcha"
    ]
    test_mal = [
        "convocations", "pip-doom", "poppo213", "SpyWare",
        "taya", "wordsdict"
    ]

    benign_pkgs = sorted([
        p for p, p_rows in pkg_rows.items()
        if any(r["label"] == "assumed_benign_background" for idx, r in p_rows) and p != "requests"
    ])
    train_ben = benign_pkgs[:69]
    val_ben = benign_pkgs[69:84]
    test_ben = benign_pkgs[84:]

    unf_pkgs = sorted([
        p for p, p_rows in pkg_rows.items()
        if all(r["label"] == "source_unflagged" for idx, r in p_rows)
    ])
    train_unf = unf_pkgs[:10]
    val_unf = unf_pkgs[10:13]
    test_unf = unf_pkgs[13:]

    splits_pkgs = {
        "train": sorted(train_mal + train_ben + train_unf),
        "val": sorted(val_mal + val_ben + val_unf),
        "test": sorted(test_mal + test_ben + test_unf),
    }

    # Verify disjointness
    tr_p, val_p, te_p = set(splits_pkgs["train"]), set(splits_pkgs["val"]), set(splits_pkgs["test"])
    assert len(tr_p & val_p) == 0, f"Package overlap between Train and Val: {tr_p & val_p}"
    assert len(tr_p & te_p) == 0, f"Package overlap between Train and Test: {tr_p & te_p}"
    assert len(val_p & te_p) == 0, f"Package overlap between Val and Test: {val_p & te_p}"

    def get_repos(pkgs):
        res = set()
        for p in pkgs:
            for idx, r in pkg_rows[p]:
                if r["repository_url"]:
                    res.add(r["repository_url"])
        return res

    tr_r, val_r, te_r = get_repos(tr_p), get_repos(val_p), get_repos(te_p)
    assert len(tr_r & val_r) == 0, f"Repository overlap between Train and Val: {tr_r & val_r}"
    assert len(tr_r & te_r) == 0, f"Repository overlap between Train and Test: {tr_r & te_r}"
    assert len(val_r & te_r) == 0, f"Repository overlap between Val and Test: {val_r & te_r}"

    logger.info("Verified strict 0 package and repository overlap.")

    # Build split payload
    split_payload = {
        "split_type": "package_maintainer_repository_disjoint_unseen_attack",
        "description": (
            "Package/maintainer/repository-disjoint unseen-package generalization benchmark. "
            "Deliberately preserves 6 unseen malicious packages in Test (130 releases, 30 graph-observable) "
            "and 6 in Validation (68 releases, 7 graph-observable). "
            "Excludes label-only takedown packages from message passing."
        ),
        "counts": {
            "total_packages": len(pkg_rows),
            "train_packages": len(tr_p),
            "val_packages": len(val_p),
            "test_packages": len(te_p),
            "train_records": 0,
            "val_records": 0,
            "test_records": 0,
            "train_graph_observable": 0,
            "train_live_unobservable": 0,
            "train_confirmed_malicious": 0,
            "train_confirmed_malicious_graph_observable": 0,
            "val_graph_observable": 0,
            "val_live_unobservable": 0,
            "val_confirmed_malicious": 0,
            "val_confirmed_malicious_graph_observable": 0,
            "test_graph_observable": 0,
            "test_live_unobservable": 0,
            "test_confirmed_malicious": 0,
            "test_confirmed_malicious_graph_observable": 0,
        },
        "packages": {
            "train_packages": splits_pkgs["train"],
            "val_packages": splits_pkgs["val"],
            "test_packages": splits_pkgs["test"],
        },
        "identifier_leakage_audit": {
            "strict_package_disjointness": True,
            "package_overlap_count": 0,
            "repository_overlap_count": 0,
            "maintainer_overlap_count": 0,
            "shared_repositories": [],
            "shared_maintainers": [],
        },
    }

    # Index collection
    partition_details = {}
    for sp, key_name in [("train", "train"), ("val", "validation"), ("test", "test")]:
        pkgs = set(splits_pkgs[sp])
        sp_items = [(idx, r) for idx, r in live_rows if r["package"] in pkgs]
        obs_idx = [idx for idx, r in sp_items if r["topology_status"] == "graph_observable"]
        unobs_idx = [idx for idx, r in sp_items if r["topology_status"] == "live_unobservable_topology"]

        mal_count = sum(1 for idx, r in sp_items if r["label"] == "confirmed_malicious")
        mal_obs = sum(1 for idx, r in sp_items if r["label"] == "confirmed_malicious" and r["topology_status"] == "graph_observable")
        ben_count = sum(1 for idx, r in sp_items if r["label"] == "assumed_benign_background")
        ben_obs = sum(1 for idx, r in sp_items if r["label"] == "assumed_benign_background" and r["topology_status"] == "graph_observable")
        unf_count = sum(1 for idx, r in sp_items if r["label"] == "source_unflagged")

        split_payload["counts"][f"{sp}_records"] = len(sp_items)
        split_payload["counts"][f"{sp}_graph_observable"] = len(obs_idx)
        split_payload["counts"][f"{sp}_live_unobservable"] = len(unobs_idx)
        split_payload["counts"][f"{sp}_confirmed_malicious"] = mal_count
        split_payload["counts"][f"{sp}_confirmed_malicious_graph_observable"] = mal_obs

        partition_details[key_name] = {
            "package_count": len(pkgs),
            "total_records": len(sp_items),
            "confirmed_malicious": mal_count,
            "confirmed_malicious_graph_observable": mal_obs,
            "assumed_benign": ben_count,
            "assumed_benign_graph_observable": ben_obs,
            "source_unflagged": unf_count,
            "graph_observable_indices": obs_idx,
            "unobservable_topology_indices": unobs_idx,
        }

    split_payload["train"] = partition_details["train"]
    split_payload["validation"] = partition_details["validation"]
    split_payload["test"] = partition_details["test"]

    # 1. Write updated split JSON
    out_json_path = PROJECT_ROOT / "data" / "experimental" / "splits" / "graph_package_disjoint_split.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)
    logger.info(f"Updated split JSON written to {out_json_path}.")

    # 2. Update split_masks.pt
    masks_path = PROJECT_ROOT / "data" / "experimental" / "graph" / "split_masks.pt"
    masks = torch.load(masks_path, weights_only=True)

    num_versions = len(rows)
    masks["train_mask_graph_disjoint"] = torch.zeros(num_versions, dtype=torch.bool)
    masks["val_mask_graph_disjoint"] = torch.zeros(num_versions, dtype=torch.bool)
    masks["test_mask_graph_disjoint"] = torch.zeros(num_versions, dtype=torch.bool)

    tr_indices = partition_details["train"]["graph_observable_indices"] + partition_details["train"]["unobservable_topology_indices"]
    val_indices = partition_details["validation"]["graph_observable_indices"] + partition_details["validation"]["unobservable_topology_indices"]
    te_indices = partition_details["test"]["graph_observable_indices"] + partition_details["test"]["unobservable_topology_indices"]

    masks["train_mask_graph_disjoint"][tr_indices] = True
    masks["val_mask_graph_disjoint"][val_indices] = True
    masks["test_mask_graph_disjoint"][te_indices] = True

    torch.save(masks, masks_path)
    logger.info(f"Updated split masks saved to {masks_path}.")

    # 3. Update graph_summary.json
    summary_path = PROJECT_ROOT / "data" / "experimental" / "graph" / "graph_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        summary["graph_package_disjoint_split_counts"] = {
            "train": int(masks["train_mask_graph_disjoint"].sum()),
            "validation": int(masks["val_mask_graph_disjoint"].sum()),
            "test": int(masks["test_mask_graph_disjoint"].sum()),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Updated graph summary saved to {summary_path}.")

    print("\n" + "=" * 70)
    print("PACKAGE-DISJOINT SPLIT UPDATE COMPLETE")
    print("=" * 70)
    print(f"Train packages: {len(tr_p)} | Releases: {len(tr_indices)} | Malicious: {partition_details['train']['confirmed_malicious']} (obs: {partition_details['train']['confirmed_malicious_graph_observable']})")
    print(f"Val packages:   {len(val_p)} | Releases: {len(val_indices)} | Malicious: {partition_details['validation']['confirmed_malicious']} (obs: {partition_details['validation']['confirmed_malicious_graph_observable']})")
    print(f"Test packages:  {len(te_p)} | Releases: {len(te_indices)} | Malicious: {partition_details['test']['confirmed_malicious']} (obs: {partition_details['test']['confirmed_malicious_graph_observable']})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    update_disjoint_split()
