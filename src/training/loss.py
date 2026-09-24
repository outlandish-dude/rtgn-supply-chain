"""
Loss Function and Imbalance Weighting for RTGN.

Implements:
- Package-balanced weighting combined with effective class imbalance weighting:
    W_pos = sum_{v in Train, y_v=1} w_pkg(v)
    W_neg = sum_{v in Train, y_v=0} w_pkg(v)
    effective_pos_weight = W_neg / W_pos
- Logs raw class counts, effective positive weight, weighted masses, and per-package totals.
- Avoids double-compensation of class imbalance and package concentration.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PackageBalancedLoss(nn.Module):
    """
    Weighted Binary Cross-Entropy with mathematically grounded effective positive weighting.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps
        self.effective_pos_weight: float = 1.0
        self.raw_pos_count: int = 0
        self.raw_neg_count: int = 0
        self.total_weighted_pos_mass: float = 0.0
        self.total_weighted_neg_mass: float = 0.0
        self.raw_ignored_count: int = 0
        self.is_initialized: bool = False

    def compute_weights(
        self,
        y_train: torch.Tensor,
        sample_weights: torch.Tensor,
        supervised_mask: Optional[torch.Tensor] = None,
        package_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute effective positive weight from training split weighted class totals.

        Strictly restricts supervision to:
        - confirmed_malicious (y=1)
        - assumed_benign_background (y=0)
        source_unflagged (y=-1 or unflagged/unlabeled) are explicitly excluded from supervised loss.

        Args:
            y_train: 1D Tensor of binary labels (0, 1, or -1 for unflagged).
            sample_weights: 1D Tensor of package sample weights.
            supervised_mask: Optional boolean mask indicating supervised targets (y in {0, 1}).
            package_ids: Optional 1D Tensor of package indices for per-package audit.
        """
        if supervised_mask is not None:
            is_pos = (y_train == 1) & supervised_mask
            is_neg = (y_train == 0) & supervised_mask
            is_ignored = ~supervised_mask
        else:
            is_pos = (y_train == 1)
            is_neg = (y_train == 0)
            is_ignored = ~is_pos & ~is_neg

        self.raw_pos_count = int(torch.sum(is_pos).item())
        self.raw_neg_count = int(torch.sum(is_neg).item())
        self.raw_ignored_count = int(torch.sum(is_ignored).item())

        pos_weights = sample_weights[is_pos]
        neg_weights = sample_weights[is_neg]

        self.total_weighted_pos_mass = float(torch.sum(pos_weights).item())
        self.total_weighted_neg_mass = float(torch.sum(neg_weights).item())

        if self.total_weighted_pos_mass > 0:
            self.effective_pos_weight = self.total_weighted_neg_mass / self.total_weighted_pos_mass
        else:
            self.effective_pos_weight = 1.0

        stats = {
            "raw_pos_count": self.raw_pos_count,
            "raw_neg_count": self.raw_neg_count,
            "raw_ignored_count": self.raw_ignored_count,
            "total_weighted_pos_mass": round(self.total_weighted_pos_mass, 4),
            "total_weighted_neg_mass": round(self.total_weighted_neg_mass, 4),
            "effective_pos_weight": round(self.effective_pos_weight, 4),
        }

        # Optional per-package mass audit
        if package_ids is not None:
            pkg_mass = {}
            for pid in torch.unique(package_ids):
                mask = (package_ids == pid) & (is_pos | is_neg)
                if mask.any():
                    pkg_mass[int(pid.item())] = round(float(sample_weights[mask].sum().item()), 4)
            stats["package_count"] = len(pkg_mass)

        self.is_initialized = True
        logger.info(
            f"PackageBalancedLoss initialized: Supervised (pos={self.raw_pos_count}, neg={self.raw_neg_count}), "
            f"Ignored/Unflagged={self.raw_ignored_count} | "
            f"Weighted Mass (pos={self.total_weighted_pos_mass:.2f}, neg={self.total_weighted_neg_mass:.2f}) | "
            f"Effective Pos Weight={self.effective_pos_weight:.4f}"
        )
        return stats

    def get_stats(self) -> Dict[str, float]:
        """Return dict of current loss and weight parameters."""
        return {
            "raw_pos_count": self.raw_pos_count,
            "raw_neg_count": self.raw_neg_count,
            "raw_ignored_count": self.raw_ignored_count,
            "total_weighted_pos_mass": round(self.total_weighted_pos_mass, 4),
            "total_weighted_neg_mass": round(self.total_weighted_neg_mass, 4),
            "effective_pos_weight": round(self.effective_pos_weight, 4),
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: 1D Tensor of unnormalized predictions.
            targets: 1D Tensor of binary labels (0, 1, or -1 for unflagged).
            sample_weights: Optional 1D Tensor of sample weights (if None, uniform).
            supervised_mask: Optional 1D boolean Tensor masking only supervised targets.
        Returns:
            Scalar loss tensor computed strictly over supervised nodes.
        """
        if not self.is_initialized:
            raise RuntimeError("compute_weights must be called on training set before forward pass.")

        if supervised_mask is not None:
            valid_logits = logits[supervised_mask]
            valid_targets = targets[supervised_mask]
            valid_weights = sample_weights[supervised_mask] if sample_weights is not None else None
        else:
            # Auto-exclude non-binary targets (e.g. source_unflagged marked with -1 or NaN)
            valid = (targets == 0) | (targets == 1)
            valid_logits = logits[valid]
            valid_targets = targets[valid]
            valid_weights = sample_weights[valid] if sample_weights is not None else None

        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs = torch.sigmoid(valid_logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        pos_loss = -self.effective_pos_weight * valid_targets * torch.log(probs)
        neg_loss = -(1.0 - valid_targets) * torch.log(1.0 - probs)
        bce_loss = pos_loss + neg_loss

        if valid_weights is not None:
            weighted_bce = bce_loss * valid_weights
            return torch.sum(weighted_bce) / (torch.sum(valid_weights) + self.eps)
        else:
            return torch.mean(bce_loss)
