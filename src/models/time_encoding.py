"""
Continuous Harmonic Time Encoding for Temporal Edges.

Converts time differences from seconds to days:
    delta_days = max(0.0, delta_seconds) / 86400.0
and computes Fourier/harmonic embeddings:
    Phi_k(delta_days) = [cos(delta_days / beta^(2k / d)), sin(delta_days / beta^(2k / d))]

Prevents feeding raw Unix timestamps or large seconds directly into neural layers.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class HarmonicTimeEncoder(nn.Module):
    """
    Continuous harmonic / Fourier temporal encoder for elapsed days.
    """

    def __init__(
        self,
        dimension: int = 32,
        max_period_days: float = 365.25 * 10.0,  # 10 years
        min_period_days: float = 1.0 / 24.0,     # 1 hour
    ) -> None:
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dimension}")

        self.dimension = dimension
        self.num_frequencies = dimension // 2

        # Log-spaced frequencies from min_period to max_period
        log_min = math.log(2.0 * math.pi / max_period_days)
        log_max = math.log(2.0 * math.pi / min_period_days)
        freqs = torch.exp(torch.linspace(log_min, log_max, self.num_frequencies))
        self.register_buffer("frequencies", freqs)

    def forward(self, delta_seconds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_seconds: Tensor of shape (..., ) representing elapsed seconds (must be >= 0).
        Returns:
            Tensor of shape (..., dimension) representing harmonic embedding of elapsed days.
        """
        # 1. Strict conversion: delta_seconds -> delta_days
        delta_days = torch.clamp(delta_seconds, min=0.0) / 86400.0

        # 2. Outer product: delta_days * frequencies
        angles = delta_days.unsqueeze(-1) * self.frequencies.unsqueeze(0)

        # 3. Concatenate cos and sin
        cos_part = torch.cos(angles)
        sin_part = torch.sin(angles)

        return torch.cat([cos_part, sin_part], dim=-1)
