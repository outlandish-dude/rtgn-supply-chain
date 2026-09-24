"""
Package Dynamic Temporal Memory Module.

Maintains evolving observed history of packages without static package-ID lookup tables.
Adheres strictly to the methodological principles:
1. No nn.Embedding(num_packages, dim): Memory initializes from a shared learned state vector.
2. Package state represents dynamic observed history, not persistent identity memorization.
3. Explicitly distinguishes Tier C cold-start (0 prior updates) from Tier C online adaptation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PackageMemoryManager(nn.Module):
    """
    Stateful package memory module powered by a Gated Recurrent Unit (GRU).

    Initial state is a shared learned vector across all packages.
    No static package identity is memorized.
    """

    def __init__(
        self,
        num_packages: int,
        memory_dim: int = 64,
        message_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_packages = num_packages
        self.memory_dim = memory_dim
        self.message_dim = message_dim

        # 1. Shared learned initial state (NO nn.Embedding lookup table)
        self.initial_state = nn.Parameter(torch.zeros(memory_dim))
        nn.init.normal_(self.initial_state, std=0.02)

        # 2. Recurrent memory cell
        self.gru_cell = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)

        # 3. Dynamic memory state buffer (non-parameter, reset per epoch/evaluation)
        self.register_buffer("memory", torch.zeros(num_packages, memory_dim))

        # 4. Observability and cold-start tracking
        self.register_buffer("update_count", torch.zeros(num_packages, dtype=torch.long))

        self.reset_memory()

    def reset_memory(self) -> None:
        """Reset all package memories to the shared initial state vector and clear update counts."""
        with torch.no_grad():
            self.memory.copy_(self.initial_state.unsqueeze(0).expand(self.num_packages, -1))
            self.update_count.zero_()

    def get_memory(self, package_ids: torch.Tensor) -> torch.Tensor:
        """
        Query memory state for given package IDs.
        For cold-start packages (0 prior updates), connects directly to self.initial_state
        to enable end-to-end gradient backpropagation.
        """
        is_cold = (self.update_count[package_ids] == 0)
        if is_cold.all():
            return self.initial_state.unsqueeze(0).expand(len(package_ids), -1)

        mem = self.memory[package_ids].clone()
        if is_cold.any():
            cold_idx = torch.where(is_cold)[0]
            mem[cold_idx] = self.initial_state.unsqueeze(0).expand(len(cold_idx), -1)
        return mem

    def is_cold_start(self, package_ids: torch.Tensor) -> torch.Tensor:
        """
        Check whether given packages are evaluated in cold-start mode (0 prior observed releases).
        Returns boolean tensor of shape (len(package_ids), ).
        """
        return self.update_count[package_ids] == 0

    def update_memory(
        self,
        package_ids: torch.Tensor,
        messages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update package memory states via GRUCell based on observed release representation.

        Args:
            package_ids: 1D Tensor of package indices to update.
            messages: Tensor of shape (len(package_ids), message_dim).
        Returns:
            Updated memory tensor of shape (len(package_ids), memory_dim).
        """
        prev_states = self.memory[package_ids]
        new_states = self.gru_cell(messages, prev_states)

        # In-place update of memory buffer
        with torch.no_grad():
            self.memory[package_ids] = new_states.detach()
            self.update_count[package_ids] += 1

        return new_states

    def detach_memory(self) -> None:
        """Detach memory states to prevent gradient backpropagation across long temporal sequences."""
        with torch.no_grad():
            self.memory = self.memory.detach()
