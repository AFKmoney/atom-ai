"""Dynamic consolidation of stable toroidal structures into persistent memory."""
from __future__ import annotations
import torch
import torch.nn as nn


class ConsolidationEngine(nn.Module):
    def __init__(self, d_model: int = 256, consolidation_threshold: float = 0.7,
                 persistence_decay: float = 0.995) -> None:
        super().__init__()
        self.d_model = d_model
        self.consolidation_threshold = consolidation_threshold
        self.persistence_decay = persistence_decay
        self.register_buffer("persistence", torch.zeros(d_model))
        self.register_buffer("persistence_strength", torch.tensor(0.0))

    def compute_stability(self, E: torch.Tensor, history: list[torch.Tensor]) -> torch.Tensor:
        if not history:
            return torch.full_like(E, 0.5)
        hist = torch.stack(history[-10:], dim=0)
        mean = hist.mean(dim=0)
        std = hist.std(dim=0, unbiased=False)
        # Bounded stability: consistent, non-zero energy is stable.
        cv = std / mean.abs().clamp_min(1e-6)
        return torch.exp(-cv).clamp(0, 1) * E.clamp(0, 1)

    def select_for_consolidation(self, E: torch.Tensor, stability: torch.Tensor) -> torch.Tensor:
        score = E.clamp(0, 1) * stability.clamp(0, 1)
        return score > self.consolidation_threshold

    def consolidate(self, alpha: torch.Tensor, E: torch.Tensor, stability: torch.Tensor):
        alpha_flat = alpha.reshape(-1, self.d_model)
        E_flat = E.reshape(-1, self.d_model).clamp(0, 1)
        stability_flat = stability.reshape(-1, self.d_model).clamp(0, 1)
        mask = self.select_for_consolidation(E_flat, stability_flat)

        if mask.any():
            selected = alpha_flat[mask]
            incoming = selected.mean(dim=0)
            strength = mask.float().mean()
            with torch.no_grad():
                self.persistence.mul_(self.persistence_decay).add_((1 - self.persistence_decay) * incoming)
                self.persistence_strength.mul_(self.persistence_decay).add_((1 - self.persistence_decay) * strength)
            persistent = self.persistence.unsqueeze(0)
        else:
            persistent = self.persistence.unsqueeze(0) if self.persistence_strength.item() > 0 else None
        return persistent, mask.any(dim=-1)

    def get_persistent_state(self) -> torch.Tensor:
        return self.persistence.detach()
