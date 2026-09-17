"""Evaluation metrics for Toroidal Fractal Intelligence."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ToroidalMetrics:
    def __init__(self) -> None:
        self.history: list[dict] = []

    def compute_perplexity(self, logits: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Compute perplexity for logits shaped [B, vocab] and targets [B]."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        target_ids = target_ids.reshape(-1)
        if logits.shape[0] != target_ids.shape[0]:
            raise ValueError(f"logits batch ({logits.shape[0]}) != targets ({target_ids.shape[0]})")
        return float(torch.exp(F.cross_entropy(logits, target_ids)).item())

    def compute_structure_complexity(self, n_atoms: int, n_aggregates: int, n_abstractions: int) -> dict:
        return {
            "n_atoms": n_atoms,
            "n_aggregates": n_aggregates,
            "n_abstractions": n_abstractions,
            "atom_to_aggregate_ratio": n_aggregates / max(n_atoms, 1),
            "structure_density": (n_atoms + n_aggregates) / max(n_abstractions, 1),
        }

    def compute_energy_efficiency(self, n_parameters: int, n_atoms: int, loss: float, flops: int) -> dict:
        return {
            "loss_per_parameter": loss / max(n_parameters, 1),
            "atoms_per_parameter": n_atoms / max(n_parameters, 1),
            "loss_per_flop": loss / max(flops, 1),
            "information_per_flop": (1.0 / max(loss, 1e-6)) / max(flops, 1),
        }

    def log_metrics(self, perplexity: float, n_atoms: int, n_aggregates: int,
                    n_abstractions: int, consolidation_count: int, loss: float,
                    n_parameters: int) -> None:
        self.history.append({
            "perplexity": perplexity,
            **self.compute_structure_complexity(n_atoms, n_aggregates, n_abstractions),
            "consolidation_count": consolidation_count,
            "loss": loss,
            "n_parameters": n_parameters,
            "epoch": len(self.history),
        })

    def get_summary(self) -> dict:
        if not self.history:
            return {}
        keys = ["perplexity", "loss", "n_atoms", "n_aggregates", "n_abstractions"]
        return {
            f"{key}_{suffix}": value
            for key in keys
            for suffix, value in (
                ("mean", np.mean([h[key] for h in self.history])),
                ("std", np.std([h[key] for h in self.history])),
                ("final", self.history[-1][key]),
            )
        }
